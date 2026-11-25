"""Batch-one execution over the SDiT manuscript policy.

The legacy manager is process-global. Use a separate process for each concurrent
worker; never share this runner between threads.
"""
from copy import deepcopy
import importlib
import time

import torch
from diffusers import HunyuanDiTPipeline, LuminaPipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args_from_dict
from .config import MODEL_IDS, InferenceConfig


class Runner:
    def __init__(self, config: InferenceConfig, model_path=None, device="cuda:0", local_files_only=False):
        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Inference requires an NVIDIA CUDA GPU")
        torch.cuda.set_device(self.device)
        self.model_id = str(model_path or MODEL_IDS[config.model])
        cls = LuminaPipeline if config.model == "lumina" else HunyuanDiTPipeline
        self.dtype = torch.bfloat16 if config.model == "lumina" else torch.float16
        self.pipe = cls.from_pretrained(self.model_id, torch_dtype=self.dtype,
                                       local_files_only=local_files_only).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.original_scheduler_config = deepcopy(dict(self.pipe.scheduler.config))
        self.tea = None
        if config.method == "teacache":
            path = "lumina_next_t2i" if config.model == "lumina" else "hunyuan_dit"
            self.tea = importlib.import_module(f"ras.utils.{path}.update_pipeline_{config.model}_teacache")
            getattr(self.tea, f"update_{config.model}_pipeline_teacache")(self.pipe)
        elif config.method != "dense":
            if config.model == "lumina":
                from ras.utils.lumina_next_t2i.update_pipeline_lumina import update_lumina_pipeline
                update_lumina_pipeline(self.pipe)
            else:
                from ras.utils.hunyuan_dit.update_pipeline_hunyuan import update_hunyuan_pipeline
                update_hunyuan_pipeline(self.pipe)
        if config.model == "lumina":
            self.pipe.transformer.to(memory_format=torch.channels_last)
            self.pipe.vae.to(memory_format=torch.channels_last)
        self.pipe.transformer.eval()
        self.pipe.vae.eval()

    def reset(self):
        c = self.config
        # Recreate process-global state and scheduler for every image request.
        ras_manager.MANAGER = ras_manager.ras_manager()
        m = ras_manager.MANAGER
        if c.method == "dense":
            return {"native_pipeline": True}
        if c.method == "teacache":
            getattr(self.tea, f"reset_{c.model}_teacache")(
                self.pipe, num_inference_steps=c.steps,
                rel_l1_thresh=c.teacache_threshold, enable_teacache=True)
            return {"teacache_threshold": c.teacache_threshold}
        sparse = c.method == "sdit"
        effective = dict(
            height=c.height, width=c.width, num_inference_steps=c.steps,
            scheduler_start_step=c.warmup, scheduler_end_step=c.steps,
            sample_ratio=c.rho, cluster_selection_ratio=c.rho, patch_size=2,
            error_reset_steps="", metric="std" if c.method == "ras" else c.metric,
            starvation_scale=c.starvation_scale, neighbor_expansion_mode="0",
            use_cluster_adaptive=c.method != "ras", use_cluster_graph_attention=sparse,
            cluster_graph_attention_hops=c.hops,
            enable_temporal_selective_denoising=True, enable_region_sparse_attention=sparse,
            cluster_attention_backend="grouped_sdpa" if sparse else "sdpa",
            use_cluster_remote_anchors=sparse and c.anchors > 0,
            cluster_remote_anchor_count=c.anchors if sparse else 0,
            cluster_remote_anchor_weight_mode="logn", cluster_remote_anchor_scale_target="v",
            enable_cluster_remote_anchor_value_coreset=True,
            enable_cluster_remote_anchor_incremental_value_cache=True,
            enable_cluster_remote_anchor_merged_key_fast_path=False,
            enable_cluster_remote_anchor_bias_feature_fast_path=False,
            enable_cluster_fused_pack=False,
            cluster_attention_kernel="sdpa",
            enable_lumina_text_cache=False, enable_cluster_grouped_sdpa=sparse,
            enable_cluster_fused_triton_attention=False, enable_module_timing=c.analysis,
            enable_cluster_attention_density=c.analysis, cluster_prediction_source="x0_pred",
            cluster_relevance_topk_ratio=0.0, recluster_every_step=False,
            recluster_every_n_steps=0, final_sync_steps=0,
        )
        args = parse_args_from_dict(effective)
        m.set_parameters(args)
        m.use_cluster_concat_attention = True
        for module in self.pipe.transformer.modules():
            processor = getattr(module, "processor", None)
            if processor is None:
                continue
            for key in ("k_cache", "v_cache"):
                if hasattr(processor, key):
                    setattr(processor, key, None)
            if hasattr(processor, "cache_in_cluster_order"):
                processor.cache_in_cluster_order = False
            reset = getattr(processor, "_reset_remote_anchor_value_cache", None)
            if reset:
                reset()
        if c.method == "ras":
            from ras.schedulers import RASFlowMatchEulerDiscreteScheduler
            self.pipe.scheduler = RASFlowMatchEulerDiscreteScheduler.from_config(self.original_scheduler_config)
        else:
            from ras.schedulers.cluster_adaptive_scheduler import ClusterAdaptiveScheduler
            from ras.schedulers.cluster_adaptive_ddpm_scheduler import ClusterAdaptiveDDPMScheduler
            cls = ClusterAdaptiveScheduler if c.model == "lumina" else ClusterAdaptiveDDPMScheduler
            self.pipe.scheduler = cls.from_config(
                self.original_scheduler_config, segmentation_algorithm="quickshift",
                num_clusters=16, cluster_selection_ratio=c.rho,
                recluster_every_step=False, recluster_every_n_steps=0, final_sync_steps=0,
                use_momentum_prediction=False, quickshift_kernel_size=c.quickshift_kernel_size,
                quickshift_max_dist=c.quickshift_max_dist, quickshift_ratio=c.quickshift_ratio,
            )
            self.pipe.scheduler.defer_dense_cluster_mask = False
        return vars(args)

    @torch.inference_mode()
    def generate(self, prompt, seed, negative_prompt=""):
        effective = self.reset()
        c = self.config
        kwargs = dict(prompt=prompt, negative_prompt=negative_prompt,
                      generator=torch.Generator(device=self.device).manual_seed(seed),
                      num_inference_steps=c.steps, height=c.height, width=c.width,
                      guidance_scale=c.guidance_scale, output_type="pil")
        if c.model == "hunyuan":
            kwargs["use_resolution_binning"] = True
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        image = self.pipe(**kwargs).images[0]
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        record = {
            "wall_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "effective_config": effective,
            "actual_image_size": list(image.size),
            "backend_report": ras_manager.MANAGER.get_cluster_backend_report(),
            "scheduler_update_history": list(getattr(self.pipe.scheduler, "step_update_history", [])),
        }
        if c.analysis:
            record["analysis"] = {
                "density": ras_manager.MANAGER.get_cluster_attention_density_report(),
                "attention_core": ras_manager.MANAGER.get_attention_core_timing_report(),
                "modules": ras_manager.MANAGER.get_module_timing_report(),
                "reorder": ras_manager.MANAGER.get_cluster_reorder_timing_report(),
            }
        if self.tea:
            record["teacache_stats"] = getattr(self.tea, f"get_{c.model}_teacache_stats")(self.pipe)
        return image, record
