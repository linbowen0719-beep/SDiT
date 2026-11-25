from typing import Optional, Tuple, Union

import numpy as np
import torch
from diffusers.schedulers import DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput

from ras.utils import ras_manager
from .cluster_adaptive_scheduler import ClusterAdaptiveScheduler


class ClusterAdaptiveDDPMScheduler(DDPMScheduler, ClusterAdaptiveScheduler):
    """DDPM scheduler wrapper with cluster-adaptive RAS bookkeeping for HunyuanDiT."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, list]] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        num_clusters: int = 16,
        cluster_selection_ratio: float = 0.5,
        recluster_every_step: bool = True,
        recluster_every_n_steps: int = 0,
        final_sync_steps: int = 3,
        segmentation_algorithm: str = "quickshift",
        use_momentum_prediction: bool = False,
        save_x_pred: bool = False,
        x_pred_save_steps: list = None,
        x_pred_output_dir: str = None,
        vae_model: any = None,
        quickshift_kernel_size: float = 2.3,
        quickshift_max_dist: float = 6.0,
        quickshift_ratio: float = 0.5,
        enable_partition_metadata_cache: bool = True,
        enable_cpu_graph_metadata: bool = True,
        enable_host_selection_schedule: bool = True,
        defer_dense_cluster_mask: bool = False,
        **kwargs,
    ):
        DDPMScheduler.__init__(
            self,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            variance_type=variance_type,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            clip_sample_range=clip_sample_range,
            sample_max_value=sample_max_value,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            **kwargs,
        )

        self.segmentation_algorithm = segmentation_algorithm
        self.num_clusters = num_clusters
        self.top_k_clusters = max(1, int(num_clusters * cluster_selection_ratio))
        self.quickshift_kernel_size = quickshift_kernel_size
        self.quickshift_max_dist = quickshift_max_dist
        self.quickshift_ratio = quickshift_ratio
        self.cluster_labels = None
        self.cluster_drop_cnt = None
        self.clustered = False
        self.actual_num_clusters = None

        self.cluster_selection_ratio = cluster_selection_ratio
        self.recluster_every_step = recluster_every_step
        self.recluster_every_n_steps = int(recluster_every_n_steps or 0)
        self.last_recluster_step = None
        self.final_sync_steps = final_sync_steps
        self.use_momentum_prediction = use_momentum_prediction
        # The shared selection implementation uses these fields even though
        # this wrapper initializes the DDPM parent rather than the flow parent.
        self.enable_partition_metadata_cache = bool(enable_partition_metadata_cache)
        self.enable_cpu_graph_metadata = bool(enable_cpu_graph_metadata)
        self.enable_host_selection_schedule = bool(enable_host_selection_schedule)
        self.defer_dense_cluster_mask = bool(defer_dense_cluster_mask)
        self._active_selection_cpu = None
        self._cached_selection_cpu = None
        self._partition_metadata_cache = None
        self._cluster_mask_density_cache = None
        self._cluster_far_ids_cpu_cache = None

        self.save_x_pred = save_x_pred
        self.x_pred_save_steps = x_pred_save_steps if x_pred_save_steps is not None else []
        self.x_pred_output_dir = x_pred_output_dir if x_pred_output_dir is not None else "outputs/debug/predictions"
        self.vae_model = vae_model

        self.total_patches = 0
        self.cumulative_updated_patches = 0
        self.step_update_history = []
        self.cluster_update_counts: Optional[np.ndarray] = None

        self.clustering_times = []
        self.metric_compute_times = []
        self.kmeans_times = []
        self.selection_times = []
        self.total_clustering_time = 0.0
        self.attn_mask_density_history = []
        self.window_area_ratio_history = []

        self.cached_noise = None
        self.cached_index_signature = None
        self.cached_region_weights = None
        self.prev_delta_sigma = None
        self.drop_cnt = None
        self._ras_step_index = 0

        print("✅ ClusterAdaptiveDDPMScheduler初始化完成:")
        print(f"   分割算法: {self.segmentation_algorithm}")
        print(f"   聚类数量: {self.num_clusters}")
        print(f"   选择比例: {self.cluster_selection_ratio:.0%}")
        print(f"   邻域扩展模式: {self._get_expansion_mode_desc()}")
        print(f"   每步重新聚类: {self.recluster_every_step}")
        print(f"   最后{self.final_sync_steps}步: 禁用adaptive，全部同步更新")
        print("   噪声复用策略: 直接复用 epsilon")

    def set_timesteps(self, *args, **kwargs):
        result = DDPMScheduler.set_timesteps(self, *args, **kwargs)
        self._ras_step_index = 0
        return result

    def _init_ras_config(self, latents):
        num_patches = (
            latents.shape[-2] // ras_manager.MANAGER.patch_size
            * latents.shape[-1] // ras_manager.MANAGER.patch_size
        )
        step_count = len(getattr(self, "timesteps", [])) or int(getattr(ras_manager.MANAGER, "num_steps", 0))
        self.drop_cnt = torch.zeros(num_patches, device=latents.device) - step_count
        self.cluster_drop_cnt = torch.zeros(self.num_clusters, device=latents.device)
        self.clustered = False

    def _predicted_clean_sample(self, model_output, timestep, sample):
        """Convert native model output to a clean latent for partitioning and scoring."""
        timestep_int = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep_int].to(
            device=sample.device, dtype=torch.float32
        )
        beta_prod_t = 1 - alpha_prod_t
        sample_float = sample.to(torch.float32)
        output_float = model_output.to(torch.float32)
        prediction_type = self.config.prediction_type
        if prediction_type == "epsilon":
            return (sample_float - beta_prod_t.sqrt() * output_float) / alpha_prod_t.sqrt()
        if prediction_type == "sample":
            return output_float
        if prediction_type == "v_prediction":
            return alpha_prod_t.sqrt() * sample_float - beta_prod_t.sqrt() * output_float
        raise ValueError(f"Unsupported prediction_type for clean analysis: {prediction_type}")

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        if self.drop_cnt is None or self._ras_step_index == 0:
            self._init_ras_config(sample)

        if self._ras_step_index == 0:
            ras_manager.MANAGER.reset_cache()
            ras_manager.MANAGER.reset_cluster_backend_stats()
            self.clustered = False
            self.last_recluster_step = None
            self._clear_cached_history()
            self.cluster_update_counts = None

        latent_dim, height, width = sample.shape[-3:]
        assert 0.0 < ras_manager.MANAGER.sample_ratio <= 1.0

        if ras_manager.MANAGER.temporal_selection_enabled() and ras_manager.MANAGER.is_RAS_step:
            flat_output = model_output.squeeze(0).view(latent_dim, -1)
            cached_indices = ras_manager.MANAGER.cached_index
            cached_noise = ras_manager.MANAGER.cached_scaled_noise
            if cached_indices is not None and cached_noise is not None:
                indices_empty = (
                    cached_indices.numel() == 0
                    if isinstance(cached_indices, torch.Tensor)
                    else len(cached_indices) == 0
                )
                if not indices_empty:
                    flat_output[:, cached_indices] = cached_noise
            model_output = flat_output.view(latent_dim, height, width).unsqueeze(0)

        sample_before_step = sample
        current_step = ras_manager.MANAGER.current_step
        clean_prediction = self._predicted_clean_sample(model_output, timestep, sample_before_step)
        prev_sample = DDPMScheduler.step(
            self,
            model_output,
            timestep,
            sample,
            generator=generator,
            return_dict=False,
        )[0]

        sample_float = sample_before_step.to(torch.float32)
        prev_float = prev_sample.to(torch.float32)
        diff = prev_float - sample_float
        prediction_source = getattr(ras_manager.MANAGER, "cluster_prediction_source", "prev_sample").strip().lower()
        if prediction_source in ("legacy", "prev_sample", "x_next", "next"):
            cluster_prediction = prev_float
        elif prediction_source in ("sample", "current", "xt"):
            cluster_prediction = sample_float
        elif prediction_source in ("x0_pred", "x0", "clean"):
            cluster_prediction = clean_prediction
        else:
            raise ValueError(
                f"Unsupported cluster_prediction_source={prediction_source!r}; "
                "expected 'prev_sample', 'sample', or 'x0_pred'"
            )

        analysis_callback = ras_manager.MANAGER.scheduler_analysis_callback
        if analysis_callback is not None:
            analysis_callback({
                "model": "hunyuan",
                "step": int(current_step),
                "timestep": timestep,
                "sample": sample_float.detach(),
                "clean_prediction": clean_prediction.detach(),
                "model_output": model_output.detach(),
                "prev_sample": prev_sample.detach(),
            })

        self._save_sample_tensor(sample_float, current_step)
        remaining_steps = ras_manager.MANAGER.num_steps - ras_manager.MANAGER.current_step

        temporal_enabled = ras_manager.MANAGER.temporal_selection_enabled()
        spatial_enabled = ras_manager.MANAGER.spatial_attention_enabled()
        if temporal_enabled and ras_manager.MANAGER.is_next_RAS_step:
            if remaining_steps <= self.final_sync_steps:
                print(f"   ⚠️  最后{remaining_steps}步：禁用adaptive，全部同步更新")
                ras_manager.MANAGER.sample_ratio = 1.0
                self._clear_cached_history()
                ras_manager.MANAGER.cached_index = None
                ras_manager.MANAGER.cached_scaled_noise = None
            else:
                latent_cached_indices, other_patchified_indices = self.cluster_adaptive_selection(
                    sample_float,
                    cluster_prediction,
                    height,
                    width,
                    diff=diff,
                )
                flattened_output = model_output.squeeze(0).view(latent_dim, -1)
                new_cached_noise = None
                if latent_cached_indices is not None:
                    if isinstance(latent_cached_indices, torch.Tensor) and latent_cached_indices.numel() > 0:
                        new_cached_noise = flattened_output[:, latent_cached_indices]
                    elif hasattr(latent_cached_indices, "__len__") and len(latent_cached_indices) > 0:
                        new_cached_noise = flattened_output[:, latent_cached_indices]

                self._update_cached_noise(latent_cached_indices, new_cached_noise, 1.0)

                ras_manager.MANAGER.set_active_patch_indices(other_patchified_indices)

                num_updated = len(other_patchified_indices) if other_patchified_indices is not None else 0
                total_patches = (height // ras_manager.MANAGER.patch_size) * (width // ras_manager.MANAGER.patch_size)
                self.update_patch_statistics(num_updated, total_patches)
        elif spatial_enabled and ras_manager.MANAGER.is_next_RAS_step:
            should_update_layout = (
                not self.clustered
                or self.recluster_every_step
                or (
                    self.recluster_every_n_steps > 0
                    and self.last_recluster_step is not None
                    and current_step - self.last_recluster_step >= self.recluster_every_n_steps
                )
            )
            if should_update_layout:
                self.cluster_adaptive_selection(
                    sample_float, cluster_prediction, height, width, diff=diff
                )
            num_patches = (height // ras_manager.MANAGER.patch_size) * (
                width // ras_manager.MANAGER.patch_size
            )
            ras_manager.MANAGER.cached_scaled_noise = None
            ras_manager.MANAGER.set_all_patches_active(num_patches, sample.device)
            self._clear_cached_history()
            self.update_patch_statistics(num_patches, num_patches)
        else:
            total_patches = (height // ras_manager.MANAGER.patch_size) * (width // ras_manager.MANAGER.patch_size)
            self.update_patch_statistics(total_patches, total_patches)

        if ras_manager.MANAGER.current_step in ras_manager.MANAGER.error_reset_steps:
            self.clustered = False
            self.last_recluster_step = None
            print(f"   🔄 Sync step {ras_manager.MANAGER.current_step} 完成，下一步将重新聚类")

        self._ras_step_index += 1
        ras_manager.MANAGER.increase_step()
        if ras_manager.MANAGER.current_step >= ras_manager.MANAGER.num_steps:
            ras_manager.MANAGER.reset_cache()
            self.clustered = False
            self.last_recluster_step = None
            self._clear_cached_history()
            self.print_statistics_summary()

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)
