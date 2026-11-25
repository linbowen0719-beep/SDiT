import time
import torch


class ras_manager:
    def __init__(self):
        self.patch_size = 2
        self.scheduler_start_step = 4
        self.scheduler_end_step = 30
        self.metric = "std"
        self.error_reset_steps = [12, 22]
        self.sample_ratio = 0.5
        self.starvation_scale = 0.1
        self.vae_size = 8
        self.high_ratio = 1
        self.num_steps = 30
        self.current_step = 0
        self.current_layer = 0
        self.enable_index_fusion = False
        self.is_RAS_step = False
        self.is_next_RAS_step = False

        self.cached_index = None
        self.other_index = None
        self.cached_patchified_index = None
        self.other_patchified_index = None
        self.image_rotary_emb_skip = None
        self.cached_scaled_noise = None
        self.skip_token_num_list = []

        # ============ Adaptive Complexity Fields ============
        self.use_adaptive_complexity = False
        self.complexity_threshold = 0.5
        self.syn_timesteps = None
        self.custom_timesteps = None
        self.current_syn_index = 0
        self.patch_strategies = {}  # Dict[patch_id, str]: 'a' or 'b'
        self.complexity_scores = None  # Tensor of complexity scores per patch

        # ============ Complexity Source Selection ============
        # 选择用于计算复杂度的来源: 'diff' (残差) 或 'x_pred' (预测样本)
        self.complexity_source = "x_pred"  # default: x_pred

        # ============ Neighbor Expansion Mode ============
        # 邻域扩展模式: '0' (无扩展), '4' (4-邻域), '8' (8-邻域，默认)
        self.neighbor_expansion_mode = "8"  # default: 8-neighbor

        # ============ Cluster Graph Attention ============
        self.use_cluster_graph_attention = False
        self.use_cluster_concat_attention = True
        self.use_cluster_global_local_head_split = False
        self.cluster_global_head_count = 4
        self.cluster_global_head_selection = "online"
        self.cluster_full_kv_attention_until_step = -1
        self.cluster_prediction_source = "legacy"
        self.use_cluster_remote_anchors = True
        self.enable_cluster_relevance_routing = True
        self.cluster_relevance_topk_ratio = 0.0
        self.cluster_relevance_use_size_bias = False
        self.cluster_relevance_head_reduce = "mean"
        self.enable_cluster_relevance_optimizations = False
        self.cluster_relevance_refresh_every_n_layers = 4
        self.cluster_relevance_reuse_within_step = True
        self.cluster_remote_anchor_count = 4
        self.cluster_remote_anchor_weight_mode = "none"
        self.cluster_remote_anchor_scale_target = "v"
        self.enable_cluster_remote_anchor_value_coreset = False
        self.enable_cluster_remote_anchor_incremental_value_cache = False
        self.enable_cluster_remote_anchor_merged_key_fast_path = False
        self.enable_cluster_remote_anchor_bias_feature_fast_path = False
        self.enable_cluster_late_kv_expansion = True
        self.lumina_layer_mask = None
        self.lumina_layer_mask_all_true = None
        self.enable_lumina_fused_active_io = True
        self.enable_lumina_text_cache = False
        self.lumina_text_cache_run_token = None
        self.lumina_text_cache_immutable_context = False
        self.lumina_text_cache_immutable_model = False
        self.lumina_text_cache_model_plan = None
        self.lumina_text_cache_scope = None
        self.lumina_text_cache_state = None
        self.cluster_attention_kernel = "sdpa"
        self.enable_cluster_fused_pack = False
        self.enable_cluster_compact_pack = True
        self.enable_cluster_fused_anchor_refresh = True
        self.enable_cluster_static_pack_sources = True
        self.enable_cluster_fused_kv_refresh = True
        self.cluster_pack_cache = None
        self.cluster_static_pack_cache = None
        self.cluster_pack_metadata_cpu = None
        self._cluster_host_selection_cache = None
        self.enable_cluster_grouped_sdpa = False
        self.cluster_grouped_sdpa_max_padding_ratio = 1.25
        self.enable_cluster_anchor_diagnostics = False
        self.cluster_remote_anchor_selection_budget_ms = 0.0
        self.cluster_remote_anchor_logit_bias_alpha = 1.0
        self.cluster_graph_attention_hops = 2
        self.cluster_attention_mask = None   # bool tensor [num_patches, num_patches], used for RAS steps
        self.cluster_attention_mask_factory = None
        self.cluster_attention_mask_density = None
        self.cluster_window_size = -1        # kept for statistics only
        self.cluster_key_indices = None      # dict {cluster_id: LongTensor of key patch indices}
        self.cluster_far_cluster_ids = None  # dict {cluster_id: LongTensor of far cluster ids}
        self.cluster_ids_tensor = None       # LongTensor [num_clusters], sorted cluster ids
        self.cluster_label_to_compact = None # LongTensor lookup: cluster id -> compact row/col
        self.cluster_block_starts = None     # LongTensor [num_clusters], cluster-order token block starts
        self.cluster_block_ends = None       # LongTensor [num_clusters], cluster-order token block ends
        self.cluster_block_lengths = None    # LongTensor [num_clusters], cluster-order token block lengths
        self.cluster_max_block_length = 0
        self.cluster_compact_labels = None   # LongTensor [num_patches], compact cluster id per ordered token
        self.cluster_token_indices_padded = None # LongTensor [num_clusters, max_cluster_tokens]
        self.cluster_token_indices_mask = None   # BoolTensor [num_clusters, max_cluster_tokens]
        self.cluster_local_cluster_mask = None   # BoolTensor [num_clusters, num_clusters], local graph mask
        self.cluster_active_queries_sorted = False
        self.cluster_active_query_ranges_cpu = None
        self.cluster_active_cluster_ids = None
        self.cluster_active_query_starts = None
        self.cluster_active_query_lengths = None
        self.cluster_local_block_indices_padded = None  # LongTensor [num_clusters, max_local_blocks]
        self.cluster_local_block_counts = None          # LongTensor [num_clusters]
        self.cluster_local_block_max_count = 0
        self.cluster_local_key_spans = None             # dict {cluster_id: (start, end)} for contiguous local KV
        self.cluster_local_key_span_starts = None       # LongTensor [num_clusters], -1 if unavailable
        self.cluster_local_key_span_lengths = None      # LongTensor [num_clusters]
        self.enable_cluster_fused_triton_attention = False
        self.cluster_relevance_cache_step = -1
        self.cluster_relevance_cache_layer = -1
        self.cluster_relevance_cache_active_count = -1
        self.cluster_relevance_cache_remote_ids = None
        self.cluster_relevance_cache_remote_counts = None
        self.cluster_relevance_cache_cluster_ids = None
        self.cluster_remote_anchor_indices = None  # dict {cluster_id: LongTensor of remote anchor indices}
        self.cluster_remote_anchor_weights = None  # dict {cluster_id: FloatTensor of remote anchor weights}
        self.cluster_remote_anchor_group_indices = None  # dict {cluster_id: LongTensor [anchors, group_tokens]}
        self.cluster_remote_anchor_group_mask = None     # dict {cluster_id: BoolTensor [anchors, group_tokens]}
        self.cluster_remote_anchor_source_cluster_ids = None  # dict {query_cluster_id: tuple of source cluster ids}
        self.cluster_remote_anchor_source_indices = None      # dict {source_cluster_id: LongTensor of anchor indices}
        self.cluster_remote_anchor_source_weights = None      # dict {source_cluster_id: FloatTensor of anchor weights}
        self.cluster_remote_anchor_source_group_indices = None
        self.cluster_remote_anchor_source_group_mask = None
        self.cluster_remote_anchor_all_indices = None         # packed source anchors across all clusters
        self.cluster_remote_anchor_all_weights = None
        self.cluster_remote_anchor_all_group_indices = None
        self.cluster_remote_anchor_all_group_mask = None
        self.cluster_remote_anchor_packed_positions = None    # dict {query_cluster_id: LongTensor into packed anchors}
        self.cluster_remote_anchor_merged_key_indices = None  # dict {query_cluster_id: local key indices + remote anchor key indices}
        self.cluster_remote_anchor_token_to_packed_anchor = None
        self.cluster_remote_anchor_packed_counts = None
        self.cluster_remote_anchor_group_token_order = None
        self.cluster_remote_anchor_group_lengths = None
        self.cluster_remote_anchor_group_token_order_is_cluster_order = False
        self.cluster_remote_anchor_all_logit_bias = None  # precomputed logn bias for all packed anchors
        self.cluster_remote_anchor_metadata_version = 0
        self.cluster_remote_anchor_value_update_plan = None
        self.cluster_key_padding_mask_all_true = None
        self.cluster_key_padding_mask_numel = -1
        self.cluster_labels_flat = None      # LongTensor [num_patches], cluster id per patch
        self.cluster_perm = None             # LongTensor [num_patches], ordered_idx -> original_idx
        self.cluster_inv_perm = None         # LongTensor [num_patches], original_idx -> ordered_idx
        self.cluster_sorted_labels = None    # LongTensor [num_patches], labels in cluster-ordered coords
        self.cluster_block_ranges = None     # dict {cluster_id: (start, end)} in ordered coords

        # ============ Cluster Graph Attention Density Stats ============
        # Analysis only: reductions and diagnostic mask gathers are excluded
        # from ordinary inference and latency measurements.
        self.enable_cluster_attention_density = False
        self.cluster_attn_step_true_counts = None
        self.cluster_attn_step_total_counts = None
        self.cluster_attn_step_call_counts = None
        self.cluster_attn_step_query_counts = None
        self.cluster_attn_step_key_counts = None

        # ============ Cluster Reorder Timing Stats ============
        self.cluster_reorder_time_stats = {}

        # ============ Attention Core Timing Stats ============
        self.attention_core_time_stats = {}

        # ============ QK Matmul Benchmark Timing Stats ============
        self.qk_matmul_time_stats = {}

        # ============ Module Timing Stats ============
        self.module_time_stats = {}
        self.enable_module_timing = False

        # ============ Remote Anchor Diagnostics ============
        self.cluster_anchor_metadata_summary = {}
        self.cluster_anchor_attention_stats = {}
        self.cluster_grouped_sdpa_stats = {}
        self.hunyuan_force_cluster_concat_attention = False

        # Orthogonal SDiT components.  These default to the legacy joint mode.
        self.enable_temporal_selective_denoising = True
        self.enable_region_sparse_attention = True
        self.cluster_attention_backend = "auto"
        self.cluster_backend_stats = {}

        # Optional read-only analysis hooks used by rebuttal experiments.
        self.scheduler_analysis_callback = None
        self.attention_analysis_callback = None
        self.save_scheduler_samples = False


    def set_parameters(self, args):
        self.patch_size = args.patch_size
        # self.scheduler_pattern = args.scheduler_pattern
        self.scheduler_start_step = args.scheduler_start_step
        self.scheduler_end_step = args.scheduler_end_step
        self.metric = args.metric
        # Handle error_reset_steps: support both string and list, handle empty string
        if isinstance(args.error_reset_steps, list):
            self.error_reset_steps = args.error_reset_steps
        elif args.error_reset_steps.strip():  # Non-empty string
            self.error_reset_steps = [int(i.strip()) for i in args.error_reset_steps.split(",")]
        else:  # Empty string
            self.error_reset_steps = []
        self.sample_ratio = args.sample_ratio
        self.num_steps = args.num_inference_steps
        self.skip_num_step = args.skip_num_step
        self.skip_num_step_length = args.skip_num_step_length
        self.height = args.height
        self.weight = args.width
        self.high_ratio = args.high_ratio
        self.enable_index_fusion = args.enable_index_fusion

        # Adaptive Complexity parameters
        if hasattr(args, 'use_adaptive_complexity'):
            self.use_adaptive_complexity = args.use_adaptive_complexity
            self.complexity_threshold = args.complexity_threshold
            if args.syn_timesteps:
                self.syn_timesteps = [int(t.strip()) for t in args.syn_timesteps.split(",")]
            if args.custom_timesteps:
                self.custom_timesteps = [int(t.strip()) for t in args.custom_timesteps.split(",")]

        # Complexity Source Selection
        if hasattr(args, 'complexity_source'):
            if args.complexity_source not in ["diff", "x_pred"]:
                raise ValueError(f"complexity_source must be 'diff' or 'x_pred', got {args.complexity_source}")
            self.complexity_source = args.complexity_source

        # Neighbor Expansion Mode Selection
        # Supports both mode strings ("0", "4", "8") and radius integers ("0", "1", "2", ...)
        if hasattr(args, 'neighbor_expansion_mode'):
            mode = args.neighbor_expansion_mode
            # Validate: must be "0", "4", "8" (mode) or any non-negative integer string (radius)
            try:
                radius = int(mode)
                if radius < 0:
                    raise ValueError(f"neighbor_expansion_mode radius must be non-negative, got {radius}")
            except ValueError:
                # If not an integer, must be one of the mode strings
                if mode not in ["0", "4", "8"]:
                    raise ValueError(f"neighbor_expansion_mode must be '0'/'4'/'8' (mode) or '0'/'1'/'2'/... (radius), got {mode}")
            self.neighbor_expansion_mode = mode

        if hasattr(args, 'use_cluster_graph_attention'):
            self.use_cluster_graph_attention = args.use_cluster_graph_attention
        if hasattr(args, 'cluster_graph_attention_hops'):
            self.cluster_graph_attention_hops = args.cluster_graph_attention_hops
        if hasattr(args, 'use_cluster_global_local_head_split'):
            self.use_cluster_global_local_head_split = args.use_cluster_global_local_head_split
        if hasattr(args, 'cluster_global_head_count'):
            self.cluster_global_head_count = args.cluster_global_head_count
        if hasattr(args, 'cluster_global_head_selection'):
            self.cluster_global_head_selection = args.cluster_global_head_selection
        if hasattr(args, 'cluster_full_kv_attention_until_step'):
            self.cluster_full_kv_attention_until_step = args.cluster_full_kv_attention_until_step
        if hasattr(args, 'cluster_prediction_source'):
            self.cluster_prediction_source = args.cluster_prediction_source
        if hasattr(args, 'use_cluster_remote_anchors'):
            self.use_cluster_remote_anchors = args.use_cluster_remote_anchors
        if hasattr(args, 'cluster_remote_anchor_count'):
            self.cluster_remote_anchor_count = args.cluster_remote_anchor_count
        if hasattr(args, 'cluster_remote_anchor_weight_mode'):
            self.cluster_remote_anchor_weight_mode = args.cluster_remote_anchor_weight_mode
        if hasattr(args, 'cluster_remote_anchor_scale_target'):
            self.cluster_remote_anchor_scale_target = args.cluster_remote_anchor_scale_target
        if hasattr(args, 'enable_cluster_remote_anchor_value_coreset'):
            self.enable_cluster_remote_anchor_value_coreset = args.enable_cluster_remote_anchor_value_coreset
        if hasattr(args, 'enable_cluster_remote_anchor_incremental_value_cache'):
            self.enable_cluster_remote_anchor_incremental_value_cache = args.enable_cluster_remote_anchor_incremental_value_cache
        if hasattr(args, 'enable_cluster_remote_anchor_merged_key_fast_path'):
            self.enable_cluster_remote_anchor_merged_key_fast_path = args.enable_cluster_remote_anchor_merged_key_fast_path
        if hasattr(args, 'enable_cluster_remote_anchor_bias_feature_fast_path'):
            self.enable_cluster_remote_anchor_bias_feature_fast_path = args.enable_cluster_remote_anchor_bias_feature_fast_path
        self.enable_cluster_late_kv_expansion = bool(getattr(args, 'enable_cluster_late_kv_expansion', True))
        self.enable_cluster_fused_pack = bool(getattr(args, 'enable_cluster_fused_pack', False))
        self.enable_cluster_compact_pack = bool(getattr(args, 'enable_cluster_compact_pack', True))
        self.enable_cluster_fused_anchor_refresh = bool(getattr(args, 'enable_cluster_fused_anchor_refresh', True))
        self.enable_cluster_static_pack_sources = bool(getattr(args, 'enable_cluster_static_pack_sources', True))
        self.enable_cluster_fused_kv_refresh = bool(getattr(args, 'enable_cluster_fused_kv_refresh', True))
        self.cluster_pack_cache = None
        self.cluster_static_pack_cache = None
        self.cluster_attention_mask_factory = None
        self.cluster_attention_mask_density = None
        self.cluster_pack_metadata_cpu = None
        self._cluster_host_selection_cache = None
        self.enable_lumina_fused_active_io = bool(getattr(args, 'enable_lumina_fused_active_io', True))
        self.enable_lumina_text_cache = bool(getattr(args, 'enable_lumina_text_cache', False))
        self.lumina_text_cache_run_token = None
        self.lumina_text_cache_immutable_context = False
        self.lumina_text_cache_immutable_model = False
        self.lumina_text_cache_model_plan = None
        self.lumina_text_cache_scope = None
        self.lumina_text_cache_state = None
        self.cluster_attention_kernel = getattr(args, 'cluster_attention_kernel', 'sdpa')
        if self.cluster_attention_kernel not in ('sdpa',):
            raise ValueError("Unsupported cluster_attention_kernel")
        if hasattr(args, 'enable_cluster_grouped_sdpa'):
            self.enable_cluster_grouped_sdpa = args.enable_cluster_grouped_sdpa
        if hasattr(args, 'cluster_grouped_sdpa_max_padding_ratio'):
            self.cluster_grouped_sdpa_max_padding_ratio = args.cluster_grouped_sdpa_max_padding_ratio
        if hasattr(args, 'enable_cluster_anchor_diagnostics'):
            self.enable_cluster_anchor_diagnostics = args.enable_cluster_anchor_diagnostics
        self.enable_cluster_attention_density = bool(getattr(args, 'enable_cluster_attention_density', False))
        if hasattr(args, 'cluster_remote_anchor_selection_budget_ms'):
            self.cluster_remote_anchor_selection_budget_ms = args.cluster_remote_anchor_selection_budget_ms
        if hasattr(args, 'cluster_remote_anchor_logit_bias_alpha'):
            self.cluster_remote_anchor_logit_bias_alpha = args.cluster_remote_anchor_logit_bias_alpha
        if hasattr(args, 'enable_cluster_relevance_routing'):
            self.enable_cluster_relevance_routing = args.enable_cluster_relevance_routing
        if hasattr(args, 'cluster_relevance_topk_ratio'):
            self.cluster_relevance_topk_ratio = args.cluster_relevance_topk_ratio
        if hasattr(args, 'cluster_relevance_use_size_bias'):
            self.cluster_relevance_use_size_bias = args.cluster_relevance_use_size_bias
        if hasattr(args, 'cluster_relevance_head_reduce'):
            self.cluster_relevance_head_reduce = args.cluster_relevance_head_reduce
        if hasattr(args, 'enable_cluster_relevance_optimizations'):
            self.enable_cluster_relevance_optimizations = args.enable_cluster_relevance_optimizations
        if hasattr(args, 'cluster_relevance_refresh_every_n_layers'):
            self.cluster_relevance_refresh_every_n_layers = args.cluster_relevance_refresh_every_n_layers
        if hasattr(args, 'cluster_relevance_reuse_within_step'):
            self.cluster_relevance_reuse_within_step = args.cluster_relevance_reuse_within_step
        if hasattr(args, 'enable_cluster_fused_triton_attention'):
            self.enable_cluster_fused_triton_attention = args.enable_cluster_fused_triton_attention
        if hasattr(args, 'enable_module_timing'):
            self.enable_module_timing = args.enable_module_timing
        if hasattr(args, 'enable_temporal_selective_denoising'):
            self.enable_temporal_selective_denoising = bool(args.enable_temporal_selective_denoising)
        if hasattr(args, 'enable_region_sparse_attention'):
            self.enable_region_sparse_attention = bool(args.enable_region_sparse_attention)
        if hasattr(args, 'cluster_attention_backend'):
            backend = str(args.cluster_attention_backend)
            if backend not in {"auto", "sdpa", "grouped_sdpa", "triton"}:
                raise ValueError(f"Unsupported cluster_attention_backend: {backend}")
            self.cluster_attention_backend = backend

        if self.cluster_attention_backend == "sdpa":
            self.enable_cluster_grouped_sdpa = False
            self.enable_cluster_fused_triton_attention = False
        elif self.cluster_attention_backend == "grouped_sdpa":
            self.enable_cluster_grouped_sdpa = True
            self.enable_cluster_fused_triton_attention = False
        elif self.cluster_attention_backend == "triton":
            if int(self.cluster_remote_anchor_count) > 0:
                raise ValueError(
                    "cluster_attention_backend='triton' requires m=0: "
                    "set cluster_remote_anchor_count=0"
                )
            self.enable_cluster_grouped_sdpa = False
            self.enable_cluster_fused_triton_attention = True

        self.reset_cluster_attention_density_stats()
        self.reset_cluster_reorder_timing_stats()
        self.reset_attention_core_timing_stats()
        self.reset_qk_matmul_timing_stats()
        self.reset_module_timing_stats()
        self.reset_cluster_anchor_diagnostics_stats()
        self.reset_cluster_backend_stats()
        self.generate_skip_token_list()

    def temporal_selection_enabled(self):
        """Whether this run actually drops/reuses tokens across timesteps."""
        return self.enable_temporal_selective_denoising and float(self.sample_ratio) < 1.0

    def spatial_attention_enabled(self):
        """Whether image self-attention should use the semantic region topology."""
        return self.enable_region_sparse_attention and self.use_cluster_graph_attention

    def cluster_layout_enabled(self):
        """A partition is needed by either temporal selection or sparse attention."""
        return self.temporal_selection_enabled() or self.spatial_attention_enabled()

    def reset_cluster_backend_stats(self):
        self.cluster_backend_stats = {
            "requested_backend": self.cluster_attention_backend,
            "effective_calls": {"sdpa": 0, "grouped_sdpa": 0, "triton": 0},
            "fallback_reasons": {},
        }

    def record_cluster_backend_call(self, effective_backend, fallback_reason=None):
        calls = self.cluster_backend_stats.setdefault("effective_calls", {})
        calls[effective_backend] = int(calls.get(effective_backend, 0)) + 1
        if fallback_reason:
            reasons = self.cluster_backend_stats.setdefault("fallback_reasons", {})
            reasons[fallback_reason] = int(reasons.get(fallback_reason, 0)) + 1

    def get_cluster_backend_report(self):
        report = {
            "requested_backend": self.cluster_backend_stats.get(
                "requested_backend", self.cluster_attention_backend
            ),
            "effective_calls": dict(self.cluster_backend_stats.get("effective_calls", {})),
            "fallback_reasons": dict(self.cluster_backend_stats.get("fallback_reasons", {})),
        }
        calls = report["effective_calls"]
        total = sum(int(value) for value in calls.values())
        report["total_calls"] = total
        report["effective_call_fractions"] = {
            name: (int(value) / total if total else 0.0)
            for name, value in calls.items()
        }
        report["triton_coverage"] = (int(calls.get("triton", 0)) / total) if total else 0.0
        report["is_pure_triton"] = (
            report["requested_backend"] == "triton"
            and total > 0
            and int(calls.get("triton", 0)) == total
        )
        report["requested_attention_kernel"] = self.cluster_attention_kernel
        report["effective_kernel_calls"] = {
            "sdpa": int(calls.get("sdpa", 0)) + int(calls.get("grouped_sdpa", 0)) + int(calls.get("fused_pack_sdpa", 0)),
            "triton": int(calls.get("triton", 0)) + int(calls.get("fused_pack_triton", 0)),
        }
        return report

    def get_active_cluster_query_ranges_cpu(self):
        """Materialize the immutable per-step query schedule once for all layers.

        Schedulers replace these tensors when selection changes. Retaining their
        identities also supports schedulers that assign the fields directly.
        """
        tensors = (self.cluster_active_cluster_ids,
                   self.cluster_active_query_starts,
                   self.cluster_active_query_lengths)
        if any(tensor is None for tensor in tensors):
            self.cluster_active_query_ranges_cpu = None
            return None
        cached = self.cluster_active_query_ranges_cpu
        if (cached is not None and cached[0] in (None, self.current_step)
                and all(old is new for old, new in zip(cached[1], tensors))):
            return cached[2]
        # One small transfer/synchronization, instead of three per layer.
        ranges = tuple(tuple(row) for row in torch.stack(tensors).detach().cpu().tolist())
        self.cluster_active_query_ranges_cpu = (self.current_step, tensors, ranges)
        return ranges

    def set_active_patch_indices(self, patch_indices, *, indices_cpu=None):
        """Store active queries in cluster order and derive contiguous cluster runs."""
        self.cluster_remote_anchor_value_update_plan = None
        self.cluster_active_query_ranges_cpu = None
        if indices_cpu is not None:
            from .host_cluster_schedule import set_active_from_host
            if set_active_from_host(self, patch_indices, indices_cpu):
                return self.other_patchified_index
        active = self.map_patch_indices_to_cluster_order(patch_indices)
        if (
            self.spatial_attention_enabled()
            and self.cluster_key_indices is not None
            and active is not None
        ):
            active = torch.sort(active).values
            self.cluster_active_queries_sorted = True
            if self.cluster_sorted_labels is not None and active.numel() > 0:
                labels = self.cluster_sorted_labels.index_select(
                    0, active.to(device=self.cluster_sorted_labels.device)
                )
                cluster_ids, lengths = torch.unique_consecutive(labels, return_counts=True)
                starts = torch.cat(
                    [
                        torch.zeros(1, device=lengths.device, dtype=torch.long),
                        lengths.cumsum(dim=0)[:-1],
                    ],
                    dim=0,
                )
                self.cluster_active_cluster_ids = cluster_ids
                self.cluster_active_query_starts = starts
                self.cluster_active_query_lengths = lengths
            else:
                self.cluster_active_cluster_ids = None
                self.cluster_active_query_starts = None
                self.cluster_active_query_lengths = None
        else:
            self.cluster_active_queries_sorted = False
            self.cluster_active_cluster_ids = None
            self.cluster_active_query_starts = None
            self.cluster_active_query_lengths = None
        self.other_patchified_index = active
        return active

    def set_all_patches_active(self, num_patches, device):
        original_order = torch.arange(num_patches, device=device, dtype=torch.long)
        self.cached_index = None
        self.cached_patchified_index = torch.empty(0, device=device, dtype=torch.long)
        return self.set_active_patch_indices(original_order)

    def reset_cluster_attention_density_stats(self):
        self.cluster_attn_step_true_counts = None
        self.cluster_attn_step_total_counts = None
        self.cluster_attn_step_call_counts = None
        self.cluster_attn_step_query_counts = None
        self.cluster_attn_step_key_counts = None

    def reset_cluster_reorder_timing_stats(self):
        self.cluster_reorder_time_stats = {}

    def reset_attention_core_timing_stats(self):
        self.attention_core_time_stats = {}

    def reset_qk_matmul_timing_stats(self):
        self.qk_matmul_time_stats = {}

    def reset_module_timing_stats(self):
        self.module_time_stats = {}

    def reset_cluster_anchor_diagnostics_stats(self):
        self.cluster_anchor_metadata_summary = {}
        self.cluster_anchor_attention_stats = {
            "logical_jobs": 0,
            "sum_q_len": 0,
            "sum_local_k_len": 0,
            "sum_remote_k_len": 0,
            "sum_total_k_len": 0,
            "max_remote_k_len": 0,
            "max_total_k_len": 0,
        }
        self.cluster_grouped_sdpa_stats = {
            "sdpa_calls": 0,
            "logical_jobs": 0,
            "merged_groups": 0,
            "max_group_jobs": 0,
            "sum_group_q_len": 0,
            "sum_group_k_len": 0,
        }

    def set_cluster_anchor_metadata_summary(self, summary):
        if not self.enable_cluster_anchor_diagnostics:
            return
        self.cluster_anchor_metadata_summary = dict(summary)

    def add_cluster_anchor_attention_job(self, q_len, local_k_len, remote_k_len, total_k_len):
        if not self.enable_cluster_anchor_diagnostics:
            return
        stats = self.cluster_anchor_attention_stats
        stats["logical_jobs"] += 1
        stats["sum_q_len"] += int(q_len)
        stats["sum_local_k_len"] += int(local_k_len)
        stats["sum_remote_k_len"] += int(remote_k_len)
        stats["sum_total_k_len"] += int(total_k_len)
        stats["max_remote_k_len"] = max(int(stats["max_remote_k_len"]), int(remote_k_len))
        stats["max_total_k_len"] = max(int(stats["max_total_k_len"]), int(total_k_len))

    def add_cluster_grouped_sdpa_call(self, logical_jobs, q_len, k_len):
        if not self.enable_cluster_anchor_diagnostics:
            return
        stats = self.cluster_grouped_sdpa_stats
        logical_jobs = int(logical_jobs)
        stats["sdpa_calls"] += 1
        stats["logical_jobs"] += logical_jobs
        if logical_jobs > 1:
            stats["merged_groups"] += 1
        stats["max_group_jobs"] = max(int(stats["max_group_jobs"]), logical_jobs)
        stats["sum_group_q_len"] += int(q_len)
        stats["sum_group_k_len"] += int(k_len)

    def get_cluster_anchor_diagnostics_report(self):
        attention = dict(self.cluster_anchor_attention_stats)
        grouped = dict(self.cluster_grouped_sdpa_stats)
        jobs = int(attention.get("logical_jobs", 0))
        calls = int(grouped.get("sdpa_calls", 0))
        if jobs > 0:
            attention["avg_q_len"] = attention["sum_q_len"] / jobs
            attention["avg_local_k_len"] = attention["sum_local_k_len"] / jobs
            attention["avg_remote_k_len"] = attention["sum_remote_k_len"] / jobs
            attention["avg_total_k_len"] = attention["sum_total_k_len"] / jobs
            attention["remote_k_fraction"] = attention["sum_remote_k_len"] / max(1, attention["sum_total_k_len"])
        else:
            attention.update({
                "avg_q_len": 0.0,
                "avg_local_k_len": 0.0,
                "avg_remote_k_len": 0.0,
                "avg_total_k_len": 0.0,
                "remote_k_fraction": 0.0,
            })
        grouped["call_reduction"] = (1.0 - calls / jobs) if jobs > 0 else 0.0
        grouped["avg_jobs_per_call"] = (jobs / calls) if calls > 0 else 0.0
        return {
            "metadata": dict(self.cluster_anchor_metadata_summary),
            "attention": attention,
            "grouped_sdpa": grouped,
        }

    def reset_cluster_relevance_cache(self):
        self.cluster_relevance_cache_step = -1
        self.cluster_relevance_cache_layer = -1
        self.cluster_relevance_cache_active_count = -1
        self.cluster_relevance_cache_remote_ids = None
        self.cluster_relevance_cache_remote_counts = None
        self.cluster_relevance_cache_cluster_ids = None

    def add_cluster_reorder_timing(self, name, elapsed_seconds):
        stats = self.cluster_reorder_time_stats.setdefault(name, {"count": 0, "total": 0.0})
        stats["count"] += 1
        stats["total"] += float(elapsed_seconds)

    def timed_cluster_reorder(self, name, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self.add_cluster_reorder_timing(name, time.perf_counter() - start)
        return result

    def get_cluster_reorder_timing_report(self):
        report = []
        total_time = 0.0
        total_calls = 0
        for name, stats in sorted(self.cluster_reorder_time_stats.items()):
            count = int(stats["count"])
            total = float(stats["total"])
            avg = total / count if count > 0 else 0.0
            report.append({
                "name": name,
                "count": count,
                "total_seconds": total,
                "avg_seconds": avg,
            })
            total_time += total
            total_calls += count

        return {
            "items": report,
            "total_seconds": total_time,
            "total_calls": total_calls,
        }

    def add_attention_core_timing(
        self,
        name,
        elapsed_seconds,
        q_len=None,
        k_len=None,
        v_len=None,
        num_heads=None,
        head_dim=None,
    ):
        if not self.enable_module_timing:
            return
        stats = self.attention_core_time_stats.setdefault(
            name,
            {
                "count": 0,
                "total": 0.0,
                "sum_q_len": 0,
                "sum_k_len": 0,
                "sum_v_len": 0,
                "sum_heads": 0,
                "sum_head_dim": 0,
            },
        )
        stats["count"] += 1
        stats["total"] += float(elapsed_seconds)
        if q_len is not None:
            stats["sum_q_len"] += int(q_len)
        if k_len is not None:
            stats["sum_k_len"] += int(k_len)
        if v_len is not None:
            stats["sum_v_len"] += int(v_len)
        if num_heads is not None:
            stats["sum_heads"] += int(num_heads)
        if head_dim is not None:
            stats["sum_head_dim"] += int(head_dim)

    def timed_attention_core(
        self,
        name,
        fn,
        *args,
        q_len=None,
        k_len=None,
        v_len=None,
        num_heads=None,
        head_dim=None,
        **kwargs,
    ):
        if not self.enable_module_timing:
            return fn(*args, **kwargs)
        tensor_arg = next((arg for arg in args if torch.is_tensor(arg)), None)
        if tensor_arg is not None and tensor_arg.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device=tensor_arg.device)
            start_event.record(stream)
            result = fn(*args, **kwargs)
            end_event.record(stream)
            end_event.synchronize()
            elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0
        else:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_seconds = time.perf_counter() - start

        self.add_attention_core_timing(
            name,
            elapsed_seconds,
            q_len=q_len,
            k_len=k_len,
            v_len=v_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        return result

    def get_attention_core_timing_report(self):
        report = []
        total_time = 0.0
        total_calls = 0
        for name, stats in sorted(self.attention_core_time_stats.items()):
            count = int(stats["count"])
            total = float(stats["total"])
            avg = total / count if count > 0 else 0.0
            report.append({
                "name": name,
                "count": count,
                "total_seconds": total,
                "avg_seconds": avg,
                "avg_q_len": stats["sum_q_len"] / count if count > 0 else 0.0,
                "avg_k_len": stats["sum_k_len"] / count if count > 0 else 0.0,
                "avg_v_len": stats["sum_v_len"] / count if count > 0 else 0.0,
                "avg_heads": stats["sum_heads"] / count if count > 0 else 0.0,
                "avg_head_dim": stats["sum_head_dim"] / count if count > 0 else 0.0,
            })
            total_time += total
            total_calls += count

        return {
            "items": report,
            "total_seconds": total_time,
            "total_calls": total_calls,
        }

    def add_qk_matmul_timing(
        self,
        name,
        elapsed_seconds,
        q_len=None,
        k_len=None,
        num_heads=None,
        head_dim=None,
    ):
        if not self.enable_module_timing:
            return
        stats = self.qk_matmul_time_stats.setdefault(
            name,
            {
                "count": 0,
                "total": 0.0,
                "sum_q_len": 0,
                "sum_k_len": 0,
                "sum_heads": 0,
                "sum_head_dim": 0,
            },
        )
        stats["count"] += 1
        stats["total"] += float(elapsed_seconds)
        if q_len is not None:
            stats["sum_q_len"] += int(q_len)
        if k_len is not None:
            stats["sum_k_len"] += int(k_len)
        if num_heads is not None:
            stats["sum_heads"] += int(num_heads)
        if head_dim is not None:
            stats["sum_head_dim"] += int(head_dim)

    def benchmark_qk_matmul(self, name, query, key, chunk_heads=4):
        if not self.enable_module_timing:
            return
        batch_size, num_heads, q_len, head_dim = query.shape
        k_len = key.shape[-2]

        query_3d = query.detach().reshape(batch_size * num_heads, q_len, head_dim)
        key_3d = key.detach().reshape(batch_size * num_heads, k_len, head_dim)

        if query_3d.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device=query_3d.device)
            start_event.record(stream)
            for start in range(0, query_3d.shape[0], chunk_heads):
                stop = min(start + chunk_heads, query_3d.shape[0])
                torch.bmm(query_3d[start:stop], key_3d[start:stop].transpose(1, 2))
            end_event.record(stream)
            end_event.synchronize()
            elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0
        else:
            start = time.perf_counter()
            for start_idx in range(0, query_3d.shape[0], chunk_heads):
                stop_idx = min(start_idx + chunk_heads, query_3d.shape[0])
                torch.bmm(query_3d[start_idx:stop_idx], key_3d[start_idx:stop_idx].transpose(1, 2))
            elapsed_seconds = time.perf_counter() - start

        self.add_qk_matmul_timing(
            name,
            elapsed_seconds,
            q_len=q_len,
            k_len=k_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )

    def get_qk_matmul_timing_report(self):
        report = []
        total_time = 0.0
        total_calls = 0
        for name, stats in sorted(self.qk_matmul_time_stats.items()):
            count = int(stats["count"])
            total = float(stats["total"])
            avg = total / count if count > 0 else 0.0
            report.append({
                "name": name,
                "count": count,
                "total_seconds": total,
                "avg_seconds": avg,
                "avg_q_len": stats["sum_q_len"] / count if count > 0 else 0.0,
                "avg_k_len": stats["sum_k_len"] / count if count > 0 else 0.0,
                "avg_heads": stats["sum_heads"] / count if count > 0 else 0.0,
                "avg_head_dim": stats["sum_head_dim"] / count if count > 0 else 0.0,
            })
            total_time += total
            total_calls += count

        return {
            "items": report,
            "total_seconds": total_time,
            "total_calls": total_calls,
        }

    def add_module_timing(self, name, elapsed_seconds):
        if not self.enable_module_timing:
            return
        stats = self.module_time_stats.setdefault(name, {"count": 0, "total": 0.0})
        stats["count"] += 1
        stats["total"] += float(elapsed_seconds)

    def timed_module_op(self, name, fn, *args, **kwargs):
        if not self.enable_module_timing:
            return fn(*args, **kwargs)
        tensor_arg = next((arg for arg in args if torch.is_tensor(arg)), None)
        if tensor_arg is not None and tensor_arg.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device=tensor_arg.device)
            start_event.record(stream)
            result = fn(*args, **kwargs)
            end_event.record(stream)
            end_event.synchronize()
            elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0
        else:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_seconds = time.perf_counter() - start
        self.add_module_timing(name, elapsed_seconds)
        return result

    def get_module_timing_report(self):
        report = []
        total_time = 0.0
        total_calls = 0
        for name, stats in sorted(self.module_time_stats.items()):
            count = int(stats["count"])
            total = float(stats["total"])
            avg = total / count if count > 0 else 0.0
            report.append({
                "name": name,
                "count": count,
                "total_seconds": total,
                "avg_seconds": avg,
            })
            total_time += total
            total_calls += count
        return {
            "items": report,
            "total_seconds": total_time,
            "total_calls": total_calls,
        }

    def initialize_cluster_permutation(self, labels):
        labels = labels.to(dtype=torch.long)
        perm = torch.argsort(labels, stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
        sorted_labels = labels.index_select(0, perm)

        cluster_block_ranges = {}
        unique_labels, counts = torch.unique_consecutive(sorted_labels, return_counts=True)
        start = 0
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            end = start + count
            cluster_block_ranges[int(label)] = (int(start), int(end))
            start = end

        self.cluster_perm = perm
        self.cluster_inv_perm = inv_perm
        self.cluster_sorted_labels = sorted_labels
        self.cluster_block_ranges = cluster_block_ranges
        self.cluster_labels_flat = sorted_labels
        self.cluster_ids_tensor = unique_labels
        self.cluster_block_lengths = counts.to(dtype=torch.long)
        self.cluster_max_block_length = int(counts.max().item()) if counts.numel() > 0 else 0
        if self.enable_cluster_relevance_optimizations:
            self.cluster_compact_labels = torch.repeat_interleave(
                torch.arange(unique_labels.numel(), device=labels.device, dtype=torch.long),
                self.cluster_block_lengths,
            )
        else:
            self.cluster_compact_labels = None
        self.cluster_block_starts = torch.cat(
            [
                torch.zeros(1, device=labels.device, dtype=torch.long),
                self.cluster_block_lengths.cumsum(dim=0)[:-1],
            ],
            dim=0,
        )
        self.cluster_block_ends = self.cluster_block_starts + self.cluster_block_lengths
        if unique_labels.numel() > 0:
            max_label = int(unique_labels.max().item())
            label_to_compact = torch.full(
                (max_label + 1,),
                -1,
                device=labels.device,
                dtype=torch.long,
            )
            label_to_compact.index_copy_(
                0,
                unique_labels,
                torch.arange(unique_labels.numel(), device=labels.device, dtype=torch.long),
            )
            self.cluster_label_to_compact = label_to_compact
            max_cluster_tokens = int(self.cluster_block_lengths.max().item())
            offsets = torch.arange(max_cluster_tokens, device=labels.device, dtype=torch.long)
            self.cluster_token_indices_mask = offsets[None, :] < self.cluster_block_lengths[:, None]
            token_indices_padded = self.cluster_block_starts[:, None] + offsets[None, :]
            self.cluster_token_indices_padded = torch.where(
                self.cluster_token_indices_mask,
                token_indices_padded,
                torch.zeros_like(token_indices_padded),
            )
        else:
            self.cluster_label_to_compact = torch.empty(0, device=labels.device, dtype=torch.long)
            self.cluster_token_indices_padded = torch.empty(0, 0, device=labels.device, dtype=torch.long)
            self.cluster_token_indices_mask = torch.empty(0, 0, device=labels.device, dtype=torch.bool)
        self.cluster_local_cluster_mask = None
        self.cluster_active_queries_sorted = False
        self.cluster_active_query_ranges_cpu = None
        self.cluster_active_cluster_ids = None
        self.cluster_active_query_starts = None
        self.cluster_active_query_lengths = None
        self.cluster_local_block_indices_padded = None
        self.cluster_local_block_counts = None
        self.cluster_local_block_max_count = 0
        self.cluster_local_key_spans = None
        self.cluster_local_key_span_starts = None
        self.cluster_local_key_span_lengths = None
        self.reset_cluster_relevance_cache()

    def cluster_ids_to_compact(self, cluster_ids):
        if self.cluster_label_to_compact is None:
            return None

        cluster_ids = cluster_ids.to(dtype=torch.long)
        lookup = self.cluster_label_to_compact.to(device=cluster_ids.device)
        if lookup.numel() == 0:
            return torch.full_like(cluster_ids, -1)
        valid = (cluster_ids >= 0) & (cluster_ids < lookup.numel())
        safe_cluster_ids = cluster_ids.clamp(min=0, max=lookup.numel() - 1)
        compact = lookup.index_select(0, safe_cluster_ids.reshape(-1)).view_as(cluster_ids)
        return torch.where(valid, compact, torch.full_like(compact, -1))

    def has_cluster_permutation(self):
        return self.cluster_perm is not None and self.cluster_inv_perm is not None

    def permute_patch_tensor(self, tensor, dim):
        if not self.has_cluster_permutation():
            return tensor

        dim = dim if dim >= 0 else tensor.dim() + dim
        perm = self.cluster_perm.to(device=tensor.device)
        return tensor.index_select(dim, perm)

    def inverse_permute_patch_tensor(self, tensor, dim):
        if not self.has_cluster_permutation():
            return tensor

        dim = dim if dim >= 0 else tensor.dim() + dim
        inv_perm = self.cluster_inv_perm.to(device=tensor.device)
        return tensor.index_select(dim, inv_perm)

    def map_patch_indices_to_cluster_order(self, indices):
        if indices is None or not self.has_cluster_permutation():
            return indices

        input_device = indices.device if torch.is_tensor(indices) else self.cluster_inv_perm.device
        indices_t = torch.as_tensor(indices, device=self.cluster_inv_perm.device, dtype=torch.long)
        mapped = self.cluster_inv_perm.index_select(0, indices_t.reshape(-1)).view_as(indices_t)
        return mapped.to(device=input_device)

    def _ensure_cluster_attention_density_buffers(self, device):
        num_steps = max(int(self.num_steps), 1)
        need_init = (
            self.cluster_attn_step_true_counts is None or
            self.cluster_attn_step_true_counts.device != device or
            self.cluster_attn_step_true_counts.shape[0] != num_steps
        )

        if need_init:
            self.cluster_attn_step_true_counts = torch.zeros(num_steps, device=device, dtype=torch.int64)
            self.cluster_attn_step_total_counts = torch.zeros(num_steps, device=device, dtype=torch.int64)
            self.cluster_attn_step_call_counts = torch.zeros(num_steps, device=device, dtype=torch.int64)
            self.cluster_attn_step_query_counts = torch.zeros(num_steps, device=device, dtype=torch.int64)
            self.cluster_attn_step_key_counts = torch.zeros(num_steps, device=device, dtype=torch.int64)

    def record_cluster_attention_density(self, cluster_mask, key_padding_mask, num_heads):
        if not self.enable_cluster_attention_density or cluster_mask is None:
            return

        step = int(self.current_step)
        if step < 0 or step >= int(self.num_steps):
            return

        cluster_mask = cluster_mask.to(dtype=torch.bool)
        if key_padding_mask.ndim == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        key_padding_mask = key_padding_mask.to(device=cluster_mask.device, dtype=torch.bool)

        self._ensure_cluster_attention_density_buffers(cluster_mask.device)

        column_true_counts = cluster_mask.sum(dim=0, dtype=torch.int64)
        valid_key_mask = key_padding_mask.to(dtype=torch.int64)
        true_count = (valid_key_mask * column_true_counts.unsqueeze(0)).sum(dtype=torch.int64) * int(num_heads)
        total_count = (
            cluster_mask.shape[0] * cluster_mask.shape[1] * key_padding_mask.shape[0] * int(num_heads)
        )

        self.cluster_attn_step_true_counts[step] += true_count
        self.cluster_attn_step_total_counts[step] += total_count
        self.cluster_attn_step_call_counts[step] += 1
        self.cluster_attn_step_query_counts[step] = cluster_mask.shape[0]
        self.cluster_attn_step_key_counts[step] = cluster_mask.shape[1]

    def get_cluster_attention_density_report(self):
        if self.cluster_attn_step_true_counts is None:
            return {
                "steps": [],
                "overall_density": None,
                "overall_true_count": 0,
                "overall_total_count": 0,
            }

        true_counts = self.cluster_attn_step_true_counts.detach().cpu()
        total_counts = self.cluster_attn_step_total_counts.detach().cpu()
        call_counts = self.cluster_attn_step_call_counts.detach().cpu()
        query_counts = self.cluster_attn_step_query_counts.detach().cpu()
        key_counts = self.cluster_attn_step_key_counts.detach().cpu()

        steps = []
        cumulative_true = 0
        cumulative_total = 0

        for step_idx in range(true_counts.shape[0]):
            total_count = int(total_counts[step_idx].item())
            call_count = int(call_counts[step_idx].item())
            if total_count == 0 or call_count == 0:
                continue

            true_count = int(true_counts[step_idx].item())
            cumulative_true += true_count
            cumulative_total += total_count

            density = true_count / total_count
            cumulative_density = cumulative_true / cumulative_total

            steps.append({
                "step": step_idx,
                "density": density,
                "cumulative_density": cumulative_density,
                "true_count": true_count,
                "total_count": total_count,
                "num_calls": call_count,
                "query_tokens": int(query_counts[step_idx].item()),
                "key_tokens": int(key_counts[step_idx].item()),
            })

        overall_density = cumulative_true / cumulative_total if cumulative_total > 0 else None
        return {
            "steps": steps,
            "overall_density": overall_density,
            "overall_true_count": cumulative_true,
            "overall_total_count": cumulative_total,
        }


    def generate_skip_token_list(self):
        avg_skip_token_num = int((1 - self.sample_ratio) * ((self.height // self.patch_size) // self.vae_size) * ((self.weight // self.patch_size) // self.vae_size))
        if self.skip_num_step_length == 0: # static dropping
            self.skip_token_num_list = [avg_skip_token_num for i in range(self.num_steps)]
            for i in self.error_reset_steps:
                self.skip_token_num_list[i] = 0
            for i in range(self.scheduler_start_step):
                self.skip_token_num_list[i] = 0
            print("Skip token number list: ", len(self.skip_token_num_list), self.skip_token_num_list)
            return
        for i in range(0, self.num_steps // self.skip_num_step_length + 1):
            for j in range(self.skip_num_step_length):
                if i * self.skip_num_step_length + j >= self.num_steps:
                    break
                # Temporarily use avg_skip_token_num to avoid exceeding total token count
                temp_skip_num = avg_skip_token_num
                # Original formula (commented out to prevent assertion error):
                # temp_skip_num = avg_skip_token_num + self.skip_num_step * (i - (((self.num_steps + self.scheduler_start_step) // self.skip_num_step_length) // 2))
                temp_skip_num = (temp_skip_num // 64) * 64
                self.skip_token_num_list.append(temp_skip_num)
        for i in range(self.scheduler_start_step):
            self.skip_token_num_list[i] = 0
        for i in self.error_reset_steps:
            self.skip_token_num_list[i] = 0
        for i in range(len(self.skip_token_num_list)):
            assert self.skip_token_num_list[i] >= 0, "Skip token number should be positive"
            assert self.skip_token_num_list[i] <= ((self.height // self.patch_size) // self.vae_size) * ((self.weight // self.patch_size) // self.vae_size)
        print("Skip token number list: ", len(self.skip_token_num_list), self.skip_token_num_list)

    def get_cluster_attention_mask(self):
        """Materialize the partition mask only for consumers that need it."""
        if self.cluster_attention_mask is None and self.cluster_attention_mask_factory is not None:
            self.cluster_attention_mask = self.cluster_attention_mask_factory()
            self.cluster_attention_mask_factory = None
        return self.cluster_attention_mask

    def reset_cache(self):
        # A scheduler reset invalidates memoized text tensors as well. The
        # enclosing text_cache_run owns its request token and immutable-input
        # contract; step-zero temporal resets must not end that request.
        self.lumina_text_cache_scope = None
        self.lumina_text_cache_state = None
        self.cached_index = None
        self.other_index = None
        self.cached_patchified_index = None
        self.other_patchified_index = None
        self.image_rotary_emb_skip = None
        self.cached_scaled_noise = None
        self.cluster_attention_mask = None
        self.cluster_attention_mask_factory = None
        self.cluster_attention_mask_density = None
        self.cluster_window_size = -1
        self.cluster_key_indices = None
        self.cluster_far_cluster_ids = None
        self.cluster_ids_tensor = None
        self.cluster_label_to_compact = None
        self.cluster_block_starts = None
        self.cluster_block_ends = None
        self.cluster_block_lengths = None
        self.cluster_max_block_length = 0
        self.cluster_compact_labels = None
        self.cluster_token_indices_padded = None
        self.cluster_token_indices_mask = None
        self.cluster_local_cluster_mask = None
        self.cluster_active_queries_sorted = False
        self.cluster_pack_cache = None
        self.cluster_static_pack_cache = None
        self.cluster_pack_metadata_cpu = None
        self._cluster_host_selection_cache = None
        self.cluster_active_query_ranges_cpu = None
        self.cluster_active_cluster_ids = None
        self.cluster_active_query_starts = None
        self.cluster_active_query_lengths = None
        self.cluster_local_block_indices_padded = None
        self.cluster_local_block_counts = None
        self.cluster_local_block_max_count = 0
        self.cluster_local_key_spans = None
        self.cluster_local_key_span_starts = None
        self.cluster_local_key_span_lengths = None
        self.reset_cluster_relevance_cache()
        self.cluster_remote_anchor_indices = None
        self.cluster_remote_anchor_weights = None
        self.cluster_remote_anchor_group_indices = None
        self.cluster_remote_anchor_group_mask = None
        self.cluster_remote_anchor_source_cluster_ids = None
        self.cluster_remote_anchor_source_indices = None
        self.cluster_remote_anchor_source_weights = None
        self.cluster_remote_anchor_source_group_indices = None
        self.cluster_remote_anchor_source_group_mask = None
        self.cluster_remote_anchor_all_indices = None
        self.cluster_remote_anchor_all_weights = None
        self.cluster_remote_anchor_all_group_indices = None
        self.cluster_remote_anchor_all_group_mask = None
        self.cluster_remote_anchor_packed_positions = None
        self.cluster_remote_anchor_merged_key_indices = None
        self.cluster_remote_anchor_token_to_packed_anchor = None
        self.cluster_remote_anchor_packed_counts = None
        self.cluster_remote_anchor_group_token_order = None
        self.cluster_remote_anchor_group_lengths = None
        self.cluster_remote_anchor_group_token_order_is_cluster_order = False
        self.cluster_remote_anchor_all_logit_bias = None
        self.cluster_remote_anchor_metadata_version += 1
        self.cluster_remote_anchor_value_update_plan = None
        self.cluster_key_padding_mask_all_true = None
        self.cluster_key_padding_mask_numel = -1
        self.cluster_labels_flat = None
        self.cluster_perm = None
        self.cluster_inv_perm = None
        self.cluster_sorted_labels = None
        self.cluster_block_ranges = None
        self.current_step = 0
        self.current_layer = 0
        if self.current_step >= self.scheduler_start_step and self.current_step <= self.scheduler_end_step and self.current_step not in self.error_reset_steps:
            self.is_RAS_step = True
        else:
            self.is_RAS_step = False
        if self.current_step + 1 >= self.scheduler_start_step and self.current_step + 1 <= self.scheduler_end_step and self.current_step + 1 not in self.error_reset_steps:
            self.is_next_RAS_step = True
        else:
            self.is_next_RAS_step = False

    def increase_step(self):
        self.current_step += 1
        self.current_layer = 0
        if self.current_step >= self.scheduler_start_step and self.current_step <= self.scheduler_end_step and self.current_step not in self.error_reset_steps:
            self.is_RAS_step = True
        else:
            self.is_RAS_step = False
        if self.current_step + 1 >= self.scheduler_start_step and self.current_step + 1 < self.scheduler_end_step and self.current_step + 1 not in self.error_reset_steps:
            self.is_next_RAS_step = True
        else:
            self.is_next_RAS_step = False







MANAGER = ras_manager()
