import argparse

def parse_args_from_dict(args_dict):
    """
    Convert a dictionary of arguments to argparse.Namespace

    Args:
        args_dict: Dictionary containing argument names and values

    Returns:
        argparse.Namespace object
    """
    # Create a Namespace with default values first
    parser = argparse.ArgumentParser(description="Lumina Inference")

    # Set default values for all parameters
    defaults = {
        "prompt": "",
        "negative_prompt": "",
        "num_inference_steps": 30,
        "output": "output.png",
        "seed": None,
        "sample_ratio": 0.5,
        "error_reset_steps": "12,22",
        "metric": "std",
        "high_ratio": 1,
        "height": 1024,
        "width": 1024,
        "starvation_scale": 0.1,
        "scheduler_start_step": 4,
        "scheduler_end_step": 30,
        "patch_size": 2,
        "skip_num_step": 0,
        "skip_num_step_length": 0,
        "enable_index_fusion": False,
        "use_adaptive_complexity": False,
        "complexity_threshold": 0.5,
        "syn_timesteps": "1000,933,867,800,733,667,600,533,467,400,333,267,200,133,67,0",
        "custom_timesteps": None,
        # Cluster Adaptive parameters
        "use_cluster_adaptive": False,
        "num_clusters": 16,
        "cluster_selection_ratio": 0.5,
        "recluster_every_step": True,
        "recluster_every_n_steps": 0,
        "final_sync_steps": 3,
        # Complexity Source Selection
        "complexity_source": "x_pred",  # 'diff' or 'x_pred'
        # Neighbor Expansion Mode
        "neighbor_expansion_mode": "8",  # '0' (no expansion), '4' (4-neighbor), '8' (8-neighbor)
        # Cluster Graph Attention
        "use_cluster_graph_attention": False,
        "cluster_graph_attention_hops": 2,
        "use_cluster_global_local_head_split": False,
        "cluster_global_head_count": 4,
        "cluster_global_head_selection": "online",
        "cluster_full_kv_attention_until_step": -1,
        "cluster_prediction_source": "legacy",
        "use_cluster_remote_anchors": True,
        "cluster_remote_anchor_count": 4,
        "cluster_remote_anchor_weight_mode": "none",
        "cluster_remote_anchor_scale_target": "v",
        "enable_cluster_remote_anchor_value_coreset": False,
        "enable_cluster_remote_anchor_incremental_value_cache": False,
        "enable_cluster_remote_anchor_merged_key_fast_path": False,
        "enable_cluster_remote_anchor_bias_feature_fast_path": False,
        "enable_cluster_late_kv_expansion": True,
        "enable_cluster_fused_pack": False,
        "cluster_attention_kernel": "sdpa",
        "enable_cluster_grouped_sdpa": False,
        "cluster_grouped_sdpa_max_padding_ratio": 1.25,
        "enable_cluster_anchor_diagnostics": False,
        "enable_cluster_attention_density": False,
        "cluster_remote_anchor_selection_budget_ms": 0.0,
        "cluster_remote_anchor_logit_bias_alpha": 1.0,
        "enable_cluster_relevance_routing": True,
        "cluster_relevance_topk_ratio": 0.0,
        "cluster_relevance_use_size_bias": False,
        "cluster_relevance_head_reduce": "mean",
        "enable_cluster_relevance_optimizations": False,
        "cluster_relevance_refresh_every_n_layers": 4,
        "cluster_relevance_reuse_within_step": True,
        "enable_cluster_fused_triton_attention": False,
        "enable_module_timing": False,
        # Orthogonal SDiT component switches.  Defaults preserve the submitted
        # implementation when the cluster-adaptive scheduler is selected.
        "enable_temporal_selective_denoising": True,
        "enable_region_sparse_attention": True,
        "cluster_attention_backend": "auto",
    }

    # Update with provided values
    defaults.update(args_dict)

    # Keep error_reset_steps as string - ras_manager will parse it
    # (Don't convert to list here, it will be converted in ras_manager.set_parameters)

    return argparse.Namespace(**defaults)

def parse_args():
    parser = argparse.ArgumentParser(description="Lumina Inference")
    parser.add_argument("--prompt", type=str, default="A red heart in the clouds over water, in the style of zbrush, light pink and sky-blue, I can't believe how beautiful this is, hyperbolic expression, nyc explosion coverage, unreal engine 5, robert bissell.")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--output", type=str, default="output.png", help="Output file path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=0.5, help="Average sample ratio during each step for RAS")
    parser.add_argument("--error_reset_steps", type=str, default="12,22", help="Dense steps to reset the error, use comma to separate")
    parser.add_argument("--metric", type=str, default="std", choices=["std", "l2norm"], help="Metric to calculate the patch selection, currently support std and l2norm")
    parser.add_argument("--high_ratio", type=float, default=1, help="Based on the metric selected, the ratio of the high value chosen to be cached. Range from 0 to 1")
    parser.add_argument("--height", type=int, default=1024, help="Height of the image")
    parser.add_argument("--width", type=int, default=1024, help="Width of the image")
    parser.add_argument("--starvation_scale", type=float, default=0.1, help="Starvation scale for the patch selection, higher value will make the patch selection more average")
    parser.add_argument("--scheduler_start_step", type=int, default=4, help="From which sampling step to start using RAS")
    parser.add_argument("--scheduler_end_step", type=int, default=30, help="From which sampling step to stop using RAS")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size for Lumina")
    parser.add_argument("--skip_num_step", type=int, default=0, help="Parameter for dynamic skip token numbers. The skip token number increases/decreases by this value every skip_num_step_length steps. Can be negative or positive.")
    parser.add_argument("--skip_num_step_length", type=int, default=0, help="The interval to change the skip token number")
    parser.add_argument("--enable_index_fusion", action="store_true", help="Enable index fusion for RAS")

    # ============ Adaptive Complexity Sampling Arguments ============
    parser.add_argument(
        "--use_adaptive_complexity",
        action="store_true",
        help="Enable adaptive complexity-based sampling strategy"
    )
    parser.add_argument(
        "--complexity_threshold",
        type=float,
        default=0.5,
        help="Threshold for splitting patches into complex/simple groups (0.5 = top 50%% use multi-step)"
    )
    parser.add_argument(
        "--syn_timesteps",
        type=str,
        default="1000,933,867,800,733,667,600,533,467,400,333,267,200,133,67,0",
        help="Synchronization points for complexity computation (comma-separated)"
    )
    parser.add_argument(
        "--custom_timesteps",
        type=str,
        default=None,
        help="Custom timestep schedule (comma-separated). If None, uses default 30-step schedule"
    )

    # ============ Cluster Adaptive Scheduler Arguments ============
    parser.add_argument(
        "--use_cluster_adaptive",
        action="store_true",
        help="Use cluster adaptive scheduler with 8-neighbor expansion"
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=16,
        help="Number of clusters for cluster-based scheduling"
    )
    parser.add_argument(
        "--cluster_selection_ratio",
        type=float,
        default=0.5,
        help="Ratio of clusters to select each step (0.5 = select top 50%% clusters)"
    )
    parser.add_argument(
        "--recluster_every_step",
        action="store_true",
        help="Recluster patches every step (default: True for cluster adaptive)"
    )
    parser.add_argument(
        "--recluster_every_n_steps",
        type=int,
        default=0,
        help="Recluster patches every N RAS steps when --recluster_every_step is not set (0 disables interval reclustering)"
    )
    parser.add_argument(
        "--final_sync_steps",
        type=int,
        default=3,
        help="Number of final steps to disable adaptive and sync all patches (default: 3)"
    )

    # ============ Complexity Source Selection ============
    parser.add_argument(
        "--complexity_source",
        type=str,
        default="x_pred",
        choices=["diff", "x_pred"],
        help="Choose data source for complexity calculation: 'diff' (residual) or 'x_pred' (predicted sample)"
    )

    # ============ Neighbor Expansion Mode ============
    parser.add_argument(
        "--neighbor_expansion_mode",
        type=str,
        default="8",
        help="Neighbor expansion mode: '0'/'4'/'8' (mode) or '0'/'1'/'2'/... (radius, where 1=3x3, 2=5x5, etc.)"
    )

    # ============ Global-Local Head Split ============
    parser.add_argument(
        "--use_cluster_global_local_head_split",
        action="store_true",
        help="Use full cached K/V for a small set of global heads and cluster-local K/V for the remaining heads"
    )
    parser.add_argument(
        "--cluster_global_head_count",
        type=int,
        default=4,
        help="Number of self-attention heads that keep full cached K/V context under global-local head split"
    )
    parser.add_argument(
        "--cluster_global_head_selection",
        type=str,
        default="online",
        choices=["online", "uniform", "first"],
        help="How to choose global heads: online Q/K remote-mass score, evenly spaced heads, or first heads"
    )
    parser.add_argument(
        "--cluster_full_kv_attention_until_step",
        type=int,
        default=-1,
        help="During RAS steps with current_step below this exclusive cutoff, use active-Q full-KV attention"
    )
    parser.add_argument(
        "--cluster_prediction_source",
        type=str,
        default="legacy",
        choices=["legacy", "x0_pred", "prev_sample", "sample"],
        help="Tensor source for cluster layout/scoring; legacy uses sample for layout and prev_sample for scoring"
    )

    # ============ Remote Anchor Attention ============
    parser.add_argument(
        "--disable_cluster_remote_anchors",
        dest="use_cluster_remote_anchors",
        action="store_false",
        help="Disable fixed real-token remote anchors for cluster sparse attention"
    )
    parser.set_defaults(use_cluster_remote_anchors=True)
    parser.add_argument(
        "--cluster_remote_anchor_count",
        type=int,
        default=4,
        help="Number of real anchor tokens to expose from each remote cluster"
    )
    parser.add_argument(
        "--cluster_remote_anchor_weight_mode",
        type=str,
        default="none",
        choices=["none", "const2", "logn", "sqrtn"],
        help="Optional K/V scaling mode for remote anchor tokens"
    )
    parser.add_argument(
        "--cluster_remote_anchor_scale_target",
        type=str,
        default="v",
        choices=["k", "v", "kv", "k+v"],
        help="Apply remote anchor scaling to K, V, or both"
    )
    parser.add_argument(
        "--enable_cluster_remote_anchor_value_coreset",
        action="store_true",
        help="Use feature-space remote anchor coreset selection and replace each anchor V with its assigned-token mean"
    )
    parser.add_argument(
        "--enable_cluster_remote_anchor_incremental_value_cache",
        action="store_true",
        help="Cache remote-anchor means: recompute and overwrite updated groups, reuse unchanged groups (legacy flag name)"
    )
    parser.add_argument(
        "--enable_cluster_remote_anchor_merged_key_fast_path",
        action="store_true",
        help="Gather local and remote-anchor K with one merged index tensor when value coreset mode is enabled"
    )
    parser.add_argument(
        "--enable_cluster_remote_anchor_bias_feature_fast_path",
        dest="enable_cluster_remote_anchor_bias_feature_fast_path",
        action="store_true",
        help="Encode logn remote-anchor logit bias into Q/K feature dims instead of passing an additive SDPA mask"
    )
    parser.add_argument(
        "--disable_cluster_remote_anchor_bias_feature_fast_path",
        dest="enable_cluster_remote_anchor_bias_feature_fast_path",
        action="store_false",
        help="Use the original additive SDPA mask for logn remote-anchor logit bias"
    )
    parser.set_defaults(enable_cluster_remote_anchor_bias_feature_fast_path=False)
    parser.add_argument(
        "--disable_cluster_late_kv_expansion", dest="enable_cluster_late_kv_expansion",
        action="store_false", default=True,
        help="Ablation: expand all KV heads before sparse attention preparation"
    )
    parser.add_argument(
        "--cluster_attention_kernel", choices=("sdpa",), default="sdpa",
        help="Attention kernel: PyTorch SDPA"
    )
    parser.add_argument(
        "--enable_cluster_grouped_sdpa",
        action="store_true",
        help="Batch compatible cluster SDPA jobs with exact padding masks to reduce per-cluster launches"
    )
    parser.add_argument(
        "--cluster_grouped_sdpa_max_padding_ratio",
        type=float,
        default=1.25,
        help="Maximum padded QK work / original QK work allowed when batching non-identical cluster SDPA jobs"
    )
    parser.add_argument(
        "--enable_cluster_attention_density",
        action="store_true",
        help="Analysis only: collect per-layer cluster-mask density; leave disabled for inference latency measurements"
    )
    parser.add_argument(
        "--enable_cluster_anchor_diagnostics",
        action="store_true",
        help="Collect low-overhead remote-anchor count, K/V length, and grouped-SDPA diagnostics"
    )
    parser.add_argument(
        "--cluster_remote_anchor_selection_budget_ms",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; remote anchor coreset selection is no longer truncated by a time budget"
    )
    parser.add_argument(
        "--cluster_remote_anchor_logit_bias_alpha",
        type=float,
        default=1.0,
        help="Multiplier for log(group_size) attention logit bias when value coreset mode and logn weighting are enabled"
    )

    # ============ Cluster Relevance Routing ============
    parser.add_argument(
        "--disable_cluster_relevance_routing",
        dest="enable_cluster_relevance_routing",
        action="store_false",
        help="Disable intra-layer cluster QK relevance routing for remote sparse attention clusters"
    )
    parser.set_defaults(enable_cluster_relevance_routing=True)
    parser.add_argument(
        "--cluster_relevance_topk_ratio",
        type=float,
        default=0.0,
        help="Fraction of non-local clusters to keep as QK-relevant remote clusters"
    )
    parser.add_argument(
        "--cluster_relevance_use_size_bias",
        action="store_true",
        help="Add log cluster-size bias to key-cluster relevance scores"
    )
    parser.add_argument(
        "--cluster_relevance_head_reduce",
        type=str,
        default="mean",
        choices=["mean"],
        help="How to reduce per-head cluster relevance scores into one layer-shared routing map"
    )
    parser.add_argument(
        "--enable_cluster_relevance_optimizations",
        action="store_true",
        help="Enable experimental QK relevance static-map cache and within-step routing reuse"
    )
    parser.add_argument(
        "--cluster_relevance_refresh_every_n_layers",
        type=int,
        default=4,
        help="Recompute QK relevance routing every N transformer layers and reuse between refreshes"
    )
    parser.add_argument(
        "--disable_cluster_relevance_reuse_within_step",
        dest="cluster_relevance_reuse_within_step",
        action="store_false",
        help="Disable within-step reuse of cluster relevance routing"
    )
    parser.set_defaults(cluster_relevance_reuse_within_step=True)
    parser.add_argument(
        "--enable_module_timing",
        action="store_true",
        help="Enable fine-grained module timing; this synchronizes CUDA work and should stay off for latency runs"
    )
    parser.add_argument(
        "--enable_cluster_fused_triton_attention",
        action="store_true",
        help="Enable experimental Triton fused local cluster attention for the no-remote-anchor path"
    )
    parser.add_argument(
        "--disable_temporal_selective_denoising",
        dest="enable_temporal_selective_denoising",
        action="store_false",
        help="Disable region-selective denoising and all cross-timestep prediction/KV reuse",
    )
    parser.set_defaults(enable_temporal_selective_denoising=True)
    parser.add_argument(
        "--disable_region_sparse_attention",
        dest="enable_region_sparse_attention",
        action="store_false",
        help="Disable region-sparse image self-attention while retaining temporal selection",
    )
    parser.set_defaults(enable_region_sparse_attention=True)
    parser.add_argument(
        "--cluster_attention_backend",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "grouped_sdpa", "triton"],
        help="Requested image cluster-attention backend; Triton is valid only with zero remote anchors",
    )

    return parser.parse_args()
