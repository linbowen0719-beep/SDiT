# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
import torch
from ..utils import ras_manager
from typing import Callable, Optional, Tuple, Union
from torch import nn
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F


def _cuda_timed(name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) with CUDA event timing, record into module_time_stats."""
    if not getattr(ras_manager.MANAGER, "enable_module_timing", False):
        return fn(*args, **kwargs)
    if torch.cuda.is_available():
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        result = fn(*args, **kwargs)
        end_evt.record()
        end_evt.synchronize()
        elapsed = start_evt.elapsed_time(end_evt) / 1000.0
    else:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
    ras_manager.MANAGER.add_module_timing(name, elapsed)
    return result


def _wall_timed(name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) with wall-clock timing (cpu sync path), record into module_time_stats."""
    if not getattr(ras_manager.MANAGER, "enable_module_timing", False):
        return fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ras_manager.MANAGER.add_module_timing(name, time.perf_counter() - t0)
    return result


_ANCHOR_REFRESH_MODULE = None


def _get_anchor_refresh_module():
    """Resolve the optional CUDA backend once; ordinary SDPA needs no Triton."""
    global _ANCHOR_REFRESH_MODULE
    if _ANCHOR_REFRESH_MODULE is None:
        try:
            from . import triton_anchor_refresh
        except ImportError:
            _ANCHOR_REFRESH_MODULE = False
        else:
            _ANCHOR_REFRESH_MODULE = triton_anchor_refresh
    return _ANCHOR_REFRESH_MODULE or None


class RASLuminaAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LuminaNextDiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # These fields are also used by spatial-only attention.  Initializing
        # them based on sample_ratio made sample_ratio=1.0 fail before the
        # first sparse-attention call.
        self.k_cache = None
        self.v_cache = None
        self.cache_in_cluster_order = False
        self.remote_anchor_value_mean_cache = None
        self.remote_anchor_value_cache_version = -1
        self.remote_anchor_value_cache_shape = None

    def _reset_remote_anchor_value_cache(self):
        self.remote_anchor_value_mean_cache = None
        self.remote_anchor_value_cache_version = -1
        self.remote_anchor_value_cache_shape = None

    def _remote_anchor_value_cache_enabled(self) -> bool:
        manager = ras_manager.MANAGER
        return (
            bool(getattr(manager, "use_cluster_remote_anchors", False))
            and bool(getattr(manager, "enable_cluster_remote_anchor_value_coreset", False))
            and manager.cluster_remote_anchor_token_to_packed_anchor is not None
            and manager.cluster_remote_anchor_packed_counts is not None
            and manager.cluster_remote_anchor_packed_counts.numel() > 0
        )

    @staticmethod
    def _reduce_remote_anchor_means(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Each segment has one output writer; FP32 accumulation avoids a
        # long-lived BF16 difference sum. Reduce native KV heads, before GQA.
        lengths = lengths.to(device=values.device, dtype=torch.long)
        return torch.segment_reduce(
            values.float(), "mean",
            lengths=lengths.view(1, -1).expand(values.shape[0], -1),
            axis=1, unsafe=True,
        ).to(dtype=values.dtype)

    def _ensure_remote_anchor_value_cache(self, value_cache: torch.Tensor) -> bool:
        manager = ras_manager.MANAGER
        if not self._remote_anchor_value_cache_enabled() or value_cache is None or value_cache.ndim != 4:
            self._reset_remote_anchor_value_cache()
            return False
        counts = manager.cluster_remote_anchor_packed_counts
        token_to_anchor = manager.cluster_remote_anchor_token_to_packed_anchor
        order = manager.cluster_remote_anchor_group_token_order
        lengths = manager.cluster_remote_anchor_group_lengths
        if (token_to_anchor.numel() != value_cache.shape[1]
                or order is None or lengths is None
                or order.numel() != value_cache.shape[1]
                or lengths.numel() != counts.numel()):
            self._reset_remote_anchor_value_cache()
            return False
        version = int(manager.cluster_remote_anchor_metadata_version)
        shape = (value_cache.shape[0], counts.numel(), *value_cache.shape[2:])
        if (self.remote_anchor_value_mean_cache is not None
                and self.remote_anchor_value_cache_version == version
                and self.remote_anchor_value_cache_shape == shape
                and self.remote_anchor_value_mean_cache.device == value_cache.device
                and self.remote_anchor_value_mean_cache.dtype == value_cache.dtype):
            return True
        ordered = value_cache
        if not manager.cluster_remote_anchor_group_token_order_is_cluster_order:
            ordered = value_cache.index_select(1, order.to(device=value_cache.device))
        self.remote_anchor_value_mean_cache = self._reduce_remote_anchor_means(ordered, lengths)
        self.remote_anchor_value_cache_version = version
        self.remote_anchor_value_cache_shape = shape
        return True

    def _remote_anchor_value_update_plan(self, active_idx: torch.Tensor):
        # The same immutable active-index tensor is shared by all layers of a
        # step. Validate/sort once here, not inside every attention invocation.
        manager = ras_manager.MANAGER
        mapping = manager.cluster_remote_anchor_token_to_packed_anchor
        version = int(manager.cluster_remote_anchor_metadata_version)
        plan = manager.cluster_remote_anchor_value_update_plan
        if (plan is not None and plan["active_idx"] is active_idx
                and plan["mapping"] is mapping and plan["version"] == version):
            return plan
        groups = mapping.index_select(0, active_idx.to(device=mapping.device, dtype=torch.long))
        order = torch.argsort(groups, stable=True)
        ids, lengths = torch.unique_consecutive(groups.index_select(0, order), return_counts=True)
        if ids.numel() == 0 or bool((ids < 0).any()):
            return None
        expected = manager.cluster_remote_anchor_group_lengths.index_select(0, ids)
        complete = torch.equal(lengths, expected)
        already_sorted = torch.equal(order, torch.arange(order.numel(), device=order.device))
        plan = {"active_idx": active_idx, "mapping": mapping, "version": version,
                "ids": ids, "lengths": lengths, "complete": complete,
                "order": None if already_sorted else order}
        manager.cluster_remote_anchor_value_update_plan = plan
        return plan

    def _try_fused_anchor_and_kv_cache_write(
        self, key_cache, value_cache, active_idx, new_key, new_value,
    ) -> bool:
        """Update complete-group means and token KV together, or retain fallback."""
        manager = ras_manager.MANAGER
        if (not bool(getattr(manager, "enable_cluster_fused_kv_refresh", True))
                or not bool(getattr(manager, "enable_cluster_fused_anchor_refresh", True))
                or not self._remote_anchor_value_cache_enabled()
                or key_cache is None or value_cache is None or active_idx is None
                or not new_value.is_cuda or torch.is_grad_enabled()
                or active_idx.device != new_value.device or active_idx.numel() == 0):
            return False
        refresh = _get_anchor_refresh_module()
        if refresh is None or not self._ensure_remote_anchor_value_cache(value_cache):
            return False
        plan = self._remote_anchor_value_update_plan(active_idx)
        if plan is None or not plan["complete"] or plan["ids"].device != new_value.device:
            return False
        fused_plan = plan.get("fused_refresh")
        if fused_plan is None:
            fused_plan = (refresh.AnchorRefreshPlan(
                plan["ids"], plan["offsets"], active_idx, plan["order"])
                if "offsets" in plan else refresh.build_anchor_refresh_plan(
                    plan["ids"], plan["lengths"], active_idx, plan["order"]))
            plan["fused_refresh"] = fused_plan
        return refresh.try_refresh_cached_anchor_means_and_kv_(
            new_value, self.remote_anchor_value_mean_cache, fused_plan,
            value_cache=value_cache, key_cache=key_cache, new_key=new_key)

    def _update_remote_anchor_value_cache_before_write(
        self, value_cache: torch.Tensor, active_idx: torch.Tensor, new_value: torch.Tensor,
    ) -> None:
        if value_cache is None or active_idx is None or new_value is None or active_idx.numel() == 0:
            return
        if not self._ensure_remote_anchor_value_cache(value_cache):
            return
        plan = self._remote_anchor_value_update_plan(active_idx)
        if plan is None:
            self._reset_remote_anchor_value_cache()
            return
        if plan["complete"]:
            if (bool(getattr(ras_manager.MANAGER, "enable_cluster_fused_anchor_refresh", True))
                    and new_value.is_cuda and active_idx.device == new_value.device
                    and plan["ids"].device == new_value.device
                    and not torch.is_grad_enabled()
                    and new_value.dtype in (torch.float16, torch.bfloat16)
                    and new_value.stride(-1) == 1):
                # Preserve the existing sequential FP32 accumulation, then
                # round and overwrite once. The immutable step plan is shared
                # by all layers; partial groups retain the reconstruction path.
                refresh = _get_anchor_refresh_module()
                if refresh is not None:
                    fused_plan = plan.get("fused_refresh")
                    if fused_plan is None:
                        fused_plan = refresh.build_anchor_refresh_plan(
                            plan["ids"], plan["lengths"], active_idx, plan["order"])
                        plan["fused_refresh"] = fused_plan
                    refresh.refresh_anchor_means_and_kv_(
                        new_value, self.remote_anchor_value_mean_cache, fused_plan, mode="ordered")
                    return
            # Whole groups changed: no old-V gather, differences, or atomic adds.
            ordered = new_value if plan["order"] is None else new_value.index_select(1, plan["order"])
            means = self._reduce_remote_anchor_means(ordered, plan["lengths"])
        else:
            # Boundary expansion / partial groups: reconstruct their correct
            # post-write means without modifying the caller's token KV cache.
            manager = ras_manager.MANAGER
            updated = value_cache.clone()
            updated.index_copy_(1, active_idx.to(device=value_cache.device, dtype=torch.long), new_value)
            if not manager.cluster_remote_anchor_group_token_order_is_cluster_order:
                updated = updated.index_select(1, manager.cluster_remote_anchor_group_token_order)
            all_means = self._reduce_remote_anchor_means(updated, manager.cluster_remote_anchor_group_lengths)
            means = all_means.index_select(1, plan["ids"])
        # Unique group IDs: direct overwrite; untouched groups retain their means.
        self.remote_anchor_value_mean_cache.index_copy_(1, plan["ids"], means)

    def _remote_anchor_value_block_from_cache(self, value: torch.Tensor) -> Optional[torch.Tensor]:
        manager = ras_manager.MANAGER
        if (not manager.enable_cluster_remote_anchor_incremental_value_cache
                or self.remote_anchor_value_mean_cache is None
                or manager.cluster_key_padding_mask_all_true is False):
            return None
        group_values = self.remote_anchor_value_mean_cache
        if group_values.shape[2] != value.shape[1]:
            if value.shape[1] % group_values.shape[2] != 0:
                return None
            n_rep = value.shape[1] // group_values.shape[2]
            group_values = group_values.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        return group_values.transpose(1, 2)

    def _compute_remote_anchor_group_values_segment_reduce(
        self,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        manager = ras_manager.MANAGER
        token_order = manager.cluster_remote_anchor_group_token_order
        group_lengths = manager.cluster_remote_anchor_group_lengths
        if (
            token_order is None
            or group_lengths is None
            or group_lengths.numel() == 0
            or not hasattr(torch, "segment_reduce")
        ):
            return None

        group_lengths = group_lengths.to(device=value.device, dtype=torch.long)
        if bool(getattr(manager, "cluster_remote_anchor_group_token_order_is_cluster_order", False)):
            ordered_values = value
        else:
            if token_order.numel() == 0:
                return None
            token_order = token_order.to(device=value.device, dtype=torch.long)
            ordered_values = value.index_select(2, token_order)

        value_lengths = group_lengths.view(1, 1, -1).expand(value.shape[0], value.shape[1], -1)
        if key_padding_mask is None or bool(getattr(manager, "cluster_key_padding_mask_all_true", False)):
            return torch.segment_reduce(
                ordered_values,
                "mean",
                lengths=value_lengths,
                axis=2,
                unsafe=True,
            )

        if bool(getattr(manager, "cluster_remote_anchor_group_token_order_is_cluster_order", False)):
            ordered_mask = key_padding_mask.to(dtype=value.dtype)
        else:
            ordered_mask = key_padding_mask.index_select(1, token_order).to(dtype=value.dtype)
        weighted_values = ordered_values * ordered_mask[:, None, :, None]
        group_sums = torch.segment_reduce(
            weighted_values,
            "sum",
            lengths=value_lengths,
            axis=2,
            unsafe=True,
        )
        mask_lengths = group_lengths.view(1, -1).expand(key_padding_mask.shape[0], -1)
        group_counts = torch.segment_reduce(
            ordered_mask,
            "sum",
            lengths=mask_lengths,
            axis=1,
            unsafe=True,
        ).clamp_min(1.0)
        return group_sums / group_counts[:, None, :, None]

    def _ensure_cache_cluster_order(self):
        if (
            not ras_manager.MANAGER.has_cluster_permutation()
            or self.cache_in_cluster_order
            or self.k_cache is None
            or self.v_cache is None
        ):
            return

        self.k_cache = ras_manager.MANAGER.timed_cluster_reorder(
            "permute_k_cache", ras_manager.MANAGER.permute_patch_tensor, self.k_cache, 1
        )
        self.v_cache = ras_manager.MANAGER.timed_cluster_reorder(
            "permute_v_cache", ras_manager.MANAGER.permute_patch_tensor, self.v_cache, 1
        )
        self.cache_in_cluster_order = True

    def _concat_attention_inputs(
        self,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
        key_mask: torch.Tensor,
        extra_key_block: torch.Tensor,
        extra_value_block: torch.Tensor,
        extra_key_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if key_mask is None and extra_key_mask is None:
            merged_key_mask = None
        else:
            if key_mask is None:
                key_mask = torch.ones(
                    key_block.shape[0],
                    key_block.shape[2],
                    device=key_block.device,
                    dtype=torch.bool,
                )
            if extra_key_mask is None:
                extra_key_mask = torch.ones(
                    extra_key_block.shape[0],
                    extra_key_block.shape[2],
                    device=extra_key_block.device,
                    dtype=torch.bool,
                )
            merged_key_mask = torch.cat([key_mask, extra_key_mask], dim=1)
        return (
            torch.cat([key_block, extra_key_block], dim=2),
            torch.cat([value_block, extra_value_block], dim=2),
            merged_key_mask,
        )

    def _compute_remote_anchor_logit_bias(
        self,
        anchor_weights: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if anchor_weights is None or anchor_weights.numel() == 0:
            return None

        mode = getattr(ras_manager.MANAGER, "cluster_remote_anchor_weight_mode", "none").strip().lower()
        if mode != "logn":
            return None

        alpha = float(getattr(ras_manager.MANAGER, "cluster_remote_anchor_logit_bias_alpha", 1.0))
        if alpha == 0.0:
            return None

        return (
            torch.log(anchor_weights.to(device=device, dtype=torch.float32).clamp_min(1.0)) * alpha
        ).to(device=device, dtype=dtype)

    def _append_remote_anchor_bias_feature(
        self,
        q_block: torch.Tensor,
        key_block: torch.Tensor,
        remote_anchor_bias: torch.Tensor,
        softmax_scale: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        original_dim = q_block.shape[-1]
        attention_scale = float(softmax_scale) if softmax_scale is not None else original_dim ** -0.5
        padded_dim = ((original_dim + 8) // 8) * 8
        extra_dim = padded_dim - original_dim

        q_extra = q_block.new_zeros(q_block.shape[:-1] + (extra_dim,))
        q_extra[..., 0] = 1.0

        key_extra = key_block.new_zeros(key_block.shape[:-1] + (extra_dim,))
        remote_count = int(remote_anchor_bias.numel())
        key_extra[..., -remote_count:, 0] = (
            remote_anchor_bias.to(device=key_block.device, dtype=key_block.dtype) / attention_scale
        )

        return torch.cat([q_block, q_extra], dim=-1), torch.cat([key_block, key_extra], dim=-1), attention_scale

    def _compute_remote_anchor_scales(
        self,
        anchor_weights: torch.Tensor,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        mode = getattr(ras_manager.MANAGER, "cluster_remote_anchor_weight_mode", "none").strip().lower()
        if mode == "none" or anchor_weights.numel() == 0:
            return None
        if mode == "const2":
            scales = torch.full_like(anchor_weights, 2.0)
        elif mode == "logn":
            scales = torch.log1p(anchor_weights)
        elif mode == "sqrtn":
            scales = torch.sqrt(anchor_weights)
        else:
            return None
        return scales.to(dtype=dtype).clamp_min(1.0)

    def _compute_remote_anchor_group_values(
        self,
        value: torch.Tensor,
        group_indices: torch.Tensor,
        group_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if group_indices is None or group_mask is None or group_indices.numel() == 0:
            return None

        group_indices = group_indices.to(device=value.device, dtype=torch.long)
        group_mask = group_mask.to(device=value.device, dtype=torch.bool)
        num_anchors, group_size = group_indices.shape
        flat_indices = group_indices.reshape(-1)
        grouped_values = value.index_select(2, flat_indices).view(
            value.shape[0],
            value.shape[1],
            num_anchors,
            group_size,
            value.shape[-1],
        )
        grouped_key_mask = key_padding_mask.index_select(1, flat_indices).view(
            key_padding_mask.shape[0],
            num_anchors,
            group_size,
        )
        valid_mask = grouped_key_mask & group_mask.unsqueeze(0)
        weights = valid_mask[:, None, :, :, None].to(dtype=value.dtype)
        value_sum = (grouped_values * weights).sum(dim=3)
        value_count = weights.sum(dim=3).clamp_min(1.0)
        return value_sum / value_count

    def _build_remote_anchor_biased_mask(
        self,
        key_mask: torch.Tensor,
        q_len: int,
        num_heads: int,
        remote_anchor_bias: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del q_len, num_heads
        valid_mask = key_mask[:, None, None, :]
        attn_mask = torch.zeros(valid_mask.shape, device=key_mask.device, dtype=dtype)
        if remote_anchor_bias.numel() > 0:
            key_bias = torch.zeros(key_mask.shape[1], device=key_mask.device, dtype=dtype)
            key_bias[-remote_anchor_bias.numel():] = remote_anchor_bias.to(device=key_mask.device, dtype=dtype)
            attn_mask = attn_mask + key_bias.view(1, 1, 1, -1)
        attn_mask = attn_mask.masked_fill(~valid_mask, torch.finfo(dtype).min)
        return attn_mask

    def _build_remote_anchor_bias_only_mask(
        self,
        key_len: int,
        remote_anchor_bias: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        bias = remote_anchor_bias.to(device=device, dtype=dtype)
        pad_len = key_len - bias.numel()
        return F.pad(bias, (pad_len, 0), value=0.0).view(1, 1, 1, key_len)

    def _compute_cluster_qk_score_head(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cluster_labels: torch.Tensor,
        key_cluster_labels: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must have shape [B, H, N, D]")
        if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
            raise ValueError("query and key must have matching batch/head/head_dim shapes")
        if query.device != key.device:
            raise ValueError("query and key must be on the same device")

        cluster_labels = cluster_labels.to(device=query.device, dtype=torch.long)
        if cluster_labels.numel() != query.shape[2]:
            raise ValueError(
                f"query cluster_labels length {cluster_labels.numel()} must match query length {query.shape[2]}"
            )
        if key_cluster_labels is None:
            key_cluster_labels = cluster_labels
        key_cluster_labels = key_cluster_labels.to(device=key.device, dtype=torch.long)
        if key_cluster_labels.numel() != key.shape[2]:
            raise ValueError(
                f"key_cluster_labels length {key_cluster_labels.numel()} must match key length {key.shape[2]}"
            )

        manager = ras_manager.MANAGER
        use_relevance_optimizations = bool(getattr(manager, "enable_cluster_relevance_optimizations", False))
        use_cached_cluster_map = (
            use_relevance_optimizations
            and manager.cluster_ids_tensor is not None
            and manager.cluster_compact_labels is not None
            and manager.cluster_block_lengths is not None
            and key_cluster_labels.numel() == manager.cluster_compact_labels.numel()
        )
        if use_cached_cluster_map:
            cluster_ids = manager.cluster_ids_tensor.to(device=key.device)
            key_inverse = manager.cluster_compact_labels.to(device=key.device)
            counts = manager.cluster_block_lengths.to(device=key.device)
        else:
            cluster_ids, key_inverse, counts = torch.unique(
                key_cluster_labels,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
        num_clusters = cluster_ids.numel()
        if use_cached_cluster_map:
            query_inverse = manager.cluster_ids_to_compact(cluster_labels)
            if query_inverse is None:
                query_inverse = torch.searchsorted(cluster_ids, cluster_labels)
        else:
            query_inverse = torch.searchsorted(cluster_ids, cluster_labels)
        if query_inverse.numel() > 0 and not query_inverse.is_cuda:
            if (
                query_inverse.max().item() >= num_clusters
                or query_inverse.min().item() < 0
                or not torch.equal(cluster_ids.index_select(0, query_inverse), cluster_labels)
            ):
                raise ValueError("query cluster labels must be present in key cluster labels")

        query_group = query.new_zeros(query.shape[0], query.shape[1], num_clusters, query.shape[-1])
        key_group = key.new_zeros(key.shape[0], key.shape[1], num_clusters, key.shape[-1])

        def _do_scatter():
            query_group.index_add_(2, query_inverse, query)
            key_group.index_add_(2, key_inverse, key)
        _cuda_timed("rel_qk_scatter_index_add", _do_scatter)

        query_counts = torch.bincount(query_inverse, minlength=num_clusters)
        query_counts_for_mean = query_counts.to(device=query.device, dtype=query.dtype).clamp_min(1).view(1, 1, -1, 1)
        key_counts_for_mean = counts.to(device=key.device, dtype=key.dtype).clamp_min(1).view(1, 1, -1, 1)
        query_group = query_group / query_counts_for_mean
        key_group = key_group / key_counts_for_mean

        scale = float(softmax_scale) if softmax_scale is not None else query.shape[-1] ** -0.5
        score_head = _cuda_timed(
            "rel_cluster_cc_matmul",
            torch.matmul,
            query_group,
            key_group.transpose(-1, -2),
        ) * scale
        return score_head, cluster_ids, counts.to(device=query.device)

    def compute_cluster_relevance_from_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cluster_labels: torch.Tensor,
        key_cluster_labels: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        use_size_bias: bool = False,
        head_reduce: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate cluster-to-cluster routing relevance from already-computed Q/K.

        This is only a routing estimator. Final attention is still exact token-level
        SDPA over the selected token clusters.
        """
        score_head, cluster_ids, counts = self._compute_cluster_qk_score_head(
            query,
            key,
            cluster_labels,
            key_cluster_labels=key_cluster_labels,
            softmax_scale=softmax_scale,
        )
        head_reduce = head_reduce.strip().lower()
        if head_reduce != "mean":
            raise ValueError(f"Unsupported cluster_relevance_head_reduce={head_reduce!r}; only 'mean' is supported")
        score_layer = score_head.mean(dim=1)

        if use_size_bias:
            size_bias = torch.log(counts.to(device=query.device, dtype=score_layer.dtype).clamp_min(1))
            score_layer = score_layer + size_bias.view(1, 1, -1)

        return score_layer, cluster_ids, counts.to(device=query.device)

    def select_remote_clusters(
        self,
        score_layer: torch.Tensor,
        active_cluster_ids: torch.Tensor,
        local_cluster_mask: torch.Tensor,
        cluster_ids: torch.Tensor,
        topk_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if score_layer.ndim != 3:
            raise ValueError("score_layer must have shape [B, C, C]")
        # The sparse gather path uses one kept-cluster set for the whole batch.
        # Average batch relevance maps into a single routing map before top-k.
        if score_layer.shape[0] != 1:
            score_layer = score_layer.mean(dim=0, keepdim=True)

        topk_ratio = max(float(topk_ratio), 0.0)
        active_cluster_ids = active_cluster_ids.to(device=cluster_ids.device, dtype=torch.long)
        local_cluster_mask = local_cluster_mask.to(device=score_layer.device, dtype=torch.bool)
        if active_cluster_ids.numel() == 0 or topk_ratio <= 0.0:
            empty = torch.empty(active_cluster_ids.numel(), 0, device=score_layer.device, dtype=torch.long)
            counts = torch.zeros(active_cluster_ids.numel(), device=score_layer.device, dtype=torch.long)
            return empty, counts, local_cluster_mask

        cluster_ids_for_search = cluster_ids.to(device=active_cluster_ids.device)
        active_compact = torch.searchsorted(cluster_ids_for_search, active_cluster_ids)
        if active_compact.numel() == 0:
            raise ValueError("active_cluster_ids must be present in cluster_ids")
        if not active_compact.is_cuda:
            if (
                active_compact.max().item() >= cluster_ids_for_search.numel()
                or not torch.equal(cluster_ids_for_search.index_select(0, active_compact), active_cluster_ids)
            ):
                raise ValueError("active_cluster_ids must be present in cluster_ids")

        candidate_mask = ~local_cluster_mask
        candidate_counts = candidate_mask.sum(dim=1)
        remote_counts = torch.ceil(candidate_counts.to(dtype=torch.float32) * topk_ratio).to(dtype=torch.long)
        remote_counts = torch.minimum(remote_counts, candidate_counts)
        max_remote = int(remote_counts.max().item()) if remote_counts.numel() > 0 else 0
        if max_remote == 0:
            empty = torch.empty(active_cluster_ids.numel(), 0, device=score_layer.device, dtype=torch.long)
            return empty, remote_counts, local_cluster_mask

        scores = score_layer[0].index_select(0, active_compact.to(device=score_layer.device))
        scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
        top_compact = torch.topk(scores, k=max_remote, dim=-1).indices
        remote_cluster_ids = cluster_ids.to(device=score_layer.device).index_select(0, top_compact.reshape(-1))
        remote_cluster_ids = remote_cluster_ids.view(top_compact.shape)
        return remote_cluster_ids, remote_counts, local_cluster_mask

    def _build_local_cluster_mask(
        self,
        active_cluster_ids: torch.Tensor,
        cluster_ids: torch.Tensor,
        cluster_labels: torch.Tensor,
    ) -> torch.Tensor:
        manager = ras_manager.MANAGER
        if (
            manager.cluster_local_cluster_mask is not None
            and manager.cluster_local_cluster_mask.shape[1] == cluster_ids.numel()
        ):
            cached_mask = manager.cluster_local_cluster_mask.to(device=cluster_labels.device)
            cluster_ids_for_search = cluster_ids.to(device=cluster_labels.device)
            active_cluster_ids_device = active_cluster_ids.to(device=cluster_labels.device, dtype=torch.long)
            active_compact = torch.searchsorted(cluster_ids_for_search, active_cluster_ids_device)
            return cached_mask.index_select(0, active_compact)

        local_cluster_mask = torch.zeros(
            active_cluster_ids.numel(),
            cluster_ids.numel(),
            device=cluster_labels.device,
            dtype=torch.bool,
        )
        cluster_ids_for_search = cluster_ids.to(device=cluster_labels.device)
        for row, cluster_id in enumerate(active_cluster_ids.detach().cpu().tolist()):
            key_indices = manager.cluster_key_indices[int(cluster_id)].to(device=cluster_labels.device)
            local_ids = torch.unique(cluster_labels.index_select(0, key_indices), sorted=True)
            local_compact = torch.searchsorted(cluster_ids_for_search, local_ids)
            valid = (
                (local_compact < cluster_ids_for_search.numel())
                & (cluster_ids_for_search.index_select(0, local_compact.clamp_max(cluster_ids_for_search.numel() - 1)) == local_ids)
            )
            if valid.any():
                local_cluster_mask[row, local_compact[valid]] = True
        return local_cluster_mask

    def _cluster_ids_to_token_indices(
        self,
        cluster_ids: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        manager = ras_manager.MANAGER
        if cluster_ids.numel() == 0 or manager.cluster_block_ranges is None:
            return None

        index_blocks = []
        for cluster_id in cluster_ids.detach().cpu().tolist():
            block_range = manager.cluster_block_ranges.get(int(cluster_id))
            if block_range is None:
                continue
            start, end = block_range
            if end > start:
                index_blocks.append(torch.arange(start, end, device=device, dtype=torch.long))
        if not index_blocks:
            return None
        return torch.cat(index_blocks, dim=0)

    def _remote_cluster_ids_to_padded_token_indices(
        self,
        cluster_ids: torch.Tensor,
        remote_count: torch.Tensor,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        manager = ras_manager.MANAGER
        if cluster_ids.numel() == 0:
            return None, None

        cluster_ids = cluster_ids.to(device=device, dtype=torch.long)
        if manager.cluster_token_indices_padded is None or manager.cluster_token_indices_mask is None:
            count = int(remote_count.item()) if torch.is_tensor(remote_count) else int(remote_count)
            token_indices = self._cluster_ids_to_token_indices(cluster_ids[:count], device=device)
            if token_indices is None:
                return None, None
            return token_indices, torch.ones(token_indices.numel(), device=device, dtype=torch.bool)

        compact = manager.cluster_ids_to_compact(cluster_ids)
        if compact is None:
            count = int(remote_count.item()) if torch.is_tensor(remote_count) else int(remote_count)
            token_indices = self._cluster_ids_to_token_indices(cluster_ids[:count], device=device)
            if token_indices is None:
                return None, None
            return token_indices, torch.ones(token_indices.numel(), device=device, dtype=torch.bool)

        remote_slots = cluster_ids.numel()
        valid_remote = torch.arange(remote_slots, device=device, dtype=torch.long) < remote_count.to(
            device=device,
            dtype=torch.long,
        )
        valid_remote = valid_remote & (compact >= 0)
        compact = compact.clamp_min(0)

        token_indices_padded = manager.cluster_token_indices_padded.to(device=device)
        token_indices_mask = manager.cluster_token_indices_mask.to(device=device)
        token_indices = token_indices_padded.index_select(0, compact).reshape(-1)
        token_mask = token_indices_mask.index_select(0, compact) & valid_remote[:, None]
        return token_indices, token_mask.reshape(-1)

    def _static_global_head_indices(
        self,
        num_heads: int,
        global_head_count: int,
        device: torch.device,
        selection_mode: str,
    ) -> torch.Tensor:
        global_head_count = max(0, min(int(global_head_count), int(num_heads)))
        if global_head_count == 0:
            return torch.empty(0, device=device, dtype=torch.long)
        if global_head_count == num_heads:
            return torch.arange(num_heads, device=device, dtype=torch.long)

        selection_mode = selection_mode.strip().lower()
        if selection_mode in ("uniform", "even"):
            selected = torch.linspace(
                0,
                num_heads - 1,
                steps=global_head_count,
                device=device,
            ).round().to(dtype=torch.long)
            selected = torch.unique(selected, sorted=True)
            if selected.numel() < global_head_count:
                all_heads = torch.arange(num_heads, device=device, dtype=torch.long)
                remaining_mask = torch.ones(num_heads, device=device, dtype=torch.bool)
                remaining_mask[selected] = False
                selected = torch.cat(
                    [selected, all_heads[remaining_mask][: global_head_count - selected.numel()]],
                    dim=0,
                ).sort().values
            return selected

        return torch.arange(global_head_count, device=device, dtype=torch.long)

    def _select_global_heads_from_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        active_idx: torch.Tensor,
        softmax_scale: Optional[float],
        global_head_count: int,
    ) -> torch.Tensor:
        """
        Select global heads online from current Q/K.

        The score is the average cluster-softmax probability mass each head assigns
        from active query clusters to non-local key clusters. Heads with higher
        remote mass keep full cached K/V; the remaining heads use local sparse K/V.
        """
        manager = ras_manager.MANAGER
        num_heads = query.shape[1]
        selection_mode = getattr(manager, "cluster_global_head_selection", "online")
        selection_mode = selection_mode.strip().lower()
        global_head_count = max(0, min(int(global_head_count), int(num_heads)))

        if global_head_count == 0:
            return torch.empty(0, device=query.device, dtype=torch.long)
        if global_head_count == num_heads:
            return torch.arange(num_heads, device=query.device, dtype=torch.long)
        if selection_mode in ("first", "uniform", "even"):
            return self._static_global_head_indices(
                num_heads,
                global_head_count,
                query.device,
                selection_mode,
            )
        if selection_mode != "online":
            raise ValueError(
                f"Unsupported cluster_global_head_selection={selection_mode!r}; "
                "expected 'online', 'uniform', or 'first'"
            )
        if manager.cluster_sorted_labels is None or manager.cluster_key_indices is None:
            return self._static_global_head_indices(num_heads, global_head_count, query.device, "uniform")

        cluster_labels = manager.cluster_sorted_labels.to(device=active_idx.device)
        active_labels = cluster_labels.index_select(0, active_idx)
        if active_labels.numel() == 0:
            return self._static_global_head_indices(num_heads, global_head_count, query.device, "uniform")

        active_cluster_ids = torch.unique(active_labels, sorted=True)
        try:
            score_head, cluster_ids, _ = self._compute_cluster_qk_score_head(
                query,
                key,
                active_labels.to(device=query.device),
                key_cluster_labels=cluster_labels.to(device=key.device),
                softmax_scale=softmax_scale,
            )
            local_cluster_mask = self._build_local_cluster_mask(
                active_cluster_ids,
                cluster_ids,
                cluster_labels.to(device=query.device),
            )
        except ValueError:
            return self._static_global_head_indices(num_heads, global_head_count, query.device, "uniform")

        cluster_ids_for_search = cluster_ids.to(device=active_cluster_ids.device)
        active_compact = torch.searchsorted(
            cluster_ids_for_search,
            active_cluster_ids.to(device=cluster_ids_for_search.device),
        )
        valid_active = (
            (active_compact < cluster_ids_for_search.numel())
            & (
                cluster_ids_for_search.index_select(
                    0,
                    active_compact.clamp_max(cluster_ids_for_search.numel() - 1),
                )
                == active_cluster_ids.to(device=cluster_ids_for_search.device)
            )
        )
        if not bool(valid_active.all()):
            return self._static_global_head_indices(num_heads, global_head_count, query.device, "uniform")

        remote_mask = ~local_cluster_mask.to(device=score_head.device)
        if not bool(remote_mask.any()):
            return self._static_global_head_indices(num_heads, global_head_count, query.device, "uniform")

        active_scores = score_head.index_select(2, active_compact.to(device=score_head.device))
        active_probs = torch.softmax(active_scores.to(dtype=torch.float32), dim=-1)
        remote_mass = (
            active_probs
            * remote_mask.to(dtype=active_probs.dtype).view(1, 1, remote_mask.shape[0], remote_mask.shape[1])
        ).sum(dim=-1)
        head_scores = remote_mass.mean(dim=(0, 2))
        global_heads = torch.topk(head_scores, k=global_head_count, dim=0).indices
        return global_heads.sort().values.to(device=query.device)

    def _cluster_triton_local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        active_idx: torch.Tensor,
        active_cluster_ids: torch.Tensor,
        key_padding_all_true: bool,
        softmax_scale: Optional[float],
    ) -> Optional[torch.Tensor]:
        manager = ras_manager.MANAGER
        self._last_triton_fallback_reason = None

        def _fallback(reason):
            self._last_triton_fallback_reason = reason
            return None

        if not bool(getattr(manager, "enable_cluster_fused_triton_attention", False)):
            return _fallback("triton_not_requested")
        if not key_padding_all_true:
            return _fallback("key_padding_mask")
        if getattr(self, "cache_in_cluster_order", True) is False:
            return _fallback("kv_cache_not_cluster_ordered")
        if (
            bool(getattr(manager, "use_cluster_remote_anchors", False))
            and int(getattr(manager, "cluster_remote_anchor_count", 0)) > 0
        ):
            return _fallback("remote_anchors_m_gt_0")
        if manager.cluster_token_indices_padded is None or manager.cluster_token_indices_mask is None:
            return _fallback("missing_cluster_token_metadata")
        if manager.cluster_local_block_indices_padded is None or manager.cluster_local_block_counts is None:
            return _fallback("missing_local_block_metadata")
        if manager.cluster_block_starts is None or manager.cluster_block_lengths is None:
            return _fallback("missing_cluster_block_metadata")

        max_cluster_tokens = int(getattr(manager, "cluster_max_block_length", 0))
        max_local_blocks = int(getattr(manager, "cluster_local_block_max_count", 0))
        if max_cluster_tokens <= 0 or max_local_blocks <= 0:
            return _fallback("empty_cluster_blocks")

        active_compact = manager.cluster_ids_to_compact(active_cluster_ids)
        if active_compact is None or active_compact.numel() == 0:
            return _fallback("no_active_clusters")
        if bool((active_compact < 0).any().item()):
            return _fallback("unknown_active_cluster")

        from .triton_cluster_attention import cluster_local_attention, cluster_local_span_attention

        if (
            bool(getattr(manager, "cluster_active_queries_sorted", False))
            and manager.cluster_active_query_starts is not None
            and manager.cluster_active_query_lengths is not None
            and manager.cluster_local_key_span_starts is not None
            and manager.cluster_local_key_span_lengths is not None
        ):
            span_output = cluster_local_span_attention(
                query,
                key,
                value,
                active_compact,
                manager.cluster_active_query_starts,
                manager.cluster_active_query_lengths,
                manager.cluster_local_key_span_starts,
                manager.cluster_local_key_span_lengths,
                softmax_scale,
            )
            if span_output is not None:
                return span_output

        # The generic Triton prototype is only competitive for small blocks;
        # large quickshift clusters are faster through PyTorch SDPA.
        if max_cluster_tokens > 128 or max_local_blocks > 32:
            return _fallback("generic_triton_shape_limit")

        output = cluster_local_attention(
            query,
            key,
            value,
            active_idx,
            active_compact,
            manager.cluster_token_indices_padded,
            manager.cluster_token_indices_mask,
            manager.cluster_local_block_indices_padded,
            manager.cluster_local_block_counts,
            manager.cluster_block_starts,
            manager.cluster_block_lengths,
            softmax_scale,
            max_cluster_tokens=max_cluster_tokens,
            max_local_blocks=max_local_blocks,
        )
        if output is None:
            return _fallback("triton_kernel_unavailable")
        return output

    @staticmethod
    def _cluster_sdpa(query, key, value, **kwargs):
        """Expand native KV heads only after sparse gathering and grouping."""
        if query.shape[1] != key.shape[1]:
            if query.shape[1] % key.shape[1] or key.shape[1] != value.shape[1]:
                raise ValueError("Incompatible grouped-query attention heads")
            repeats = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        return F.scaled_dot_product_attention(query, key, value, **kwargs)

    @staticmethod
    def _can_defer_cluster_kv_heads(use_cluster_online_attention, use_ras_full_kv_attention):
        manager = ras_manager.MANAGER
        relevance = (bool(getattr(manager, "enable_cluster_relevance_routing", True))
                     and float(getattr(manager, "cluster_relevance_topk_ratio", 0.2)) > 0
                     and manager.cluster_block_ranges is not None)
        return (
            bool(getattr(manager, "enable_cluster_late_kv_expansion", True))
            and use_cluster_online_attention and not use_ras_full_kv_attention
            and not relevance
            and not bool(getattr(manager, "use_cluster_global_local_head_split", False))
            and not bool(getattr(manager, "enable_cluster_fused_triton_attention", False))
            and manager.attention_analysis_callback is None
        )

    @staticmethod
    def _all_source_keys_valid(mask):
        if mask is None:
            return True
        manager = ras_manager.MANAGER
        if mask is getattr(manager, "lumina_layer_mask", None):
            return bool(manager.lumina_layer_mask_all_true)
        # Standalone callers can change values without changing shape/identity.
        return bool(mask.bool().all().item())






    def _cluster_concat_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        active_idx: torch.Tensor,
        key_padding_mask: torch.Tensor,
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        manager = ras_manager.MANAGER
        active_queries_sorted = bool(getattr(manager, "cluster_active_queries_sorted", False))
        relevance_routing_enabled = (
            bool(getattr(manager, "enable_cluster_relevance_routing", True))
            and float(getattr(manager, "cluster_relevance_topk_ratio", 0.2)) > 0.0
            and manager.cluster_block_ranges is not None
        )
        cluster_labels = manager.cluster_sorted_labels.to(device=active_idx.device)

        active_cluster_counts = None
        active_cluster_starts = None
        use_active_query_ranges = (
            active_queries_sorted
            and manager.cluster_active_cluster_ids is not None
            and manager.cluster_active_query_starts is not None
            and manager.cluster_active_query_lengths is not None
        )
        active_cluster_ids_cpu = None
        if use_active_query_ranges:
            active_cluster_ids_device = manager.cluster_active_cluster_ids.to(device=active_idx.device)
            active_cluster_ids_cpu, active_cluster_starts, active_cluster_counts = (
                manager.get_active_cluster_query_ranges_cpu()
            )
            active_labels = None
        else:
            active_labels = cluster_labels.index_select(0, active_idx)
            if active_queries_sorted:
                active_cluster_ids_device, active_cluster_counts_device = torch.unique_consecutive(
                    active_labels,
                    return_counts=True,
                )
                active_cluster_counts = active_cluster_counts_device.detach().cpu().tolist()
            else:
                active_cluster_ids_device = torch.unique(active_labels, sorted=True)

        if relevance_routing_enabled and active_labels is None:
            active_labels = cluster_labels.index_select(0, active_idx)
        if use_active_query_ranges and relevance_routing_enabled:
            # Relevance routing may change the effective remote KV per active cluster;
            # keep the simpler unsorted-query code path for that experimental mode.
            active_cluster_ids_device, active_cluster_counts_device = torch.unique_consecutive(
                active_labels,
                return_counts=True,
            )
            active_cluster_counts = active_cluster_counts_device.detach().cpu().tolist()
            active_cluster_starts = None
            active_cluster_ids_cpu = None
        key_padding_all_true = bool(getattr(manager, "cluster_key_padding_mask_all_true", False))
        triton_fallback_reason = None
        if not relevance_routing_enabled:
            triton_output = self._cluster_triton_local_attention(
                query,
                key,
                value,
                active_idx,
                active_cluster_ids_device,
                key_padding_all_true,
                softmax_scale,
            )
            if triton_output is not None:
                manager.record_cluster_backend_call("triton")
                return triton_output
            if bool(getattr(manager, "enable_cluster_fused_triton_attention", False)):
                triton_fallback_reason = getattr(
                    self, "_last_triton_fallback_reason", "unknown_triton_fallback"
                )

        output = torch.zeros_like(query)
        if active_cluster_ids_cpu is None:
            active_cluster_ids_cpu = active_cluster_ids_device.detach().cpu().tolist()

        remote_cluster_ids = None
        remote_cluster_counts = None
        if relevance_routing_enabled:
            use_relevance_optimizations = bool(getattr(manager, "enable_cluster_relevance_optimizations", False))
            if not use_relevance_optimizations:
                score_layer, cluster_ids, _ = _wall_timed(
                    "rel_compute_relevance_from_qk",
                    self.compute_cluster_relevance_from_qk,
                    query,
                    key,
                    active_labels.to(device=query.device),
                    key_cluster_labels=cluster_labels.to(device=key.device),
                    softmax_scale=softmax_scale,
                    use_size_bias=bool(getattr(manager, "cluster_relevance_use_size_bias", False)),
                    head_reduce=getattr(manager, "cluster_relevance_head_reduce", "mean"),
                )
                local_cluster_mask = _wall_timed(
                    "rel_build_local_mask",
                    self._build_local_cluster_mask,
                    active_cluster_ids_device,
                    cluster_ids,
                    cluster_labels.to(device=query.device),
                )
                remote_cluster_ids, remote_cluster_counts, _ = _wall_timed(
                    "rel_select_remote_clusters",
                    self.select_remote_clusters,
                    score_layer,
                    active_cluster_ids_device,
                    local_cluster_mask,
                    cluster_ids,
                    float(getattr(manager, "cluster_relevance_topk_ratio", 0.2)),
                )
            else:
                refresh_interval = max(1, int(getattr(manager, "cluster_relevance_refresh_every_n_layers", 4)))
                reuse_enabled = bool(getattr(manager, "cluster_relevance_reuse_within_step", True))
                current_step = int(getattr(manager, "current_step", -1))
                current_layer = int(getattr(manager, "current_layer", 0))
                cache_valid = (
                    reuse_enabled
                    and manager.cluster_relevance_cache_remote_ids is not None
                    and manager.cluster_relevance_cache_remote_counts is not None
                    and int(getattr(manager, "cluster_relevance_cache_step", -1)) == current_step
                    and int(getattr(manager, "cluster_relevance_cache_active_count", -1))
                    == int(active_cluster_ids_device.numel())
                )
                should_refresh = (
                    not cache_valid
                    or refresh_interval <= 1
                    or current_layer % refresh_interval == 0
                )
                if not should_refresh:
                    remote_cluster_ids = manager.cluster_relevance_cache_remote_ids.to(device=query.device)
                    remote_cluster_counts = manager.cluster_relevance_cache_remote_counts.to(device=query.device)
                else:
                    score_layer, cluster_ids, _ = _wall_timed(
                        "rel_compute_relevance_from_qk",
                        self.compute_cluster_relevance_from_qk,
                        query,
                        key,
                        active_labels.to(device=query.device),
                        key_cluster_labels=cluster_labels.to(device=key.device),
                        softmax_scale=softmax_scale,
                        use_size_bias=bool(getattr(manager, "cluster_relevance_use_size_bias", False)),
                        head_reduce=getattr(manager, "cluster_relevance_head_reduce", "mean"),
                    )
                    local_cluster_mask = _wall_timed(
                        "rel_build_local_mask",
                        self._build_local_cluster_mask,
                        active_cluster_ids_device,
                        cluster_ids,
                        cluster_labels.to(device=query.device),
                    )
                    remote_cluster_ids, remote_cluster_counts, _ = _wall_timed(
                        "rel_select_remote_clusters",
                        self.select_remote_clusters,
                        score_layer,
                        active_cluster_ids_device,
                        local_cluster_mask,
                        cluster_ids,
                        float(getattr(manager, "cluster_relevance_topk_ratio", 0.2)),
                    )
                    if reuse_enabled:
                        manager.cluster_relevance_cache_step = current_step
                        manager.cluster_relevance_cache_layer = current_layer
                        manager.cluster_relevance_cache_active_count = int(active_cluster_ids_device.numel())
                        manager.cluster_relevance_cache_remote_ids = remote_cluster_ids.detach()
                        manager.cluster_relevance_cache_remote_counts = remote_cluster_counts.detach()
                        manager.cluster_relevance_cache_cluster_ids = cluster_ids.detach()

        track_merge_timing = bool(getattr(manager, "enable_module_timing", False))
        use_value_coreset = bool(getattr(manager, "enable_cluster_remote_anchor_value_coreset", False))
        remote_anchor_effective = (
            bool(getattr(manager, "use_cluster_remote_anchors", False))
            and int(getattr(manager, "cluster_remote_anchor_count", 0)) > 0
        )
        use_remote_anchor_source_cache = (
            not relevance_routing_enabled
            and remote_anchor_effective
            and use_value_coreset
            and manager.cluster_remote_anchor_source_cluster_ids is not None
            and manager.cluster_remote_anchor_source_indices is not None
        )
        use_remote_anchor_packed_cache = (
            use_remote_anchor_source_cache
            and manager.cluster_remote_anchor_packed_positions is not None
            and manager.cluster_remote_anchor_all_indices is not None
        )
        weight_mode = getattr(manager, "cluster_remote_anchor_weight_mode", "none").strip().lower()
        scale_target = getattr(manager, "cluster_remote_anchor_scale_target", "v").strip().lower()
        if scale_target not in ("k", "v", "kv", "k+v"):
            scale_target = "v"
        remote_key_needs_scaling = weight_mode not in ("none", "logn") and scale_target in ("k", "kv", "k+v")
        use_packed_merged_key_indices = (
            use_remote_anchor_packed_cache
            and bool(getattr(manager, "enable_cluster_remote_anchor_merged_key_fast_path", False))
            and not remote_key_needs_scaling
            and manager.cluster_remote_anchor_merged_key_indices is not None
        )
        use_local_key_span_fast_path = manager.cluster_local_key_spans is not None
        packed_anchor_block_cache = None
        source_anchor_block_cache = {}
        grouped_sdpa_enabled = bool(getattr(manager, "enable_cluster_grouped_sdpa", False))
        sdpa_groups = {}
        sdpa_group_order = []

        def _small_tensor_tuple(tensor):
            if tensor is None:
                return ()
            if not torch.is_tensor(tensor):
                return tuple(int(item) for item in tensor)
            if tensor.numel() == 0:
                return ()
            return tuple(int(item) for item in tensor.detach().cpu().reshape(-1).tolist())

        def _small_float_tensor_tuple(tensor):
            if tensor is None:
                return ()
            if not torch.is_tensor(tensor):
                return tuple(float(item) for item in tensor)
            if tensor.numel() == 0:
                return ()
            return tuple(float(item) for item in tensor.detach().cpu().reshape(-1).tolist())

        def _store_attention_output(attention_output, q_positions, q_start, q_end):
            if q_positions is None:
                output[:, :, q_start:q_end] = attention_output
            else:
                output.index_copy_(
                    2,
                    q_positions,
                    attention_output,
                )

        def _pad_sequence_dim(tensor, target_len):
            pad_len = int(target_len) - int(tensor.shape[2])
            if pad_len <= 0:
                return tensor
            return F.pad(tensor, (0, 0, 0, pad_len))

        def _pad_mask_dim(mask, target_len, dim, value):
            pad_len = int(target_len) - int(mask.shape[dim])
            if pad_len <= 0:
                return mask
            pad_shape = list(mask.shape)
            pad_shape[dim] = pad_len
            return torch.cat([mask, mask.new_full(pad_shape, value)], dim=dim)

        def _expand_job_mask(mask, batch_size, max_q_len, max_k_len, device, dtype):
            if mask is None:
                return None
            mask = mask.to(device=device)
            if mask.shape[0] == 1 and batch_size != 1:
                mask = mask.expand(batch_size, *mask.shape[1:])
            if mask.shape[2] not in (1, max_q_len):
                pad_value = False if mask.dtype == torch.bool else torch.finfo(mask.dtype).min
                mask = _pad_mask_dim(mask, max_q_len, 2, pad_value)
            pad_value = False if mask.dtype == torch.bool else torch.finfo(mask.dtype).min
            mask = _pad_mask_dim(mask, max_k_len, 3, pad_value)
            if mask.dtype != torch.bool and mask.dtype != dtype:
                mask = mask.to(dtype=dtype)
            return mask

        def _make_padding_mask(batch_size, real_k_len, max_k_len, device):
            mask = torch.zeros(batch_size, 1, 1, max_k_len, device=device, dtype=torch.bool)
            mask[..., :real_k_len] = True
            return mask

        def _make_sdpa_batch_signature(attention_query, attention_key, value_block, local_attention_mask, attention_scale):
            if local_attention_mask is None:
                mask_kind = "none"
            elif local_attention_mask.dtype == torch.bool:
                mask_kind = "bool"
            else:
                mask_kind = f"float:{local_attention_mask.dtype}"
            return (
                int(attention_query.shape[0]),
                int(attention_query.shape[1]),
                int(attention_query.shape[-1]),
                int(attention_key.shape[-1]),
                int(value_block.shape[-1]),
                str(attention_query.dtype),
                str(attention_key.dtype),
                str(value_block.dtype),
                mask_kind,
                None if attention_scale is None else float(attention_scale),
            )

        def _run_sdpa_job(job):
            attention_output = self._cluster_sdpa(
                job["query"],
                job["key"],
                job["value"],
                attn_mask=job["mask"],
                scale=job["scale"],
            )
            manager.add_cluster_grouped_sdpa_call(
                1,
                int(job["query"].shape[2]),
                int(job["key"].shape[2]),
            )
            sdpa_reason = triton_fallback_reason
            if sdpa_reason is None and getattr(manager, "cluster_attention_backend", "auto") == "grouped_sdpa":
                sdpa_reason = "grouping_not_applicable_to_single_job"
            manager.record_cluster_backend_call("sdpa", sdpa_reason)
            _store_attention_output(
                attention_output,
                job["q_positions"],
                job["q_start"],
                job["q_end"],
            )

        def _run_batched_sdpa_jobs(jobs):
            batch_size = int(jobs[0]["query"].shape[0])
            max_q_len = max(int(job["query"].shape[2]) for job in jobs)
            max_k_len = max(int(job["key"].shape[2]) for job in jobs)
            mask_kind = "none"
            for job in jobs:
                if job["mask"] is not None:
                    mask_kind = "bool" if job["mask"].dtype == torch.bool else "float"
                    break

            padded_queries = []
            padded_keys = []
            padded_values = []
            masks = []
            needs_padding_mask = any(int(job["key"].shape[2]) != max_k_len for job in jobs)
            for job in jobs:
                q_len = int(job["query"].shape[2])
                k_len = int(job["key"].shape[2])
                padded_queries.append(_pad_sequence_dim(job["query"], max_q_len))
                padded_keys.append(_pad_sequence_dim(job["key"], max_k_len))
                padded_values.append(_pad_sequence_dim(job["value"], max_k_len))

                if mask_kind == "none":
                    if needs_padding_mask:
                        masks.append(_make_padding_mask(batch_size, k_len, max_k_len, job["key"].device))
                else:
                    masks.append(
                        _expand_job_mask(
                            job["mask"],
                            batch_size,
                            max_q_len,
                            max_k_len,
                            job["key"].device,
                            job["query"].dtype,
                        )
                    )

            group_query = torch.cat(padded_queries, dim=0)
            group_key = torch.cat(padded_keys, dim=0)
            group_value = torch.cat(padded_values, dim=0)
            if mask_kind == "none":
                group_mask = torch.cat(masks, dim=0) if needs_padding_mask else None
            else:
                group_mask = torch.cat(masks, dim=0)

            group_output = self._cluster_sdpa(
                group_query,
                group_key,
                group_value,
                attn_mask=group_mask,
                scale=jobs[0]["scale"],
            )
            manager.add_cluster_grouped_sdpa_call(
                len(jobs),
                sum(int(job["query"].shape[2]) for job in jobs),
                max_k_len,
            )
            manager.record_cluster_backend_call("grouped_sdpa", triton_fallback_reason)

            batch_start = 0
            for job in jobs:
                q_len = int(job["query"].shape[2])
                batch_end = batch_start + batch_size
                _store_attention_output(
                    group_output[batch_start:batch_end, :, :q_len],
                    job["q_positions"],
                    job["q_start"],
                    job["q_end"],
                )
                batch_start = batch_end

        def _sdpa_group_padding_ratio(jobs):
            exact_work = 0
            max_q_len = 0
            max_k_len = 0
            for job in jobs:
                q_len = int(job["query"].shape[2])
                k_len = int(job["key"].shape[2])
                exact_work += q_len * k_len
                max_q_len = max(max_q_len, q_len)
                max_k_len = max(max_k_len, k_len)
            padded_work = len(jobs) * max_q_len * max_k_len
            return padded_work / max(1, exact_work)

        def _split_padding_bounded_sdpa_groups(jobs):
            max_padding_ratio = float(getattr(manager, "cluster_grouped_sdpa_max_padding_ratio", 1.25))
            if max_padding_ratio <= 1.0 or len(jobs) <= 1:
                return [[job] for job in jobs]

            sorted_jobs = sorted(
                jobs,
                key=lambda job: (int(job["key"].shape[2]), int(job["query"].shape[2])),
            )
            bounded_groups = []
            current = []
            for job in sorted_jobs:
                candidate = current + [job]
                if len(candidate) == 1 or _sdpa_group_padding_ratio(candidate) <= max_padding_ratio:
                    current = candidate
                    continue
                bounded_groups.append(current)
                current = [job]
            if current:
                bounded_groups.append(current)
            return bounded_groups

        def _enqueue_or_run_sdpa_job(signature, job):
            if not grouped_sdpa_enabled or signature is None:
                _run_sdpa_job(job)
                return

            group = sdpa_groups.get(signature)
            if group is None:
                group = {"jobs": []}
                sdpa_groups[signature] = group
                sdpa_group_order.append(signature)
            group["jobs"].append(job)

        def _flush_grouped_sdpa_jobs():
            for signature in sdpa_group_order:
                group = sdpa_groups[signature]
                jobs = group["jobs"]
                if len(jobs) == 1:
                    _run_sdpa_job(jobs[0])
                    continue
                for bounded_jobs in _split_padding_bounded_sdpa_groups(jobs):
                    if len(bounded_jobs) == 1:
                        _run_sdpa_job(bounded_jobs[0])
                    else:
                        _run_batched_sdpa_jobs(bounded_jobs)

        def _append_signature(parts, item):
            if parts is not None:
                parts.append(item)

        def _get_packed_anchor_block(include_key: bool = True):
            nonlocal packed_anchor_block_cache
            if packed_anchor_block_cache is not None and (
                not include_key or packed_anchor_block_cache[0] is not None
            ):
                return packed_anchor_block_cache

            anchor_indices = manager.cluster_remote_anchor_all_indices
            if anchor_indices is None or anchor_indices.numel() == 0:
                packed_anchor_block_cache = None
                return None

            anchor_indices = anchor_indices.to(device=key.device)
            remote_key_block = key.index_select(2, anchor_indices) if include_key else None
            remote_value_block = self._remote_anchor_value_block_from_cache(value)
            if remote_value_block is None:
                remote_value_block = self._compute_remote_anchor_group_values_segment_reduce(
                    value,
                    key_padding_mask,
                )

            group_indices = manager.cluster_remote_anchor_all_group_indices
            group_mask = manager.cluster_remote_anchor_all_group_mask
            if (
                remote_value_block is None
                and group_indices is not None
                and group_mask is not None
                and group_indices.numel() > 0
                and group_indices.shape[0] == anchor_indices.numel()
            ):
                remote_value_block = self._compute_remote_anchor_group_values(
                    value,
                    group_indices,
                    group_mask,
                    key_padding_mask,
                )
            if remote_value_block is None:
                remote_value_block = value.index_select(2, anchor_indices)

            remote_anchor_logit_bias = None
            remote_anchor_weights = manager.cluster_remote_anchor_all_weights
            if remote_anchor_weights is not None and remote_anchor_weights.numel() > 0:
                remote_anchor_weights = remote_anchor_weights.to(device=value.device)
                weight_mode = getattr(manager, "cluster_remote_anchor_weight_mode", "none").strip().lower()
                if weight_mode == "logn":
                    # Use bias precomputed at scheduling time; cast to current dtype/device.
                    precomputed = getattr(manager, "cluster_remote_anchor_all_logit_bias", None)
                    if precomputed is not None:
                        remote_anchor_logit_bias = precomputed.to(device=key.device, dtype=query.dtype)
                    else:
                        remote_anchor_logit_bias = self._compute_remote_anchor_logit_bias(
                            remote_anchor_weights,
                            dtype=query.dtype,
                            device=key.device,
                        )
                else:
                    remote_anchor_scales = self._compute_remote_anchor_scales(
                        remote_anchor_weights,
                        dtype=remote_value_block.dtype,
                    )
                    if remote_anchor_scales is not None:
                        scale_target = getattr(
                            manager,
                            "cluster_remote_anchor_scale_target",
                            "v",
                        ).strip().lower()
                        if scale_target not in ("k", "v", "kv", "k+v"):
                            scale_target = "v"
                        remote_anchor_scales = remote_anchor_scales.view(1, 1, -1, 1)
                        if scale_target in ("k", "kv", "k+v"):
                            if remote_key_block is None:
                                remote_key_block = key.index_select(2, anchor_indices)
                            remote_key_block = remote_key_block * remote_anchor_scales
                        if scale_target in ("v", "kv", "k+v"):
                            remote_value_block = remote_value_block * remote_anchor_scales

            remote_key_mask = None if key_padding_all_true else key_padding_mask.index_select(1, anchor_indices)
            packed_anchor_block_cache = (
                remote_key_block,
                remote_value_block,
                remote_key_mask,
                remote_anchor_logit_bias,
            )
            return packed_anchor_block_cache

        def _get_source_anchor_block(source_cluster_id: int):
            source_cluster_id = int(source_cluster_id)
            if source_cluster_id in source_anchor_block_cache:
                return source_anchor_block_cache[source_cluster_id]

            anchor_indices = manager.cluster_remote_anchor_source_indices.get(source_cluster_id)
            if anchor_indices is None or anchor_indices.numel() == 0:
                source_anchor_block_cache[source_cluster_id] = None
                return None

            anchor_indices = anchor_indices.to(device=key.device)
            remote_key_block = key.index_select(2, anchor_indices)
            remote_value_block = None

            group_indices = None
            group_mask = None
            if manager.cluster_remote_anchor_source_group_indices is not None:
                group_indices = manager.cluster_remote_anchor_source_group_indices.get(source_cluster_id)
            if manager.cluster_remote_anchor_source_group_mask is not None:
                group_mask = manager.cluster_remote_anchor_source_group_mask.get(source_cluster_id)
            if (
                group_indices is not None
                and group_mask is not None
                and group_indices.shape[0] == anchor_indices.numel()
            ):
                remote_value_block = self._compute_remote_anchor_group_values(
                    value,
                    group_indices,
                    group_mask,
                    key_padding_mask,
                )
            if remote_value_block is None:
                remote_value_block = value.index_select(2, anchor_indices)

            remote_anchor_logit_bias = None
            if manager.cluster_remote_anchor_source_weights is not None:
                remote_anchor_weights = manager.cluster_remote_anchor_source_weights.get(source_cluster_id)
                if remote_anchor_weights is not None and remote_anchor_weights.numel() > 0:
                    remote_anchor_weights = remote_anchor_weights.to(device=value.device)
                    weight_mode = getattr(manager, "cluster_remote_anchor_weight_mode", "none").strip().lower()
                    if weight_mode == "logn":
                        remote_anchor_logit_bias = self._compute_remote_anchor_logit_bias(
                            remote_anchor_weights,
                            dtype=query.dtype,
                            device=key.device,
                        )
                    else:
                        remote_anchor_scales = self._compute_remote_anchor_scales(
                            remote_anchor_weights,
                            dtype=remote_value_block.dtype,
                        )
                        if remote_anchor_scales is not None:
                            scale_target = getattr(
                                manager,
                                "cluster_remote_anchor_scale_target",
                                "v",
                            ).strip().lower()
                            if scale_target not in ("k", "v", "kv", "k+v"):
                                scale_target = "v"
                            remote_anchor_scales = remote_anchor_scales.view(1, 1, -1, 1)
                            if scale_target in ("k", "kv", "k+v"):
                                remote_key_block = remote_key_block * remote_anchor_scales
                            if scale_target in ("v", "kv", "k+v"):
                                remote_value_block = remote_value_block * remote_anchor_scales

            remote_key_mask = key_padding_mask.index_select(1, anchor_indices)
            source_anchor_block_cache[source_cluster_id] = (
                remote_key_block,
                remote_value_block,
                remote_key_mask,
                remote_anchor_logit_bias,
            )
            return source_anchor_block_cache[source_cluster_id]

        for row, cluster_id in enumerate(active_cluster_ids_cpu):
            signature_parts = None
            if active_cluster_counts is not None:
                q_count = int(active_cluster_counts[row])
                if q_count == 0:
                    continue
                q_start = int(active_cluster_starts[row]) if active_cluster_starts is not None else int(sum(active_cluster_counts[:row]))
                q_end = q_start + q_count
                q_positions = None
                q_block = query[:, :, q_start:q_end]
            else:
                q_positions = torch.nonzero(active_labels == cluster_id, as_tuple=False).squeeze(-1)
                if q_positions.numel() == 0:
                    continue
                q_start = None
                q_end = None
                q_block = query.index_select(2, q_positions)
            local_key_span = (
                manager.cluster_local_key_spans.get(int(cluster_id))
                if use_local_key_span_fast_path
                else None
            )
            if local_key_span is not None:
                _append_signature(signature_parts, ("local_span", int(local_key_span[0]), int(local_key_span[1])))
            else:
                _append_signature(signature_parts, ("local_indices", int(cluster_id)))
            key_indices = (
                None
                if local_key_span is not None
                else manager.cluster_key_indices[int(cluster_id)].to(device=key.device)
            )
            packed_positions = None
            packed_merged_key_indices = None
            use_cluster_packed_key_fast_path = False
            if use_remote_anchor_packed_cache:
                packed_positions = manager.cluster_remote_anchor_packed_positions.get(int(cluster_id))
            if (
                not relevance_routing_enabled
                and use_packed_merged_key_indices
                and manager.use_cluster_remote_anchors
                and manager.cluster_remote_anchor_indices is not None
            ):
                packed_merged_key_indices = manager.cluster_remote_anchor_merged_key_indices.get(int(cluster_id))
                use_cluster_packed_key_fast_path = (
                    packed_positions is not None
                    and packed_positions.numel() > 0
                    and packed_merged_key_indices is not None
                    and key_indices is not None
                    and packed_merged_key_indices.numel() > key_indices.numel()
                )
            remote_indices = None
            remote_valid_mask = None
            remote_anchor_logit_bias = None
            if remote_cluster_ids is not None and remote_cluster_counts is not None:
                if track_merge_timing:
                    _t_merge = time.perf_counter()
                remote_indices, remote_valid_mask = self._remote_cluster_ids_to_padded_token_indices(
                    remote_cluster_ids[row],
                    remote_cluster_counts[row],
                    device=key.device,
                )
                if track_merge_timing:
                    manager.add_module_timing("rel_per_cluster_remote_merge", time.perf_counter() - _t_merge)
                if signature_parts is not None:
                    signature_parts.append((
                        "relevance_remote",
                        _small_tensor_tuple(remote_cluster_ids[row]),
                        int(remote_cluster_counts[row].item()) if torch.is_tensor(remote_cluster_counts[row]) else int(remote_cluster_counts[row]),
                    ))
            else:
                _append_signature(signature_parts, ("relevance_remote", ()))
            if local_key_span is not None:
                span_start, span_end = local_key_span
                key_block = key[:, :, span_start:span_end]
                value_block = value[:, :, span_start:span_end]
                key_mask = None if key_padding_all_true and remote_indices is None else key_padding_mask[:, span_start:span_end]
                local_k_len = int(span_end - span_start)
            else:
                key_block = None if use_cluster_packed_key_fast_path else key.index_select(2, key_indices)
                value_block = value.index_select(2, key_indices)
                key_mask = (
                    None
                    if key_padding_all_true and remote_indices is None
                    else key_padding_mask.index_select(1, key_indices)
                )
                local_k_len = int(key_indices.numel())
            if remote_indices is not None and remote_valid_mask is not None:
                remote_key_block = key.index_select(2, remote_indices)
                remote_value_block = value.index_select(2, remote_indices)
                remote_key_mask = key_padding_mask.index_select(1, remote_indices)
                remote_key_mask = remote_key_mask & remote_valid_mask.view(1, -1)
                key_block, value_block, key_mask = self._concat_attention_inputs(
                    key_block,
                    value_block,
                    key_mask,
                    remote_key_block,
                    remote_value_block,
                    remote_key_mask,
                )

            # Keep the original 1-hop exact local attention untouched, and only
            # append a tiny fixed set of remote anchors after the exact kept KV.
            if (
                not relevance_routing_enabled
                and manager.use_cluster_remote_anchors
                and manager.cluster_remote_anchor_indices is not None
            ):
                if use_remote_anchor_packed_cache:
                    if packed_positions is not None and packed_positions.numel() > 0:
                        if signature_parts is not None:
                            signature_parts.append(("packed_anchor_positions", _small_tensor_tuple(packed_positions)))
                        packed_block = _get_packed_anchor_block(include_key=not use_cluster_packed_key_fast_path)
                        if packed_block is not None:
                            packed_positions = packed_positions.to(device=key.device)
                            all_key_block, all_value_block, all_key_mask, all_logit_bias = packed_block
                            remote_value_block = all_value_block.index_select(2, packed_positions)
                            if all_logit_bias is not None:
                                remote_anchor_logit_bias = all_logit_bias.index_select(0, packed_positions)
                            if use_cluster_packed_key_fast_path:
                                key_block = key.index_select(
                                    2,
                                    packed_merged_key_indices.to(device=key.device),
                                )
                                value_block = torch.cat([value_block, remote_value_block], dim=2)
                                if key_mask is not None and all_key_mask is not None:
                                    key_mask = torch.cat(
                                        [key_mask, all_key_mask.index_select(1, packed_positions)],
                                        dim=1,
                                    )
                            else:
                                remote_key_block = all_key_block.index_select(2, packed_positions)
                                remote_key_mask = (
                                    None if all_key_mask is None else all_key_mask.index_select(1, packed_positions)
                                )
                                if key_mask is None and remote_key_mask is None:
                                    key_block = torch.cat([key_block, remote_key_block], dim=2)
                                    value_block = torch.cat([value_block, remote_value_block], dim=2)
                                else:
                                    key_block, value_block, key_mask = self._concat_attention_inputs(
                                        key_block,
                                        value_block,
                                        key_mask,
                                        remote_key_block,
                                        remote_value_block,
                                        remote_key_mask,
                                    )
                    else:
                        _append_signature(signature_parts, ("packed_anchor_positions", ()))
                elif use_remote_anchor_source_cache:
                    source_cluster_ids = manager.cluster_remote_anchor_source_cluster_ids.get(int(cluster_id))
                    _append_signature(signature_parts, ("source_anchor_clusters", tuple(source_cluster_ids) if source_cluster_ids else ()))
                    if source_cluster_ids:
                        remote_key_blocks = []
                        remote_value_blocks = []
                        remote_key_masks = []
                        remote_anchor_logit_bias_blocks = []
                        for source_cluster_id in source_cluster_ids:
                            source_block = _get_source_anchor_block(int(source_cluster_id))
                            if source_block is None:
                                continue
                            remote_key_block, remote_value_block, remote_key_mask, source_logit_bias = source_block
                            remote_key_blocks.append(remote_key_block)
                            remote_value_blocks.append(remote_value_block)
                            remote_key_masks.append(remote_key_mask)
                            if source_logit_bias is not None:
                                remote_anchor_logit_bias_blocks.append(source_logit_bias)
                        if remote_key_blocks:
                            remote_key_block = torch.cat(remote_key_blocks, dim=2)
                            remote_value_block = torch.cat(remote_value_blocks, dim=2)
                            remote_key_mask = torch.cat(remote_key_masks, dim=1)
                            if remote_anchor_logit_bias_blocks:
                                remote_anchor_logit_bias = torch.cat(remote_anchor_logit_bias_blocks, dim=0)
                            key_block, value_block, key_mask = self._concat_attention_inputs(
                                key_block,
                                value_block,
                                key_mask,
                                remote_key_block,
                                remote_value_block,
                                remote_key_mask,
                            )
                else:
                    remote_anchor_indices = manager.cluster_remote_anchor_indices.get(int(cluster_id))
                    if remote_anchor_indices is not None and remote_anchor_indices.numel() > 0:
                        if signature_parts is not None:
                            signature_parts.append(("anchor_indices", _small_tensor_tuple(remote_anchor_indices)))
                        remote_anchor_indices = remote_anchor_indices.to(device=key.device)
                        remote_key_block = key.index_select(2, remote_anchor_indices)
                        remote_value_block = None
                        if (
                            use_value_coreset
                            and manager.cluster_remote_anchor_group_indices is not None
                            and manager.cluster_remote_anchor_group_mask is not None
                        ):
                            remote_anchor_group_indices = manager.cluster_remote_anchor_group_indices.get(int(cluster_id))
                            remote_anchor_group_mask = manager.cluster_remote_anchor_group_mask.get(int(cluster_id))
                            if (
                                remote_anchor_group_indices is not None
                                and remote_anchor_group_mask is not None
                                and remote_anchor_group_indices.shape[0] == remote_anchor_indices.numel()
                            ):
                                remote_value_block = self._compute_remote_anchor_group_values(
                                    value,
                                    remote_anchor_group_indices,
                                    remote_anchor_group_mask,
                                    key_padding_mask,
                                )
                        if remote_value_block is None:
                            remote_value_block = value.index_select(2, remote_anchor_indices)
                        remote_anchor_weights = manager.cluster_remote_anchor_weights.get(int(cluster_id))
                        if remote_anchor_weights is not None and remote_anchor_weights.numel() > 0:
                            if signature_parts is not None:
                                signature_parts.append(("anchor_weights", _small_float_tensor_tuple(remote_anchor_weights)))
                            remote_anchor_weights = remote_anchor_weights.to(device=value.device)
                            weight_mode = getattr(manager, "cluster_remote_anchor_weight_mode", "none").strip().lower()
                            if use_value_coreset and weight_mode == "logn":
                                remote_anchor_logit_bias = self._compute_remote_anchor_logit_bias(
                                    remote_anchor_weights,
                                    dtype=query.dtype,
                                    device=key.device,
                                )
                            else:
                                remote_anchor_scales = self._compute_remote_anchor_scales(
                                    remote_anchor_weights,
                                    dtype=remote_value_block.dtype,
                                )
                                if remote_anchor_scales is not None:
                                    scale_target = getattr(
                                        manager,
                                        "cluster_remote_anchor_scale_target",
                                        "v",
                                    ).strip().lower()
                                    if scale_target not in ("k", "v", "kv", "k+v"):
                                        scale_target = "v"
                                    remote_anchor_scales = remote_anchor_scales.view(1, 1, -1, 1)
                                    if scale_target in ("k", "kv", "k+v"):
                                        remote_key_block = remote_key_block * remote_anchor_scales
                                    if scale_target in ("v", "kv", "k+v"):
                                        remote_value_block = remote_value_block * remote_anchor_scales
                        key_block, value_block, key_mask = self._concat_attention_inputs(
                            key_block,
                            value_block,
                            key_mask,
                            remote_key_block,
                            remote_value_block,
                            key_padding_mask.index_select(1, remote_anchor_indices),
                        )
                    else:
                        _append_signature(signature_parts, ("anchor_indices", ()))
            else:
                _append_signature(signature_parts, ("anchors_disabled",))

            if key_block is None:
                key_block = key.index_select(2, key_indices)

            attention_query = q_block
            attention_key = key_block
            attention_scale = softmax_scale
            use_bias_feature_fast_path = (
                remote_anchor_logit_bias is not None
                and remote_anchor_logit_bias.numel() > 0
                and key_mask is None
                and bool(getattr(manager, "enable_cluster_remote_anchor_bias_feature_fast_path", False))
            )
            if use_bias_feature_fast_path:
                attention_query, attention_key, attention_scale = self._append_remote_anchor_bias_feature(
                    q_block,
                    key_block,
                    remote_anchor_logit_bias,
                    softmax_scale,
                )
                local_attention_mask = None
            elif remote_anchor_logit_bias is not None and remote_anchor_logit_bias.numel() > 0:
                if key_mask is None:
                    local_attention_mask = self._build_remote_anchor_bias_only_mask(
                        key_block.shape[2],
                        remote_anchor_logit_bias,
                        key.device,
                        dtype=query.dtype,
                    )
                else:
                    local_attention_mask = self._build_remote_anchor_biased_mask(
                        key_mask,
                        q_block.shape[2],
                        q_block.shape[1],
                        remote_anchor_logit_bias,
                        dtype=query.dtype,
                    )
            else:
                local_attention_mask = None if key_mask is None else key_mask[:, None, None, :]

            _append_signature(signature_parts, ("bias_feature", bool(use_bias_feature_fast_path)))
            _append_signature(signature_parts, ("mask", local_attention_mask is not None))
            _append_signature(signature_parts, ("qk_dim", int(attention_query.shape[-1])))
            _append_signature(signature_parts, ("scale", None if attention_scale is None else float(attention_scale)))
            signature = (
                _make_sdpa_batch_signature(
                    attention_query,
                    attention_key,
                    value_block,
                    local_attention_mask,
                    attention_scale,
                )
                if grouped_sdpa_enabled
                else None
            )
            total_k_len = int(value_block.shape[2])
            manager.add_cluster_anchor_attention_job(
                int(attention_query.shape[2]),
                local_k_len,
                max(0, total_k_len - local_k_len),
                total_k_len,
            )
            _enqueue_or_run_sdpa_job(
                signature,
                {
                    "query": attention_query,
                    "key": attention_key,
                    "value": value_block,
                    "mask": local_attention_mask,
                    "scale": attention_scale,
                    "q_positions": q_positions,
                    "q_start": q_start,
                    "q_end": q_end,
                },
            )

        _flush_grouped_sdpa_jobs()
        return output

    def _cluster_global_local_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        active_idx: torch.Tensor,
        key_padding_mask: torch.Tensor,
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        num_heads = query.shape[1]
        global_head_count = int(getattr(ras_manager.MANAGER, "cluster_global_head_count", 4))
        global_head_count = max(0, min(global_head_count, num_heads))

        if global_head_count == 0:
            return self._cluster_concat_attention(query, key, value, active_idx, key_padding_mask, softmax_scale)

        global_heads = self._select_global_heads_from_qk(
            query,
            key,
            active_idx,
            softmax_scale,
            global_head_count,
        )
        all_heads = torch.arange(num_heads, device=query.device, dtype=torch.long)
        local_head_mask = torch.ones(num_heads, device=query.device, dtype=torch.bool)
        local_head_mask[global_heads] = False
        local_heads = all_heads[local_head_mask]

        output = torch.empty_like(query)
        global_attention_mask = key_padding_mask[:, None, None, :].expand(
            -1,
            global_heads.numel(),
            query.shape[2],
            -1,
        )
        global_output = F.scaled_dot_product_attention(
            query.index_select(1, global_heads),
            key.index_select(1, global_heads),
            value.index_select(1, global_heads),
            attn_mask=global_attention_mask,
            scale=softmax_scale,
        )
        output.index_copy_(1, global_heads, global_output)

        if local_heads.numel() == 0:
            return output

        local_output = self._cluster_concat_attention(
            query.index_select(1, local_heads),
            key.index_select(1, local_heads),
            value.index_select(1, local_heads),
            active_idx,
            key_padding_mask,
            softmax_scale,
        )
        output.index_copy_(1, local_heads, local_output)
        return output

    def _active_query_full_kv_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        active_idx: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        # Query remains sparse/active-only, but every active query attends exact
        # token-level full cached K/V. This is the quality-preserving bridge
        # before switching to local sparse KV.
        del active_idx
        attention_mask = None
        if key_padding_mask is not None:
            attention_mask = key_padding_mask[:, None, None, :].expand(
                -1,
                query.shape[1],
                query.shape[2],
                -1,
            )
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=softmax_scale,
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[torch.Tensor] = None,
        key_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        is_self_attention = encoder_hidden_states is None or hidden_states.shape == encoder_hidden_states.shape
        temporal_ras_active = (
            ras_manager.MANAGER.temporal_selection_enabled()
            and ras_manager.MANAGER.is_RAS_step
        )
        spatial_ras_active = (
            ras_manager.MANAGER.spatial_attention_enabled()
            and ras_manager.MANAGER.is_RAS_step
        )
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        if is_self_attention and ras_manager.MANAGER.temporal_selection_enabled():
            self._ensure_cache_cluster_order()

        v_fuse_linear = temporal_ras_active and \
            is_self_attention and self.v_cache is not None \
            and ras_manager.MANAGER.enable_index_fusion
        k_fuse_linear = v_fuse_linear and key_rotary_emb is None

        # Get Query-Key-Value Pair
        query = ras_manager.MANAGER.timed_module_op(
            "self_attn_query_linear" if is_self_attention else "cross_attn_query_linear",
            attn.to_q,
            hidden_states,
        )

        text_cached = text_binding = None
        if (not is_self_attention and key_rotary_emb is None
                and getattr(ras_manager.MANAGER, "lumina_text_cache_scope", None) is not None):
            from .lumina_next_t2i.text_cache import cross_kv_cache
            text_cached, text_binding = cross_kv_cache(
                ras_manager.MANAGER, attn, encoder_hidden_states)

        if text_cached is not None:
            value = text_cached[2]
        elif v_fuse_linear:
            from .fused_kernels_lumina import _partially_linear
            _partially_linear(
                encoder_hidden_states,
                attn.to_v.weight,
                attn.to_v.bias,
                ras_manager.MANAGER.other_patchified_index,
                self.v_cache.view(batch_size, self.v_cache.shape[1], -1)
            )
        else:
            value = ras_manager.MANAGER.timed_module_op(
                "self_attn_value_linear" if is_self_attention else "cross_attn_value_linear",
                attn.to_v,
                encoder_hidden_states,
            )

        if text_cached is not None:
            key = text_cached[1]
        elif k_fuse_linear:
            _partially_linear(
                encoder_hidden_states,
                attn.to_k.weight,
                attn.to_k.bias,
                ras_manager.MANAGER.other_patchified_index,
                self.k_cache.view(batch_size, self.k_cache.shape[1], -1)
            )
        else:
            key = ras_manager.MANAGER.timed_module_op(
                "self_attn_key_linear" if is_self_attention else "cross_attn_key_linear",
                attn.to_k,
                encoder_hidden_states,
            )

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Apply Query-Key Norm if needed
        if attn.norm_q is not None:
            query = ras_manager.MANAGER.timed_module_op(
                "self_attn_query_norm" if is_self_attention else "cross_attn_query_norm",
                attn.norm_q,
                query,
            )
        if attn.norm_k is not None and text_cached is None:
            key = ras_manager.MANAGER.timed_module_op(
                "self_attn_key_norm" if is_self_attention else "cross_attn_key_norm",
                attn.norm_k,
                key,
            )

        if text_binding is not None:
            from .lumina_next_t2i.text_cache import store_cross_kv
            store_cross_kv(text_binding, key, value)

        query = query.view(batch_size, -1, attn.heads, head_dim)

        if not k_fuse_linear:
            key = key.view(batch_size, -1, kv_heads, head_dim)
        if not v_fuse_linear:
            value = value.view(batch_size, -1, kv_heads, head_dim)

        if ras_manager.MANAGER.temporal_selection_enabled() and ras_manager.MANAGER.current_step == 0 and is_self_attention:
            self.k_cache = None
            self.v_cache = None
            self.cache_in_cluster_order = False
            self._reset_remote_anchor_value_cache()

        if ras_manager.MANAGER.temporal_selection_enabled() and ras_manager.MANAGER.current_step > ras_manager.MANAGER.scheduler_end_step and is_self_attention:
            self.k_cache = None
            self.v_cache = None
            self.cache_in_cluster_order = False
            self._reset_remote_anchor_value_cache()

        # Apply RoPE if needed
        if query_rotary_emb is not None:
            if temporal_ras_active:
                query = ras_manager.MANAGER.timed_module_op(
                    "self_attn_query_rope",
                    apply_rotary_emb,
                    query,
                    ras_manager.MANAGER.image_rotary_emb_skip,
                    use_real=False,
                )
            else:
                query = ras_manager.MANAGER.timed_module_op(
                    "self_attn_query_rope" if is_self_attention else "cross_attn_query_rope",
                    apply_rotary_emb,
                    query,
                    query_rotary_emb,
                    use_real=False,
                )
        if key_rotary_emb is not None:
            if temporal_ras_active:
                if ras_manager.MANAGER.enable_index_fusion:
                    from .fused_kernels_lumina import _partially_apply_rope
                    _partially_apply_rope(
                        key,
                        ras_manager.MANAGER.image_rotary_emb_skip,
                        ras_manager.MANAGER.other_patchified_index,
                        self.k_cache,
                    )
                else:
                    key = ras_manager.MANAGER.timed_module_op(
                        "self_attn_key_rope",
                        apply_rotary_emb,
                        key,
                        ras_manager.MANAGER.image_rotary_emb_skip,
                        use_real=False,
                    )
            else:
                key = ras_manager.MANAGER.timed_module_op(
                    "self_attn_key_rope" if is_self_attention else "cross_attn_key_rope",
                    apply_rotary_emb,
                    key,
                    key_rotary_emb,
                    use_real=False,
                )

        if ras_manager.MANAGER.temporal_selection_enabled() and (ras_manager.MANAGER.current_step == ras_manager.MANAGER.scheduler_start_step - 1 or ras_manager.MANAGER.current_step in ras_manager.MANAGER.error_reset_steps) and is_self_attention:
            self.k_cache = key
            self.v_cache = value
            self.cache_in_cluster_order = ras_manager.MANAGER.has_cluster_permutation()
            self._reset_remote_anchor_value_cache()

        if temporal_ras_active and is_self_attention:
            if not ras_manager.MANAGER.enable_index_fusion:
                fused_cache_write = False
                if (bool(getattr(ras_manager.MANAGER, "enable_cluster_remote_anchor_incremental_value_cache", False))
                        and ras_manager.MANAGER.use_cluster_remote_anchors):
                    fused_cache_write = self._try_fused_anchor_and_kv_cache_write(
                        self.k_cache, self.v_cache, ras_manager.MANAGER.other_patchified_index, key, value)
                if not fused_cache_write:
                    if bool(getattr(ras_manager.MANAGER, "enable_cluster_remote_anchor_incremental_value_cache", False)):
                        self._update_remote_anchor_value_cache_before_write(
                            self.v_cache,
                            ras_manager.MANAGER.other_patchified_index,
                            value,
                        )
                    ras_manager.MANAGER.timed_module_op(
                        "self_attn_cache_update",
                        lambda cache_k, cache_v, active_idx, new_k, new_v: (
                            cache_k.__setitem__((slice(None), active_idx), new_k),
                            cache_v.__setitem__((slice(None), active_idx), new_v),
                        ),
                        self.k_cache,
                        self.v_cache,
                        ras_manager.MANAGER.other_patchified_index,
                        key,
                        value,
                    )
            key = self.k_cache
            value = self.v_cache

        query, key = query.to(dtype), key.to(dtype)

        if temporal_ras_active and is_self_attention:
            if is_self_attention:
                sequence_length = key.shape[1]
            else:
                sequence_length = base_sequence_length


        # Apply proportional attention if true
        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attn_q_len = query.shape[1]
        use_cluster_online_attention = (
            is_self_attention
            and spatial_ras_active
            and ras_manager.MANAGER.use_cluster_concat_attention
            and ras_manager.MANAGER.cluster_key_indices is not None
            and ras_manager.MANAGER.cluster_sorted_labels is not None
            and ras_manager.MANAGER.has_cluster_permutation()
        )
        use_ras_full_kv_attention = (
            use_cluster_online_attention
            and int(getattr(ras_manager.MANAGER, "cluster_full_kv_attention_until_step", -1))
            > int(ras_manager.MANAGER.current_step)
        )

        # Keep native KV heads through sparse preparation. The legacy repeat
        # remains for dense, routing, analysis and other unsupported paths.
        defer_kv_heads = self._can_defer_cluster_kv_heads(
            use_cluster_online_attention, use_ras_full_kv_attention,
        )
        if not defer_kv_heads:
            n_rep = attn.heads // kv_heads
            if n_rep >= 1:
                key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        if attention_mask is None:
            key_padding_mask = torch.ones(
                batch_size,
                sequence_length,
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            key_padding_mask = attention_mask.bool().view(batch_size, -1)
        analysis_callback = ras_manager.MANAGER.attention_analysis_callback
        if analysis_callback is not None and is_self_attention:
            analysis_callback({
                "model": "lumina",
                "step": int(ras_manager.MANAGER.current_step),
                "layer": int(ras_manager.MANAGER.current_layer),
                "query": query.detach(),
                "key": key.detach(),
                "key_padding_mask": key_padding_mask.detach(),
                "softmax_scale": softmax_scale,
            })
        if use_cluster_online_attention:
            ras_manager.MANAGER.cluster_key_padding_mask_all_true = (
                self._all_source_keys_valid(attention_mask)
            )
            ras_manager.MANAGER.cluster_key_padding_mask_numel = int(key_padding_mask.numel())
        if use_cluster_online_attention:
            attention_mask = None
        else:
            def _build_attention_mask(mask_tensor, bsz, n_heads, q_length):
                attention_mask_ = mask_tensor.view(bsz, 1, 1, -1)
                return attention_mask_.expand(-1, n_heads, q_length, -1)

            attention_mask = ras_manager.MANAGER.timed_module_op(
                "attention_mask_build",
                _build_attention_mask,
                key_padding_mask,
                batch_size,
                attn.heads,
                attn_q_len,
            )

        # Online attention consumes compact neighbor lists. Construct the dense
        # mask only for the actual dense fallback or explicitly enabled analysis.
        cluster_mask = None
        if (is_self_attention and spatial_ras_active and not use_ras_full_kv_attention
                and (not use_cluster_online_attention or ras_manager.MANAGER.enable_cluster_attention_density)):
            cluster_mask = ras_manager.MANAGER.get_cluster_attention_mask()
        if cluster_mask is not None:
            if temporal_ras_active:
                active_idx = ras_manager.MANAGER.other_patchified_index
                cluster_mask = cluster_mask[active_idx, :]             # [active, num_patches]

            # 记录当前 diffusion step 内所有 self-attention 调用累计的实际 mask 密度。
            ras_manager.MANAGER.record_cluster_attention_density(
                cluster_mask=cluster_mask,
                key_padding_mask=key_padding_mask,
                num_heads=attn.heads,
            )

            if not use_cluster_online_attention:
                def _apply_cluster_mask(attn_mask, cluster_mask_, bsz, n_heads):
                    cluster_mask_ = cluster_mask_[None, None, :, :].expand(bsz, n_heads, -1, -1)
                    return attn_mask & cluster_mask_

                attention_mask = ras_manager.MANAGER.timed_module_op(
                    "cluster_mask_apply",
                    _apply_cluster_mask,
                    attention_mask,
                    cluster_mask,
                    batch_size,
                    attn.heads,
                )

        if is_self_attention:
            attention_core_name = "self_attn_ras_sdpa" if (
                temporal_ras_active or spatial_ras_active
            ) else "self_attn_full_sdpa"
            qk_benchmark_name = "self_attn_ras_qk_bmm" if (
                temporal_ras_active or spatial_ras_active
            ) else "self_attn_full_qk_bmm"
        else:
            attention_core_name = "cross_attn_sdpa"
            qk_benchmark_name = "cross_attn_qk_bmm"
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if use_cluster_online_attention:
            if use_ras_full_kv_attention:
                attention_fn = self._active_query_full_kv_attention
                attention_name = "self_attn_ras_full_kv_sdpa"
            elif bool(getattr(ras_manager.MANAGER, "use_cluster_global_local_head_split", False)):
                attention_fn = self._cluster_global_local_head_attention
                attention_name = "self_attn_ras_global_local_head_split"
            else:
                attention_fn = self._cluster_concat_attention
                attention_name = "self_attn_ras_cluster_concat"
            hidden_states = ras_manager.MANAGER.timed_attention_core(
                attention_name,
                attention_fn,
                query,
                key,
                value,
                ras_manager.MANAGER.other_patchified_index,
                key_padding_mask,
                softmax_scale,
                q_len=query.shape[-2],
                k_len=key.shape[-2],
                v_len=value.shape[-2],
                num_heads=query.shape[1],
                head_dim=query.shape[-1],
            )
        else:
            ras_manager.MANAGER.benchmark_qk_matmul(qk_benchmark_name, query, key)
            hidden_states = ras_manager.MANAGER.timed_attention_core(
                attention_core_name,
                F.scaled_dot_product_attention,
                query,
                key,
                value,
                attn_mask=attention_mask,
                scale=softmax_scale,
                q_len=query.shape[-2],
                k_len=key.shape[-2],
                v_len=value.shape[-2],
                num_heads=query.shape[1],
                head_dim=query.shape[-1],
            )
        hidden_states = hidden_states.transpose(1, 2).to(dtype)
        return hidden_states




class RASHunyuanAttnProcessor2_0(RASLuminaAttnProcessor2_0):
    """HunyuanDiT attention processor with RAS self-attention and dense text cross-attention."""

    def _select_hunyuan_rotary(
        self,
        image_rotary_emb,
        indices: Optional[torch.Tensor],
        device: torch.device,
    ):
        if image_rotary_emb is None:
            return None
        if indices is None:
            if isinstance(image_rotary_emb, tuple):
                return tuple(item.to(device) for item in image_rotary_emb)
            return image_rotary_emb.to(device)

        if isinstance(image_rotary_emb, tuple):
            return tuple(
                item.to(device).index_select(0, indices.to(device=device, dtype=torch.long))
                for item in image_rotary_emb
            )
        return image_rotary_emb.to(device).index_select(0, indices.to(device=device, dtype=torch.long))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        del args, kwargs

        manager = ras_manager.MANAGER
        residual = hidden_states
        is_self_attention = encoder_hidden_states is None
        temporal_ras_active = manager.temporal_selection_enabled() and manager.is_RAS_step
        spatial_ras_active = manager.spatial_attention_enabled() and manager.is_RAS_step

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, channel, height, width = hidden_states.shape[0], None, None, None

        query_length = hidden_states.shape[1]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            key_length = query_length
        else:
            key_length = encoder_hidden_states.shape[1]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if is_self_attention and manager.temporal_selection_enabled():
            self._ensure_cache_cluster_order()

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        dtype = query.dtype

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            active_rotary = (
                temporal_ras_active
                and getattr(manager, "image_rotary_emb_skip", None) is not None
                and query.shape[2] == int(manager.other_patchified_index.numel())
            )
            query_rotary_emb = (
                manager.image_rotary_emb_skip
                if active_rotary
                else self._select_hunyuan_rotary(image_rotary_emb, None, query.device)
            )
            query = apply_rotary_emb(query, query_rotary_emb, use_real=True)
            if is_self_attention:
                key_rotary_emb = query_rotary_emb if active_rotary else self._select_hunyuan_rotary(
                    image_rotary_emb,
                    None,
                    key.device,
                )
                key = apply_rotary_emb(key, key_rotary_emb, use_real=True)

        if manager.temporal_selection_enabled() and manager.current_step == 0 and is_self_attention:
            self.k_cache = None
            self.v_cache = None
            self.cache_in_cluster_order = False
            self._reset_remote_anchor_value_cache()

        if (
            manager.temporal_selection_enabled()
            and is_self_attention
            and (
                manager.current_step == manager.scheduler_start_step - 1
                or manager.current_step in manager.error_reset_steps
            )
        ):
            self.k_cache = key.transpose(1, 2).contiguous()
            self.v_cache = value.transpose(1, 2).contiguous()
            self.cache_in_cluster_order = manager.has_cluster_permutation()
            self._reset_remote_anchor_value_cache()

        use_ras = (
            is_self_attention
            and temporal_ras_active
            and manager.other_patchified_index is not None
            and self.k_cache is not None
            and self.v_cache is not None
        )
        if use_ras:
            key_update = key.transpose(1, 2).contiguous()
            value_update = value.transpose(1, 2).contiguous()
            if bool(getattr(manager, "enable_cluster_remote_anchor_incremental_value_cache", False)):
                self._update_remote_anchor_value_cache_before_write(
                    self.v_cache,
                    manager.other_patchified_index,
                    value_update,
                )
            self.k_cache[:, manager.other_patchified_index] = key_update
            self.v_cache[:, manager.other_patchified_index] = value_update
            key = self.k_cache.transpose(1, 2)
            value = self.v_cache.transpose(1, 2)
            key_length = key.shape[2]

        if manager.temporal_selection_enabled() and manager.current_step > manager.scheduler_end_step and is_self_attention:
            self.k_cache = None
            self.v_cache = None
            self.cache_in_cluster_order = False
            self._reset_remote_anchor_value_cache()

        if attention_mask is None:
            key_padding_mask = torch.ones(batch_size, key_length, device=query.device, dtype=torch.bool)
        else:
            key_padding_mask = attention_mask.bool().view(batch_size, -1)

        analysis_callback = manager.attention_analysis_callback
        if analysis_callback is not None and is_self_attention:
            analysis_callback({
                "model": "hunyuan",
                "step": int(manager.current_step),
                "layer": int(manager.current_layer),
                "query": query.detach(),
                "key": key.detach(),
                "key_padding_mask": key_padding_mask.detach(),
                "softmax_scale": None,
            })

        use_cluster_online_attention = (
            is_self_attention
            and spatial_ras_active
            and manager.use_cluster_concat_attention
            and manager.cluster_key_indices is not None
            and manager.cluster_sorted_labels is not None
            and manager.has_cluster_permutation()
        )

        if use_cluster_online_attention and (
            getattr(manager, "cluster_key_padding_mask_all_true", None) is None
            or int(getattr(manager, "cluster_key_padding_mask_numel", -1)) != int(key_padding_mask.numel())
        ):
            manager.cluster_key_padding_mask_all_true = bool(key_padding_mask.all().item())
            manager.cluster_key_padding_mask_numel = int(key_padding_mask.numel())

        if (use_cluster_online_attention and manager.enable_cluster_attention_density
                and manager.cluster_attention_mask is not None):
            cluster_mask = manager.cluster_attention_mask.index_select(
                0,
                manager.other_patchified_index.to(device=manager.cluster_attention_mask.device),
            )
            manager.record_cluster_attention_density(
                cluster_mask=cluster_mask,
                key_padding_mask=key_padding_mask,
                num_heads=query.shape[1],
            )

        if use_cluster_online_attention:
            if (
                int(getattr(manager, "cluster_full_kv_attention_until_step", -1))
                > int(manager.current_step)
            ):
                attention_fn = self._active_query_full_kv_attention
                attention_name = "hunyuan_self_ras_full_kv_sdpa"
            elif bool(getattr(manager, "use_cluster_global_local_head_split", False)):
                attention_fn = self._cluster_global_local_head_attention
                attention_name = "hunyuan_self_ras_global_local_head_split"
            else:
                attention_fn = self._cluster_concat_attention
                attention_name = "hunyuan_self_ras_cluster_concat"
            core_key_padding_mask = key_padding_mask
            if (
                attention_name
                == "hunyuan_self_ras_full_kv_sdpa"
                and bool(getattr(manager, "cluster_key_padding_mask_all_true", False))
            ):
                core_key_padding_mask = None
            hidden_states = manager.timed_attention_core(
                attention_name,
                attention_fn,
                query,
                key,
                value,
                manager.other_patchified_index,
                core_key_padding_mask,
                None,
                q_len=query.shape[-2],
                k_len=key.shape[-2],
                v_len=value.shape[-2],
                num_heads=query.shape[1],
                head_dim=query.shape[-1],
            )
        else:
            if attention_mask is not None:
                attention_mask = key_padding_mask.view(batch_size, 1, 1, -1).expand(
                    -1,
                    attn.heads,
                    query.shape[2],
                    -1,
                )
            qk_name = (
                "hunyuan_self_ras_qk_bmm"
                if use_ras
                else "hunyuan_self_full_qk_bmm"
                if is_self_attention
                else "hunyuan_cross_qk_bmm"
            )
            core_name = (
                "hunyuan_self_ras_sdpa"
                if use_ras
                else "hunyuan_self_full_sdpa"
                if is_self_attention
                else "hunyuan_cross_sdpa"
            )
            manager.benchmark_qk_matmul(qk_name, query, key)
            hidden_states = manager.timed_attention_core(
                core_name,
                F.scaled_dot_product_attention,
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                q_len=query.shape[-2],
                k_len=key.shape[-2],
                v_len=value.shape[-2],
                num_heads=query.shape[1],
                head_dim=query.shape[-1],
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
