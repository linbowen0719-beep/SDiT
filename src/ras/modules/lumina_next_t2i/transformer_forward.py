# Copyright 2024 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
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

import torch
from typing import Any, Dict, Optional
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from ras.utils import ras_manager

def ras_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict=True,
        ) -> torch.Tensor:
    """
        Modified Forward pass of LuminaNextDiT for RAS.

        Parameters:
            hidden_states (torch.Tensor): Input tensor of shape (N, C, H, W).
            timestep (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            encoder_hidden_states (torch.Tensor): Tensor of caption features of shape (N, D).
            encoder_mask (torch.Tensor): Tensor of caption masks of shape (N, L).
    """
    hidden_states, mask, img_size, image_rotary_emb = self.patch_embedder(hidden_states, image_rotary_emb)
    image_rotary_emb = image_rotary_emb.to(hidden_states.device)

    temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)

    source_encoder_mask = encoder_mask
    encoder_mask = encoder_mask.bool()
    manager = ras_manager.MANAGER
    temporal_active = manager.temporal_selection_enabled() and manager.is_RAS_step
    active_raster_indices = None
    if manager.has_cluster_permutation():
        # A selected row in cluster order corresponds to perm[active] in raster
        # order. Gather it directly instead of materializing every hidden row.
        # Keep the original path for padded or otherwise mismatched layouts.
        height, width = img_size[0]
        output_tokens = (height // self.patch_size) * (width // self.patch_size)
        if (temporal_active and getattr(manager, "enable_lumina_fused_active_io", True)
                and hidden_states.shape[1] == manager.cluster_perm.numel() == output_tokens):
            active_raster_indices = manager.timed_cluster_reorder(
                "compose_active_indices", torch.index_select,
                manager.cluster_perm, 0, manager.other_patchified_index,
            )
        else:
            hidden_states = manager.timed_cluster_reorder(
                "permute_hidden_states", manager.permute_patch_tensor, hidden_states, 1
            )
        mask = ras_manager.MANAGER.timed_cluster_reorder(
            "permute_mask", ras_manager.MANAGER.permute_patch_tensor, mask, -1
        )
        image_rotary_emb = ras_manager.MANAGER.timed_cluster_reorder(
            "permute_image_rotary_emb", ras_manager.MANAGER.permute_patch_tensor, image_rotary_emb, 1
        )

    if temporal_active:
        gather_indices = manager.other_patchified_index if active_raster_indices is None else active_raster_indices
        hidden_states = hidden_states[:, gather_indices]
        # RoPE may change with the timestep. Select from this forward's input;
        # only the index mapping, not the positional embedding, is reusable.
        manager.image_rotary_emb_skip = image_rotary_emb[:, manager.other_patchified_index]
    # This image mask is immutable for the following layer loop. Check once
    # per forward, not once per layer, and retain its identity for safe reuse.
    manager = ras_manager.MANAGER
    if manager.enable_cluster_late_kv_expansion or manager.enable_cluster_fused_pack:
        manager.lumina_layer_mask = mask
        manager.lumina_layer_mask_all_true = bool(mask.bool().all().item())
    track_relevance_layer = bool(
        getattr(ras_manager.MANAGER, "enable_cluster_relevance_optimizations", False)
        or getattr(ras_manager.MANAGER, "attention_analysis_callback", None) is not None
    )
    text_scope = None
    if getattr(manager, "enable_lumina_text_cache", False):
        from .text_cache import begin_text_cache_scope, end_text_cache_scope
        text_scope = begin_text_cache_scope(
            manager, self, encoder_hidden_states, source_encoder_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )
    try:
        for layer_idx, layer in enumerate(self.layers):
            if track_relevance_layer:
                ras_manager.MANAGER.current_layer = layer_idx
            hidden_states = layer(
                hidden_states,
                mask,
                image_rotary_emb,
                encoder_hidden_states,
                encoder_mask,
                temb=temb,
                cross_attention_kwargs=cross_attention_kwargs,
            )
    finally:
        if text_scope is not None:
            end_text_cache_scope(manager, text_scope)
        manager.lumina_layer_mask = None
        manager.lumina_layer_mask_all_true = None

    hidden_states = self.norm_out(hidden_states, temb)

    # unpatchify
    height_tokens = width_tokens = self.patch_size
    height, width = img_size[0]
    batch_size = hidden_states.size(0)
    sequence_length = (height // height_tokens) * (width // width_tokens)

    if temporal_active:
        final_hidden_states = torch.zeros((hidden_states.shape[0], sequence_length, hidden_states.shape[2]), device=hidden_states.device, dtype=hidden_states.dtype)
        scatter_indices = manager.other_patchified_index if active_raster_indices is None else active_raster_indices
        final_hidden_states[:, scatter_indices] = hidden_states
        hidden_states = final_hidden_states
    if manager.has_cluster_permutation() and active_raster_indices is None:
        hidden_states = ras_manager.MANAGER.timed_cluster_reorder(
            "inverse_permute_hidden_states", ras_manager.MANAGER.inverse_permute_patch_tensor, hidden_states, 1
        )
    hidden_states = hidden_states[:, :sequence_length].view(
        batch_size, height // height_tokens, width // width_tokens, height_tokens, width_tokens, self.out_channels
    )
    output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
