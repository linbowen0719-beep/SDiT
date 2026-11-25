from typing import Optional

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from ras.utils import ras_manager


def _rotary_to_device(image_rotary_emb, device):
    if image_rotary_emb is None:
        return None
    if isinstance(image_rotary_emb, tuple):
        return tuple(item.to(device) for item in image_rotary_emb)
    return image_rotary_emb.to(device)


def _permute_rotary(image_rotary_emb, manager):
    if image_rotary_emb is None or not manager.has_cluster_permutation():
        return image_rotary_emb
    if isinstance(image_rotary_emb, tuple):
        return tuple(manager.permute_patch_tensor(item, 0) for item in image_rotary_emb)
    return manager.permute_patch_tensor(image_rotary_emb, 0)


def _select_rotary(image_rotary_emb, indices):
    if image_rotary_emb is None:
        return None
    if isinstance(image_rotary_emb, tuple):
        return tuple(item.index_select(0, indices.to(device=item.device, dtype=torch.long)) for item in image_rotary_emb)
    return image_rotary_emb.index_select(0, indices.to(device=image_rotary_emb.device, dtype=torch.long))


def ras_forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states=None,
    text_embedding_mask=None,
    encoder_hidden_states_t5=None,
    text_embedding_mask_t5=None,
    image_meta_size=None,
    style=None,
    image_rotary_emb=None,
    controlnet_block_samples=None,
    return_dict=True,
):
    manager = ras_manager.MANAGER
    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)
    sequence_length = int(hidden_states.shape[1])
    image_rotary_emb = _rotary_to_device(image_rotary_emb, hidden_states.device)

    temb = self.time_extra_emb(
        timestep,
        encoder_hidden_states_t5,
        image_meta_size,
        style,
        hidden_dtype=timestep.dtype,
    )

    batch_size, text_sequence_length, _ = encoder_hidden_states_t5.shape
    encoder_hidden_states_t5 = self.text_embedder(
        encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
    )
    encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, text_sequence_length, -1)

    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1)
    text_embedding_mask = torch.cat([text_embedding_mask, text_embedding_mask_t5], dim=-1)
    text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()
    encoder_hidden_states = torch.where(text_embedding_mask, encoder_hidden_states, self.text_embedding_padding)

    if manager.has_cluster_permutation():
        hidden_states = manager.timed_cluster_reorder(
            "hunyuan_permute_hidden_states",
            manager.permute_patch_tensor,
            hidden_states,
            1,
        )
        image_rotary_emb = _permute_rotary(image_rotary_emb, manager)

    manager.image_rotary_emb_skip = None
    if manager.temporal_selection_enabled() and manager.is_RAS_step:
        hidden_states = hidden_states[:, manager.other_patchified_index]
        manager.image_rotary_emb_skip = _select_rotary(image_rotary_emb, manager.other_patchified_index)

    track_relevance_layer = (
        manager.enable_cluster_relevance_optimizations
        or manager.attention_analysis_callback is not None
    )

    skips = []
    for layer, block in enumerate(self.blocks):
        if track_relevance_layer:
            manager.current_layer = layer

        if layer > self.config.num_layers // 2:
            if controlnet_block_samples is not None:
                skip = skips.pop() + controlnet_block_samples.pop()
            else:
                skip = skips.pop()
            hidden_states = block(
                hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                skip=skip,
            )
        else:
            hidden_states = block(
                hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

        if layer < (self.config.num_layers // 2 - 1):
            skips.append(hidden_states)

    if controlnet_block_samples is not None and len(controlnet_block_samples) != 0:
        raise ValueError("The number of controls is not equal to the number of skip connections.")

    hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
    hidden_states = self.proj_out(hidden_states)

    patch_size = self.pos_embed.patch_size
    latent_h = height // patch_size
    latent_w = width // patch_size

    if manager.temporal_selection_enabled() and manager.is_RAS_step:
        final_hidden_states = torch.zeros(
            (hidden_states.shape[0], sequence_length, hidden_states.shape[2]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        final_hidden_states[:, manager.other_patchified_index] = hidden_states
        hidden_states = final_hidden_states

    if manager.has_cluster_permutation():
        hidden_states = manager.timed_cluster_reorder(
            "hunyuan_inverse_permute_hidden_states",
            manager.inverse_permute_patch_tensor,
            hidden_states,
            1,
        )

    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], latent_h, latent_w, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, latent_h * patch_size, latent_w * patch_size)
    )

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
