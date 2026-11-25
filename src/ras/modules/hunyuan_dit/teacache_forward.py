import time
from typing import Optional

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput


def _rotary_to_device(image_rotary_emb, device):
    if image_rotary_emb is None:
        return None
    if isinstance(image_rotary_emb, tuple):
        return tuple(item.to(device) for item in image_rotary_emb)
    return image_rotary_emb.to(device)


def reset_teacache_state(transformer):
    transformer._teacache_cnt = 0
    transformer._teacache_accumulated_rel_l1_distance = 0.0
    transformer._teacache_previous_modulated_input = None
    transformer._teacache_previous_residual = None
    transformer._teacache_stats = {
        "dense_steps": 0,
        "cached_steps": 0,
        "total_steps": 0,
        "relative_l1_distances": [],
        "accumulated_distances": [],
        "step_records": [],
        "dense_time": 0.0,
        "cached_time": 0.0,
    }


def get_teacache_stats(transformer):
    stats = dict(getattr(transformer, "_teacache_stats", {}) or {})
    total_steps = int(stats.get("total_steps", 0))
    cached_steps = int(stats.get("cached_steps", 0))
    dense_steps = int(stats.get("dense_steps", 0))
    stats["cache_hit_rate"] = cached_steps / total_steps if total_steps > 0 else 0.0
    stats["dense_steps"] = dense_steps
    stats["cached_steps"] = cached_steps
    stats["total_steps"] = total_steps
    return stats


def _ensure_teacache_state(transformer):
    if not hasattr(transformer, "_teacache_stats"):
        reset_teacache_state(transformer)


def _record_step(transformer, record):
    _ensure_teacache_state(transformer)
    stats = transformer._teacache_stats
    stats["total_steps"] += 1
    if record["should_calc"]:
        stats["dense_steps"] += 1
        stats["dense_time"] += record["elapsed_seconds"]
    else:
        stats["cached_steps"] += 1
        stats["cached_time"] += record["elapsed_seconds"]
    if record.get("relative_l1_distance") is not None:
        stats["relative_l1_distances"].append(record["relative_l1_distance"])
    stats["accumulated_distances"].append(record["accumulated_distance"])
    stats["step_records"].append(record)


def teacache_forward(
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
    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)
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

    enable_teacache = bool(getattr(self, "_teacache_enabled", False))
    should_calc = True
    skip_reason = "disabled"
    relative_l1_distance = None
    step_start = time.perf_counter()

    if enable_teacache:
        _ensure_teacache_state(self)
        step_index = int(getattr(self, "_teacache_cnt", 0))
        num_steps = int(getattr(self, "_teacache_num_steps", 0))
        rel_l1_thresh = float(getattr(self, "_teacache_rel_l1_thresh", 0.0))

        modulated_input = self.blocks[0].norm1(hidden_states, temb)
        previous_modulated_input = getattr(self, "_teacache_previous_modulated_input", None)
        previous_residual = getattr(self, "_teacache_previous_residual", None)
        cache_ready = (
            previous_modulated_input is not None
            and previous_residual is not None
            and previous_modulated_input.shape == modulated_input.shape
            and previous_residual.shape == hidden_states.shape
        )
        is_first_step = step_index == 0
        is_last_step = num_steps > 0 and step_index == num_steps - 1
        has_controlnet = controlnet_block_samples is not None

        if is_first_step:
            skip_reason = "first_step"
            self._teacache_accumulated_rel_l1_distance = 0.0
        elif is_last_step:
            skip_reason = "last_step"
            self._teacache_accumulated_rel_l1_distance = 0.0
        elif has_controlnet:
            skip_reason = "controlnet"
            self._teacache_accumulated_rel_l1_distance = 0.0
        elif not cache_ready:
            skip_reason = "cache_not_ready"
            self._teacache_accumulated_rel_l1_distance = 0.0
        else:
            denominator = previous_modulated_input.abs().mean().clamp_min(1e-6)
            relative_l1_distance = (
                (modulated_input - previous_modulated_input).abs().mean() / denominator
            ).detach().float().item()
            self._teacache_accumulated_rel_l1_distance += float(relative_l1_distance)
            if self._teacache_accumulated_rel_l1_distance < rel_l1_thresh:
                should_calc = False
                skip_reason = "cache_hit"
            else:
                skip_reason = "threshold_reached"
                self._teacache_accumulated_rel_l1_distance = 0.0

        self._teacache_previous_modulated_input = modulated_input.detach().clone()
    else:
        step_index = int(getattr(self, "_teacache_cnt", 0))

    if enable_teacache and not should_calc:
        hidden_states = hidden_states + self._teacache_previous_residual
    else:
        original_hidden_states = hidden_states.clone() if enable_teacache else None
        skips = []
        for layer, block in enumerate(self.blocks):
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

        if enable_teacache:
            self._teacache_previous_residual = (hidden_states - original_hidden_states).detach().clone()

    if enable_teacache:
        elapsed_seconds = time.perf_counter() - step_start
        _record_step(
            self,
            {
                "step": step_index,
                "should_calc": bool(should_calc),
                "reason": skip_reason,
                "relative_l1_distance": relative_l1_distance,
                "accumulated_distance": float(getattr(self, "_teacache_accumulated_rel_l1_distance", 0.0)),
                "elapsed_seconds": elapsed_seconds,
            },
        )
        self._teacache_cnt = step_index + 1
        if num_steps > 0 and self._teacache_cnt >= num_steps:
            self._teacache_cnt = 0

    hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
    hidden_states = self.proj_out(hidden_states)

    patch_size = self.pos_embed.patch_size
    latent_h = height // patch_size
    latent_w = width // patch_size

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
