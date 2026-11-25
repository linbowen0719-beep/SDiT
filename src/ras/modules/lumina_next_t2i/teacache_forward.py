import time
from typing import Any, Dict

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput


LUMINA_NEXT_TEACACHE_COEFFICIENTS = (
    393.76566581,
    -603.50993606,
    209.10239044,
    -23.00726601,
    0.86377344,
)


def _polyval(coefficients, value: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + float(coefficient)
    return result


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
        "rescaled_distances": [],
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
    if record.get("rescaled_distance") is not None:
        stats["rescaled_distances"].append(record["rescaled_distance"])
    stats["accumulated_distances"].append(record["accumulated_distance"])
    stats["step_records"].append(record)


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_mask: torch.Tensor,
    image_rotary_emb: torch.Tensor,
    cross_attention_kwargs: Dict[str, Any] = None,
    return_dict=True,
) -> torch.Tensor:
    if cross_attention_kwargs is None:
        cross_attention_kwargs = {}

    hidden_states, mask, img_size, image_rotary_emb = self.patch_embedder(hidden_states, image_rotary_emb)
    image_rotary_emb = image_rotary_emb.to(hidden_states.device)
    temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)
    encoder_mask = encoder_mask.bool()

    enable_teacache = bool(getattr(self, "_teacache_enabled", False))
    should_calc = True
    skip_reason = "disabled"
    relative_l1_distance = None
    rescaled_distance = None
    step_start = time.perf_counter()

    if enable_teacache:
        _ensure_teacache_state(self)
        step_index = int(getattr(self, "_teacache_cnt", 0))
        num_steps = int(getattr(self, "_teacache_num_steps", 0))
        rel_l1_thresh = float(getattr(self, "_teacache_rel_l1_thresh", 0.0))

        modulated_input = self.layers[0].norm1(hidden_states, temb)[0]
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

        if is_first_step:
            skip_reason = "first_step"
            self._teacache_accumulated_rel_l1_distance = 0.0
        elif is_last_step:
            skip_reason = "last_step"
            self._teacache_accumulated_rel_l1_distance = 0.0
        elif not cache_ready:
            skip_reason = "cache_not_ready"
            self._teacache_accumulated_rel_l1_distance = 0.0
        else:
            denominator = previous_modulated_input.abs().mean().clamp_min(1e-6)
            relative_l1_distance = (
                (modulated_input - previous_modulated_input).abs().mean() / denominator
            ).detach().float().item()
            coefficients = getattr(self, "_teacache_coefficients", LUMINA_NEXT_TEACACHE_COEFFICIENTS)
            raw_rescaled_distance = _polyval(coefficients, relative_l1_distance)
            rescaled_distance = abs(raw_rescaled_distance)
            self._teacache_accumulated_rel_l1_distance += float(rescaled_distance)
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
        original_hidden_states = hidden_states.clone()
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                mask,
                image_rotary_emb,
                encoder_hidden_states,
                encoder_mask,
                temb=temb,
                cross_attention_kwargs=cross_attention_kwargs,
            )
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
                "rescaled_distance": rescaled_distance,
                "accumulated_distance": float(getattr(self, "_teacache_accumulated_rel_l1_distance", 0.0)),
                "elapsed_seconds": elapsed_seconds,
            },
        )
        self._teacache_cnt = step_index + 1
        if num_steps > 0 and self._teacache_cnt >= num_steps:
            self._teacache_cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)

    height_tokens = width_tokens = self.patch_size
    height, width = img_size[0]
    batch_size = hidden_states.size(0)
    sequence_length = (height // height_tokens) * (width // width_tokens)

    hidden_states = hidden_states[:, :sequence_length].view(
        batch_size,
        height // height_tokens,
        width // width_tokens,
        height_tokens,
        width_tokens,
        self.out_channels,
    )
    output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
