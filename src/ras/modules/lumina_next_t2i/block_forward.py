import torch
from typing import Any, Dict, Optional

from ras.utils import ras_manager
from .text_cache import context_norm


def ras_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    image_rotary_emb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_mask: torch.Tensor,
    temb: torch.Tensor,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
):
    if cross_attention_kwargs is None:
        cross_attention_kwargs = {}

    residual = hidden_states

    norm_hidden_states, gate_msa, scale_mlp, gate_mlp = ras_manager.MANAGER.timed_module_op(
        "block_norm1", self.norm1, hidden_states, temb
    )
    self_attn_output = self.attn1(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_hidden_states,
        attention_mask=attention_mask,
        query_rotary_emb=image_rotary_emb,
        key_rotary_emb=image_rotary_emb,
        **cross_attention_kwargs,
    )

    if getattr(ras_manager.MANAGER, "lumina_text_cache_scope", None) is not None:
        norm_encoder_hidden_states = context_norm(
            ras_manager.MANAGER, self, encoder_hidden_states,
            lambda: ras_manager.MANAGER.timed_module_op(
                "block_norm1_context", self.norm1_context, encoder_hidden_states))
    else:
        norm_encoder_hidden_states = ras_manager.MANAGER.timed_module_op(
            "block_norm1_context", self.norm1_context, encoder_hidden_states
        )
    cross_attn_output = self.attn2(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=encoder_mask,
        query_rotary_emb=image_rotary_emb,
        key_rotary_emb=None,
        **cross_attention_kwargs,
    )
    cross_attn_output = cross_attn_output * self.gate.tanh().view(1, 1, -1, 1)
    mixed_attn_output = self_attn_output + cross_attn_output
    mixed_attn_output = mixed_attn_output.flatten(-2)

    hidden_states = ras_manager.MANAGER.timed_module_op(
        "block_attn_out_proj", self.attn2.to_out[0], mixed_attn_output
    )
    norm2_hidden_states = ras_manager.MANAGER.timed_module_op("block_norm2", self.norm2, hidden_states)
    hidden_states = ras_manager.MANAGER.timed_module_op(
        "block_residual_msa",
        lambda res, gate, normed: res + gate.unsqueeze(1).tanh() * normed,
        residual,
        gate_msa,
        norm2_hidden_states,
    )

    ffn_input = ras_manager.MANAGER.timed_module_op("block_ffn_norm1", self.ffn_norm1, hidden_states)
    ffn_input = ras_manager.MANAGER.timed_module_op(
        "block_ffn_scale",
        lambda x, scale: x * (1 + scale.unsqueeze(1)),
        ffn_input,
        scale_mlp,
    )
    mlp_output = ras_manager.MANAGER.timed_module_op("block_mlp", self.feed_forward, ffn_input)
    mlp_output = ras_manager.MANAGER.timed_module_op("block_ffn_norm2", self.ffn_norm2, mlp_output)
    hidden_states = ras_manager.MANAGER.timed_module_op(
        "block_residual_mlp",
        lambda hs, gate, mlp: hs + gate.unsqueeze(1).tanh() * mlp,
        hidden_states,
        gate_mlp,
        mlp_output,
    )

    return hidden_states
