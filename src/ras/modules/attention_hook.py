"""
自定义 Attention Hook，用于保存 softmax 后的注意力矩阵

策略：
1. 在原有 LuminaAttnProcessor2_0 基础上添加 hook
2. 手动计算 Q @ K^T 和 softmax，保存注意力矩阵
3. 原有推理仍使用 F.scaled_dot_product_attention（不影响性能）
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
from ..utils import ras_manager


class AttentionMatrixHook:
    """
    用于保存注意力矩阵的 Hook

    在推理过程中收集每一层自注意力的 softmax 后的注意力矩阵
    """

    def __init__(
        self,
        save_dir: str = "outputs/debug/attention",
        save_steps: list = [5, 15, 25],  # 只保存这些时间步
        save_last_layer_only: bool = True,  # 只保存最后一层
        num_layers: int = 32,  # DiT 总层数
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_steps = save_steps
        self.save_last_layer_only = save_last_layer_only
        self.num_layers = num_layers

        # 存储注意力矩阵: {step: attn_matrix}
        self.attention_probs = {}  # Dict: step -> [B, H, T_img, T_img]

        self.current_step = 0  # 从 0 开始计数（第一个 denoising step 是 step 0）
        self.current_layer = 0  # 当前正在处理的层

        print(f"   📋 AttentionHook 初始化: current_step 从 {self.current_step} 开始")

    def reset_step(self):
        """在每个 denoising step 开始时调用"""
        self.current_step += 1
        self.current_layer = 0

    def increment_layer(self):
        """每处理完一层后调用"""
        self.current_layer += 1

        # 自动检测新的 step (当 layer 回到 0 时，表示开始新的 step)
        if self.current_layer >= self.num_layers:
            self.current_step += 1
            self.current_layer = 0

    def compute_and_save_attention(
        self,
        query: torch.Tensor,  # [B, H, T, D]
        key: torch.Tensor,    # [B, H, T, D]
        attention_mask: Optional[torch.Tensor],
        softmax_scale: float,
        is_self_attention: bool = True,
    ):
        """
        手动计算注意力矩阵并保存

        Args:
            query: [B, H, T, D]
            key: [B, H, T, D]
            attention_mask: [B, H, T, T] 或 None
            softmax_scale: 缩放因子
            is_self_attention: 是否为自注意力
        """
        if not is_self_attention:
            self.increment_layer()
            return

        # 调试：打印当前状态（仅在目标步数附近）
        if self.current_step in self.save_steps or (self.current_step + 1) in self.save_steps:
            if self.current_layer == 0:  # 只在第一层打印
                print(f"   🔍 Debug: current_step={self.current_step}, current_layer={self.current_layer}, save_steps={self.save_steps}")

        # 检查是否需要保存当前步
        if self.current_step not in self.save_steps:
            self.increment_layer()
            return

        # 检查是否只保存最后一层
        is_last_layer = (self.current_layer == self.num_layers - 1)
        if self.save_last_layer_only and not is_last_layer:
            self.increment_layer()
            return

        with torch.no_grad():
            # 计算 Q @ K^T
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale

            # 应用 mask
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))

            # Softmax
            attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, T, T]

            # 保存到字典 (只保留图像 token 部分)
            self.attention_probs[self.current_step] = attn_probs.detach().cpu()

            print(f"   📸 收集注意力矩阵: Step {self.current_step}, Layer {self.current_layer}")

        self.increment_layer()

    def save_all_attention_data(self):
        """
        保存所有收集的注意力矩阵到单个文件
        """
        if len(self.attention_probs) == 0:
            print("⚠️  没有收集到注意力数据")
            return

        save_path = self.save_dir / "selected_attention_probs.pt"

        # 保存为字典: {step: tensor}
        data_to_save = {
            "attention_probs": self.attention_probs,  # Dict: step -> [B, H, T, T]
            "saved_steps": list(self.attention_probs.keys()),
            "config": {
                "save_steps": self.save_steps,
                "save_last_layer_only": self.save_last_layer_only,
                "num_layers": self.num_layers,
            }
        }

        torch.save(data_to_save, save_path)

        print(f"\n{'='*80}")
        print(f"✅ 保存注意力矩阵数据: {save_path}")
        print(f"   保存的步数: {list(self.attention_probs.keys())}")
        print(f"   总矩阵数: {len(self.attention_probs)}")

        # 打印每个步数的形状
        for step, attn in self.attention_probs.items():
            print(f"   Step {step} shape: {attn.shape} (B, H, T, T)")

        print(f"{'='*80}\n")

    def clear_data(self):
        """清空所有数据"""
        self.attention_probs = {}
        self.current_step = 0
        self.current_layer = 0


class LuminaAttnProcessorWithHook:
    """
    修改自 LuminaAttnProcessor2_0，增加注意力矩阵收集功能

    保持原有推理逻辑，额外手动计算注意力矩阵用于保存
    """

    def __init__(self, hook: AttentionMatrixHook):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hook = hook

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[torch.Tensor] = None,
        key_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Apply Query-Key Norm if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply RoPE if needed (use sliced emb on RAS steps where query is a subset)
        if query_rotary_emb is not None:
            if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step:
                query = apply_rotary_emb(query, ras_manager.MANAGER.image_rotary_emb_skip, use_real=False)
            else:
                query = apply_rotary_emb(query, query_rotary_emb, use_real=False)
        if key_rotary_emb is not None:
            key = apply_rotary_emb(key, key_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Apply proportional attention if true
        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # perform Grouped-query Attention (GQA)
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask_bool = None
        if attention_mask is not None:
            attention_mask_bool = attention_mask.bool().view(batch_size, 1, 1, -1)
            attention_mask_bool = attention_mask_bool.expand(-1, attn.heads, sequence_length, -1)

        query_t = query.transpose(1, 2)  # [B, H, T, D]
        key_t = key.transpose(1, 2)      # [B, H, T, D]
        value_t = value.transpose(1, 2)  # [B, H, T, D]

        # ========== Hook: 手动计算并保存注意力矩阵 ==========
        # 判断是否为自注意力：检查 encoder_hidden_states 和 hidden_states 是否相同
        # 方法1: 检查是否是同一个对象
        is_same_object = encoder_hidden_states is hidden_states
        # 方法2: 检查形状和数值是否接近
        is_self_attention = (encoder_hidden_states.shape == hidden_states.shape and
                            (is_same_object or torch.allclose(encoder_hidden_states, hidden_states, atol=1e-3)))

        # 调试：检查 is_self_attention (每个 step 的第一层打印一次)
        if self.hook is not None and self.hook.current_layer == 0:
            shapes_match = encoder_hidden_states.shape == hidden_states.shape

            print(f"   🔍 Debug Step {self.hook.current_step}: "
                  f"is_same_object={is_same_object}, shapes_match={shapes_match}, "
                  f"is_self_attention={is_self_attention}")

        if is_self_attention and self.hook is not None:
            scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))
            self.hook.compute_and_save_attention(
                query=query_t,
                key=key_t,
                attention_mask=attention_mask_bool,
                softmax_scale=scale,
                is_self_attention=True,
            )

        # ========== 原有推理逻辑：使用融合实现 ==========
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query_t, key_t, value_t, attn_mask=attention_mask_bool, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).to(dtype)

        return hidden_states
