import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional CUDA fast path
    triton = None
    tl = None


def _bucket_power_of_two(value: int, minimum: int = 16, maximum: int = 128) -> Optional[int]:
    if value <= 0 or value > maximum:
        return None
    bucket = max(minimum, 1 << (int(value) - 1).bit_length())
    return bucket if bucket <= maximum else None


if triton is not None:

    @triton.jit
    def _cluster_local_span_attention_kernel(
        query,
        key,
        value,
        output,
        q_starts,
        q_lengths,
        key_span_starts,
        key_span_lengths,
        scale,
        H: tl.constexpr,
        D: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_q: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_n: tl.constexpr,
        v_stride_d: tl.constexpr,
        o_stride_b: tl.constexpr,
        o_stride_h: tl.constexpr,
        o_stride_q: tl.constexpr,
        o_stride_d: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        Q_SLOT_COUNT: tl.constexpr,
    ):
        q_tile = tl.program_id(0)
        batch_head = tl.program_id(1)
        cluster_row = tl.program_id(2)

        batch = batch_head // H
        head = batch_head - batch * H

        q_start = tl.load(q_starts + cluster_row)
        q_len = tl.load(q_lengths + cluster_row)
        key_start = tl.load(key_span_starts + cluster_row)
        key_len = tl.load(key_span_lengths + cluster_row)

        q_offsets = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)
        q_valid = q_offsets < q_len
        q_pos = q_start + q_offsets

        q_ptrs = (
            query
            + batch * q_stride_b
            + head * q_stride_h
            + q_pos[:, None] * q_stride_q
            + dim_offsets[None, :] * q_stride_d
        )
        q = tl.load(
            q_ptrs,
            mask=q_valid[:, None] & (dim_offsets[None, :] < D),
            other=0.0,
        )

        neg_inf = -3.4028234663852886e38
        m_i = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        key_offset = 0
        while key_offset < key_len:
            n_offsets = key_offset + tl.arange(0, BLOCK_N)
            key_valid = n_offsets < key_len
            key_pos = key_start + n_offsets

            k_ptrs = (
                key
                + batch * k_stride_b
                + head * k_stride_h
                + key_pos[None, :] * k_stride_n
                + dim_offsets[:, None] * k_stride_d
            )
            k = tl.load(
                k_ptrs,
                mask=key_valid[None, :] & (dim_offsets[:, None] < D),
                other=0.0,
            )
            scores = tl.dot(q, k) * scale
            scores = tl.where(q_valid[:, None] & key_valid[None, :], scores, neg_inf)

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)

            v_ptrs = (
                value
                + batch * v_stride_b
                + head * v_stride_h
                + key_pos[:, None] * v_stride_n
                + dim_offsets[None, :] * v_stride_d
            )
            v = tl.load(
                v_ptrs,
                mask=key_valid[:, None] & (dim_offsets[None, :] < D),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
            key_offset += BLOCK_N

        out = acc / l_i[:, None]
        out = tl.where(l_i[:, None] > 0.0, out, 0.0)
        o_ptrs = (
            output
            + batch * o_stride_b
            + head * o_stride_h
            + q_pos[:, None] * o_stride_q
            + dim_offsets[None, :] * o_stride_d
        )
        tl.store(
            o_ptrs,
            out,
            mask=q_valid[:, None] & (dim_offsets[None, :] < D),
        )

    @triton.jit
    def _cluster_local_attention_kernel(
        query,
        key,
        value,
        output,
        q_positions,
        q_valid_mask,
        key_block_starts,
        key_block_lengths,
        key_block_counts,
        scale,
        B: tl.constexpr,
        H: tl.constexpr,
        Q: tl.constexpr,
        D: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_q: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_n: tl.constexpr,
        v_stride_d: tl.constexpr,
        o_stride_b: tl.constexpr,
        o_stride_h: tl.constexpr,
        o_stride_q: tl.constexpr,
        o_stride_d: tl.constexpr,
        qpos_stride_a: tl.constexpr,
        qpos_stride_s: tl.constexpr,
        qmask_stride_a: tl.constexpr,
        qmask_stride_s: tl.constexpr,
        kstart_stride_a: tl.constexpr,
        kstart_stride_s: tl.constexpr,
        klen_stride_a: tl.constexpr,
        klen_stride_s: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        Q_SLOT_COUNT: tl.constexpr,
        KEY_BLOCK_COUNT: tl.constexpr,
        KEY_BLOCK_TOKEN_COUNT: tl.constexpr,
    ):
        q_tile = tl.program_id(0)
        batch_head = tl.program_id(1)
        cluster_row = tl.program_id(2)

        batch = batch_head // H
        head = batch_head - batch * H

        q_slots = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)
        q_valid = tl.load(
            q_valid_mask + cluster_row * qmask_stride_a + q_slots * qmask_stride_s,
            mask=q_slots < Q_SLOT_COUNT,
            other=0,
        ).to(tl.int1)
        q_pos = tl.load(
            q_positions + cluster_row * qpos_stride_a + q_slots * qpos_stride_s,
            mask=q_slots < Q_SLOT_COUNT,
            other=0,
        )
        q_pos = tl.where(q_valid, q_pos, 0)

        q_ptrs = (
            query
            + batch * q_stride_b
            + head * q_stride_h
            + q_pos[:, None] * q_stride_q
            + dim_offsets[None, :] * q_stride_d
        )
        q = tl.load(
            q_ptrs,
            mask=q_valid[:, None] & (dim_offsets[None, :] < D),
            other=0.0,
        )

        neg_inf = -3.4028234663852886e38
        m_i = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        block_count = tl.load(key_block_counts + cluster_row)

        for block_slot in range(0, KEY_BLOCK_COUNT):
            valid_block = block_slot < block_count
            block_start = tl.load(
                key_block_starts + cluster_row * kstart_stride_a + block_slot * kstart_stride_s,
                mask=valid_block,
                other=0,
            )
            block_len = tl.load(
                key_block_lengths + cluster_row * klen_stride_a + block_slot * klen_stride_s,
                mask=valid_block,
                other=0,
            )
            key_offset = 0
            while key_offset < block_len:
                n_offsets = key_offset + tl.arange(0, BLOCK_N)
                key_valid = valid_block & (n_offsets < block_len)
                key_pos = block_start + n_offsets

                k_ptrs = (
                    key
                    + batch * k_stride_b
                    + head * k_stride_h
                    + key_pos[None, :] * k_stride_n
                    + dim_offsets[:, None] * k_stride_d
                )
                k = tl.load(
                    k_ptrs,
                    mask=key_valid[None, :] & (dim_offsets[:, None] < D),
                    other=0.0,
                )
                scores = tl.dot(q, k) * scale
                scores = tl.where(q_valid[:, None] & key_valid[None, :], scores, neg_inf)

                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                p = tl.exp(scores - m_new[:, None])
                alpha = tl.exp(m_i - m_new)

                v_ptrs = (
                    value
                    + batch * v_stride_b
                    + head * v_stride_h
                    + key_pos[:, None] * v_stride_n
                    + dim_offsets[None, :] * v_stride_d
                )
                v = tl.load(
                    v_ptrs,
                    mask=key_valid[:, None] & (dim_offsets[None, :] < D),
                    other=0.0,
                )
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_new
                key_offset += BLOCK_N

        out = acc / l_i[:, None]
        out = tl.where(l_i[:, None] > 0.0, out, 0.0)
        o_ptrs = (
            output
            + batch * o_stride_b
            + head * o_stride_h
            + q_pos[:, None] * o_stride_q
            + dim_offsets[None, :] * o_stride_d
        )
        tl.store(
            o_ptrs,
            out,
            mask=q_valid[:, None] & (dim_offsets[None, :] < D),
        )


def cluster_local_span_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    active_compact_cluster_ids: torch.Tensor,
    q_starts: torch.Tensor,
    q_lengths: torch.Tensor,
    local_key_span_starts: torch.Tensor,
    local_key_span_lengths: torch.Tensor,
    softmax_scale: Optional[float],
) -> Optional[torch.Tensor]:
    if triton is None:
        return None
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return None
    if query.dtype not in (torch.float16, torch.bfloat16):
        return None
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return None

    batch_size, num_heads, query_tokens, head_dim = query.shape
    if key.shape[0] != batch_size or key.shape[1] != num_heads or key.shape[-1] != head_dim:
        return None
    if value.shape != key.shape:
        return None

    active_count = int(active_compact_cluster_ids.numel())
    if active_count == 0:
        return torch.zeros_like(query)

    device = query.device
    active_compact_cluster_ids = active_compact_cluster_ids.to(device=device, dtype=torch.long)
    q_starts = q_starts.to(device=device, dtype=torch.long)
    q_lengths = q_lengths.to(device=device, dtype=torch.long)
    local_key_span_starts = local_key_span_starts.to(device=device, dtype=torch.long)
    local_key_span_lengths = local_key_span_lengths.to(device=device, dtype=torch.long)

    active_q_starts = q_starts
    active_q_lengths = q_lengths
    active_key_starts = local_key_span_starts.index_select(0, active_compact_cluster_ids)
    active_key_lengths = local_key_span_lengths.index_select(0, active_compact_cluster_ids)
    if bool((active_key_starts < 0).any().item()) or bool((active_key_lengths <= 0).any().item()):
        return None

    active_q_max = int(active_q_lengths.max().item()) if active_q_lengths.numel() > 0 else 0
    q_slot_count = _bucket_power_of_two(active_q_max, minimum=16, maximum=4096)
    block_d = _bucket_power_of_two(head_dim, minimum=16, maximum=128)
    if q_slot_count is None or block_d is None:
        return None

    output = torch.empty_like(query)
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(float(head_dim))
    block_m = 16
    block_n = 64
    grid = (triton.cdiv(q_slot_count, block_m), batch_size * num_heads, active_count)
    _cluster_local_span_attention_kernel[grid](
        query,
        key,
        value,
        output,
        active_q_starts,
        active_q_lengths,
        active_key_starts,
        active_key_lengths,
        scale,
        H=num_heads,
        D=head_dim,
        q_stride_b=query.stride(0),
        q_stride_h=query.stride(1),
        q_stride_q=query.stride(2),
        q_stride_d=query.stride(3),
        k_stride_b=key.stride(0),
        k_stride_h=key.stride(1),
        k_stride_n=key.stride(2),
        k_stride_d=key.stride(3),
        v_stride_b=value.stride(0),
        v_stride_h=value.stride(1),
        v_stride_n=value.stride(2),
        v_stride_d=value.stride(3),
        o_stride_b=output.stride(0),
        o_stride_h=output.stride(1),
        o_stride_q=output.stride(2),
        o_stride_d=output.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        Q_SLOT_COUNT=q_slot_count,
        num_warps=4,
        num_stages=3,
    )
    return output


def cluster_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    active_idx: torch.Tensor,
    active_compact_cluster_ids: torch.Tensor,
    token_indices_padded: torch.Tensor,
    token_indices_mask: torch.Tensor,
    local_block_indices_padded: torch.Tensor,
    local_block_counts: torch.Tensor,
    block_starts: torch.Tensor,
    block_lengths: torch.Tensor,
    softmax_scale: Optional[float],
    *,
    max_cluster_tokens: int,
    max_local_blocks: int,
) -> Optional[torch.Tensor]:
    if triton is None:
        return None
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return None
    if query.dtype not in (torch.float16, torch.bfloat16):
        return None
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return None
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return None

    batch_size, num_heads, query_tokens, head_dim = query.shape
    if key.shape[0] != batch_size or key.shape[1] != num_heads or key.shape[-1] != head_dim:
        return None
    if value.shape != key.shape:
        return None

    key_block_count = _bucket_power_of_two(max_local_blocks, minimum=1, maximum=32)
    key_block_token_count = _bucket_power_of_two(max_cluster_tokens, minimum=16, maximum=4096)
    block_d = _bucket_power_of_two(head_dim, minimum=16, maximum=128)
    if key_block_count is None or key_block_token_count is None or block_d is None:
        return None

    active_count = int(active_compact_cluster_ids.numel())
    if active_count == 0:
        return torch.zeros_like(query)

    device = query.device
    active_idx = active_idx.to(device=device, dtype=torch.long)
    active_compact_cluster_ids = active_compact_cluster_ids.to(device=device, dtype=torch.long)
    token_indices_padded = token_indices_padded.to(device=device, dtype=torch.long)
    token_indices_mask = token_indices_mask.to(device=device)
    local_block_indices_padded = local_block_indices_padded.to(device=device, dtype=torch.long)
    local_block_counts = local_block_counts.to(device=device, dtype=torch.long)
    block_starts = block_starts.to(device=device, dtype=torch.long)
    block_lengths = block_lengths.to(device=device, dtype=torch.long)

    actual_q_slots = int(token_indices_padded.shape[1])
    actual_key_blocks = int(local_block_indices_padded.shape[1])
    if actual_key_blocks > key_block_count:
        return None

    active_pos_lookup = torch.full(
        (key.shape[2],),
        -1,
        device=device,
        dtype=torch.long,
    )
    active_pos_lookup.scatter_(
        0,
        active_idx,
        torch.arange(query_tokens, device=device, dtype=torch.long),
    )

    active_token_indices = token_indices_padded.index_select(0, active_compact_cluster_ids)
    active_token_mask = token_indices_mask.index_select(0, active_compact_cluster_ids)
    candidate_q_positions = active_pos_lookup.index_select(
        0,
        active_token_indices.reshape(-1),
    ).view(active_count, actual_q_slots)
    candidate_q_mask = active_token_mask & (candidate_q_positions >= 0)

    active_q_counts = candidate_q_mask.sum(dim=1, dtype=torch.long)
    active_q_max = int(active_q_counts.max().item()) if active_q_counts.numel() > 0 else 0
    q_slot_count = _bucket_power_of_two(active_q_max, minimum=16, maximum=4096)
    if q_slot_count is None:
        return None

    invalid_q_position = torch.full_like(candidate_q_positions, query_tokens)
    packed_q_positions = torch.sort(
        torch.where(candidate_q_mask, candidate_q_positions, invalid_q_position),
        dim=1,
    ).values
    q_positions = torch.zeros((active_count, q_slot_count), device=device, dtype=torch.long)
    q_valid_mask = torch.zeros((active_count, q_slot_count), device=device, dtype=torch.bool)
    copied_q_slots = min(actual_q_slots, q_slot_count)
    q_positions[:, :copied_q_slots] = packed_q_positions[:, :copied_q_slots].clamp_max(query_tokens - 1)
    q_valid_mask[:, :] = torch.arange(q_slot_count, device=device)[None, :] < active_q_counts[:, None]

    active_local_block_ids = local_block_indices_padded.index_select(0, active_compact_cluster_ids)
    key_block_ids = torch.zeros((active_count, key_block_count), device=device, dtype=torch.long)
    key_block_ids[:, :actual_key_blocks] = active_local_block_ids
    flat_block_ids = key_block_ids.reshape(-1)
    key_block_starts = block_starts.index_select(0, flat_block_ids).view(active_count, key_block_count)
    key_block_lengths = block_lengths.index_select(0, flat_block_ids).view(active_count, key_block_count)
    active_block_counts = local_block_counts.index_select(0, active_compact_cluster_ids)

    output = torch.zeros_like(query)
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(float(head_dim))
    block_m = 16
    block_n = 64
    grid = (triton.cdiv(q_slot_count, block_m), batch_size * num_heads, active_count)
    _cluster_local_attention_kernel[grid](
        query,
        key,
        value,
        output,
        q_positions,
        q_valid_mask,
        key_block_starts,
        key_block_lengths,
        active_block_counts,
        scale,
        B=batch_size,
        H=num_heads,
        Q=query_tokens,
        D=head_dim,
        q_stride_b=query.stride(0),
        q_stride_h=query.stride(1),
        q_stride_q=query.stride(2),
        q_stride_d=query.stride(3),
        k_stride_b=key.stride(0),
        k_stride_h=key.stride(1),
        k_stride_n=key.stride(2),
        k_stride_d=key.stride(3),
        v_stride_b=value.stride(0),
        v_stride_h=value.stride(1),
        v_stride_n=value.stride(2),
        v_stride_d=value.stride(3),
        o_stride_b=output.stride(0),
        o_stride_h=output.stride(1),
        o_stride_q=output.stride(2),
        o_stride_d=output.stride(3),
        qpos_stride_a=q_positions.stride(0),
        qpos_stride_s=q_positions.stride(1),
        qmask_stride_a=q_valid_mask.stride(0),
        qmask_stride_s=q_valid_mask.stride(1),
        kstart_stride_a=key_block_starts.stride(0),
        kstart_stride_s=key_block_starts.stride(1),
        klen_stride_a=key_block_lengths.stride(0),
        klen_stride_s=key_block_lengths.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        Q_SLOT_COUNT=q_slot_count,
        KEY_BLOCK_COUNT=key_block_count,
        KEY_BLOCK_TOKEN_COUNT=key_block_token_count,
        num_warps=4,
        num_stages=3,
    )
    return output
