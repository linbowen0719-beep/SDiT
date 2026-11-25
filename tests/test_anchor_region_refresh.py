import sys
from pathlib import Path

import pytest
import torch

from ras.modules import attention_processor as ap
from ras.utils.ras_manager import ras_manager


def setup(monkeypatch, mapping, dtype=torch.float32):
    m = ras_manager()
    monkeypatch.setattr(ap.ras_manager, "MANAGER", m)
    m.use_cluster_remote_anchors = True
    m.enable_cluster_remote_anchor_value_coreset = True
    m.enable_cluster_remote_anchor_incremental_value_cache = True
    m.cluster_key_padding_mask_all_true = True
    mapping = torch.tensor(mapping)
    order = torch.argsort(mapping, stable=True)
    lengths = torch.bincount(mapping)
    m.cluster_remote_anchor_token_to_packed_anchor = mapping
    m.cluster_remote_anchor_packed_counts = lengths.float()
    m.cluster_remote_anchor_group_lengths = lengths
    m.cluster_remote_anchor_group_token_order = order
    m.cluster_remote_anchor_group_token_order_is_cluster_order = torch.equal(order, torch.arange(len(mapping)))
    torch.manual_seed(11)
    v = torch.randn(2, len(mapping), 2, 4).to(dtype)
    return m, ap.RASLuminaAttnProcessor2_0(), v


def reference(m, v):
    return torch.stack([v[:, m.cluster_remote_anchor_token_to_packed_anchor == i].float().mean(1)
                        for i in range(m.cluster_remote_anchor_group_lengths.numel())], dim=1).to(v.dtype)


@pytest.mark.parametrize("mapping,active", [
    ([0, 0, 1, 1, 1, 2], [2, 3, 4]),  # Whole region, contiguous.
    ([0, 0, 1, 1, 1, 2], [5, 1, 0]),  # Multiple complete groups, unsorted.
    ([0, 1, 0, 2, 1, 2], [4, 1]),     # Non-contiguous coreset members.
    ([0, 0, 1, 1, 1, 2], [1, 3]),     # Partial groups / boundary expansion.
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_refresh_matches_full_recompute_and_preserves_untouched_groups(monkeypatch, mapping, active, dtype):
    m, p, v = setup(monkeypatch, mapping, dtype)
    assert p._ensure_remote_anchor_value_cache(v)
    indices = torch.tensor(active)
    untouched = [i for i in range(m.cluster_remote_anchor_group_lengths.numel()) if i not in set(m.cluster_remote_anchor_token_to_packed_anchor[indices].tolist())]
    for _ in range(8):
        old = v.clone()
        old_mean = p.remote_anchor_value_mean_cache.clone()
        fresh = torch.randn_like(v[:, indices])
        p._update_remote_anchor_value_cache_before_write(v, indices, fresh)
        torch.testing.assert_close(v, old, rtol=0, atol=0)  # Caller still owns token-cache write.
        v[:, indices] = fresh
        torch.testing.assert_close(p.remote_anchor_value_mean_cache, reference(m, v))
        torch.testing.assert_close(p.remote_anchor_value_mean_cache[:, untouched], old_mean[:, untouched], rtol=0, atol=0)
    block = p._remote_anchor_value_block_from_cache(torch.empty(2, 4, len(mapping), 4))
    expected = reference(m, v).repeat_interleave(2, dim=2).transpose(1, 2)
    torch.testing.assert_close(block, expected)
    m.cluster_key_padding_mask_all_true = False
    assert p._remote_anchor_value_block_from_cache(torch.empty(2, 4, len(mapping), 4)) is None


def test_rebuild_and_shared_plan_lifecycle(monkeypatch):
    m, p, v = setup(monkeypatch, [0, 0, 1, 1])
    idx = torch.tensor([0, 1])
    fresh = torch.randn_like(v[:, idx])
    p._update_remote_anchor_value_cache_before_write(v, idx, fresh)
    plan = m.cluster_remote_anchor_value_update_plan
    other = ap.RASLuminaAttnProcessor2_0()
    other._update_remote_anchor_value_cache_before_write(v, idx, fresh)
    assert m.cluster_remote_anchor_value_update_plan is plan
    torch.testing.assert_close(p.remote_anchor_value_mean_cache, other.remote_anchor_value_mean_cache)
    m.cluster_remote_anchor_metadata_version += 1
    new_cache = torch.randn_like(v)
    p._update_remote_anchor_value_cache_before_write(new_cache, idx, fresh)
    assert m.cluster_remote_anchor_value_update_plan is not plan
    new_cache[:, idx] = fresh
    torch.testing.assert_close(p.remote_anchor_value_mean_cache, reference(m, new_cache))
    p._reset_remote_anchor_value_cache()
    assert p.remote_anchor_value_mean_cache is None
    m.enable_cluster_remote_anchor_incremental_value_cache = False
    assert p._remote_anchor_value_block_from_cache(v.transpose(1, 2)) is None
