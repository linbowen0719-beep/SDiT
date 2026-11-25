"""Host-produced schedules must preserve arbitrary group assignment and order."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from ras.utils.ras_manager import ras_manager
from ras.modules import attention_processor as ap
from ras.schedulers import cluster_adaptive_scheduler as sched


def manager(device, mapping=True):
    m = ras_manager()
    m.use_cluster_graph_attention = True
    m.initialize_cluster_permutation(torch.tensor([4, 0, 4, 0, 2, 4, 2, 0, 4], device=device))
    m.cluster_key_indices = {}
    if mapping:
        # Anchor groups intentionally non-contiguous in cluster-ordered tokens.
        m.cluster_remote_anchor_token_to_packed_anchor = torch.tensor([0,1,0,2,2,3,4,3,4],device=device)
        m.cluster_remote_anchor_group_lengths = torch.tensor([2,1,2,2,2], device=device)
    return m


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA'))])
@pytest.mark.parametrize('selection', [[0,2,5,8], [0,1,4,7], list(range(9)), []])
@pytest.mark.parametrize('mapping', [True, False])
def test_preserves_active_runs_and_anchor_order(device, selection, mapping):
    m = manager(device, mapping)
    source = torch.tensor(selection, device=device, dtype=torch.long)
    expected = m.set_active_patch_indices(source)
    old_runs = m.get_active_cluster_query_ranges_cpu()
    active = m.set_active_patch_indices(source, indices_cpu=np.array(selection, dtype=np.int64))
    assert torch.equal(active, expected)
    assert m.get_active_cluster_query_ranges_cpu() == old_runs
    m.current_step += 1
    assert m.get_active_cluster_query_ranges_cpu() == old_runs
    plan = m.cluster_remote_anchor_value_update_plan
    if not mapping or not selection:
        assert plan is None
        return
    groups = m.cluster_remote_anchor_token_to_packed_anchor[expected]
    order = torch.argsort(groups, stable=True)
    ids, counts = torch.unique_consecutive(groups[order], return_counts=True)
    assert torch.equal(plan['ids'], ids)
    assert torch.equal(plan['lengths'], counts)
    assert plan['complete'] == torch.equal(counts, m.cluster_remote_anchor_group_lengths[ids])
    assert torch.equal(plan['offsets'], torch.cat([counts.new_zeros(1), counts.cumsum(0)]))
    if plan['order'] is None:
        assert torch.equal(order, torch.arange(order.numel(), device=device))
    else:
        assert torch.equal(plan['order'], order)


def test_partition_snapshot_reuses_and_invalidates():
    m = manager('cpu')
    idx = torch.tensor([0,2,5,8])
    m.set_active_patch_indices(idx, indices_cpu=idx.numpy())
    first = m._cluster_host_selection_cache
    m.set_active_patch_indices(idx, indices_cpu=idx.numpy())
    assert m._cluster_host_selection_cache is first
    m.cluster_remote_anchor_token_to_packed_anchor[5] = 4
    m.set_active_patch_indices(idx, indices_cpu=idx.numpy())
    assert m._cluster_host_selection_cache is not first
    first = m._cluster_host_selection_cache
    m.cluster_remote_anchor_metadata_version += 1
    m.set_active_patch_indices(idx, indices_cpu=idx.numpy())
    assert m._cluster_host_selection_cache is not first
    first = m._cluster_host_selection_cache
    m.cluster_inv_perm = m.cluster_inv_perm.clone()
    m.set_active_patch_indices(idx, indices_cpu=idx.numpy())
    assert m._cluster_host_selection_cache is not first
    m.reset_cache()
    assert m._cluster_host_selection_cache is None


def test_rejects_duplicate_indices_and_falls_back_without_spatial_attention():
    m = manager('cpu')
    with pytest.raises(ValueError, match='unique'):
        m.set_active_patch_indices(torch.tensor([0,0]), indices_cpu=[0,0])
    m.use_cluster_graph_attention = False
    idx = torch.tensor([0,2])
    expected = m.map_patch_indices_to_cluster_order(idx)
    assert torch.equal(m.set_active_patch_indices(idx, indices_cpu=[0,2]), expected)


def make_manager(labels, mapping=None, device="cpu"):
    manager = ras_manager()
    manager.use_cluster_graph_attention = True
    manager.initialize_cluster_permutation(torch.as_tensor(labels, dtype=torch.long, device=device))
    manager.cluster_key_indices = {}
    if mapping is not None:
        mapping = torch.as_tensor(mapping, dtype=torch.long, device=device)
        manager.use_cluster_remote_anchors = True
        manager.enable_cluster_remote_anchor_value_coreset = True
        manager.cluster_remote_anchor_token_to_packed_anchor = mapping
        manager.cluster_remote_anchor_group_lengths = torch.bincount(mapping)
        manager.cluster_remote_anchor_packed_counts = torch.bincount(mapping)
        manager.cluster_remote_anchor_group_token_order = torch.argsort(mapping, stable=True)
        manager.cluster_remote_anchor_group_token_order_is_cluster_order = bool(torch.all(mapping[1:] >= mapping[:-1]))
    return manager


def test_random_arbitrary_memberships_match_old_sort_and_grouping():
    rng = np.random.default_rng(2105)
    for _ in range(40):
        labels = rng.choice([0, 4, 9], size=27)
        mapping = rng.integers(0, 5, size=27)
        mapping[:5] = np.arange(5)
        manager = make_manager(labels, mapping)
        for length in (0, 1, 8, 27):
            host = rng.choice(27, size=length, replace=False)
            indices = torch.from_numpy(host.copy())
            expected = manager.set_active_patch_indices(indices).clone()
            expected_ranges = manager.get_active_cluster_query_ranges_cpu()
            actual = manager.set_active_patch_indices(indices, indices_cpu=host)
            assert torch.equal(actual, expected)
            assert manager.get_active_cluster_query_ranges_cpu() == expected_ranges
            manager.current_step += 1
            assert manager.get_active_cluster_query_ranges_cpu() == expected_ranges


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
@pytest.mark.parametrize("selection", [[0, 3, 4], list(range(9))])
def test_partial_and_complete_anchor_update_matches_gpu_plan_algorithm(monkeypatch, selection, device):
    # Interleaved groups exercise stable argsort as well as partial reconstruction.
    manager = make_manager([4, 0, 4, 0, 2, 4, 2, 0, 4], [0, 1, 0, 2, 2, 3, 4, 3, 4], device=device)
    monkeypatch.setattr(ap.ras_manager, "MANAGER", manager)
    torch.manual_seed(93)
    old = torch.randn(2, 9, 2, 8, device=device)
    indices = torch.tensor(selection, dtype=torch.long, device=device)
    active = manager.set_active_patch_indices(indices)
    new = torch.randn(2, active.numel(), 2, 8, device=device)
    legacy = ap.RASLuminaAttnProcessor2_0()
    legacy._update_remote_anchor_value_cache_before_write(old, active, new)
    expected = legacy.remote_anchor_value_mean_cache.clone()
    old_copy = old.clone()
    active_host = manager.set_active_patch_indices(indices, indices_cpu=selection)
    host = ap.RASLuminaAttnProcessor2_0()
    host._update_remote_anchor_value_cache_before_write(old, active_host, new)
    assert torch.equal(host.remote_anchor_value_mean_cache, expected)
    assert torch.equal(old, old_copy), "partial-group fallback must not overwrite the caller cache"
    assert manager.cluster_remote_anchor_value_update_plan["complete"] == (len(selection) == 9)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
@pytest.mark.parametrize("recluster", [False, True])
def test_host_selection_keeps_topk_latent_order_and_starvation(monkeypatch, patch_size, recluster):
    labels = np.array([0, 2, 1, 2, 0, 1, 1, 0, 2, 0, 2, 1, 0, 2, 1], dtype=np.int64)
    height, width = patch_size * 3, patch_size * 5
    sample = torch.zeros(1, 4, height, width)
    scores = [torch.ones(15), torch.arange(15).float(), torch.ones(15)]
    runs = []
    for enabled in (False, True):
        manager = ras_manager()
        monkeypatch.setattr(sched.ras_manager, "MANAGER", manager)
        manager.patch_size = patch_size
        manager.neighbor_expansion_mode = "0"
        manager.use_cluster_graph_attention = False
        manager.starvation_scale = .3
        scheduler = sched.ClusterAdaptiveScheduler(num_clusters=3, cluster_selection_ratio=.67,
            recluster_every_step=recluster, enable_host_selection_schedule=enabled)
        scheduler.actual_num_clusters = 3
        scheduler.spatial_aware_clustering = lambda *args: labels.copy()
        out = []
        for step, score in enumerate(scores):
            manager.current_step = step
            scheduler._compute_patch_complexity_from_x_pred = lambda *a, score=score, **k: score
            cached, active = scheduler.cluster_adaptive_selection(sample, sample, height, width)
            out.append((cached.clone(), active.clone(), scheduler.patch_drop_cnt.clone()))
            if enabled:
                assert np.array_equal(scheduler._cached_selection_cpu[1], cached.numpy())
                assert np.array_equal(scheduler._active_selection_cpu[1], active.numpy())
        runs.append(out)
    for left, right in zip(*runs):
        for old, new in zip(left, right):
            assert torch.equal(old, new)


def test_recluster_identity_changes_replace_host_snapshot_in_inference_mode():
    with torch.inference_mode():
        manager = make_manager([0, 1, 0, 1, 2, 2], [0, 0, 1, 1, 2, 2])
        indices = torch.tensor([0, 1, 4])
        manager.set_active_patch_indices(indices, indices_cpu=[0, 1, 4])
        first = manager._cluster_host_selection_cache
        manager.initialize_cluster_permutation(torch.tensor([2, 0, 1, 0, 2, 1]))
        expected = manager.set_active_patch_indices(indices).clone()
        actual = manager.set_active_patch_indices(indices, indices_cpu=[0, 1, 4])
        assert torch.equal(actual, expected)
        assert manager._cluster_host_selection_cache is not first


def test_empty_and_nonspatial_selection_clear_pending_anchor_and_query_plans():
    m = manager("cpu")
    m.set_active_patch_indices(torch.tensor([0, 2, 5, 8]), indices_cpu=[0, 2, 5, 8])
    assert m.cluster_remote_anchor_value_update_plan is not None
    assert m.get_active_cluster_query_ranges_cpu() is not None
    m.set_active_patch_indices(torch.empty(0, dtype=torch.long), indices_cpu=[])
    assert m.cluster_remote_anchor_value_update_plan is None
    assert m.get_active_cluster_query_ranges_cpu() is None
    m.set_active_patch_indices(torch.tensor([0, 2]), indices_cpu=[0, 2])
    m.use_cluster_graph_attention = False
    m.set_active_patch_indices(torch.tensor([0, 2]), indices_cpu=[0, 2])
    assert m.cluster_remote_anchor_value_update_plan is None
    assert m.get_active_cluster_query_ranges_cpu() is None
    assert not m.cluster_active_queries_sorted


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("complete", [False, True])
def test_host_offsets_fused_refresh_and_partial_fallback_are_exact(monkeypatch, complete):
    m = make_manager([4, 0, 4, 0, 2, 4, 2, 0, 4],
                     [0, 1, 0, 2, 2, 3, 4, 3, 4], device="cuda")
    monkeypatch.setattr(ap.ras_manager, "MANAGER", m)
    mapping = m.cluster_remote_anchor_token_to_packed_anchor
    selected = ((mapping == 0) | (mapping == 2)).nonzero().flatten()
    if not complete:
        selected = selected[::2]
    raster = m.cluster_perm[selected]
    cpu_source = raster.cpu().numpy()
    torch.manual_seed(231)
    key = torch.randn(2, 9, 2, 72, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    with torch.inference_mode():
        active = m.set_active_patch_indices(raster)
        new_key = torch.randn(2, active.numel(), 2, 72, device="cuda", dtype=key.dtype)
        new_value = torch.randn_like(new_key)
        reference = ap.RASLuminaAttnProcessor2_0()
        m.enable_cluster_fused_anchor_refresh = False
        m.enable_cluster_fused_kv_refresh = False
        reference._update_remote_anchor_value_cache_before_write(value, active, new_value)
        expected_key, expected_value = key.clone(), value.clone()
        expected_key.index_copy_(1, active, new_key)
        expected_value.index_copy_(1, active, new_value)

        host_active = m.set_active_patch_indices(raster, indices_cpu=cpu_source)
        assert "offsets" in m.cluster_remote_anchor_value_update_plan
        m.enable_cluster_fused_anchor_refresh = True
        m.enable_cluster_fused_kv_refresh = True
        fast = ap.RASLuminaAttnProcessor2_0()
        actual_key, actual_value = key.clone(), value.clone()
        handled = fast._try_fused_anchor_and_kv_cache_write(
            actual_key, actual_value, host_active, new_key, new_value)
        assert handled is complete
        if not handled:
            assert torch.equal(actual_key, key) and torch.equal(actual_value, value)
            fast._update_remote_anchor_value_cache_before_write(actual_value, host_active, new_value)
            actual_key.index_copy_(1, host_active, new_key)
            actual_value.index_copy_(1, host_active, new_value)
        assert torch.equal(actual_key, expected_key)
        assert torch.equal(actual_value, expected_value)
        assert torch.equal(fast.remote_anchor_value_mean_cache, reference.remote_anchor_value_mean_cache)
