"""Fixed graph metadata and deferred dense fallback must preserve token order."""
import sys
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from ras.schedulers.cluster_adaptive_scheduler import ClusterAdaptiveScheduler
from ras.utils import ras_manager


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]
FIELDS = ["cluster_key_indices", "cluster_far_cluster_ids", "cluster_local_cluster_mask",
          "cluster_local_block_indices_padded", "cluster_local_block_counts",
          "cluster_local_block_max_count", "cluster_local_key_spans",
          "cluster_local_key_span_starts", "cluster_local_key_span_lengths"]


def setup(monkeypatch, device):
    manager = ras_manager.ras_manager()
    monkeypatch.setattr(ras_manager, "MANAGER", manager)
    scheduler = ClusterAdaptiveScheduler(num_clusters=4, recluster_every_step=False)
    labels = np.array([5, 2, 5, 9, 2, 9, 9, 11], dtype=np.int64)
    scheduler.cluster_labels = labels
    labels_t = torch.from_numpy(labels).to(device)
    manager.initialize_cluster_permutation(labels_t)
    return scheduler, manager, labels_t


def assert_equal(a, b):
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and a.device == b.device and a.shape == b.shape
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            assert_equal(a[key], b[key])
    else:
        assert a == b


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["disjoint", "full", "self"])
def test_cpu_graph_matches_original_gpu_construction(monkeypatch, device, mode):
    scheduler, manager, labels = setup(monkeypatch, device)
    if mode == "disjoint":
        neighborhoods = {2: frozenset([2, 9]), 5: frozenset([2, 5]),
                         9: frozenset([2, 9, 11]), 11: frozenset([9, 11])}
    elif mode == "full":
        neighborhoods = {cid: frozenset([2, 5, 9, 11]) for cid in [2, 5, 9, 11]}
    else:
        neighborhoods = {cid: frozenset([cid]) for cid in [2, 5, 9, 11]}
    scheduler._build_cluster_graph_metadata_legacy(labels, neighborhoods, len(labels), labels.device)
    expected = {name: getattr(manager, name) for name in FIELDS}
    scheduler._build_cluster_graph_metadata_cpu(neighborhoods, labels.device)
    for name in FIELDS:
        assert_equal(expected[name], getattr(manager, name))
    assert scheduler._cluster_far_ids_cpu_cache[0] is manager.cluster_far_cluster_ids
    for cid, ids in manager.cluster_far_cluster_ids.items():
        assert scheduler._cluster_far_ids_cpu_cache[1][cid] == tuple(ids.tolist())


@pytest.mark.parametrize("device", DEVICES)
def test_lazy_dense_mask_is_exact_and_owns_its_partition(monkeypatch, device):
    scheduler, manager, labels = setup(monkeypatch, device)
    neighborhoods = {2: frozenset([2, 5]), 5: frozenset([2, 5]),
                     9: frozenset([9, 11]), 11: frozenset([9, 11])}
    permutation = manager.cluster_perm
    expected = scheduler._build_patch_attention_mask(
        scheduler.cluster_labels, neighborhoods, len(labels), labels.device)
    expected = expected.index_select(0, permutation).index_select(1, permutation)
    calls = []
    original = scheduler._build_patch_attention_mask

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_build_patch_attention_mask", record)
    # This reproduces the agreed manager API even on a checkout before its
    # getter lands, keeping the scheduler contract test independent of AP.
    def get_mask(self):
        if self.cluster_attention_mask is None and self.cluster_attention_mask_factory is not None:
            self.cluster_attention_mask = self.cluster_attention_mask_factory()
            self.cluster_attention_mask_factory = None
        return self.cluster_attention_mask

    monkeypatch.setattr(manager, "get_cluster_attention_mask", types.MethodType(get_mask, manager), raising=False)
    scheduler.defer_dense_cluster_mask = True
    scheduler._prepare_cluster_attention_mask(neighborhoods, len(labels), labels.device)
    assert manager.cluster_attention_mask is None and len(calls) == 0
    assert scheduler._cluster_graph_density() == expected.float().mean().item()
    assert len(calls) == 0
    # A delayed factory must not consult a future scheduler/manager partition.
    scheduler.cluster_labels[:] = 2
    manager.cluster_perm = torch.arange(len(labels), device=labels.device)
    actual = manager.get_cluster_attention_mask()
    assert torch.equal(actual, expected)
    assert manager.get_cluster_attention_mask() is actual and len(calls) == 1
    assert manager.cluster_attention_mask_factory is None


@pytest.mark.parametrize("defer,has_getter", [(False, True), (True, False)])
def test_dense_mask_stays_eager_when_disabled_or_fallback_getter_is_missing(monkeypatch, defer, has_getter):
    scheduler, manager, labels = setup(monkeypatch, "cpu")
    neighborhoods = {cid: frozenset([cid]) for cid in [2, 5, 9, 11]}
    scheduler.defer_dense_cluster_mask = defer
    monkeypatch.setattr(manager, "get_cluster_attention_mask", (lambda: None) if has_getter else None, raising=False)
    scheduler._prepare_cluster_attention_mask(neighborhoods, len(labels), labels.device)
    assert manager.cluster_attention_mask is not None
    assert manager.cluster_attention_mask_factory is None
    assert manager.cluster_attention_mask_density is None


def test_invalid_graph_fails_before_metadata_is_replaced(monkeypatch):
    scheduler, manager, _ = setup(monkeypatch, "cpu")
    old_keys = manager.cluster_key_indices
    with pytest.raises(ValueError, match="same labels"):
        scheduler._build_cluster_graph_metadata_cpu({2: frozenset([2])}, "cpu")
    assert manager.cluster_key_indices is old_keys


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (2, 2), (7, 11), (64, 64)])
def test_vectorized_adjacency_and_nhop_equal_all_eight_directions(monkeypatch, shape):
    scheduler, _, _ = setup(monkeypatch, "cpu")
    generator = np.random.default_rng(51)
    # Sparse labels also exercise compact edge encoding independently of IDs.
    labels = generator.choice(np.array([-3, 2, 7, 29, 105]), size=np.prod(shape)).astype(np.int64)
    scheduler.enable_cpu_graph_metadata = False
    expected = scheduler._build_cluster_adjacency_graph(labels, *shape)
    scheduler.enable_cpu_graph_metadata = True
    actual = scheduler._build_cluster_adjacency_graph(labels, *shape)
    assert actual == expected
    for hops in (0, 1, 2, 3):
        assert scheduler._compute_nhop_neighborhoods(actual, hops) == scheduler._compute_nhop_neighborhoods(expected, hops)


@pytest.mark.parametrize("device", DEVICES)
def test_full_scheduler_graph_flags_preserve_selection_and_representatives(monkeypatch, device):
    torch.manual_seed(91)
    sample = torch.randn(1, 4, 16, 16).to(device)
    scores = [torch.rand(64).to(device) for _ in range(3)]
    results = []
    for cpu_graph, defer in [(False, False), (True, False), (True, True)]:
        manager = ras_manager.ras_manager()
        monkeypatch.setattr(ras_manager, "MANAGER", manager)
        manager.patch_size = 2
        manager.starvation_scale = .3
        manager.use_cluster_graph_attention = True
        manager.use_cluster_concat_attention = True
        manager.cluster_graph_attention_hops = 1
        manager.neighbor_expansion_mode = "0"
        manager.cluster_remote_anchor_count = 1
        manager.enable_cluster_remote_anchor_value_coreset = True
        manager.cluster_remote_anchor_weight_mode = "logn"
        manager.enable_cluster_fused_pack = True
        scheduler = ClusterAdaptiveScheduler(
            num_clusters=4, cluster_selection_ratio=.5, recluster_every_step=False,
            enable_cpu_graph_metadata=cpu_graph, defer_dense_cluster_mask=defer)
        scheduler.actual_num_clusters = 4
        labels = np.tile(np.repeat(np.arange(4, dtype=np.int64), 2), 8)
        scheduler.spatial_aware_clustering = lambda *args: labels.copy()
        selections = []
        for step, step_scores in enumerate(scores):
            manager.current_step = step
            scheduler._compute_patch_complexity_from_x_pred = lambda *a, **k: step_scores
            selections.extend(scheduler.cluster_adaptive_selection(sample, sample, 16, 16))
        fields = {name: getattr(manager, name) for name in FIELDS}
        for name, value in vars(manager).items():
            if name.startswith("cluster_remote_anchor_") and (torch.is_tensor(value) or isinstance(value, dict)):
                fields[name] = value
        if defer:
            assert manager.cluster_attention_mask is None
            assert manager.cluster_attention_mask_factory is not None
            mask = manager.cluster_attention_mask_factory()
        else:
            mask = manager.cluster_attention_mask
        fields.update(selections={i: value for i, value in enumerate(selections)},
                      drop_count=scheduler.patch_drop_cnt, mask=mask,
                      update_counts=scheduler.cluster_update_counts.tolist(),
                      density_history=scheduler.attn_mask_density_history)
        results.append(fields)
    assert_equal(results[0], results[1])
    assert_equal(results[0], results[2])
