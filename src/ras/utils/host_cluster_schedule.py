"""Host schedules from scheduler-owned membership; no activation readback.

The optional host indices must be the exact immutable CPU source of the supplied
GPU indices. This is an internal scheduler contract, not a tensor-content guess.
Partition tensors are immutable in inference; replacing them or changing their
metadata version invalidates the snapshot. Outside inference, tensor versions
also catch in-place edits.
"""
import numpy as np
import torch


def _version(tensor):
    if tensor is None:
        return None
    try:
        return tensor._version
    except RuntimeError:
        return None


def set_active_from_host(manager, patch_indices, indices_cpu):
    if (not manager.spatial_attention_enabled() or manager.cluster_key_indices is None
            or not manager.has_cluster_permutation()
            or manager.cluster_sorted_labels is None or patch_indices is None):
        return False
    host = np.asarray(indices_cpu, dtype=np.int64)
    if host.ndim != 1 or host.size != patch_indices.numel():
        raise ValueError("host selection must match the supplied patch index vector")
    refs = (manager.cluster_inv_perm, manager.cluster_sorted_labels,
            manager.cluster_remote_anchor_token_to_packed_anchor,
            manager.cluster_remote_anchor_group_lengths)
    versions = tuple(_version(x) for x in refs)
    version = int(manager.cluster_remote_anchor_metadata_version)
    cached = getattr(manager, "_cluster_host_selection_cache", None)
    if (cached is None or cached[0] != version or cached[2] != versions
            or any(a is not b for a, b in zip(cached[1], refs))):
        # One transfer per partition, amortized across all adaptive steps/layers.
        present = [x for x in refs if x is not None]
        packed = torch.cat([x.reshape(-1).to(dtype=torch.long) for x in present]).cpu().numpy()
        arrays = []; offset = 0
        for x in refs:
            n = 0 if x is None else x.numel()
            arrays.append(None if x is None else packed[offset:offset+n])
            offset += n
        cached = (version, refs, versions, arrays)
        manager._cluster_host_selection_cache = cached
    inv, labels, mapping, group_lengths = cached[3]
    if host.size and (host.min() < 0 or host.max() >= inv.size):
        raise ValueError("patch index out of partition bounds")
    active = np.sort(inv[host], kind="stable")
    # Caller provides a set of unique raster indices. Reject duplicates because
    # concurrent cache writes and group-completeness checks require uniqueness.
    if active.size > 1 and np.any(active[1:] == active[:-1]):
        raise ValueError("host selection must contain unique patches")
    selected_labels = labels[active]
    ids, starts, lengths = np.unique(selected_labels, return_index=True, return_counts=True)
    vectors = dict(active=active, ids=ids, starts=starts, lengths=lengths)
    anchor = None
    if mapping is not None and group_lengths is not None and active.size:
        groups = mapping[active]
        order = np.argsort(groups, kind="stable")
        gids, glens = np.unique(groups, return_counts=True)
        if gids.size and gids[0] >= 0 and gids[-1] < group_lengths.size:
            complete = bool(np.array_equal(glens, group_lengths[gids]))
            sorted_groups = bool(np.all(order == np.arange(active.size)))
            vectors.update(gids=gids, glens=glens,
                           offsets=np.concatenate(([0], np.cumsum(glens))))
            if not sorted_groups:
                vectors["order"] = order
            anchor = (complete, sorted_groups)
    sizes = {key: value.size for key, value in vectors.items()}
    gpu = torch.from_numpy(np.concatenate(tuple(vectors.values())).astype(np.int64, copy=False)).to(patch_indices.device)
    tensors = {}; offset = 0
    for key, size in sizes.items():
        tensors[key] = gpu[offset:offset+size]; offset += size
    manager.other_patchified_index = tensors["active"]
    manager.cluster_active_queries_sorted = True
    manager.cluster_active_cluster_ids = tensors["ids"] if active.size else None
    manager.cluster_active_query_starts = tensors["starts"] if active.size else None
    manager.cluster_active_query_lengths = tensors["lengths"] if active.size else None
    if active.size:
        refs = (tensors["ids"], tensors["starts"], tensors["lengths"])
        ranges = (tuple(ids.tolist()), tuple(starts.tolist()), tuple(lengths.tolist()))
        # Selection is published at the end of step t for use at step t+1.
        # get_active_cluster_query_ranges_cpu accepts this identity-owned record.
        manager.cluster_active_query_ranges_cpu = (None, refs, ranges)
    if anchor is not None:
        complete, sorted_groups = anchor
        manager.cluster_remote_anchor_value_update_plan = {
            "active_idx": tensors["active"],
            "mapping": manager.cluster_remote_anchor_token_to_packed_anchor,
            "version": version, "ids": tensors["gids"], "lengths": tensors["glens"],
            "complete": complete, "order": None if sorted_groups else tensors["order"],
            "offsets": tensors["offsets"],
        }
    return True
