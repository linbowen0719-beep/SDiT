"""Partition-owned fallback tensors, materialized only when attention reads them.

The scheduler publishes immutable metadata: replacing a partition creates new
maps. These maps snapshot dictionary entries and retain their tensor references;
they never consult a later manager/scheduler state. Producers must not modify
the retained tensors in place, including inference tensors without version
counters. Grouped SDPA materializes these maps only when needed.
"""
from collections.abc import Mapping

import torch


def pad_and_concat_anchor_groups(group_blocks, mask_blocks, device):
    """Original padding/cat ordering, shared by eager and lazy construction."""
    if not group_blocks:
        return (torch.empty(0, 0, device=device, dtype=torch.long),
                torch.empty(0, 0, device=device, dtype=torch.bool))
    max_group_size = max(int(block.shape[1]) for block in group_blocks)
    padded_groups = []
    padded_masks = []
    for group_block, mask_block in zip(group_blocks, mask_blocks):
        pad_width = max_group_size - int(group_block.shape[1])
        if pad_width > 0:
            group_pad = torch.zeros(group_block.shape[0], pad_width,
                                    device=device, dtype=torch.long)
            mask_pad = torch.zeros(mask_block.shape[0], pad_width,
                                   device=device, dtype=torch.bool)
            group_block = torch.cat([group_block, group_pad], dim=1)
            mask_block = torch.cat([mask_block, mask_pad], dim=1)
        padded_groups.append(group_block)
        padded_masks.append(mask_block)
    return torch.cat(padded_groups, dim=0), torch.cat(padded_masks, dim=0)


class _GroupField(Mapping):
    def __init__(self, owner, field):
        self._owner = owner
        self._field = field

    def __getitem__(self, key):
        return self._owner.materialize(key)[self._field]

    def __iter__(self):
        return iter(self._owner._blocks)

    def __len__(self):
        return len(self._owner._blocks)

    def __contains__(self, key):
        return key in self._owner._blocks

    @property
    def materialized_count(self):
        return len(self._owner._cache)


class LazyRemoteAnchorGroups:
    """Two Mapping views sharing one exact padded-group result per query."""
    def __init__(self, query_sources, source_groups, source_masks, device):
        self._blocks = {
            cid: tuple((source_groups[sid], source_masks[sid]) for sid in sources)
            for cid, sources in query_sources.items()
        }
        self._device = torch.device(device)
        self._cache = {}

    # The owner must not retain these views: owner -> view -> owner would keep
    # GPU buffers alive until Python's cyclic GC, including when GC is disabled.
    @property
    def indices(self):
        return _GroupField(self, 0)

    @property
    def mask(self):
        return _GroupField(self, 1)

    def materialize(self, key):
        blocks = self._blocks[key]  # Preserve normal Mapping KeyError/get semantics.
        if key not in self._cache:
            self._cache[key] = pad_and_concat_anchor_groups(
                [block[0] for block in blocks], [block[1] for block in blocks], self._device)
        return self._cache[key]


class LazyMergedAnchorKeys(Mapping):
    """Original local-then-remote concatenation, with partition-local ownership."""
    def __init__(self, local_keys, remote_keys):
        # Eager construction only publishes entries with a local key vector.
        self._rows = {cid: (local_keys[cid], remote)
                      for cid, remote in remote_keys.items() if local_keys.get(cid) is not None}
        self._cache = {}

    def __getitem__(self, key):
        local, remote = self._rows[key]
        if key not in self._cache:
            self._cache[key] = torch.cat([local, remote], dim=0) if remote.numel() else local
        return self._cache[key]

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __contains__(self, key):
        return key in self._rows

    @property
    def materialized_count(self):
        return len(self._cache)
