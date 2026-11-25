"""Density diagnostics must not alter attention or tax its normal online path."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from ras.modules import attention_processor
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args, parse_args_from_dict


def test_density_is_explicitly_opt_in_and_resets_between_runs(monkeypatch):
    manager = ras_manager.ras_manager()
    assert not manager.enable_cluster_attention_density
    # A disabled recorder must not inspect its inputs or allocate buffers.
    manager.record_cluster_attention_density(object(), object(), 2)
    assert manager.get_cluster_attention_density_report()["overall_density"] is None
    manager.set_parameters(parse_args_from_dict({"enable_cluster_attention_density": True}))
    assert manager.enable_cluster_attention_density
    manager.set_parameters(parse_args_from_dict({}))
    assert not manager.enable_cluster_attention_density
    monkeypatch.setattr(sys, "argv", ["inference", "--enable_cluster_attention_density"])
    assert parse_args().enable_cluster_attention_density


@pytest.mark.parametrize("online", [False, True])
def test_density_toggle_preserves_sparse_attention_and_required_fallback_mask(monkeypatch, online):
    torch.manual_seed(7)
    manager = ras_manager.ras_manager()
    monkeypatch.setattr(attention_processor.ras_manager, "MANAGER", manager)
    manager.use_cluster_graph_attention = True
    manager.use_cluster_concat_attention = online
    manager.use_cluster_remote_anchors = False
    manager.current_step = 6
    manager.is_RAS_step = True
    labels = torch.tensor([0, 0, 1, 1])
    manager.initialize_cluster_permutation(labels)
    manager.cluster_key_indices = {0: torch.tensor([0, 1]), 1: torch.tensor([2, 3])}
    manager.set_active_patch_indices(torch.tensor([0, 1]))
    mask = labels[:, None] == labels[None, :]
    manager.cluster_attention_mask = mask
    attn = Attention(query_dim=8, heads=2, dim_head=4)
    hidden = torch.randn(1, 2, 8)
    key_cache = torch.randn(1, 4, 2, 4)
    value_cache = torch.randn(1, 4, 2, 4)

    def run(enabled):
        manager.enable_cluster_attention_density = enabled
        manager.reset_cluster_attention_density_stats()
        processor = attention_processor.RASLuminaAttnProcessor2_0()
        processor.k_cache = key_cache.clone()
        processor.v_cache = value_cache.clone()
        processor.cache_in_cluster_order = True
        with torch.no_grad():
            result = processor(attn, hidden)
            q = attn.to_q(hidden).view(1, 2, 2, 4).transpose(1, 2)
            expected = F.scaled_dot_product_attention(
                q, processor.k_cache.transpose(1, 2), processor.v_cache.transpose(1, 2),
                attn_mask=mask[:2][None, None],
            ).transpose(1, 2)
        torch.testing.assert_close(result, expected)
        return result

    enabled = run(True)
    report = manager.get_cluster_attention_density_report()
    assert report["overall_density"] == 0.5
    assert len(report["steps"]) == 1

    if online:
        # The online kernel uses compact indices. Even reading/gathering the
        # diagnostic-only dense mask is forbidden with analysis disabled.
        class UnreadableDiagnosticMask:
            def __getitem__(self, key):
                raise AssertionError("diagnostic mask gathered during normal inference")
        manager.cluster_attention_mask = UnreadableDiagnosticMask()
    disabled = run(False)
    torch.testing.assert_close(enabled, disabled, rtol=0, atol=0)
    assert manager.get_cluster_attention_density_report()["overall_density"] is None
