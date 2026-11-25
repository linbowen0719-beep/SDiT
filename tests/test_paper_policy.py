"""Numerical and scheduling contracts of the manuscript configuration."""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sdit.config import InferenceConfig
from sdit.policy import detail_residual_score, spatial_anchor_groups, token_coordinates
from ras.schedulers.cluster_adaptive_scheduler import ClusterAdaptiveScheduler
from ras.schedulers.cluster_adaptive_ddpm_scheduler import ClusterAdaptiveDDPMScheduler
from ras.utils import ras_manager


def test_priority_matches_independent_laplacian_and_residual():
    clean = torch.arange(30, dtype=torch.float32).reshape(1, 2, 3, 5).square()
    current = clean + torch.linspace(-2, 3, 30).reshape_as(clean)
    # Independent explicit per-pixel evaluation with replicated boundaries.
    expected = torch.zeros(1, 3, 5)
    for y in range(3):
        for x in range(5):
            lap = (clean[0, :, max(y-1, 0), x] + clean[0, :, min(y+1, 2), x]
                   + clean[0, :, y, max(x-1, 0)] + clean[0, :, y, min(x+1, 4)]
                   - 4*clean[0, :, y, x])
            expected[0, y, x] = (lap.abs() + (current[0, :, y, x]-clean[0, :, y, x]).abs()).mean()/2
    torch.testing.assert_close(detail_residual_score(clean, current), expected)


def test_priority_selects_regions_and_applies_starvation(monkeypatch):
    m = ras_manager.ras_manager()
    m.metric = 'paper'; m.patch_size = 2; m.neighbor_expansion_mode = '0'
    m.complexity_source = 'x_pred'; m.starvation_scale = .1
    m.enable_region_sparse_attention = False; m.use_cluster_graph_attention = False
    monkeypatch.setattr(ras_manager, 'MANAGER', m)
    scheduler = ClusterAdaptiveScheduler(num_clusters=2, cluster_selection_ratio=.5,
                                        recluster_every_step=False, final_sync_steps=0)
    scheduler.clustered = True; scheduler.actual_num_clusters = 2
    scheduler.cluster_labels = np.array([0,0,1,1]); scheduler.patch_drop_cnt = torch.tensor([0.,0.,10.,10.])
    clean = torch.zeros(1,2,4,4)
    current = torch.ones_like(clean); current[:,:,:2] *= 2
    # Detail=0; region scores are 1 and 0.5*exp(1), selecting region 1.
    _, active = scheduler.cluster_adaptive_selection(current,clean,4,4)
    assert set(active.tolist()) == {2,3}
    assert scheduler.patch_drop_cnt.tolist() == [1.,1.,0.,0.]


def test_single_anchor_is_geometric_centroid_nearest():
    xy = token_coordinates(4, 5, 'cpu')
    ids = torch.tensor([0, 1, 2, 3, 4, 6, 7, 8, 12])
    rep, groups, mask, weights = spatial_anchor_groups(ids, xy, 1)
    expected = ids[(xy[ids] - xy[ids].mean(0)).square().sum(-1).argmin()]
    assert rep.item() == expected.item()
    assert torch.equal(groups[mask],ids)
    assert weights.tolist() == [9.]


@pytest.mark.parametrize('count',[0,1,4,16])
def test_anchor_subgroups_partition_every_token_once(count):
    xy=token_coordinates(3,4,'cpu'); ids=torch.tensor([0,1,2,3,4,7,8,9,10,11])
    reps,groups,mask,weights=spatial_anchor_groups(ids,xy,count)
    assert len(reps)==min(count,len(ids))
    if count:
        assert torch.equal(groups[mask].sort().values,ids)
        assert torch.all(weights > 0)
        assert weights.sum()==len(ids)
        for r, g, valid in zip(reps,groups,mask):
            assert r in g[valid]


def test_log_count_and_mean_value_match_dense_equal_logits():
    generator=torch.Generator().manual_seed(17)
    q=torch.randn(1,2,3,8,generator=generator)
    local_k=torch.randn(1,2,2,8,generator=generator)
    local_v=torch.randn(1,2,2,8,generator=generator)
    remote_k=torch.randn(1,2,1,8,generator=generator)
    remote_v=torch.randn(1,2,5,8,generator=generator)
    dense=F.scaled_dot_product_attention(q,torch.cat([local_k,remote_k.expand(-1,-1,5,-1)],2),torch.cat([local_v,remote_v],2))
    bias=torch.tensor([0.,0.,np.log(5)],dtype=q.dtype)
    compressed=F.scaled_dot_product_attention(q,torch.cat([local_k,remote_k],2),torch.cat([local_v,remote_v.mean(2,keepdim=True)],2),attn_mask=bias[None,None,None,:])
    torch.testing.assert_close(compressed,dense)


def test_prediction_reuse_does_not_scale_cached_output():
    scheduler=ClusterAdaptiveScheduler(use_momentum_prediction=False)
    scheduler.prev_delta_sigma=.1
    m=ras_manager.ras_manager(); old=ras_manager.MANAGER; ras_manager.MANAGER=m
    try:
        m.cached_scaled_noise=torch.tensor([[1.,-3.]])
        scheduler.cached_noise=m.cached_scaled_noise.clone()
        torch.testing.assert_close(scheduler._compute_scaled_noise(.01),m.cached_scaled_noise)
    finally:
        ras_manager.MANAGER=old


def test_clean_ddpm_partition_input_not_next_sample(monkeypatch):
    m=ras_manager.ras_manager(); m.metric='paper';m.sample_ratio=.5
    m.num_steps=2;m.scheduler_start_step=1;m.scheduler_end_step=2
    m.cluster_prediction_source='x0_pred'
    monkeypatch.setattr(ras_manager,'MANAGER',m)
    scheduler=ClusterAdaptiveDDPMScheduler(recluster_every_step=False,final_sync_steps=0)
    scheduler.set_timesteps(2)
    sample=torch.randn(1,4,8,8); noise=torch.randn_like(sample)
    timestep=scheduler.timesteps[0]
    expected=scheduler._predicted_clean_sample(noise,timestep,sample)
    captured=[]
    def capture(current, clean, height, width, diff=None):
        captured.append(clean.clone())
        return torch.empty(0,dtype=torch.long),torch.arange(16)
    monkeypatch.setattr(scheduler,'cluster_adaptive_selection',capture)
    scheduler.step(noise,timestep,sample,generator=torch.Generator().manual_seed(4))
    assert len(captured)==1
    torch.testing.assert_close(captured[0],expected)


def test_no_external_attention_backend_imports():
    root=Path(__file__).parents[1]/'src'
    for file in root.rglob('*.py'):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith('flash_attn'),file
            elif isinstance(node,ast.Import):
                assert all(not n.name.startswith('flash_attn') for n in node.names),file
    with pytest.raises(ValueError):
        InferenceConfig(backend='flash')


def test_grouped_attention_matches_explicit_local_and_remote_formula(monkeypatch):
    from ras.utils.ras_argparser import parse_args_from_dict
    from ras.modules.attention_processor import RASLuminaAttnProcessor2_0
    m=ras_manager.ras_manager()
    monkeypatch.setattr(ras_manager,'MANAGER',m)
    m.set_parameters(parse_args_from_dict(dict(
        metric='paper',neighbor_expansion_mode='0',cluster_prediction_source='x0_pred',
        use_cluster_graph_attention=True,cluster_graph_attention_hops=0,
        use_cluster_remote_anchors=True,cluster_remote_anchor_count=1,
        cluster_remote_anchor_weight_mode='logn',enable_cluster_remote_anchor_value_coreset=True,
        enable_cluster_grouped_sdpa=True,cluster_relevance_topk_ratio=0.0)))
    m.use_cluster_concat_attention=True
    scheduler=ClusterAdaptiveScheduler(num_clusters=3,cluster_selection_ratio=1,
                                      recluster_every_step=False,final_sync_steps=0)
    scheduler.actual_num_clusters=3
    # Deliberately noncontiguous raster labels to exercise permutation.
    labels=np.array([0,1,1,0,2,2])
    monkeypatch.setattr(scheduler,'spatial_aware_clustering',lambda *args:labels)
    clean=torch.randn(1,2,4,6)
    _,active=scheduler.cluster_adaptive_selection(clean+1,clean,4,6)
    m.set_active_patch_indices(active)
    generator=torch.Generator().manual_seed(5)
    q=torch.randn(1,2,6,4,generator=generator)
    k=torch.randn(1,2,6,4,generator=generator)
    v=torch.randn(1,2,6,4,generator=generator)
    processor=RASLuminaAttnProcessor2_0()
    actual=processor._cluster_concat_attention(q,k,v,m.other_patchified_index,
                                               torch.ones(1,6,dtype=torch.bool),.5)
    expected=torch.empty_like(actual)
    xy=token_coordinates(2,3,'cpu')
    for qi,token in enumerate(m.other_patchified_index):
        region=int(m.cluster_sorted_labels[token])
        local=torch.where(m.cluster_sorted_labels==region)[0]
        keys=[k[:,:,local]];values=[v[:,:,local]];bias=[torch.zeros(len(local))]
        for remote in range(3):
            if remote==region:continue
            raster=torch.from_numpy(np.flatnonzero(labels==remote))
            representative=raster[(xy[raster]-xy[raster].mean(0)).square().sum(-1).argmin()].view(1)
            representative=m.map_patch_indices_to_cluster_order(representative)
            members=m.map_patch_indices_to_cluster_order(raster)
            keys.append(k[:,:,representative]);values.append(v[:,:,members].mean(2,keepdim=True))
            bias.append(torch.tensor([float(np.log(len(raster)))]))
        expected[:,:,qi:qi+1]=F.scaled_dot_product_attention(q[:,:,qi:qi+1],torch.cat(keys,2),
            torch.cat(values,2),attn_mask=torch.cat(bias)[None,None,None,:],scale=.5)
    torch.testing.assert_close(actual,expected)
