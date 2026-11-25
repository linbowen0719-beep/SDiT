"""Experimental whole-group anchor means and optional fused token KV refresh.

This module owns no cache policy. The caller supplies a validated complete-group
update plan, unique group IDs, and unique active token indices. Partial-group
updates must keep their existing reconstruction path. ``ordered`` preserves the
FP32 token accumulation order; ``tree`` changes it and is explicitly approximate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class AnchorRefreshPlan:
    group_ids: torch.Tensor
    offsets: torch.Tensor
    active_indices: torch.Tensor
    order: torch.Tensor | None
    # Geometry-only launch bindings: never retain layer activations or caches.
    _launch_cache: dict = field(default_factory=dict,repr=False,compare=False)


def build_anchor_refresh_plan(group_ids, lengths, active_indices, order=None, *, complete=True):
    """Convert an existing validated AP plan once; no CPU synchronization.

    The caller guarantees positive lengths summing to active_indices.numel(),
    unique in-range group IDs/active indices, and an optional permutation order.
    Reuse this object across layers sharing the same immutable active plan.
    """
    if not complete:
        raise ValueError("anchor refresh requires complete groups")
    vectors=(group_ids,lengths,active_indices) + (() if order is None else (order,))
    if any(x.ndim != 1 or x.dtype not in (torch.int32,torch.int64) or not x.is_cuda
           or x.device != group_ids.device for x in vectors):
        raise ValueError("plan vectors must be 1D int32/int64 on one CUDA device")
    if group_ids.numel() != lengths.numel() or not group_ids.numel() or not active_indices.numel():
        raise ValueError("nonempty IDs/lengths and active indices required")
    if order is not None and order.numel() != active_indices.numel():
        raise ValueError("order length must match active indices")
    offsets=torch.cat((torch.zeros(1,device=lengths.device,dtype=torch.int64),lengths.cumsum(0)))
    return AnchorRefreshPlan(group_ids.contiguous(),offsets,active_indices.contiguous(),
                             None if order is None else order.contiguous())


@triton.jit(do_not_specialize=[
    "NSB","NSN","NSH","MSB","MSN","MSH","VSB","VSN","VSH",
    "KSB","KSN","KSH","NKSB","NKSN","NKSH",
])
def _anchor_mean_tile(
    NewV, Means, IDs, Offsets, Active, Order, VCache, KCache, NewK,
    gi,b,feature_tile,
    H:tl.constexpr,D:tl.constexpr,
    NSB,NSN,NSH,MSB,MSN,MSH,VSB,VSN,VSH,KSB,KSN,KSH,NKSB,NKSN,NKSH,
    HAS_ORDER:tl.constexpr,WRITE_V:tl.constexpr,WRITE_K:tl.constexpr,
    TREE:tl.constexpr,BF:tl.constexpr,BT:tl.constexpr,
):
    f=feature_tile*BF+tl.arange(0,BF)
    h=f//D;d=f%D;fm=f<H*D
    start=tl.load(Offsets+gi);end=tl.load(Offsets+gi+1)
    group=tl.load(IDs+gi)
    acc=tl.zeros((BF,),tl.float32)
    if TREE:
        for block in range(tl.cdiv(end-start,BT)):
            t=start+block*BT+tl.arange(0,BT)
            tm=t<end
            if HAS_ORDER:
                src=tl.load(Order+t,mask=tm,other=0)
            else:
                src=t
            x=tl.load(NewV+b*NSB+src[:,None]*NSN+h[None,:]*NSH+d[None,:],
                      mask=tm[:,None]&fm[None,:],other=0)
            acc+=tl.sum(x.to(tl.float32),axis=0)
            if WRITE_V or WRITE_K:
                dst=tl.load(Active+src,mask=tm,other=0)
                if WRITE_V:
                    tl.store(VCache+b*VSB+dst[:,None]*VSN+h[None,:]*VSH+d[None,:],x,
                             mask=tm[:,None]&fm[None,:])
                if WRITE_K:
                    nk=tl.load(NewK+b*NKSB+src[:,None]*NKSN+h[None,:]*NKSH+d[None,:],
                               mask=tm[:,None]&fm[None,:],other=0)
                    tl.store(KCache+b*KSB+dst[:,None]*KSN+h[None,:]*KSH+d[None,:],nk,
                             mask=tm[:,None]&fm[None,:])
    else:
        for t in range(start,end):
            if HAS_ORDER:
                src=tl.load(Order+t)
            else:
                src=t
            x=tl.load(NewV+b*NSB+src*NSN+h*NSH+d,mask=fm,other=0)
            acc=acc+x.to(tl.float32)
            if WRITE_V or WRITE_K:
                dst=tl.load(Active+src)
                if WRITE_V:
                    tl.store(VCache+b*VSB+dst*VSN+h*VSH+d,x,mask=fm)
                if WRITE_K:
                    nk=tl.load(NewK+b*NKSB+src*NKSN+h*NKSH+d,mask=fm,other=0)
                    tl.store(KCache+b*KSB+dst*KSN+h*KSH+d,nk,mask=fm)
    mean=tl.div_rn(acc,(end-start).to(tl.float32))
    tl.store(Means+b*MSB+group*MSN+h*MSH+d,mean,mask=fm)


@triton.jit(do_not_specialize=[
    "NGROUPS","QLEN","NSB","NSN","NSH","MSB","MSN","MSH","VSB","VSN","VSH",
    "KSB","KSN","KSH","NKSB","NKSN","NKSH",
])
def _refresh_anchor_kernel(
    NewV, Means, IDs, Offsets, Active, Order, VCache, KCache, NewK,
    B:tl.constexpr,H:tl.constexpr,D:tl.constexpr,NGROUPS,QLEN,
    NSB,NSN,NSH,MSB,MSN,MSH,VSB,VSN,VSH,KSB,KSN,KSH,NKSB,NKSN,NKSH,
    HAS_ORDER:tl.constexpr,WRITE_V:tl.constexpr,WRITE_K:tl.constexpr,
    TREE:tl.constexpr,PARALLEL_COPY:tl.constexpr,BF:tl.constexpr,BT:tl.constexpr,
):
    pid=tl.program_id(0);nf=tl.cdiv(H*D,BF);mean_tiles=NGROUPS*B*nf
    if pid<mean_tiles:
        gi=pid//(B*nf);b=(pid//nf)%B;ft=pid%nf
        _anchor_mean_tile(
            NewV,Means,IDs,Offsets,Active,Order,VCache,KCache,NewK,gi,b,ft,H,D,
            NSB,NSN,NSH,MSB,MSN,MSH,VSB,VSN,VSH,KSB,KSN,KSH,NKSB,NKSN,NKSH,
            HAS_ORDER,WRITE_V and not PARALLEL_COPY,WRITE_K and not PARALLEL_COPY,TREE,BF,BT,
        )
    elif PARALLEL_COPY:
        # Copy CTAs are independent of mean CTAs: both read immutable new K/V,
        # and their destinations are disjoint means versus token caches.
        x=(pid-mean_tiles)*256+tl.arange(0,256);mask=x<B*QLEN*H*D
        d=x%D;h=(x//D)%H;src=(x//(H*D))%QLEN;copy_b=x//(QLEN*H*D)
        dst=tl.load(Active+src,mask=mask,other=0)
        if WRITE_V:
            value=tl.load(NewV+copy_b*NSB+src*NSN+h*NSH+d,mask=mask,other=0)
            tl.store(VCache+copy_b*VSB+dst*VSN+h*VSH+d,value,mask=mask)
        if WRITE_K:
            key=tl.load(NewK+copy_b*NKSB+src*NKSN+h*NKSH+d,mask=mask,other=0)
            tl.store(KCache+copy_b*KSB+dst*KSN+h*KSH+d,key,mask=mask)


def refresh_anchor_means_and_kv_(
    new_value:torch.Tensor, mean_cache:torch.Tensor, plan:AnchorRefreshPlan,
    *, value_cache:torch.Tensor|None=None, key_cache:torch.Tensor|None=None,
    new_key:torch.Tensor|None=None, mode="ordered", block_features=None,
    parallel_kv=False, validate=True,
):
    """In-place means[B,A,H,D], optionally V/K[B,N,H,D]; return means.

    Inputs new_value/new_key are [B,Q,H,D] in active-index order. ``ordered``
    accumulates each group's FP32 sum in the same sequence order used by the
    existing segment_reduce path, then uses round-to-nearest division. Bitwise
    parity must still be verified for the deployed PyTorch/dtype combination.
    ``tree`` reorders additions and must not be enabled as an exact path.
    Source tensors must not alias destination caches when indices are reordered.
    parallel_kv=True schedules independent copy CTAs in the same launch; it
    preserves ordered mean arithmetic without serializing KV stores into it.
    validate=False is reserved for a caller that already validated identical
    tensor geometry/dtype/device and the immutable plan.
    """
    if validate and mode not in ("ordered","tree"):
        raise ValueError("mode must be ordered or tree")
    if validate and (new_value.ndim!=4 or mean_cache.ndim!=4 or not new_value.is_cuda):
        raise ValueError("new values and means must have four dimensions on CUDA")
    b,q,h,d=new_value.shape
    if validate and (min(b,q,h,d)<1 or new_value.dtype not in (torch.bfloat16,torch.float16,torch.float32)):
        raise ValueError("nonempty fp16/bf16/fp32 values required")
    if validate and (mean_cache.shape[0]!=b or mean_cache.shape[2:]!=(h,d)):
        raise ValueError("mean cache shape mismatch")
    if validate and plan.active_indices.numel()!=q:
        raise ValueError("active plan length differs from new values")
    for x in ((mean_cache,plan.group_ids,plan.offsets,plan.active_indices,plan.order) if validate else ()):
        if x is not None and x.device!=new_value.device:
            raise ValueError("all tensors must be on the same CUDA device")
    tensors=[new_value,mean_cache]
    for cache in ((value_cache,key_cache) if validate else ()):
        if cache is not None:
            if cache.ndim!=4 or cache.shape[0]!=b or cache.shape[2:]!=(h,d):
                raise ValueError("token cache shape mismatch")
            tensors.append(cache)
    if validate and (key_cache is None)!=(new_key is None):
        raise ValueError("key cache and new key must be supplied together")
    if validate and new_key is not None:
        if new_key.shape!=new_value.shape:
            raise ValueError("new K/V shapes must match")
        tensors.append(new_key)
    if validate and any(x.device!=new_value.device or x.dtype!=new_value.dtype or x.stride(-1)!=1 for x in tensors):
        raise ValueError("floating tensors need matching dtype/device and contiguous head dimension")
    bf=block_features or (32 if mode=="ordered" else 16)
    if validate and bf not in (16,32,64,128):
        raise ValueError("block_features must be 16,32,64,128")
    dummy=new_value
    strides=lambda x: x.stride()[:3] if x is not None else (0,0,0)
    parallel=bool(parallel_kv and (value_cache is not None or key_cache is not None))
    grid=plan.group_ids.numel()*b*triton.cdiv(h*d,bf)+(triton.cdiv(b*q*h*d,256) if parallel else 0)
    _refresh_anchor_kernel[(grid,)](
        new_value,mean_cache,plan.group_ids,plan.offsets,plan.active_indices,
        plan.order if plan.order is not None else plan.active_indices,
        value_cache if value_cache is not None else dummy,
        key_cache if key_cache is not None else dummy,new_key if new_key is not None else dummy,
        b,h,d,plan.group_ids.numel(),q,*strides(new_value),*strides(mean_cache),*strides(value_cache),
        *strides(key_cache),*strides(new_key),plan.order is not None,
        value_cache is not None,key_cache is not None,mode=="tree",parallel,bf,128,
        num_warps=1 if mode=="ordered" else 4,enable_fp_fusion=False,
    )
    return mean_cache


_MISSING_LAUNCH = object()


def _cached_parallel_launch(plan, tensors, geometry):
    """Validate a new layout once and cache scalar launch arguments only."""
    new_value,mean_cache,value_cache,key_cache,new_key=tensors
    device=new_value.device;dtype=new_value.dtype
    if not new_value.is_cuda or dtype not in (torch.float16,torch.bfloat16,torch.float32):
        return None
    if any(len(shape)!=4 or any(n<1 for n in shape) or stride[-1]!=1
           or dt!=dtype or dev!=device for shape,stride,dt,dev in geometry):
        return None
    b,q,h,d=new_value.shape
    if (new_key.shape!=new_value.shape or value_cache.shape!=key_cache.shape
            or value_cache.shape[0]!=b or value_cache.shape[2:]!=(h,d)
            or mean_cache.shape[0]!=b or mean_cache.shape[2:]!=(h,d)
            or q!=plan.active_indices.numel() or q>value_cache.shape[1]):
        return None
    vectors=(plan.group_ids,plan.offsets,plan.active_indices)+( () if plan.order is None else (plan.order,))
    if any(x.device!=device or x.ndim!=1 or not x.is_contiguous()
           or x.dtype not in (torch.int32,torch.int64) for x in vectors):
        return None
    groups=plan.group_ids.numel()
    if not groups or plan.offsets.numel()!=groups+1 or groups>mean_cache.shape[1]:
        return None
    if plan.order is not None and plan.order.numel()!=q:
        return None
    # Read-only sources may be strided or expanded. Destination geometry must
    # prove no internal overlap. Ambiguous sliced layouts safely fall back.
    if any(torch._debug_has_internal_overlap(t)!=0 for t in (mean_cache,value_cache,key_cache)):
        return None
    bf=32
    grid=groups*b*triton.cdiv(h*d,bf)+triton.cdiv(b*q*h*d,256)
    strides=tuple(s for _,stride,_,_ in geometry for s in stride[:3])
    scalar_args=(b,h,d,groups,q,*strides,plan.order is not None,True,True,False,True,bf,128)
    return _refresh_anchor_kernel[(grid,)],scalar_args


def try_refresh_cached_anchor_means_and_kv_(
    new_value:torch.Tensor,mean_cache:torch.Tensor,plan:AnchorRefreshPlan|None,
    *,value_cache:torch.Tensor,key_cache:torch.Tensor,new_key:torch.Tensor,
    complete:bool=True,
) -> bool:
    """Try one exact ordered-mean + independent KV-copy launch; return handled.

    True means all three updates have been enqueued on the current CUDA stream;
    the caller MUST skip its separate mean update and token K/V writes. False
    means no work was launched and no destination was modified, so the caller
    can run its complete original fallback. Runtime launch errors propagate.

    Contract: inference only; plan metadata remains immutable and was validated
    by the caller (positive complete segments, permutation order, unique valid
    group/active indices). Tensors use [B,Q/N/A,H,D], matching dtype/device and
    last-dimension stride 1. Non-contiguous source layouts are supported. Every
    call rejects source/destination or destination/destination storage aliasing;
    valid disjoint views of one storage conservatively fall back. Destination
    layouts must have provably non-overlapping elements. The plan caches only
    validated geometry and scalar launch arguments, so sharing it across layers
    retains no layer tensors and requires no GPU synchronization.
    """
    if not complete or plan is None or torch.is_grad_enabled():
        return False
    tensors=(new_value,mean_cache,value_cache,key_cache,new_key)
    if any(not isinstance(t,torch.Tensor) for t in tensors):
        return False
    geometry=tuple((t.shape,t.stride(),t.dtype,t.device) for t in tensors)
    binding=plan._launch_cache.get(geometry,_MISSING_LAUNCH)
    if binding is _MISSING_LAUNCH:
        binding=_cached_parallel_launch(plan,tensors,geometry)
        plan._launch_cache[geometry]=binding
    if binding is None:
        return False
    # Storage identity is checked on every invocation because new activations
    # change across layers even when their cached geometry is identical.
    pointers=tuple(t.untyped_storage().data_ptr() for t in tensors)
    if (len(set(pointers[1:4]))!=3 or pointers[0] in pointers[1:4]
            or pointers[4] in pointers[1:4]):
        return False
    launch,scalar_args=binding
    launch(new_value,mean_cache,plan.group_ids,plan.offsets,plan.active_indices,
           plan.order if plan.order is not None else plan.active_indices,
           value_cache,key_cache,new_key,*scalar_args,
           num_warps=1,enable_fp_fusion=False)
    return True
