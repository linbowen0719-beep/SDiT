"""Optional, request-scoped reuse of immutable Lumina text computations.

Only the controlled, callback-free pipeline may assert immutable_context for
inference tensors (which have no version counter). Standalone calls otherwise
fall back. Parameters must have version counters; untracked mutation through
``.data`` or external storage is outside this explicit immutable-run contract.
No attention output, representative choice, or reduction is approximated.

immutable_model=True is a separate explicit request contract: no
parameter/buffer, module topology, operator attributes/forward, adapter, training
mode, or local/global hook may be modified until the request exits, including by
other threads. The first scope validates the model; later scopes do not detect
contract violations. Default False retains the original per-step model guards.
The prepared model belongs to the request, independently of resettable memo state.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from diffusers.models.normalization import RMSNorm


@dataclass
class _RunState:
    token: object
    norms: dict = field(default_factory=dict)
    projections: dict = field(default_factory=dict)


@dataclass
class _Scope:
    state: _RunState
    source: torch.Tensor
    mask: object
    signatures: dict
    contexts: dict = field(default_factory=dict)


@dataclass
class _ModelPlan:
    token: object
    transformer: object
    operators: tuple
    base: object = None
    signatures: object = None


@contextmanager
def text_cache_run(manager, *, immutable_context=False, immutable_model=False):
    """Enter one generation; reset_cache may discard memo state within it."""
    names = ("lumina_text_cache_run_token", "lumina_text_cache_immutable_context",
             "lumina_text_cache_state", "lumina_text_cache_scope",
             "lumina_text_cache_immutable_model", "lumina_text_cache_model_plan")
    previous = tuple(getattr(manager, name, None) for name in names)
    manager.lumina_text_cache_run_token = object()
    manager.lumina_text_cache_immutable_context = bool(immutable_context)
    manager.lumina_text_cache_state = None
    manager.lumina_text_cache_scope = None
    manager.lumina_text_cache_immutable_model = bool(immutable_model)
    manager.lumina_text_cache_model_plan = None
    try:
        yield
    finally:
        for name, value in zip(names, previous):
            setattr(manager, name, value)


def _tensor_signature(tensor, *, allow_unversioned=False):
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError:
        if not allow_unversioned:
            raise ValueError("an immutable-run contract is required")
        version = None
    return (id(tensor), version, tensor.data_ptr(), tensor.shape, tensor.stride(),
            tensor.dtype, tensor.device, tensor)


def _has_hooks(module):
    return any(getattr(module, name, None) for name in (
        "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"))


def _operator_signature(module):
    if module is None:
        return None
    supported = (nn.Linear, nn.LayerNorm, nn.RMSNorm, RMSNorm)
    if type(module) not in supported or module.training or _has_hooks(module):
        raise ValueError("only ordinary eval linear/norm operators can be reused")
    tensors = tuple((name, _tensor_signature(tensor))
                    for name, tensor in (*module._parameters.items(), *module._buffers.items()))
    forward = getattr(module.forward, "__func__", module.forward)
    options = tuple((name, getattr(module, name, None))
                    for name in ("eps", "normalized_shape", "elementwise_affine"))
    return id(module), forward, tensors, options


def _model_operators(transformer):
    """Validate the complete hook surface plus just the reused eval operators."""
    if any(module._forward_hooks or module._forward_pre_hooks for module in transformer.modules()):
        raise ValueError("model hooks may modify cached inputs or weights")
    operators = []
    for block in transformer.layers:
        attn = block.attn2
        if block.training or attn.training or _has_hooks(block) or _has_hooks(attn):
            raise ValueError("mutable/training block callbacks cannot use text reuse")
        norm = _operator_signature(block.norm1_context)
        projections = tuple(_operator_signature(op) for op in (attn.to_k, attn.to_v, attn.norm_k))
        operators.append((block, norm, attn, projections))
    return tuple(operators)


def _bind_signatures(base, operators):
    signatures = {}
    for block, norm, attn, projections in operators:
        signatures[block] = (base, norm)
        signatures[attn] = (base, norm, projections)
    return signatures


def begin_text_cache_scope(manager, transformer, encoder_hidden_states, encoder_mask,
                           *, cross_attention_kwargs=None):
    """Snapshot inexpensive guards once per step, before the immutable layer loop.

    The source and module references remain alive in the scope/cache. Parameter
    changes via normal in-place operations, replacement, dtype/device changes,
    training and hooks invalidate reuse in the default dynamic-model mode. With
    immutable_model=True, the caller instead guarantees a fixed model for the
    whole request, including across memo resets. Prompt/CFG replacements and
    versioned source writes still invalidate reuse in both modes. Unversioned
    sources require the explicit callback-free immutable-context contract.
    """
    manager.lumina_text_cache_scope = None
    token = getattr(manager, "lumina_text_cache_run_token", None)
    global_hooks = torch.nn.modules.module
    # Lumina's default proportional attention supplies this option. It changes
    # only the attention scale, after the cached linear/key-norm computations.
    # Keep it in the signature and reject every unrecognized callback/LoRA option.
    attention_options = None
    if not cross_attention_kwargs:
        attention_options = ()
    elif (isinstance(cross_attention_kwargs, dict)
          and set(cross_attention_kwargs) == {"base_sequence_length"}
          and type(cross_attention_kwargs["base_sequence_length"]) is int
          and cross_attention_kwargs["base_sequence_length"] > 0):
        attention_options = (("base_sequence_length", cross_attention_kwargs["base_sequence_length"]),)
    if (not getattr(manager, "enable_lumina_text_cache", False) or token is None
            or torch.is_grad_enabled() or transformer.training or _has_hooks(transformer)
            or attention_options is None
            or getattr(manager, "attention_analysis_callback", None) is not None
            or getattr(manager, "scheduler_analysis_callback", None) is not None
            or global_hooks._global_forward_hooks or global_hooks._global_forward_pre_hooks):
        manager.lumina_text_cache_state = None
        manager.lumina_text_cache_model_plan = None
        return None
    state = getattr(manager, "lumina_text_cache_state", None)
    if state is None or state.token is not token:
        state = _RunState(token)
        manager.lumina_text_cache_state = state
    try:
        allow = bool(getattr(manager, "lumina_text_cache_immutable_context", False))
        source = _tensor_signature(encoder_hidden_states, allow_unversioned=allow)
        mask = _tensor_signature(encoder_mask, allow_unversioned=allow)
        device_type = encoder_hidden_states.device.type
        execution = (torch.is_autocast_enabled(device_type),
                     torch.get_autocast_dtype(device_type),
                     torch.get_float32_matmul_precision())
        stream = (torch.cuda.current_stream(encoder_hidden_states.device).cuda_stream
                  if encoder_hidden_states.is_cuda else None)
        base = source, mask, execution, stream, attention_options
        if bool(getattr(manager, "lumina_text_cache_immutable_model", False)):
            plan = getattr(manager, "lumina_text_cache_model_plan", None)
            if plan is None or plan.token is not token or plan.transformer is not transformer:
                plan = _ModelPlan(token, transformer, _model_operators(transformer))
                manager.lumina_text_cache_model_plan = plan
            if plan.signatures is None or plan.base != base:
                plan.base = base
                plan.signatures = _bind_signatures(base, plan.operators)
            signatures = plan.signatures
        else:
            manager.lumina_text_cache_model_plan = None
            signatures = _bind_signatures(base, _model_operators(transformer))
    except (ValueError, RuntimeError, AttributeError):
        manager.lumina_text_cache_state = None
        manager.lumina_text_cache_model_plan = None
        return None
    scope = _Scope(state, encoder_hidden_states, encoder_mask, signatures)
    manager.lumina_text_cache_scope = scope
    return scope


def end_text_cache_scope(manager, scope):
    if getattr(manager, "lumina_text_cache_scope", None) is scope:
        manager.lumina_text_cache_scope = None


def context_norm(manager, block, source, compute):
    """Return exactly the ordinary norm result, reused only inside a valid scope."""
    scope = getattr(manager, "lumina_text_cache_scope", None)
    if scope is None or source is not scope.source or block not in scope.signatures:
        return compute()
    signature = scope.signatures[block]
    cached = scope.state.norms.get(block)
    if cached is None or cached[0] != signature:
        value = compute()
        # Retain input references as well as identity/version fingerprints.
        cached = (signature, value, source, scope.mask)
        scope.state.norms[block] = cached
    value = cached[1]
    scope.contexts[block.attn2] = value
    return value


def cross_kv_cache(manager, attn, context):
    """Return (cache entry, binding) for this block's registered context only."""
    scope = getattr(manager, "lumina_text_cache_scope", None)
    if (scope is None or scope.contexts.get(attn) is not context
            or attn not in scope.signatures):
        return None, None
    signature = scope.signatures[attn]
    cached = scope.state.projections.get(attn)
    if cached is not None and cached[0] == signature and cached[3] is context:
        return cached, None
    return None, (scope.state, attn, signature, context)


def store_cross_kv(binding, key, value):
    if binding is not None:
        state, attn, signature, context = binding
        state.projections[attn] = (signature, key, value, context)
