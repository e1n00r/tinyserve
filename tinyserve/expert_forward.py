"""Expert forward computation: weight swapping, inlined forwards, layout args.

All "how to compute an expert" logic lives here. ExpertPipeline (dispatch
logic) imports from this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .expert_store import TensorLayout

logger = logging.getLogger(__name__)

try:
    from .csrc import get_expert_loop as _get_expert_loop
except ImportError:
    logger.warning("C++ expert loop extension not available, using Python fallback")

    def _get_expert_loop():
        return None


# Quick GELU approximation coefficient: sigmoid(1.702 * x) ≈ GELU(x).
_QUICK_GELU_COEFF = 1.702

_template_weight_storage: dict[int, dict[str, torch.Tensor]] = {}


def _get_template_storage(template: nn.Module, layout) -> dict[str, torch.Tensor]:
    """Get or create dedicated storage tensors for the template.

    Ensures swap_weights_and_forward always writes into its own memory,
    never into cache slot views left behind by forward_from_packed.
    """
    tid = id(template)
    if tid not in _template_weight_storage:
        storage = {}
        for name, (shape, dtype) in layout.specs.items():
            parts = name.split(".")
            mod = template
            for part in parts[:-1]:
                mod = getattr(mod, part)
            param = getattr(mod, parts[-1])
            storage[name] = torch.empty(shape, dtype=dtype, device=param.device)
        _template_weight_storage[tid] = storage
    return _template_weight_storage[tid]


def swap_weights_and_forward(
    template: nn.Module,
    buf,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Copy buffer tensors into the template module's parameters, run forward.

    Rebinds param.data to dedicated storage before copying, so we never
    write through into GPU cache slot views left by forward_from_packed.
    """
    with torch.no_grad():
        own = _get_template_storage(template, buf.layout)
        for name, (shape, dtype) in buf.layout.specs.items():
            parts = name.split(".")
            mod = template
            for part in parts[:-1]:
                mod = getattr(mod, part)
            param = getattr(mod, parts[-1])
            storage = own[name]
            storage.copy_(buf.get_tensor(name))
            param.data = storage
    return template(hidden_states)


def _precompute_param_refs(
    template: nn.Module,
    layout: TensorLayout,
) -> list[tuple[nn.Parameter, int, int, tuple[int, ...], torch.dtype]]:
    """Precompute parameter references + offsets to avoid repeated string ops."""
    refs = []
    for name, (shape, dtype) in layout.specs.items():
        parts = name.split(".")
        mod = template
        for part in parts[:-1]:
            mod = getattr(mod, part)
        param = getattr(mod, parts[-1])
        offset = layout.offsets[name]
        nbytes = layout.sizes[name]
        refs.append((param, offset, nbytes, shape, dtype))
    return refs


def forward_from_packed(
    template: nn.Module,
    packed: torch.Tensor,
    param_refs: list,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Set template params to views of packed tensor — zero copy — then forward."""
    for param, offset, nbytes, shape, dtype in param_refs:
        param.data = packed[offset : offset + nbytes].view(dtype).view(shape)
    return template(hidden_states)


def _build_inline_forward(layout, act_fn):
    """Build an inlined expert forward function with baked-in offsets.

    Eliminates per-call: dict comprehension, getattr loop, shape checks.
    Returns a function: (packed, hidden) -> output.
    """
    specs = layout.specs

    if "gate_up_proj" not in specs or "down_proj" not in specs:
        return None

    # MXFP4 uses blocks+scales (uint8) — handled by _build_gpu_int4_forward.
    if "gate_up_proj_scales" in specs:
        return None

    gu_off = layout.offsets["gate_up_proj"]
    gu_sz = layout.sizes["gate_up_proj"]
    gu_shape, gu_dtype = specs["gate_up_proj"]
    dn_off = layout.offsets["down_proj"]
    dn_sz = layout.sizes["down_proj"]
    dn_shape, dn_dtype = specs["down_proj"]
    # Pre-decide transpose. The template forward checks:
    # if w.shape[0] == hidden_dim: w = w.t() before passing to F.linear.
    # F.linear(x, w) does x @ w.T. So we need w in [out, in] for F.linear.
    # After the template's transpose, it passes to F.linear, so effectively:
    #   stored as [in, out] → t() → [out, in] → F.linear does x @ [out,in].T = x @ [in,out]
    # This means: if shape[0] == gu_shape's in_features (hidden_dim), do t() then F.linear.
    # Simpler: just replicate the template's check exactly.
    # For gate_up: if shape[0] == hidden_dim → transpose before F.linear
    # hidden_dim = gu_shape[0] if gu_shape[0] < gu_shape[1] else gu_shape[1]
    # (gate_up_proj maps hidden → 2*intermediate, so hidden is the smaller dim)
    hidden_dim = min(gu_shape)
    gu_needs_t = gu_shape[0] == hidden_dim
    # For down_proj: maps intermediate → hidden, so intermediate is input
    # The template checks: if w.shape[0] == gated.shape[-1] (intermediate_dim)
    # intermediate = max(gu_shape) // 2 for interleaved, or just check dn_shape
    # The template does: if w_dn.shape[0] == gated.shape[-1]. gated = intermediate_size.
    # For gate_up: output is 2*intermediate. After SwiGLU: intermediate.
    # So gated_dim = max(gu_shape) // 2 (chunk) or max(gu_shape) // 2 (interleaved)
    gated_dim = max(gu_shape) // 2  # both chunk and interleaved halve the output
    dn_needs_t = dn_shape[0] == gated_dim

    has_bias = "gate_up_proj_bias" in specs
    if has_bias:
        gub_off = layout.offsets["gate_up_proj_bias"]
        gub_sz = layout.sizes["gate_up_proj_bias"]
        gub_shape, gub_dtype = specs["gate_up_proj_bias"]
        dnb_off = layout.offsets["down_proj_bias"]
        dnb_sz = layout.sizes["down_proj_bias"]
        dnb_shape, dnb_dtype = specs["down_proj_bias"]

    linear = nn.functional.linear

    if act_fn is not None:
        # Standard SiLU/GELU gate
        def _forward(packed, h):
            w_gu = packed[gu_off : gu_off + gu_sz].view(gu_dtype).view(gu_shape)
            if gu_needs_t:
                w_gu = w_gu.t()
            b_gu = packed[gub_off : gub_off + gub_sz].view(gub_dtype).view(gub_shape) if has_bias else None
            gate_up = linear(h, w_gu, b_gu)
            gate, up = gate_up.chunk(2, dim=-1)
            gated = act_fn(gate) * up
            w_dn = packed[dn_off : dn_off + dn_sz].view(dn_dtype).view(dn_shape)
            if dn_needs_t:
                w_dn = w_dn.t()
            b_dn = packed[dnb_off : dnb_off + dnb_sz].view(dnb_dtype).view(dnb_shape) if has_bias else None
            return linear(gated, w_dn, b_dn)
    else:
        # GPT-OSS custom SwiGLU
        def _forward(packed, h):
            w_gu = packed[gu_off : gu_off + gu_sz].view(gu_dtype).view(gu_shape)
            if gu_needs_t:
                w_gu = w_gu.t()
            b_gu = packed[gub_off : gub_off + gub_sz].view(gub_dtype).view(gub_shape) if has_bias else None
            gate_up = linear(h, w_gu, b_gu)
            gate = gate_up[..., ::2].clamp(max=7.0)
            up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
            gated = (up + 1) * gate * torch.sigmoid(gate * _QUICK_GELU_COEFF)
            w_dn = packed[dn_off : dn_off + dn_sz].view(dn_dtype).view(dn_shape)
            if dn_needs_t:
                w_dn = w_dn.t()
            b_dn = packed[dnb_off : dnb_off + dnb_sz].view(dnb_dtype).view(dnb_shape) if has_bias else None
            return linear(gated, w_dn, b_dn)

    return _forward


def _build_mxfp4_inline_forward(layout, act_fn):
    """Build an inline forward for MXFP4 layouts using dot_scaled_vecmat.

    Eliminates forward_from_packed's param.data rebinding loop for MXFP4
    cache hits. Returns (packed, hidden) -> output callable.
    """
    specs = layout.specs
    if "gate_up_proj_scales" not in specs:
        return None

    gu_off = layout.offsets["gate_up_proj"]
    gu_sz = layout.sizes["gate_up_proj"]
    gu_shape = specs["gate_up_proj"][0]
    gs_off = layout.offsets["gate_up_proj_scales"]
    gs_sz = layout.sizes["gate_up_proj_scales"]
    gs_shape = specs["gate_up_proj_scales"][0]
    dn_off = layout.offsets["down_proj"]
    dn_sz = layout.sizes["down_proj"]
    dn_shape = specs["down_proj"][0]
    ds_off = layout.offsets["down_proj_scales"]
    ds_sz = layout.sizes["down_proj_scales"]
    ds_shape = specs["down_proj_scales"][0]

    from ._model_hooks import _mxfp4_linear

    if act_fn is not None:

        def _forward(packed, h):
            w_gu = packed[gu_off : gu_off + gu_sz].view(torch.uint8).view(gu_shape)
            s_gu = packed[gs_off : gs_off + gs_sz].view(torch.uint8).view(gs_shape)
            gate_up = _mxfp4_linear(h, w_gu, s_gu)
            gate, up = gate_up.chunk(2, dim=-1)
            gated = act_fn(gate) * up
            w_dn = packed[dn_off : dn_off + dn_sz].view(torch.uint8).view(dn_shape)
            s_dn = packed[ds_off : ds_off + ds_sz].view(torch.uint8).view(ds_shape)
            return _mxfp4_linear(gated, w_dn, s_dn)
    else:
        # GPT-OSS custom SwiGLU (interleaved, not chunked)
        def _forward(packed, h):
            w_gu = packed[gu_off : gu_off + gu_sz].view(torch.uint8).view(gu_shape)
            s_gu = packed[gs_off : gs_off + gs_sz].view(torch.uint8).view(gs_shape)
            gate_up = _mxfp4_linear(h, w_gu, s_gu)
            gate = gate_up[..., ::2].clamp(max=7.0)
            up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
            gated = (up + 1) * gate * torch.sigmoid(gate * _QUICK_GELU_COEFF)
            w_dn = packed[dn_off : dn_off + dn_sz].view(torch.uint8).view(dn_shape)
            s_dn = packed[ds_off : ds_off + ds_sz].view(torch.uint8).view(ds_shape)
            return _mxfp4_linear(gated, w_dn, s_dn)

    return _forward


def _build_gpu_int4_forward(layout, act_fn):
    """Build a GPU INT4 forward for MXFP4 layouts.

    Returns a callable (packed, hidden) -> output that converts MXFP4 cache
    slot data to INT4 on first call (cached by data_ptr), then runs
    torch._weight_int4pack_mm on GPU. ~5x faster than BF16 F.linear.

    Returns None if layout is not MXFP4 or GPU INT4 ops unavailable.
    """
    # GPU INT4 disabled: conversion cache consumes too much VRAM on 8GB GPUs.
    # The MXFP4 template forward (dot_scaled) is used instead.
    # TODO: enable when VRAM headroom detection is implemented.
    return None


_DTYPE_TO_INT = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.bfloat16: 15,
}


def _build_cpp_layout_args(layout, act_fn):
    """Build the layout arguments needed by the C++ fast_expert_forward."""
    specs = layout.specs

    if "gate_up_proj" not in specs or "down_proj" not in specs:
        return None
    if "gate_up_proj_scales" in specs:
        return None

    gu_off = layout.offsets["gate_up_proj"]
    gu_sz = layout.sizes["gate_up_proj"]
    gu_shape, gu_dtype = specs["gate_up_proj"]
    dn_off = layout.offsets["down_proj"]
    dn_sz = layout.sizes["down_proj"]
    dn_shape, dn_dtype = specs["down_proj"]

    hidden_dim = min(gu_shape)
    gu_needs_t = gu_shape[0] == hidden_dim
    gated_dim = max(gu_shape) // 2
    dn_needs_t = dn_shape[0] == gated_dim

    has_bias = "gate_up_proj_bias" in specs
    gub_off = gub_sz = dnb_off = dnb_sz = 0
    gub_shape = [1]
    dnb_shape = [1]
    gub_dtype_int = dnb_dtype_int = _DTYPE_TO_INT.get(gu_dtype, 6)
    if has_bias:
        gub_off = layout.offsets["gate_up_proj_bias"]
        gub_sz = layout.sizes["gate_up_proj_bias"]
        gub_shape_t, gub_dt = specs["gate_up_proj_bias"]
        gub_shape = list(gub_shape_t)
        gub_dtype_int = _DTYPE_TO_INT.get(gub_dt, 6)
        dnb_off = layout.offsets["down_proj_bias"]
        dnb_sz = layout.sizes["down_proj_bias"]
        dnb_shape_t, dnb_dt = specs["down_proj_bias"]
        dnb_shape = list(dnb_shape_t)
        dnb_dtype_int = _DTYPE_TO_INT.get(dnb_dt, 6)

    activation = "silu" if act_fn is not None else "swiglu"

    return {
        "gu_offset": gu_off,
        "gu_size": gu_sz,
        "gu_shape": list(gu_shape),
        "gu_dtype_int": _DTYPE_TO_INT.get(gu_dtype, 6),
        "gu_needs_transpose": gu_needs_t,
        "dn_offset": dn_off,
        "dn_size": dn_sz,
        "dn_shape": list(dn_shape),
        "dn_dtype_int": _DTYPE_TO_INT.get(dn_dtype, 6),
        "dn_needs_transpose": dn_needs_t,
        "has_bias": has_bias,
        "gub_offset": gub_off,
        "gub_size": gub_sz,
        "gub_shape": gub_shape,
        "gub_dtype_int": gub_dtype_int,
        "dnb_offset": dnb_off,
        "dnb_size": dnb_sz,
        "dnb_shape": dnb_shape,
        "dnb_dtype_int": dnb_dtype_int,
        "activation": activation,
    }
