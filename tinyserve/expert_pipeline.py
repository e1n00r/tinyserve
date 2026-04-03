"""Model-agnostic expert pipeline with template weight swapping.

Works with any nn.Module expert: swaps weights from the buffer into a
template module, calls forward(), accumulates weighted outputs.
"""

from contextlib import nullcontext

import torch
import torch.nn as nn

from .expert_store import ExpertBuffer, ExpertStore, ExpertCache
from .profiler import OffloadProfiler
from .ram_cache import madvise_willneed

try:
    from .csrc import get_expert_loop as _get_expert_loop
except Exception:
    _get_expert_loop = lambda: None  # noqa: E731

try:
    from tinyserve._fast_cache import classify_hits_misses as _cy_classify
except ImportError:
    _cy_classify = None

try:
    from tinyserve._fast_cache import group_tokens_by_expert as _cy_group
except ImportError:
    _cy_group = None


_TEMPLATE_STORAGE: dict[int, dict[str, torch.Tensor]] = {}


def _get_template_storage(template: nn.Module, layout) -> dict[str, torch.Tensor]:
    """Get or create dedicated storage tensors for the template.

    Ensures swap_weights_and_forward always writes into its own memory,
    never into cache slot views left behind by forward_from_packed.
    """
    tid = id(template)
    if tid not in _TEMPLATE_STORAGE:
        storage = {}
        for name, (shape, dtype) in layout.specs.items():
            parts = name.split(".")
            mod = template
            for part in parts[:-1]:
                mod = getattr(mod, part)
            param = getattr(mod, parts[-1])
            storage[name] = torch.empty(shape, dtype=dtype, device=param.device)
        _TEMPLATE_STORAGE[tid] = storage
    return _TEMPLATE_STORAGE[tid]


def swap_weights_and_forward(
    template: nn.Module,
    buf: ExpertBuffer,
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
    layout: "TensorLayout",  # noqa: F821
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
            gated = (up + 1) * gate * torch.sigmoid(gate * 1.702)
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

    from .offloaded_model import _mxfp4_linear

    if act_fn is not None:
        def _forward(packed, h):
            w_gu = packed[gu_off:gu_off + gu_sz].view(torch.uint8).view(gu_shape)
            s_gu = packed[gs_off:gs_off + gs_sz].view(torch.uint8).view(gs_shape)
            gate_up = _mxfp4_linear(h, w_gu, s_gu)
            gate, up = gate_up.chunk(2, dim=-1)
            gated = act_fn(gate) * up
            w_dn = packed[dn_off:dn_off + dn_sz].view(torch.uint8).view(dn_shape)
            s_dn = packed[ds_off:ds_off + ds_sz].view(torch.uint8).view(ds_shape)
            return _mxfp4_linear(gated, w_dn, s_dn)
    else:
        # GPT-OSS custom SwiGLU (interleaved, not chunked)
        def _forward(packed, h):
            w_gu = packed[gu_off:gu_off + gu_sz].view(torch.uint8).view(gu_shape)
            s_gu = packed[gs_off:gs_off + gs_sz].view(torch.uint8).view(gs_shape)
            gate_up = _mxfp4_linear(h, w_gu, s_gu)
            gate = gate_up[..., ::2].clamp(max=7.0)
            up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
            gated = (up + 1) * gate * torch.sigmoid(gate * 1.702)
            w_dn = packed[dn_off:dn_off + dn_sz].view(torch.uint8).view(dn_shape)
            s_dn = packed[ds_off:ds_off + ds_sz].view(torch.uint8).view(ds_shape)
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


class ExpertPipeline:
    """Double-buffered PCIe pipeline with LRU cache for any expert module."""

    def __init__(
        self,
        store: ExpertStore,
        template: nn.Module,
        device: torch.device,
        buf_a: ExpertBuffer,
        buf_b: ExpertBuffer,
        transfer_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        cache: ExpertCache | None = None,
        shared_stream: torch.cuda.Stream | None = None,
        ram_cache=None,
        cpu_expert=None,
    ):
        self.store = store
        self.template = template
        self.device = device

        self.buf_a = buf_a
        self.buf_b = buf_b

        self.transfer_stream = transfer_stream
        self.compute_stream = compute_stream

        self.cache = cache
        self.ram_cache = ram_cache
        self.cpu_expert = cpu_expert
        self.cpu_on_miss: bool = False
        self._buddy_tables: dict | None = None  # Per-layer tables from calibration
        self.cache_bias: float = 0.0  # logit bias magnitude for cache-aware routing (I4)
        self.profiler: OffloadProfiler | None = None
        # Precomputed param refs for zero-copy forward from cache slots.
        bf16_layout = store._bf16_layout if store._fp8 else store.layout
        self._param_refs = _precompute_param_refs(template, bf16_layout)
        self._act_fn = getattr(template, "_act_fn", None)
        # Inlined forward: baked-in offsets, no dict/getattr per call.
        # For MXFP4 layouts, use MXFP4 inline forward (direct dot_scaled_vecmat,
        # no param.data rebinding). GPU INT4 disabled on 8GB GPUs.
        self._inline_fwd = _build_inline_forward(bf16_layout, self._act_fn)
        if self._inline_fwd is None:
            self._inline_fwd = _build_mxfp4_inline_forward(bf16_layout, self._act_fn)
        if self._inline_fwd is None:
            self._inline_fwd = _build_gpu_int4_forward(bf16_layout, self._act_fn)
        self.shared_stream = shared_stream if shared_stream is not None else torch.cuda.Stream(device)
        self._prefetch_stream = torch.cuda.Stream(device)
        self._prefetch_events: dict[int, torch.cuda.Event] = {}  # slot -> H2D-complete event
        # Shared FP8 prefetch staging buffer — injected from outside (one per model,
        # not one per layer) since only one prefetch runs at a time during decode.
        # Set by OffloadedModel.from_module after construction.
        self._prefetch_fp8_stage: torch.Tensor | None = None
        self._expert_output_buf: torch.Tensor | None = None
        # C++ expert loop: precomputed layout args + module handle.
        self._cpp_layout_args = _build_cpp_layout_args(bf16_layout, self._act_fn)
        self._cpp_ext = _get_expert_loop() if self._cpp_layout_args is not None else None

    def execute_layer_experts(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._expert_output_buf is None or self._expert_output_buf.shape != hidden_states.shape:
            self._expert_output_buf = torch.zeros_like(hidden_states)
        else:
            self._expert_output_buf.zero_()
        for tok in range(hidden_states.shape[0]):
            self._execute_token_experts(
                hidden_states[tok : tok + 1],
                self._expert_output_buf,
                tok,
                layer_idx,
                expert_indices[tok],
                routing_weights[tok],
            )
        return self._expert_output_buf.clone()

    def execute_layer_experts_batched(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Batched expert dispatch for prefill: load each expert once, batch all tokens.

        Groups tokens by expert_id, loads each unique expert once, runs a single
        batched forward for all tokens routed to that expert, then scatters weighted
        results back. Reduces expert loads from O(seq_len * top_k) to O(num_unique_experts).
        """
        seq_len = hidden_states.shape[0]
        if seq_len == 0:
            return hidden_states.clone()

        output = torch.zeros_like(hidden_states)
        top_k = expert_indices.shape[1]

        eid_list = expert_indices.tolist()
        if _cy_group is not None:
            expert_groups = _cy_group(eid_list, seq_len, top_k)
        else:
            expert_groups: dict[int, list[tuple[int, int]]] = {}
            for tok in range(seq_len):
                for k in range(top_k):
                    eid = eid_list[tok][k]
                    if eid not in expert_groups:
                        expert_groups[eid] = []
                    expert_groups[eid].append((tok, k))

        cache = self.cache

        for eid, group in expert_groups.items():
            tok_indices = [g[0] for g in group]
            weight_indices = [g[1] for g in group]

            h_batch = hidden_states[tok_indices]

            out_batch = None
            if cache is not None:
                slot = cache.lookup(layer_idx, eid)
                if slot is not None:
                    packed = cache.get_packed(slot)
                    if self._inline_fwd is not None:
                        out_batch = self._inline_fwd(packed, h_batch)
                    else:
                        out_batch = forward_from_packed(
                            self.template, packed, self._param_refs, h_batch
                        )

            if out_batch is None:
                buf = self.buf_a
                self.store.copy_to_buffer(buf, layer_idx, eid, non_blocking=False)
                torch.cuda.synchronize()

                if self._inline_fwd is not None:
                    out_batch = self._inline_fwd(buf.packed, h_batch)
                else:
                    out_batch = swap_weights_and_forward(self.template, buf, h_batch)

                if cache is not None:
                    slot = cache.allocate(layer_idx, eid)
                    cache.get_packed(slot).copy_(buf.packed)

            for i, (tok_idx, k) in enumerate(zip(tok_indices, weight_indices)):
                output[tok_idx] += routing_weights[tok_idx, k] * out_batch[i]

        if cache is not None and hasattr(cache, "flush_slot_updates"):
            cache.flush_slot_updates()
        return output

    def _execute_token_experts(
        self,
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids: torch.Tensor | list[int],
        weights: torch.Tensor,
    ):
        if self.cache is None:
            if isinstance(expert_ids, torch.Tensor):
                expert_ids = expert_ids.tolist()
            self._pipeline_experts(h, output, tok_idx, layer_idx, expert_ids, weights, list(range(len(expert_ids))))
            _evt = torch.cuda.Event()
            _evt.record(self.compute_stream)
            torch.cuda.current_stream().wait_event(_evt)
            return

        cache = self.cache
        _prof = self.profiler
        _inline = self._inline_fwd

        # GPU slot map lookup — no CUDA sync.
        if isinstance(expert_ids, torch.Tensor) and hasattr(cache, "lookup_slots"):
            with _prof.phase("cache_lookup") if _prof else nullcontext():
                slots = cache.lookup_slots(layer_idx, expert_ids)
                # ONE .tolist() for both slots and expert_ids — single CUDA sync.
                slots_list = slots.tolist()

            hits = []
            misses = []
            expert_ids_list = expert_ids.tolist()  # piggybacks on same sync
            if _cy_classify is not None:
                hits, misses = _cy_classify(expert_ids_list, slots_list)
                for i, slot in hits:
                    cache._policy.lookup((layer_idx, expert_ids_list[i]))
                    cache.hits += 1
                    cache._layer_hits[layer_idx] = cache._layer_hits.get(layer_idx, 0) + 1
                    cache._expert_access_count[(layer_idx, expert_ids_list[i])] = cache._expert_access_count.get((layer_idx, expert_ids_list[i]), 0) + 1
                for i in misses:
                    cache.misses += 1
                    cache._layer_misses[layer_idx] = cache._layer_misses.get(layer_idx, 0) + 1
                    cache._expert_access_count[(layer_idx, expert_ids_list[i])] = cache._expert_access_count.get((layer_idx, expert_ids_list[i]), 0) + 1
            else:
                for i, (eid, slot) in enumerate(zip(expert_ids_list, slots_list)):
                    if slot >= 0:
                        hits.append((i, slot))
                        # Update policy recency (Python-only, no GPU sync).
                        cache._policy.lookup((layer_idx, eid))
                        cache.hits += 1
                        cache._layer_hits[layer_idx] = cache._layer_hits.get(layer_idx, 0) + 1
                        cache._expert_access_count[(layer_idx, eid)] = cache._expert_access_count.get((layer_idx, eid), 0) + 1
                    else:
                        misses.append(i)
                        cache.misses += 1
                        cache._layer_misses[layer_idx] = cache._layer_misses.get(layer_idx, 0) + 1
                        cache._expert_access_count[(layer_idx, eid)] = cache._expert_access_count.get((layer_idx, eid), 0) + 1
        else:
            # Fallback: original Python path for list inputs or old-style cache.
            if isinstance(expert_ids, torch.Tensor):
                expert_ids = expert_ids.tolist()
            expert_ids_list = expert_ids
            hits = []
            misses = []
            with _prof.phase("cache_lookup") if _prof else nullcontext():
                for i, eid in enumerate(expert_ids_list):
                    slot = cache.lookup(layer_idx, eid)
                    if slot is not None:
                        hits.append((i, slot))
                    else:
                        misses.append(i)

        # C++ fast path: all hits, no pending prefetch events, extension loaded.
        _cpp = self._cpp_ext
        if (
            _cpp is not None
            and not misses
            and hits
            and not any(s in self._prefetch_events for _, s in hits)
        ):
            _args = self._cpp_layout_args
            slots_tensor = torch.tensor(
                [s for _, s in hits], dtype=torch.int32, device=h.device
            )
            weights_tensor = torch.tensor(
                [weights[i].item() for i, _ in hits], dtype=torch.float32, device=h.device
            )
            out = _cpp.fast_expert_forward(
                h,
                slots_tensor,
                weights_tensor,
                cache._packed,
                _args["gu_offset"],
                _args["gu_size"],
                _args["gu_shape"],
                _args["gu_dtype_int"],
                _args["gu_needs_transpose"],
                _args["dn_offset"],
                _args["dn_size"],
                _args["dn_shape"],
                _args["dn_dtype_int"],
                _args["dn_needs_transpose"],
                _args["has_bias"],
                _args["gub_offset"],
                _args["gub_size"],
                _args["gub_shape"],
                _args["gub_dtype_int"],
                _args["dnb_offset"],
                _args["dnb_size"],
                _args["dnb_shape"],
                _args["dnb_dtype_int"],
                _args["activation"],
            )
            output[tok_idx] += out.squeeze(0)
            if cache is not None and hasattr(cache, "flush_slot_updates"):
                cache.flush_slot_updates()
            return

        for i, slot in hits:
            with _prof.phase("hit_compute") if _prof else nullcontext():
                if slot in self._prefetch_events:
                    torch.cuda.current_stream().wait_event(self._prefetch_events.pop(slot))
                packed = cache.get_packed(slot)
                out = None
                if _inline is not None:
                    out = _inline(packed, h)
                if out is None:
                    out = forward_from_packed(self.template, packed, self._param_refs, h)
                output[tok_idx] += weights[i] * out.squeeze(0)

        if not misses:
            if cache is not None and hasattr(cache, "flush_slot_updates"):
                cache.flush_slot_updates()
            return

        # Fiddler (arxiv 2402.07033): at batch=1, route ALL misses through CPU compute.
        # Sending activations (~KB) to CPU is faster than H2D weight transfer (~MB).
        # cpu_on_miss=True enables this path; False preserves original GPU pipeline.
        if self.cpu_on_miss and self.cpu_expert is not None and h.shape[0] == 1:
            for i in misses:
                eid = expert_ids_list[i]

                # BuddyMoE: try cached substitute first (zero stall).
                buddy_tbl = (self._buddy_tables or {}).get(layer_idx)
                if buddy_tbl is not None and cache is not None:
                    if cache._slot_map_cpu is not None and layer_idx < cache._slot_map_cpu.shape[0]:
                        cached_experts = set(int(e) for e in range(cache._slot_map_cpu.shape[1])
                                            if cache._slot_map_cpu[layer_idx, e] >= 0)
                    else:
                        cached_experts = set()
                    buddy_eid = buddy_tbl.find_cached_buddy(eid, cached_experts)
                    if buddy_eid is not None:
                        buddy_slot = cache.lookup(layer_idx, buddy_eid)
                        if buddy_slot is not None:
                            packed = cache.get_packed(buddy_slot)
                            out = None
                            if _inline is not None:
                                out = _inline(packed, h)
                            if out is None:
                                out = forward_from_packed(self.template, packed, self._param_refs, h)
                            output[tok_idx] += weights[i] * out.squeeze(0)
                            continue

                # No buddy available — fall back to CPU compute.
                expert_packed = self.store.get_expert_data(layer_idx, eid)
                with _prof.phase("cpu_compute") if _prof else nullcontext():
                    out = self.cpu_expert.forward(h, expert_packed)
                output[tok_idx] += weights[i] * out.squeeze(0)
                if cache is not None:
                    gpu_slot = cache.allocate(layer_idx, eid)
                    cache.get_packed(gpu_slot).copy_(expert_packed[:cache.expert_bytes], non_blocking=True)
            if cache is not None and hasattr(cache, "flush_slot_updates"):
                cache.flush_slot_updates()
            return

        # Split misses: RAM-cached experts go to GPU pipeline (fast H2D from pinned),
        # truly cold experts (not in RAM) use CPU compute to avoid mmap page fault stall.
        if self.cpu_expert is not None and self.ram_cache is not None and h.shape[0] == 1:
            ram = self.ram_cache
            cold_misses = []
            for i in misses:
                eid = expert_ids_list[i]
                ram.wait_pending(layer_idx, eid)
                slot = ram.lookup(layer_idx, eid)
                if slot is not None:
                    # Expert is in pinned RAM — GPU pipeline is faster (async H2D + overlap)
                    cold_misses.append(i)  # let _pipeline_experts handle it via store._data
                else:
                    # Truly cold: not in RAM. Load via pread (or mmap fallback)
                    # then CPU compute from cached pinned data.
                    ram_slot = ram.load_sync(layer_idx, eid, self.store._data[layer_idx, eid])
                    expert_data = ram.get_slot_data(ram_slot)
                    out = self.cpu_expert.forward(h, expert_data)
                    output[tok_idx] += weights[i] * out.squeeze(0)
                    # Populate GPU cache too
                    if cache is not None:
                        gpu_slot = cache.allocate(layer_idx, eid)
                        cache.get_packed(gpu_slot).copy_(ram.get_slot_data(ram.lookup(layer_idx, eid)), non_blocking=True)
            if not cold_misses:
                if cache is not None and hasattr(cache, "flush_slot_updates"):
                    cache.flush_slot_updates()
                return
            misses = cold_misses

        self._pipeline_experts(h, output, tok_idx, layer_idx, expert_ids_list, weights, misses)
        _evt = torch.cuda.Event()
        _evt.record(self.compute_stream)
        torch.cuda.current_stream().wait_event(_evt)
        if cache is not None and hasattr(cache, "flush_slot_updates"):
            cache.flush_slot_updates()

    def _pipeline_experts(
        self,
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids: list[int],
        weights: torch.Tensor,
        indices: list[int],
    ):
        bufs = [self.buf_a, self.buf_b]
        cache = self.cache
        buf_done: list[torch.cuda.Event | None] = [None, None]
        _prof = self.profiler

        def _load_expert(buf, li, eid):
            """Load expert into GPU buffer — from RAMCache (pinned, fast) or store (mmap).

            If background fill is running and expert is not yet in RAM,
            wait briefly for the fill thread to load it (avoids mmap page faults).
            """
            if self.ram_cache is not None:
                ram = self.ram_cache
                slot = ram.lookup(li, eid)
                if slot is not None:
                    # Pinned RAM -> GPU at full PCIe bandwidth
                    buf.packed.copy_(ram.get_slot_data(slot), non_blocking=True)
                    return
                # Background fill running? Wait briefly — expert may arrive soon.
                if not ram.fill_complete:
                    ram.wait_for_fill(timeout=0.05)
                    slot = ram.lookup(li, eid)
                    if slot is not None:
                        buf.packed.copy_(ram.get_slot_data(slot), non_blocking=True)
                        return
            self.store.copy_to_buffer(buf, li, eid, non_blocking=True)

        load_done = torch.cuda.Event()
        with _prof.phase("h2d_transfer") if _prof else nullcontext(), torch.cuda.stream(self.transfer_stream):
            _load_expert(bufs[0], layer_idx, expert_ids[indices[0]])
            load_done.record(self.transfer_stream)

        for mi in range(len(indices)):
            buf_idx = mi & 1
            cur_buf = bufs[buf_idx]
            idx = indices[mi]
            eid = expert_ids[idx]

            self.compute_stream.wait_event(load_done)

            if mi < len(indices) - 1:
                next_buf_idx = 1 - buf_idx
                load_done = torch.cuda.Event()
                with _prof.phase("h2d_transfer") if _prof else nullcontext():
                    with torch.cuda.stream(self.transfer_stream):
                        if buf_done[next_buf_idx] is not None:
                            self.transfer_stream.wait_event(buf_done[next_buf_idx])
                        _load_expert(
                            bufs[next_buf_idx],
                            layer_idx,
                            expert_ids[indices[mi + 1]],
                        )
                        load_done.record(self.transfer_stream)

            with _prof.phase("gpu_compute") if _prof else nullcontext():
                with torch.cuda.stream(self.compute_stream):
                    out = swap_weights_and_forward(self.template, cur_buf, h)
                    output[tok_idx] += weights[idx] * out.squeeze(0)

            with torch.cuda.stream(self.compute_stream):
                if cache is not None:
                    with _prof.phase("cache_store") if _prof else nullcontext():
                        slot = cache.allocate(layer_idx, eid)
                        cache.get_packed(slot).copy_(cur_buf.packed)

                buf_done[buf_idx] = torch.cuda.Event()
                buf_done[buf_idx].record(self.compute_stream)

    def execute_batched_experts(
        self,
        items: "list[BatchItem]",
        layer_idx: int,
    ) -> "list[torch.Tensor]":
        """Execute expert forwards for multiple requests with expert-level batching.

        Delegates to ExpertBatcher for grouping by expert_id, loading once,
        batching hidden states, and scattering weighted results.
        """
        from .expert_batcher import ExpertBatcher

        batcher = ExpertBatcher(self)
        return batcher.batch_execute(items, layer_idx)

    def schedule_prefetch(self, layer_idx: int, expert_ids: "list[int] | torch.Tensor") -> None:
        """Pre-load predicted experts into the VRAM cache for the next token.

        Uses temporal locality: the same experts are likely active next token.
        Runs on a dedicated prefetch stream — fully overlapped with other layers'
        attention + expert compute between now and the next call to this layer.

        FP8 path: H2D raw FP8 bytes → _prefetch_fp8_stage (half-size, fast),
        then GPU dequant → cache slot. No CPU blocking.
        BF16 path: direct pinned-CPU → VRAM-cache DMA.
        """
        if self.cache is None:
            return
        # Convert tensor to list once — this sync is overlapped with attention compute.
        if isinstance(expert_ids, torch.Tensor):
            expert_ids = expert_ids.tolist()
        # RAM cache prefetch: async load from mmap into pinned RAM.
        # Prime page cache with madvise(WILLNEED) before the threadpool copy.
        if self.ram_cache is not None and getattr(self.store, "_disk_offload", False):
            has_fast_reader = self.ram_cache._fast_reader is not None
            for eid in expert_ids:
                if not self.ram_cache.contains(layer_idx, eid):
                    if has_fast_reader:
                        self.ram_cache.prefetch_async(layer_idx, eid)
                    else:
                        madvise_willneed(self.store._data[layer_idx, eid])
                        self.ram_cache.prefetch_async(layer_idx, eid, self.store._data[layer_idx, eid])
        elif self.ram_cache is not None:
            for eid in expert_ids:
                if not self.ram_cache.contains(layer_idx, eid):
                    self.ram_cache.prefetch_async(layer_idx, eid, self.store._data[layer_idx, eid])
        self._prefetch_events.clear()
        with torch.cuda.stream(self._prefetch_stream):
            for eid in expert_ids:
                if self.cache.contains(layer_idx, eid):
                    continue
                slot = self.cache.allocate(layer_idx, eid)
                cache_slot = self.cache.get_packed(slot)
                if self.store._fp8:
                    # Step 1: H2D raw FP8 bytes (half the BF16 size).
                    self._prefetch_fp8_stage.copy_(self.store._data[layer_idx, eid], non_blocking=True)
                    # Step 2: GPU dequant per tensor → cache slot (BF16).
                    for name, (fp8_shape, fp8_dtype) in self.store.layout.specs.items():
                        fp8_off = self.store.layout.offsets[name]
                        fp8_sz = self.store.layout.sizes[name]
                        bf16_off = self.store._bf16_layout.offsets[name]
                        bf16_sz = self.store._bf16_layout.sizes[name]
                        bf16_shape, bf16_dtype = self.store._bf16_layout.specs[name]
                        src = self._prefetch_fp8_stage[fp8_off : fp8_off + fp8_sz].view(fp8_dtype).view(fp8_shape)
                        dst = cache_slot[bf16_off : bf16_off + bf16_sz].view(bf16_dtype).view(bf16_shape)
                        if fp8_dtype.is_floating_point:
                            dst.copy_(src.to(bf16_dtype))
                        else:
                            dst.copy_(src.view(torch.uint8).view(bf16_shape))
                else:
                    cache_slot.copy_(self.store._data[layer_idx, eid], non_blocking=True)
                evt = torch.cuda.Event()
                evt.record(self._prefetch_stream)
                self._prefetch_events[slot] = evt
