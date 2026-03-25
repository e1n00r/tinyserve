"""Model-agnostic expert pipeline with template weight swapping.

Works with any nn.Module expert: swaps weights from the buffer into a
template module, calls forward(), accumulates weighted outputs.
"""

from contextlib import nullcontext

import torch
import torch.nn as nn

from .generic_store import GenericExpertBuffer, GenericExpertStore, GenericLRUCache
from .profiler import OffloadProfiler
from .ram_cache import madvise_willneed


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
    buf: GenericExpertBuffer,
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
    with torch.no_grad():
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

    # MXFP4 uses blocks+scales (uint8), not standard weight matrices.
    # Fall back to template forward which handles _mxfp4_linear.
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
    _intermediate_dim = hidden_dim  # for gpt-oss: hidden==intermediate, but check from gate_up  # noqa: F841
    # Actually intermediate = max(gu_shape) // 2 for interleaved, or just check dn_shape
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


class GenericExpertPipeline:
    """Double-buffered PCIe pipeline with LRU cache for any expert module."""

    def __init__(
        self,
        store: GenericExpertStore,
        template: nn.Module,
        device: torch.device,
        buf_a: GenericExpertBuffer,
        buf_b: GenericExpertBuffer,
        transfer_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        cache: GenericLRUCache | None = None,
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
        self.cache_bias: float = 0.0  # logit bias magnitude for cache-aware routing (I4)
        self.profiler: OffloadProfiler | None = None
        # Precomputed param refs for zero-copy forward from cache slots.
        bf16_layout = store._bf16_layout if store._fp8 else store.layout
        self._param_refs = _precompute_param_refs(template, bf16_layout)
        self._act_fn = getattr(template, "_act_fn", None)
        # Inlined forward: baked-in offsets, no dict/getattr per call.
        self._inline_fwd = _build_inline_forward(bf16_layout, self._act_fn)
        self.shared_stream = shared_stream if shared_stream is not None else torch.cuda.Stream(device)
        self._prefetch_stream = torch.cuda.Stream(device)
        self._prefetch_events: dict[int, torch.cuda.Event] = {}  # slot -> H2D-complete event
        # Shared FP8 prefetch staging buffer — injected from outside (one per model,
        # not one per layer) since only one prefetch runs at a time during decode.
        # Set by OffloadedModel.from_module after construction.
        self._prefetch_fp8_stage: torch.Tensor | None = None

    def execute_layer_experts(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_output = torch.zeros_like(hidden_states)
        for tok in range(hidden_states.shape[0]):
            self._execute_token_experts(
                hidden_states[tok : tok + 1],
                expert_output,
                tok,
                layer_idx,
                expert_indices[tok],
                routing_weights[tok],
            )
        return expert_output

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
            for i, (eid, slot) in enumerate(zip(expert_ids_list, slots_list)):
                if slot >= 0:
                    hits.append((i, slot))
                    # Update policy recency (Python-only, no GPU sync).
                    cache._policy.lookup((layer_idx, eid))
                    cache.hits += 1
                else:
                    misses.append(i)
                    cache.misses += 1
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

        for i, slot in hits:
            with _prof.phase("hit_compute") if _prof else nullcontext():
                if slot in self._prefetch_events:
                    torch.cuda.current_stream().wait_event(self._prefetch_events.pop(slot))
                packed = cache.get_packed(slot)
                if _inline is not None:
                    out = _inline(packed, h)
                else:
                    out = forward_from_packed(self.template, packed, self._param_refs, h)
                output[tok_idx] += weights[i] * out.squeeze(0)

        if not misses:
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
                    # Truly cold: not in RAM, would page-fault from mmap.
                    # CPU compute (1.9ms) beats mmap page fault (20ms) + GPU transfer.
                    expert_data = self.store._data[layer_idx, eid]
                    out = self.cpu_expert.forward(h, expert_data)
                    output[tok_idx] += weights[i] * out.squeeze(0)
                    # Load into RAM cache for next time
                    ram.load_sync(layer_idx, eid, expert_data)
                    # Populate GPU cache too
                    if cache is not None:
                        gpu_slot = cache.allocate(layer_idx, eid)
                        cache.get_packed(gpu_slot).copy_(ram.get_slot_data(ram.lookup(layer_idx, eid)), non_blocking=True)
            if not cold_misses:
                return
            misses = cold_misses

        self._pipeline_experts(h, output, tok_idx, layer_idx, expert_ids_list, weights, misses)
        _evt = torch.cuda.Event()
        _evt.record(self.compute_stream)
        torch.cuda.current_stream().wait_event(_evt)

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
            """Load expert into GPU buffer — from RAMCache (pinned, fast) or store (mmap)."""
            if self.ram_cache is not None:
                slot = self.ram_cache.lookup(li, eid)
                if slot is not None:
                    # Pinned RAM → GPU at full PCIe bandwidth
                    buf.packed.copy_(self.ram_cache.get_slot_data(slot), non_blocking=True)
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
            for eid in expert_ids:
                if not self.ram_cache.contains(layer_idx, eid):
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
