"""Model-agnostic expert pipeline with template weight swapping.

Works with any nn.Module expert: swaps weights from the buffer into a
template module, calls forward(), accumulates weighted outputs.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .expert_cache import ExpertCache
from .expert_forward import (
    _build_cpp_layout_args,
    _build_gpu_int4_forward,
    _build_inline_forward,
    _build_mxfp4_inline_forward,
    _get_expert_loop,
    _precompute_param_refs,
    forward_from_packed,
    swap_weights_and_forward,
)
from .expert_store import ExpertBuffer, ExpertStore
from .profiler import OffloadProfiler
from .ram_cache import madvise_willneed

if TYPE_CHECKING:
    from .expert_batcher import BatchItem

logger = logging.getLogger(__name__)

try:
    from tinyserve._fast_cache import classify_hits_misses as _cython_classify_hits
except ImportError:
    logger.debug("Cython classify_hits_misses not available, using Python fallback")
    _cython_classify_hits = None

try:
    from tinyserve._fast_cache import group_tokens_by_expert as _cython_group_by_expert
except ImportError:
    logger.debug("Cython group_tokens_by_expert not available, using Python fallback")
    _cython_group_by_expert = None

try:
    from tinyserve._fast_cache import forward_cache_hits as _cython_forward_hits
except ImportError:
    logger.debug("Cython forward_cache_hits not available, using Python fallback")
    _cython_forward_hits = None


class ExpertPipeline:
    """Double-buffered PCIe pipeline with LRU cache for any expert module."""

    def __init__(
        self,
        store: ExpertStore,
        template: nn.Module,
        device: torch.device,
        staging_buffer_a: ExpertBuffer,
        staging_buffer_b: ExpertBuffer,
        transfer_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        cache: ExpertCache | None = None,
        shared_stream: torch.cuda.Stream | None = None,
        ram_cache=None,
        cpu_expert=None,
        max_top_k: int = 8,
    ):
        self.store = store
        self.template = template
        self.device = device

        self.staging_buffer_a = staging_buffer_a
        self.staging_buffer_b = staging_buffer_b

        self.transfer_stream = transfer_stream
        self.compute_stream = compute_stream

        self.cache = cache
        self.ram_cache = ram_cache
        self.cpu_expert = cpu_expert
        self.cpu_on_miss: bool = False
        self._buddy_tables: dict | None = None  # Per-layer tables from calibration
        self.cache_bias: float = 0.0  # logit bias magnitude for cache-aware routing (I4)
        self.profiler: OffloadProfiler | None = None
        self._expert_output_buf: torch.Tensor | None = None

        # Native quant detection: MmapExpertStore / FusedMmapExpertStore have ggml_types
        self._native_quant = hasattr(store, "ggml_types")
        if self._native_quant:
            from .ggml_forward import GGMLExpertForward

            self._act_fn = nn.SiLU()
            self._nq_forward = GGMLExpertForward(
                store.layout,
                store.ggml_types,
                self._act_fn,
                getattr(store, "proj_shapes", {}),
            )
            # Mixed quant: per-layer type overrides
            if hasattr(store, "_layer_ggml_types"):
                self._nq_forward._layer_ggml_types = store._layer_ggml_types
            self._param_refs = None
            self._inline_fwd = None
            self._cpp_layout_args = None
            self._cpp_ext = None
        else:
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
            # C++ expert loop: precomputed layout args + module handle.
            self._cpp_layout_args = _build_cpp_layout_args(bf16_layout, self._act_fn)
            self._cpp_ext = _get_expert_loop() if self._cpp_layout_args is not None else None

        self.shared_stream = shared_stream if shared_stream is not None else torch.cuda.Stream(device)
        self._prefetch_stream = torch.cuda.Stream(device)
        self._prefetch_events: dict[int, torch.cuda.Event] = {}  # slot -> H2D-complete event
        # Shared FP8 prefetch staging buffer — injected from outside (one per model,
        # not one per layer) since only one prefetch runs at a time during decode.
        # Set by OffloadedModel.from_module after construction.
        self._prefetch_fp8_stage: torch.Tensor | None = None
        # Pre-allocated decode hot-path buffers: avoids torch.tensor() per token.
        # max_top_k covers top_k and fate_top_k (top_k+1). Sized once; sliced per call.
        self._slots_buf = torch.empty(max_top_k, dtype=torch.int32, device=device)
        self._weights_buf = torch.empty(max_top_k, dtype=torch.float32, device=device)

    def execute_layer_experts(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._native_quant:
            return self._execute_layer_experts_native(
                hidden_states, layer_idx, expert_indices, routing_weights,
            )
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
        if self._native_quant:
            return self._execute_layer_experts_batched_native(
                hidden_states, layer_idx, expert_indices, routing_weights,
            )
        seq_len = hidden_states.shape[0]
        if seq_len == 0:
            return hidden_states.clone()

        output = torch.zeros_like(hidden_states)
        top_k = expert_indices.shape[1]

        eid_list = expert_indices.tolist()
        if _cython_group_by_expert is not None:
            expert_groups = _cython_group_by_expert(eid_list, seq_len, top_k)
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
                        out_batch = forward_from_packed(self.template, packed, self._param_refs, h_batch)

            if out_batch is None:
                buf = self.staging_buffer_a
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

    def _execute_layer_experts_native(
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

        output = self._expert_output_buf
        cache = self.cache

        for tok in range(hidden_states.shape[0]):
            h = hidden_states[tok : tok + 1]
            eids = expert_indices[tok]
            weights = routing_weights[tok]

            if isinstance(eids, torch.Tensor):
                eids_list = eids.tolist()
            else:
                eids_list = list(eids)

            for k, eid in enumerate(eids_list):
                slot = cache.lookup(layer_idx, eid) if cache else None

                if slot is not None:
                    packed = cache.get_packed(slot)
                    out = self._nq_forward.forward(packed, h, layer_idx=layer_idx)
                else:
                    buf = self.staging_buffer_a
                    self.store.copy_to_buffer(buf, layer_idx, eid, non_blocking=False)
                    torch.cuda.synchronize()
                    out = self._nq_forward.forward(buf.packed, h)
                    if cache is not None:
                        slot = cache.allocate(layer_idx, eid)
                        cache.get_packed(slot).copy_(buf.packed)

                output[tok] += weights[k] * out.squeeze(0)

        if cache is not None and hasattr(cache, "flush_slot_updates"):
            cache.flush_slot_updates()
        return output.clone()

    def _execute_layer_experts_batched_native(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        if seq_len == 0:
            return hidden_states.clone()

        output = torch.zeros_like(hidden_states)
        top_k = expert_indices.shape[1]
        cache = self.cache

        eid_list = expert_indices.tolist()
        expert_groups: dict[int, list[tuple[int, int]]] = {}
        for tok in range(seq_len):
            for k in range(top_k):
                eid = eid_list[tok][k]
                if eid not in expert_groups:
                    expert_groups[eid] = []
                expert_groups[eid].append((tok, k))

        for eid, group in expert_groups.items():
            tok_indices = [g[0] for g in group]
            weight_indices = [g[1] for g in group]
            h_batch = hidden_states[tok_indices]

            out_batch = None
            if cache is not None:
                slot = cache.lookup(layer_idx, eid)
                if slot is not None:
                    packed = cache.get_packed(slot)
                    out_batch = self._nq_forward.forward(packed, h_batch, layer_idx=layer_idx)

            if out_batch is None:
                buf = self.staging_buffer_a
                self.store.copy_to_buffer(buf, layer_idx, eid, non_blocking=False)
                torch.cuda.synchronize()
                out_batch = self._nq_forward.forward(buf.packed, h_batch, layer_idx=layer_idx)
                if cache is not None:
                    slot = cache.allocate(layer_idx, eid)
                    cache.get_packed(slot).copy_(buf.packed)

            for i, (tok_idx, k) in enumerate(zip(tok_indices, weight_indices)):
                output[tok_idx] += routing_weights[tok_idx, k] * out_batch[i]

        if cache is not None and hasattr(cache, "flush_slot_updates"):
            cache.flush_slot_updates()
        return output

    def _classify_hits_misses(
        self,
        cache: ExpertCache,
        layer_idx: int,
        expert_ids: torch.Tensor | list[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Classify expert_ids into cache hits and misses.

        Returns:
            (hits, misses, expert_ids_list)
            hits: list of (position_idx, cache_slot)
            misses: list of position indices that missed
            expert_ids_list: expert IDs as a Python list
        """
        _prof = self.profiler

        if isinstance(expert_ids, torch.Tensor) and hasattr(cache, "lookup_slots"):
            with _prof.phase("cache_lookup") if _prof else nullcontext():
                slots = cache.lookup_slots(layer_idx, expert_ids)
                # Consolidate two .tolist() calls (= two CUDA syncs) into one:
                # stack expert_ids + slots into a single [2, top_k] tensor and
                # call .tolist() once.  Unpack the two rows on the CPU side.
                _eids_int = expert_ids.int().to(slots.device)
                _both = torch.stack([_eids_int, slots], dim=0).tolist()
                expert_ids_list: list[int] = _both[0]
                slots_list: list[int] = _both[1]

            hits: list[tuple[int, int]] = []
            misses: list[int] = []
            if _cython_classify_hits is not None:
                hits, misses = _cython_classify_hits(expert_ids_list, slots_list)
                for i, slot in hits:
                    cache._policy.lookup((layer_idx, expert_ids_list[i]))
                    cache.hits += 1
                    cache._layer_hits[layer_idx] = cache._layer_hits.get(layer_idx, 0) + 1
                    cache._expert_access_count[(layer_idx, expert_ids_list[i])] = (
                        cache._expert_access_count.get((layer_idx, expert_ids_list[i]), 0) + 1
                    )
                for i in misses:
                    cache.misses += 1
                    cache._layer_misses[layer_idx] = cache._layer_misses.get(layer_idx, 0) + 1
                    cache._expert_access_count[(layer_idx, expert_ids_list[i])] = (
                        cache._expert_access_count.get((layer_idx, expert_ids_list[i]), 0) + 1
                    )
            else:
                for i, (eid, slot) in enumerate(zip(expert_ids_list, slots_list)):
                    if slot >= 0:
                        hits.append((i, slot))
                        cache._policy.lookup((layer_idx, eid))
                        cache.hits += 1
                        cache._layer_hits[layer_idx] = cache._layer_hits.get(layer_idx, 0) + 1
                        cache._expert_access_count[(layer_idx, eid)] = (
                            cache._expert_access_count.get((layer_idx, eid), 0) + 1
                        )
                    else:
                        misses.append(i)
                        cache.misses += 1
                        cache._layer_misses[layer_idx] = cache._layer_misses.get(layer_idx, 0) + 1
                        cache._expert_access_count[(layer_idx, eid)] = (
                            cache._expert_access_count.get((layer_idx, eid), 0) + 1
                        )
        else:
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

        return hits, misses, expert_ids_list

    def _forward_cache_hits(
        self,
        hits: list[tuple[int, int]],
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        weights: torch.Tensor,
        cache: ExpertCache,
    ) -> None:
        """Process all cache hits via C++, Cython, or Python fallback."""
        _prof = self.profiler
        _inline = self._inline_fwd
        _cpp = self._cpp_ext

        if _cpp is not None and hits and not any(s in self._prefetch_events for _, s in hits):
            _args = self._cpp_layout_args
            n_hits = len(hits)
            for _j, (_i, _s) in enumerate(hits):
                self._slots_buf[_j] = _s
                self._weights_buf[_j] = weights[_i]
            slots_tensor = self._slots_buf[:n_hits]
            weights_tensor = self._weights_buf[:n_hits]
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
            return

        if _cython_forward_hits is not None and not (_prof and _prof.enabled):

            def _fallback(p, h_):
                return forward_from_packed(self.template, p, self._param_refs, h_)

            _cython_forward_hits(
                hits,
                h,
                output,
                tok_idx,
                weights,
                cache._packed,
                _inline,
                _fallback,
                self._prefetch_events,
                torch.cuda.current_stream().wait_event,
            )
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

    def _handle_miss_fallback(
        self,
        misses: list[int],
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids_list: list[int],
        weights: torch.Tensor,
        cache: ExpertCache,
    ):
        """Fiddler CPU fallback: buddy substitution then CPU compute for each miss."""
        _prof = self.profiler
        _inline = self._inline_fwd

        for i in misses:
            eid = expert_ids_list[i]

            buddy_tbl = (self._buddy_tables or {}).get(layer_idx)
            if buddy_tbl is not None:
                if cache._slot_map_cpu is not None and layer_idx < cache._slot_map_cpu.shape[0]:
                    cached_experts = set(
                        int(e) for e in range(cache._slot_map_cpu.shape[1]) if cache._slot_map_cpu[layer_idx, e] >= 0
                    )
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

            expert_packed = self.store.get_expert_data(layer_idx, eid)
            with _prof.phase("cpu_compute") if _prof else nullcontext():
                out = self.cpu_expert.forward(h, expert_packed)
            output[tok_idx] += weights[i] * out.squeeze(0)
            gpu_slot = cache.allocate(layer_idx, eid)
            cache.get_packed(gpu_slot).copy_(expert_packed[: cache.expert_bytes], non_blocking=True)

    def _handle_miss_gpu_pipeline(
        self,
        misses: list[int],
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids_list: list[int],
        weights: torch.Tensor,
        cache: ExpertCache,
    ):
        """Double-buffered H2D pipeline for misses, with optional RAM-split routing."""
        if self.cpu_expert is not None and self.ram_cache is not None and h.shape[0] == 1:
            ram = self.ram_cache
            cold_misses = []
            for i in misses:
                eid = expert_ids_list[i]
                ram.wait_pending(layer_idx, eid)
                slot = ram.lookup(layer_idx, eid)
                if slot is not None:
                    cold_misses.append(i)
                else:
                    ram_slot = ram.load_sync(layer_idx, eid, self.store._data[layer_idx, eid])
                    expert_data = ram.get_slot_data(ram_slot)
                    out = self.cpu_expert.forward(h, expert_data)
                    output[tok_idx] += weights[i] * out.squeeze(0)
                    if cache is not None:
                        gpu_slot = cache.allocate(layer_idx, eid)
                        cache.get_packed(gpu_slot).copy_(
                            ram.get_slot_data(ram.lookup(layer_idx, eid)), non_blocking=True
                        )
            if not cold_misses:
                return
            misses = cold_misses

        self._pipeline_experts(h, output, tok_idx, layer_idx, expert_ids_list, weights, misses)
        _evt = torch.cuda.Event()
        _evt.record(self.compute_stream)
        torch.cuda.current_stream().wait_event(_evt)

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
        hits, misses, expert_ids_list = self._classify_hits_misses(cache, layer_idx, expert_ids)

        self._forward_cache_hits(hits, h, output, tok_idx, weights, cache)
        if not misses:
            if hasattr(cache, "flush_slot_updates"):
                cache.flush_slot_updates()
            return

        # Sync: hit accumulation ran on default stream, miss pipeline uses
        # compute_stream. Both write output[tok_idx] — must serialize.
        if hits:
            _hit_done = torch.cuda.Event()
            _hit_done.record()
            self.compute_stream.wait_event(_hit_done)

        if self.cpu_on_miss and self.cpu_expert is not None and h.shape[0] == 1:
            self._handle_miss_fallback(misses, h, output, tok_idx, layer_idx, expert_ids_list, weights, cache)
            if hasattr(cache, "flush_slot_updates"):
                cache.flush_slot_updates()
            return

        self._handle_miss_gpu_pipeline(misses, h, output, tok_idx, layer_idx, expert_ids_list, weights, cache)
        if hasattr(cache, "flush_slot_updates"):
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
        bufs = [self.staging_buffer_a, self.staging_buffer_b]
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
        items: list[BatchItem],
        layer_idx: int,
    ) -> list[torch.Tensor]:
        """Execute expert forwards for multiple requests with expert-level batching.

        Delegates to ExpertBatcher for grouping by expert_id, loading once,
        batching hidden states, and scattering weighted results.
        """
        from .expert_batcher import ExpertBatcher

        batcher = ExpertBatcher(self)
        return batcher.batch_execute(items, layer_idx)

    def schedule_prefetch(self, layer_idx: int, expert_ids: list[int] | torch.Tensor) -> None:
        """Pre-load predicted experts into the VRAM cache for the next token.

        Uses temporal locality: the same experts are likely active next token.
        Runs on a dedicated prefetch stream — fully overlapped with other layers'
        attention + expert compute between now and the next call to this layer.

        FP8 path: H2D raw FP8 bytes → _prefetch_fp8_stage (half-size, fast),
        then GPU dequant → cache slot. No CPU blocking.
        BF16 path: direct pinned-CPU → VRAM-cache DMA.
        """
        if self.cache is None or self._native_quant:
            return  # native-quant path handles its own cache fill in _execute_layer_experts_native
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
