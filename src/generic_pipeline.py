"""Model-agnostic expert pipeline with template weight swapping.

Works with any nn.Module expert: swaps weights from the buffer into a
template module, calls forward(), accumulates weighted outputs.
"""

import torch
import torch.nn as nn

from .generic_store import GenericExpertBuffer, GenericExpertStore, GenericLRUCache


def swap_weights_and_forward(
    template: nn.Module,
    buf: GenericExpertBuffer,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Copy buffer tensors into the template module's parameters, run forward."""
    with torch.no_grad():
        for name, (shape, dtype) in buf.layout.specs.items():
            parts = name.split(".")
            mod = template
            for part in parts[:-1]:
                mod = getattr(mod, part)
            getattr(mod, parts[-1]).copy_(buf.get_tensor(name))
    return template(hidden_states)


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
    ):
        self.store = store
        self.template = template
        self.device = device

        self.buf_a = buf_a
        self.buf_b = buf_b

        self.transfer_stream = transfer_stream
        self.compute_stream = compute_stream

        self.cache = cache
        self.cache_bias: float = 0.0  # logit bias magnitude for cache-aware routing (I4)
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
                hidden_states[tok:tok + 1],
                expert_output,
                tok,
                layer_idx,
                expert_indices[tok].tolist(),
                routing_weights[tok],
            )
        return expert_output

    def _execute_token_experts(
        self,
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids: list[int],
        weights: torch.Tensor,
    ):
        if self.cache is None:
            self._pipeline_experts(h, output, tok_idx, layer_idx, expert_ids, weights,
                                   list(range(len(expert_ids))))
            # Fix B: GPU-side inter-stream sync — CPU thread free to continue.
            _evt = torch.cuda.Event()
            _evt.record(self.compute_stream)
            torch.cuda.current_stream().wait_event(_evt)
            return

        cache = self.cache
        hits: list[tuple[int, int]] = []
        misses: list[int] = []
        for i, eid in enumerate(expert_ids):
            slot = cache.lookup(layer_idx, eid)
            if slot is not None:
                hits.append((i, slot))
            else:
                misses.append(i)

        for i, slot in hits:
            if slot in self._prefetch_events:
                # Prefetch H2D may still be in flight — sync before reading the cache slot.
                torch.cuda.current_stream().wait_event(self._prefetch_events.pop(slot))
            cache.load_to_buffer(slot, self.buf_a)
            out = swap_weights_and_forward(self.template, self.buf_a, h)
            output[tok_idx] += weights[i] * out.squeeze(0)

        if not misses:
            return

        self._pipeline_experts(h, output, tok_idx, layer_idx, expert_ids, weights, misses)
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

        load_done = torch.cuda.Event()
        with torch.cuda.stream(self.transfer_stream):
            self.store.copy_to_buffer(
                bufs[0], layer_idx, expert_ids[indices[0]], non_blocking=True
            )
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
                with torch.cuda.stream(self.transfer_stream):
                    if buf_done[next_buf_idx] is not None:
                        self.transfer_stream.wait_event(buf_done[next_buf_idx])
                    self.store.copy_to_buffer(
                        bufs[next_buf_idx], layer_idx,
                        expert_ids[indices[mi + 1]], non_blocking=True,
                    )
                    load_done.record(self.transfer_stream)

            with torch.cuda.stream(self.compute_stream):
                out = swap_weights_and_forward(self.template, cur_buf, h)
                output[tok_idx] += weights[idx] * out.squeeze(0)

                if cache is not None:
                    slot = cache.allocate(layer_idx, eid)
                    cache.get_packed(slot).copy_(cur_buf.packed)

                buf_done[buf_idx] = torch.cuda.Event()
                buf_done[buf_idx].record(self.compute_stream)

    def schedule_prefetch(self, layer_idx: int, expert_ids: list[int]) -> None:
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
        self._prefetch_events.clear()
        with torch.cuda.stream(self._prefetch_stream):
            for eid in expert_ids:
                if self.cache.contains(layer_idx, eid):
                    continue
                slot = self.cache.allocate(layer_idx, eid)
                cache_slot = self.cache.get_packed(slot)
                if self.store._fp8:
                    # Step 1: H2D raw FP8 bytes (half the BF16 size).
                    self._prefetch_fp8_stage.copy_(
                        self.store._data[layer_idx, eid], non_blocking=True
                    )
                    # Step 2: GPU dequant per tensor → cache slot (BF16).
                    for name, (fp8_shape, fp8_dtype) in self.store.layout.specs.items():
                        fp8_off = self.store.layout.offsets[name]
                        fp8_sz = self.store.layout.sizes[name]
                        bf16_off = self.store._bf16_layout.offsets[name]
                        bf16_sz = self.store._bf16_layout.sizes[name]
                        bf16_shape, bf16_dtype = self.store._bf16_layout.specs[name]
                        src = self._prefetch_fp8_stage[fp8_off:fp8_off + fp8_sz].view(fp8_dtype).view(fp8_shape)
                        dst = cache_slot[bf16_off:bf16_off + bf16_sz].view(bf16_dtype).view(bf16_shape)
                        if fp8_dtype.is_floating_point:
                            dst.copy_(src.to(bf16_dtype))
                        else:
                            dst.copy_(src.view(torch.uint8).view(bf16_shape))
                else:
                    cache_slot.copy_(self.store._data[layer_idx, eid], non_blocking=True)
                evt = torch.cuda.Event()
                evt.record(self._prefetch_stream)
                self._prefetch_events[slot] = evt
