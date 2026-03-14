"""Expert pipeline with CUDA streams and LRU cache.

Cache hits: compute directly from VRAM (~0.1ms per expert with dot_scaled).
Cache misses: mmap->GPU transfer + compute, double-buffered (~3.8ms per expert).
"""

import torch

from .expert_store import ExpertBuffer, ExpertStore
from .experts import expert_forward
from .lru_cache import ExpertLRUCache


class ExpertPipeline:

    def __init__(
        self,
        expert_store: ExpertStore,
        device: torch.device,
        cache_capacity: int = 0,
    ):
        self.store = expert_store
        self.device = device
        self.dtype = torch.bfloat16

        self.buf_a = ExpertBuffer(device)
        self.buf_b = ExpertBuffer(device)

        self.transfer_stream = torch.cuda.Stream(device)
        self.compute_stream = torch.cuda.Stream(device)

        self.cache = ExpertLRUCache(cache_capacity, device) if cache_capacity > 0 else None

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
            self.compute_stream.synchronize()
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

        for miss_idx in misses:
            self.store.prefetch(layer_idx, expert_ids[miss_idx])

        for i, slot in hits:
            out = expert_forward(
                h,
                cache.gate_up_blocks[slot],
                cache.gate_up_scales[slot],
                cache.gate_up_bias[slot],
                cache.down_blocks[slot],
                cache.down_scales[slot],
                cache.down_bias[slot],
                dtype=self.dtype,
            )
            output[tok_idx] += weights[i] * out.squeeze(0)

        if not misses:
            return

        self._pipeline_experts(h, output, tok_idx, layer_idx, expert_ids, weights, misses)
        self.compute_stream.synchronize()

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
        """Double-buffer PCIe pipeline for expert transfers.

        Overlap: while compute_stream processes buf[i], transfer_stream loads
        into buf[1-i]. Transfer waits for the previous compute on that same
        buffer to finish before overwriting it.
        """
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
                out = expert_forward(
                    h,
                    cur_buf.gate_up_blocks, cur_buf.gate_up_scales,
                    cur_buf.gate_up_bias, cur_buf.down_blocks,
                    cur_buf.down_scales, cur_buf.down_bias,
                    dtype=self.dtype,
                )
                output[tok_idx] += weights[idx] * out.squeeze(0)

                if cache is not None:
                    slot = cache.allocate(layer_idx, eid)
                    cache.get_packed(slot).copy_(cur_buf.packed)

                buf_done[buf_idx] = torch.cuda.Event()
                buf_done[buf_idx].record(self.compute_stream)
