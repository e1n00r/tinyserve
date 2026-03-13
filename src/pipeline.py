"""Expert pipeline with CUDA streams and LRU cache.

Cache hits: compute directly from VRAM (0.64ms per expert).
Cache misses: mmap→GPU transfer + compute, double-buffered (2ms + 0.4ms overlap).
"""

import torch

from .config import NUM_EXPERTS_PER_TOK, NUM_LAYERS
from .expert_store import ExpertBuffer, ExpertStore
from .experts import expert_forward
from .lru_cache import ExpertLRUCache


class ExpertPipeline:
    """Pipelined expert execution with VRAM LRU cache."""

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
        num_tokens = hidden_states.shape[0]
        expert_output = torch.zeros_like(hidden_states)

        for tok in range(num_tokens):
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
            self._execute_no_cache(h, output, tok_idx, layer_idx, expert_ids, weights)
            return

        cache = self.cache

        # Partition into cache hits and misses
        hits: list[tuple[int, int]] = []
        misses: list[int] = []
        for i, eid in enumerate(expert_ids):
            slot = cache.lookup(layer_idx, eid)
            if slot is not None:
                hits.append((i, slot))
            else:
                misses.append(i)

        # Compute cache hits on default stream
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

        # Handle misses
        if len(misses) == 1:
            # Single miss: direct copy + compute (skip double-buffer overhead)
            m = misses[0]
            eid = expert_ids[m]
            buf = self.buf_a
            self.store.copy_to_buffer(buf, layer_idx, eid, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            out = expert_forward(
                h,
                buf.gate_up_blocks, buf.gate_up_scales, buf.gate_up_bias,
                buf.down_blocks, buf.down_scales, buf.down_bias,
                dtype=self.dtype,
            )
            output[tok_idx] += weights[m] * out.squeeze(0)
            self._cache_insert(cache, layer_idx, eid, buf)
        elif misses:
            self._pipeline_misses(h, output, tok_idx, layer_idx, expert_ids, weights, misses)
            self.compute_stream.synchronize()

    def _cache_insert(self, cache, layer_idx, eid, buf):
        """Insert expert from buffer into LRU cache."""
        slot = cache.allocate(layer_idx, eid)
        cache.gate_up_blocks[slot].copy_(buf.gate_up_blocks)
        cache.gate_up_scales[slot].copy_(buf.gate_up_scales)
        cache.gate_up_bias[slot].copy_(buf.gate_up_bias)
        cache.down_blocks[slot].copy_(buf.down_blocks)
        cache.down_scales[slot].copy_(buf.down_scales)
        cache.down_bias[slot].copy_(buf.down_bias)

    def _pipeline_misses(
        self,
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids: list[int],
        weights: torch.Tensor,
        misses: list[int],
    ):
        """Double-buffer PCIe pipeline for cache misses."""
        cache = self.cache
        bufs = [self.buf_a, self.buf_b]

        load_done = torch.cuda.Event()
        with torch.cuda.stream(self.transfer_stream):
            self.store.copy_to_buffer(
                bufs[0], layer_idx, expert_ids[misses[0]], non_blocking=True
            )
            load_done.record(self.transfer_stream)

        for mi in range(len(misses)):
            cur = mi & 1
            nxt = 1 - cur
            m = misses[mi]
            eid = expert_ids[m]

            self.compute_stream.wait_event(load_done)

            if mi < len(misses) - 1:
                load_done = torch.cuda.Event()
                with torch.cuda.stream(self.transfer_stream):
                    self.store.copy_to_buffer(
                        bufs[nxt], layer_idx, expert_ids[misses[mi + 1]], non_blocking=True
                    )
                    load_done.record(self.transfer_stream)

            with torch.cuda.stream(self.compute_stream):
                buf = bufs[cur]
                out = expert_forward(
                    h,
                    buf.gate_up_blocks, buf.gate_up_scales, buf.gate_up_bias,
                    buf.down_blocks, buf.down_scales, buf.down_bias,
                    dtype=self.dtype,
                )
                output[tok_idx] += weights[m] * out.squeeze(0)

                if cache is not None:
                    self._cache_insert(cache, layer_idx, eid, buf)

    def _execute_no_cache(
        self,
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids: list[int],
        weights: torch.Tensor,
    ):
        """Double-buffer pipeline without cache."""
        k = NUM_EXPERTS_PER_TOK
        bufs = [self.buf_a, self.buf_b]

        load_done = torch.cuda.Event()
        with torch.cuda.stream(self.transfer_stream):
            self.store.copy_to_buffer(bufs[0], layer_idx, expert_ids[0], non_blocking=True)
            load_done.record(self.transfer_stream)

        for i in range(k):
            cur = i & 1
            nxt = 1 - cur

            self.compute_stream.wait_event(load_done)

            if i < k - 1:
                load_done = torch.cuda.Event()
                with torch.cuda.stream(self.transfer_stream):
                    self.store.copy_to_buffer(
                        bufs[nxt], layer_idx, expert_ids[i + 1], non_blocking=True
                    )
                    load_done.record(self.transfer_stream)

            with torch.cuda.stream(self.compute_stream):
                buf = bufs[cur]
                out = expert_forward(
                    h,
                    buf.gate_up_blocks, buf.gate_up_scales, buf.gate_up_bias,
                    buf.down_blocks, buf.down_scales, buf.down_bias,
                    dtype=self.dtype,
                )
                output[tok_idx] += weights[i] * out.squeeze(0)

        self.compute_stream.synchronize()
