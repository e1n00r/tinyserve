"""Double-buffered expert pipeline with CUDA streams and LRU cache.

Overlaps expert PCIe transfer with expert GPU compute using two streams.
Cache hits skip PCIe entirely — expert data is read directly from the
LRU cache's pre-allocated GPU memory.
"""

import torch

from .config import NUM_EXPERTS_PER_TOK
from .expert_store import ExpertBuffer, ExpertStore
from .experts import expert_forward
from .lru_cache import ExpertLRUCache


class ExpertPipeline:
    """Double-buffered expert execution with LRU cache and overlapped transfer."""

    def __init__(
        self,
        expert_store: ExpertStore,
        device: torch.device,
        cache_capacity: int = 0,
    ):
        self.store = expert_store
        self.device = device
        self.dtype = torch.bfloat16

        # Two GPU buffers for double-buffering PCIe transfers
        self.buf_a = ExpertBuffer(device)
        self.buf_b = ExpertBuffer(device)

        # CUDA streams
        self.transfer_stream = torch.cuda.Stream(device)
        self.compute_stream = torch.cuda.Stream(device)

        # LRU cache (optional, capacity=0 disables)
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

    def _compute_expert_from_cache(
        self,
        h: torch.Tensor,
        slot: int,
    ) -> torch.Tensor:
        """Run expert forward using data from the LRU cache slot."""
        return expert_forward(
            h,
            self.cache.gate_up_blocks[slot],
            self.cache.gate_up_scales[slot],
            self.cache.gate_up_bias[slot],
            self.cache.down_blocks[slot],
            self.cache.down_scales[slot],
            self.cache.down_bias[slot],
            dtype=self.dtype,
        )

    def _copy_buf_to_cache(self, buf: ExpertBuffer, slot: int):
        """Copy expert data from transfer buffer into cache slot (on GPU, fast)."""
        self.cache.gate_up_blocks[slot].copy_(buf.gate_up_blocks)
        self.cache.gate_up_scales[slot].copy_(buf.gate_up_scales)
        self.cache.gate_up_bias[slot].copy_(buf.gate_up_bias)
        self.cache.down_blocks[slot].copy_(buf.down_blocks)
        self.cache.down_scales[slot].copy_(buf.down_scales)
        self.cache.down_bias[slot].copy_(buf.down_bias)

    def _execute_token_experts(
        self,
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids: list[int],
        weights: torch.Tensor,
    ):
        k = NUM_EXPERTS_PER_TOK

        if self.cache is None:
            self._execute_no_cache(h, output, tok_idx, layer_idx, expert_ids, weights)
            return

        # With cache: separate experts into hits and misses, then process
        # For simplicity, process sequentially but skip PCIe for hits
        for i in range(k):
            eid = expert_ids[i]
            slot = self.cache.lookup(layer_idx, eid)

            if slot is not None:
                # Cache hit — compute directly from cache, no PCIe
                with torch.cuda.stream(self.compute_stream):
                    out = self._compute_expert_from_cache(h, slot)
                    output[tok_idx] += weights[i] * out.squeeze(0)
            else:
                # Cache miss — transfer from CPU, compute, then cache
                self.compute_stream.synchronize()  # ensure previous compute done
                with torch.cuda.stream(self.transfer_stream):
                    self.store.copy_to_buffer(
                        self.buf_a, layer_idx, eid, non_blocking=True
                    )
                self.transfer_stream.synchronize()

                with torch.cuda.stream(self.compute_stream):
                    out = expert_forward(
                        h,
                        self.buf_a.gate_up_blocks,
                        self.buf_a.gate_up_scales,
                        self.buf_a.gate_up_bias,
                        self.buf_a.down_blocks,
                        self.buf_a.down_scales,
                        self.buf_a.down_bias,
                        dtype=self.dtype,
                    )
                    output[tok_idx] += weights[i] * out.squeeze(0)

                # Insert into cache (GPU→GPU copy, fast)
                slot = self.cache.allocate(layer_idx, eid)
                self._copy_buf_to_cache(self.buf_a, slot)

        self.compute_stream.synchronize()

    def _execute_no_cache(
        self,
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids: list[int],
        weights: torch.Tensor,
    ):
        """Double-buffered pipeline without cache (Phase 3 behavior)."""
        k = NUM_EXPERTS_PER_TOK
        bufs = [self.buf_a, self.buf_b]
        compute_done = [torch.cuda.Event() for _ in range(2)]

        load_done = torch.cuda.Event()
        with torch.cuda.stream(self.transfer_stream):
            self.store.copy_to_buffer(bufs[0], layer_idx, expert_ids[0], non_blocking=True)
            load_done.record(self.transfer_stream)

        for i in range(k):
            cur = i % 2
            nxt = (i + 1) % 2

            self.compute_stream.wait_event(load_done)

            if i < k - 1:
                next_load_done = torch.cuda.Event()
                with torch.cuda.stream(self.transfer_stream):
                    if i >= 1:
                        self.transfer_stream.wait_event(compute_done[nxt])
                    self.store.copy_to_buffer(
                        bufs[nxt], layer_idx, expert_ids[i + 1], non_blocking=True
                    )
                    next_load_done.record(self.transfer_stream)
                load_done = next_load_done

            with torch.cuda.stream(self.compute_stream):
                out = expert_forward(
                    h,
                    bufs[cur].gate_up_blocks,
                    bufs[cur].gate_up_scales,
                    bufs[cur].gate_up_bias,
                    bufs[cur].down_blocks,
                    bufs[cur].down_scales,
                    bufs[cur].down_bias,
                    dtype=self.dtype,
                )
                output[tok_idx] += weights[i] * out.squeeze(0)
                compute_done[cur].record(self.compute_stream)

        self.compute_stream.synchronize()
