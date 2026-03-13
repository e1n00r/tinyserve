"""Expert pipeline with CUDA streams, LRU cache, CPU fallback, and prefetching.

Three-way concurrency during expert execution:
  GPU compute stream: cache-hit experts via Triton fused kernel
  CPU threads:        cache-miss experts via PyTorch CPU ops (bypasses PCIe)
  GPU transfer stream: prefetch next-layer experts into LRU cache

Expert deferral: the lowest-weight expert can be deferred to execute
concurrently with the next layer's attention, overlapping its cost.
"""

from concurrent.futures import ThreadPoolExecutor

import torch

from .config import NUM_EXPERTS_PER_TOK, NUM_LAYERS
from .cpu_expert import expert_forward_cpu
from .expert_store import ExpertBuffer, ExpertStore
from .experts import expert_forward
from .lru_cache import ExpertLRUCache
from .prefetch import ExpertPredictor


class ExpertPipeline:
    """Pipelined expert execution with cache, CPU fallback, and prefetching."""

    def __init__(
        self,
        expert_store: ExpertStore,
        device: torch.device,
        cache_capacity: int = 0,
        enable_prefetch: bool = True,
        max_deferred: int = 1,
    ):
        self.store = expert_store
        self.device = device
        self.dtype = torch.bfloat16

        self.buf_a = ExpertBuffer(device)
        self.buf_b = ExpertBuffer(device)

        self.transfer_stream = torch.cuda.Stream(device)
        self.compute_stream = torch.cuda.Stream(device)

        self.cache = ExpertLRUCache(cache_capacity, device) if cache_capacity > 0 else None

        # CPU thread pool for deferred expert compute
        # (CPU too slow for regular misses — use PCIe pipeline instead)
        self.cpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu_expert")

        # Cross-layer prefetching
        # Prefetching disabled: with 359 cache slots and 144 experts/token,
        # prefetch evictions destroy hit rate (44% vs 79% without).
        # Only viable with much larger cache or better prediction accuracy.
        self.enable_prefetch = False
        if self.enable_prefetch:
            self.predictor = ExpertPredictor()

        # Expert deferral — disabled by default (CPU expert is too slow
        # for this model; only viable when CPU compute < next layer time)
        self.max_deferred = 0  # max_deferred

    def execute_layer_experts(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        deferred_results: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        """Execute experts for one layer.

        Args:
            deferred_results: list of (position, weight, cpu_future) from previous layer

        Returns:
            (expert_output, new_deferred) where new_deferred contains futures
            for experts deferred to the next layer.
        """
        num_tokens = hidden_states.shape[0]
        expert_output = torch.zeros_like(hidden_states)

        # Apply deferred results from previous layer
        if deferred_results:
            for weight, future in deferred_results:
                cpu_out = future.result()  # blocks until CPU done
                expert_output[0] += weight * cpu_out.to(self.device).squeeze(0).to(self.dtype)

        new_deferred = []
        for tok in range(num_tokens):
            tok_deferred = self._execute_token_experts(
                hidden_states[tok:tok + 1],
                expert_output,
                tok,
                layer_idx,
                expert_indices[tok].tolist(),
                routing_weights[tok],
            )
            new_deferred.extend(tok_deferred)

        # Prefetch next layer's likely experts
        if self.enable_prefetch and layer_idx < NUM_LAYERS - 1:
            expert_ids = expert_indices[0].tolist()
            self.predictor.record(layer_idx, expert_ids)
            self._prefetch_next_layer(layer_idx, expert_ids)

        return expert_output, new_deferred

    def _execute_token_experts(
        self,
        h: torch.Tensor,
        output: torch.Tensor,
        tok_idx: int,
        layer_idx: int,
        expert_ids: list[int],
        weights: torch.Tensor,
    ) -> list:
        """Returns list of (weight, future) for deferred experts."""
        if self.cache is None:
            self._execute_no_cache(h, output, tok_idx, layer_idx, expert_ids, weights)
            return []

        cache = self.cache

        # Partition into cache hits and misses
        hits: list[tuple[int, int]] = []    # (position_in_topk, cache_slot)
        misses: list[int] = []               # position_in_topk
        for i, eid in enumerate(expert_ids):
            slot = cache.lookup(layer_idx, eid)
            if slot is not None:
                hits.append((i, slot))
            else:
                misses.append(i)

        # Deferral: defer lowest-weight miss to next layer
        deferred: list[tuple[torch.Tensor, object]] = []
        if self.max_deferred > 0 and misses and layer_idx < NUM_LAYERS - 1:
            # Sort misses by weight (ascending) — defer the lightest
            misses_sorted = sorted(misses, key=lambda i: weights[i].item())
            n_defer = min(self.max_deferred, len(misses_sorted))
            defer_set = set()
            for di in range(n_defer):
                m = misses_sorted[di]
                if weights[m].item() < 0.20:  # only defer low-weight experts
                    defer_set.add(m)

            if defer_set:
                # Launch deferred experts on CPU immediately
                h_cpu = h.detach().cpu()
                for m in defer_set:
                    eid = expert_ids[m]
                    future = self.cpu_executor.submit(
                        expert_forward_cpu,
                        h_cpu,
                        self.store.gate_up_blocks[layer_idx][eid],
                        self.store.gate_up_scales[layer_idx][eid],
                        self.store.gate_up_bias[layer_idx][eid],
                        self.store.down_blocks[layer_idx][eid],
                        self.store.down_scales[layer_idx][eid],
                        self.store.down_bias[layer_idx][eid],
                    )
                    deferred.append((weights[m], future))

                misses = [m for m in misses if m not in defer_set]

        # GPU: compute cache hits
        with torch.cuda.stream(self.compute_stream):
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

        # GPU: pipeline remaining misses via PCIe double-buffer
        if misses:
            self._pipeline_misses(h, output, tok_idx, layer_idx, expert_ids, weights, misses)

        self.compute_stream.synchronize()

        return deferred

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
        """Double-buffer PCIe pipeline for cache misses (fallback when CPU disabled)."""
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
                    slot = cache.allocate(layer_idx, eid)
                    cache.gate_up_blocks[slot].copy_(buf.gate_up_blocks)
                    cache.gate_up_scales[slot].copy_(buf.gate_up_scales)
                    cache.gate_up_bias[slot].copy_(buf.gate_up_bias)
                    cache.down_blocks[slot].copy_(buf.down_blocks)
                    cache.down_scales[slot].copy_(buf.down_scales)
                    cache.down_bias[slot].copy_(buf.down_bias)

    def _prefetch_next_layer(self, layer_idx: int, current_expert_ids: list[int]):
        """Prefetch predicted experts for layer_idx+1 into LRU cache.

        Conservative: only prefetches experts that are in the current layer's
        top-k AND not already cached. Uses `contains` to avoid modifying
        LRU order or cache stats. Limits to 2 prefetches per layer to avoid
        excessive cache evictions.
        """
        cache = self.cache
        if cache is None:
            return

        next_layer = layer_idx + 1
        predicted = self.predictor.predict(layer_idx, current_expert_ids, k=4)

        prefetched = 0
        for eid in predicted:
            if prefetched >= 2:
                break
            if cache.contains(next_layer, eid):
                continue  # already cached, don't touch LRU order

            # Allocate cache slot and transfer on background stream
            slot = cache.allocate(next_layer, eid)
            with torch.cuda.stream(self.transfer_stream):
                src = self.store
                cache.gate_up_blocks[slot].copy_(
                    src.gate_up_blocks[next_layer][eid], non_blocking=True
                )
                cache.gate_up_scales[slot].copy_(
                    src.gate_up_scales[next_layer][eid], non_blocking=True
                )
                cache.gate_up_bias[slot].copy_(
                    src.gate_up_bias[next_layer][eid], non_blocking=True
                )
                cache.down_blocks[slot].copy_(
                    src.down_blocks[next_layer][eid], non_blocking=True
                )
                cache.down_scales[slot].copy_(
                    src.down_scales[next_layer][eid], non_blocking=True
                )
                cache.down_bias[slot].copy_(
                    src.down_bias[next_layer][eid], non_blocking=True
                )
            prefetched += 1

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
