"""Expert batching for concurrent request throughput.

When multiple requests decode simultaneously, they often need the same expert.
Instead of loading the expert N times, load once and batch all hidden states
through it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .expert_execution import ExpertPipeline, forward_from_packed, swap_weights_and_forward


@dataclass
class BatchItem:
    hidden_states: torch.Tensor  # [1, hidden_dim]
    expert_indices: torch.Tensor  # [top_k]
    routing_weights: torch.Tensor  # [top_k]
    request_idx: int


class ExpertBatcher:
    """Batches expert forwards across multiple concurrent requests."""

    def __init__(self, pipeline: ExpertPipeline):
        self._pipeline = pipeline

    def batch_execute(self, items: list[BatchItem], layer_idx: int) -> list[torch.Tensor]:
        """Execute expert forwards for multiple requests with expert-level batching.

        Groups hidden states by expert_id. For each unique expert:
        1. Load expert weights once (cache hit or miss)
        2. Batch all hidden states needing this expert: [N, hidden_dim]
        3. One expert forward: [N, hidden_dim] -> [N, hidden_dim]
        4. Scatter weighted results back to per-request outputs

        Returns list of output tensors, one per request.
        """
        if not items:
            return []

        # Group by expert_id
        # expert_groups[eid] = [(item_list_idx, weight_position_idx, hidden_states, weight_value)]
        expert_groups: dict[int, list[tuple[int, int, torch.Tensor, float]]] = {}

        for item_idx, item in enumerate(items):
            expert_ids = item.expert_indices.tolist()
            for wi, eid in enumerate(expert_ids):
                if eid not in expert_groups:
                    expert_groups[eid] = []
                expert_groups[eid].append((item_idx, wi, item.hidden_states, item.routing_weights[wi].item()))

        # Initialize per-request outputs
        outputs = [torch.zeros_like(item.hidden_states) for item in items]

        pipeline = self._pipeline
        cache = pipeline.cache

        for eid, group in expert_groups.items():
            # Batch hidden states: [N, hidden_dim]
            h_batch = torch.cat([g[2] for g in group], dim=0)

            # Try cache hit first
            out_batch = None
            if cache is not None:
                slot = cache.locate((layer_idx, eid))
                if slot is not None:
                    packed = cache.get_packed(slot)
                    if pipeline._inline_fwd is not None:
                        out_batch = pipeline._inline_fwd(packed, h_batch)
                    else:
                        out_batch = forward_from_packed(pipeline.template, packed, pipeline._param_refs, h_batch)

            # Cache miss: load via store -> buffer -> forward
            if out_batch is None:
                out_batch = self._load_and_forward(h_batch, layer_idx, eid)

            # Scatter results back weighted
            for i, (item_idx, _wi, _h, weight) in enumerate(group):
                outputs[item_idx] += weight * out_batch[i : i + 1]

        return outputs

    def _load_and_forward(self, h_batch: torch.Tensor, layer_idx: int, eid: int) -> torch.Tensor:
        """Load expert from store to GPU buffer, forward the batch, optionally cache."""
        pipeline = self._pipeline
        buf = pipeline.staging_buffer_a

        pipeline.store.copy_to_buffer(buf, layer_idx, eid, non_blocking=False)
        torch.cuda.synchronize()

        out = swap_weights_and_forward(pipeline.template, buf, h_batch)

        # Store in cache if available
        if pipeline.cache is not None:
            slot = pipeline.cache.claim_slot_for((layer_idx, eid))
            pipeline.cache.get_packed(slot).copy_(buf.packed)

        return out
