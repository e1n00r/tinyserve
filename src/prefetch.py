"""Cross-layer expert prefetching.

Predicts which experts layer L+1 will need while layer L is computing.
Uses a combination of:
1. Current layer's routed experts (adjacent layers often route similarly)
2. Per-layer frequency tracking (some experts are universally "hot")
"""

import numpy as np

from .config import NUM_EXPERTS, NUM_LAYERS


class ExpertPredictor:
    """Predicts next-layer expert usage for prefetching."""

    def __init__(self, num_layers: int = NUM_LAYERS, num_experts: int = NUM_EXPERTS):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.counts = np.zeros((num_layers, num_experts), dtype=np.int64)
        self._top_cache: list[list[int] | None] = [None] * num_layers

    def record(self, layer_idx: int, expert_ids: list[int]):
        """Record which experts were actually routed for frequency tracking."""
        for eid in expert_ids:
            self.counts[layer_idx, eid] += 1
        self._top_cache[layer_idx] = None

    def predict(self, layer_idx: int, current_expert_ids: list[int], k: int = 6) -> list[int]:
        """Predict k experts for layer_idx+1.

        Strategy: union of current layer's experts + frequency-based top experts.
        Returns at most k expert IDs for the next layer.
        """
        next_layer = layer_idx + 1
        if next_layer >= self.num_layers:
            return []

        # Start with current layer's experts (strongest signal)
        candidates = list(current_expert_ids)

        # Add frequency-based predictions for next layer
        if self._top_cache[next_layer] is None:
            if self.counts[next_layer].sum() > 0:
                self._top_cache[next_layer] = np.argsort(
                    self.counts[next_layer]
                )[-k * 2:][::-1].tolist()
            else:
                self._top_cache[next_layer] = []

        seen = set(candidates)
        for eid in self._top_cache[next_layer]:
            if eid not in seen:
                candidates.append(eid)
                seen.add(eid)
            if len(candidates) >= k:
                break

        return candidates[:k]
