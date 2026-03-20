"""GPU-resident LRU cache for hot experts."""

import torch

from .cache_policy import make_policy
from .config import EXPERT_BYTES
from .expert_store import _create_expert_views


class ExpertLRUCache:
    """LRU cache storing frequently-used experts in GPU VRAM.

    Uses contiguous packed storage per slot. Cache insert/read is a
    single GPU->GPU copy of the packed buffer.
    """

    def __init__(self, capacity: int, device: torch.device, policy: str = "lru"):
        self.capacity = capacity
        self.device = device

        self._policy = make_policy(policy, capacity)
        self._free_slots: list[int] = list(range(capacity - 1, -1, -1))

        self._packed = torch.empty((capacity, EXPERT_BYTES), dtype=torch.uint8, device=device)
        (self.gate_up_blocks, self.gate_up_scales, self.gate_up_bias,
         self.down_blocks, self.down_scales, self.down_bias
         ) = _create_expert_views(self._packed, prefix_shape=(capacity,))

        self.hits = 0
        self.misses = 0

    def get_packed(self, slot: int) -> torch.Tensor:
        return self._packed[slot]

    def lookup(self, layer_idx: int, expert_idx: int) -> int | None:
        slot = self._policy.lookup((layer_idx, expert_idx))
        if slot is not None:
            self.hits += 1
        else:
            self.misses += 1
        return slot

    def allocate(self, layer_idx: int, expert_idx: int) -> int:
        key = (layer_idx, expert_idx)
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            evict_key, slot = self._policy.select_evict()
            self._policy.remove(evict_key)
        self._policy.insert(key, slot)
        return slot

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_stats(self):
        self.hits = 0
        self.misses = 0

    @staticmethod
    def estimate_capacity(available_bytes: int) -> int:
        return available_bytes // EXPERT_BYTES
