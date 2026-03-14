"""GPU-resident LRU cache for hot experts."""

from collections import OrderedDict

import torch

from .config import EXPERT_BYTES
from .expert_store import _create_expert_views


class ExpertLRUCache:
    """LRU cache storing frequently-used experts in GPU VRAM.

    Uses contiguous packed storage per slot. Cache insert/read is a
    single GPU->GPU copy of the packed buffer.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device

        self._lru: OrderedDict[tuple[int, int], int] = OrderedDict()
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
        key = (layer_idx, expert_idx)
        if key in self._lru:
            self._lru.move_to_end(key)
            self.hits += 1
            return self._lru[key]
        self.misses += 1
        return None

    def allocate(self, layer_idx: int, expert_idx: int) -> int:
        key = (layer_idx, expert_idx)
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            _, slot = self._lru.popitem(last=False)
        self._lru[key] = slot
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
