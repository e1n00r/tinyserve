"""Phase 4: GPU-resident LRU cache for hot experts."""

from collections import OrderedDict

import torch

from .config import (
    DOWN_BIAS_SHAPE,
    DOWN_BLOCKS_SHAPE,
    DOWN_SCALES_SHAPE,
    EXPERT_BYTES,
    GATE_UP_BIAS_SHAPE,
    GATE_UP_BLOCKS_SHAPE,
    GATE_UP_SCALES_SHAPE,
)


class ExpertLRUCache:
    """LRU cache storing frequently-used experts in GPU VRAM.

    Pre-allocates a fixed number of slots on GPU. Each slot holds one
    expert's MXFP4-packed data (blocks, scales, biases). Cache hits
    skip PCIe transfer entirely.
    """

    def __init__(self, capacity: int, device: torch.device):
        """
        Args:
            capacity: max number of experts to cache
            device: GPU device
        """
        self.capacity = capacity
        self.device = device

        # LRU tracking: key=(layer_idx, expert_idx) -> slot_index
        # OrderedDict maintains insertion/access order for LRU
        self._lru: OrderedDict[tuple[int, int], int] = OrderedDict()

        # Free slot stack
        self._free_slots: list[int] = list(range(capacity - 1, -1, -1))

        # Pre-allocated GPU storage: [capacity, ...] for each tensor type
        # Shapes per expert (without the expert dim)
        gu_b = GATE_UP_BLOCKS_SHAPE[1:]  # (5760, 90, 16)
        gu_s = GATE_UP_SCALES_SHAPE[1:]  # (5760, 90)
        dn_b = DOWN_BLOCKS_SHAPE[1:]     # (2880, 90, 16)
        dn_s = DOWN_SCALES_SHAPE[1:]     # (2880, 90)

        self.gate_up_blocks = torch.empty((capacity, *gu_b), dtype=torch.uint8, device=device)
        self.gate_up_scales = torch.empty((capacity, *gu_s), dtype=torch.uint8, device=device)
        self.gate_up_bias = torch.empty((capacity, GATE_UP_BIAS_SHAPE[1]), dtype=torch.float32, device=device)
        self.down_blocks = torch.empty((capacity, *dn_b), dtype=torch.uint8, device=device)
        self.down_scales = torch.empty((capacity, *dn_s), dtype=torch.uint8, device=device)
        self.down_bias = torch.empty((capacity, DOWN_BIAS_SHAPE[1]), dtype=torch.float32, device=device)

        self.hits = 0
        self.misses = 0

    def lookup(self, layer_idx: int, expert_idx: int) -> int | None:
        """Check if expert is cached. Returns slot index or None.

        Moves the entry to most-recently-used on hit.
        """
        key = (layer_idx, expert_idx)
        if key in self._lru:
            self._lru.move_to_end(key)
            self.hits += 1
            return self._lru[key]
        self.misses += 1
        return None

    def allocate(self, layer_idx: int, expert_idx: int) -> int:
        """Allocate a cache slot for a new expert, evicting LRU if full.

        Returns the slot index. Caller must copy data into the slot.
        """
        key = (layer_idx, expert_idx)

        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            # Evict least recently used
            evicted_key, slot = self._lru.popitem(last=False)

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
        """How many experts fit in the given VRAM budget."""
        return available_bytes // EXPERT_BYTES
