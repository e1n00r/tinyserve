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
    PACK_DN_BIAS_OFF,
    PACK_DN_BLOCKS_OFF,
    PACK_DN_SCALES_OFF,
    PACK_GU_BIAS_OFF,
    PACK_GU_BLOCKS_OFF,
    PACK_GU_SCALES_OFF,
)

_gu_b = GATE_UP_BLOCKS_SHAPE[1] * GATE_UP_BLOCKS_SHAPE[2] * GATE_UP_BLOCKS_SHAPE[3]
_gu_s = GATE_UP_SCALES_SHAPE[1] * GATE_UP_SCALES_SHAPE[2]
_gu_bias = GATE_UP_BIAS_SHAPE[1] * 4
_dn_b = DOWN_BLOCKS_SHAPE[1] * DOWN_BLOCKS_SHAPE[2] * DOWN_BLOCKS_SHAPE[3]
_dn_s = DOWN_SCALES_SHAPE[1] * DOWN_SCALES_SHAPE[2]
_dn_bias = DOWN_BIAS_SHAPE[1] * 4


class ExpertLRUCache:
    """LRU cache storing frequently-used experts in GPU VRAM.

    Uses contiguous packed storage per slot. Cache insert/read is a
    single GPU→GPU copy of the packed buffer.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device

        self._lru: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._free_slots: list[int] = list(range(capacity - 1, -1, -1))

        # Contiguous packed storage: [capacity, EXPERT_BYTES] uint8
        self._packed = torch.empty((capacity, EXPERT_BYTES), dtype=torch.uint8, device=device)

        # Create views for each slot's tensors
        self.gate_up_blocks = self._packed[:, PACK_GU_BLOCKS_OFF:PACK_GU_BLOCKS_OFF + _gu_b].view(
            capacity, *GATE_UP_BLOCKS_SHAPE[1:])
        self.gate_up_scales = self._packed[:, PACK_GU_SCALES_OFF:PACK_GU_SCALES_OFF + _gu_s].view(
            capacity, *GATE_UP_SCALES_SHAPE[1:])
        self.gate_up_bias = self._packed[:, PACK_GU_BIAS_OFF:PACK_GU_BIAS_OFF + _gu_bias].reshape(
            capacity, _gu_bias).view(torch.float32).view(capacity, *GATE_UP_BIAS_SHAPE[1:])
        self.down_blocks = self._packed[:, PACK_DN_BLOCKS_OFF:PACK_DN_BLOCKS_OFF + _dn_b].view(
            capacity, *DOWN_BLOCKS_SHAPE[1:])
        self.down_scales = self._packed[:, PACK_DN_SCALES_OFF:PACK_DN_SCALES_OFF + _dn_s].view(
            capacity, *DOWN_SCALES_SHAPE[1:])
        self.down_bias = self._packed[:, PACK_DN_BIAS_OFF:PACK_DN_BIAS_OFF + _dn_bias].reshape(
            capacity, _dn_bias).view(torch.float32).view(capacity, *DOWN_BIAS_SHAPE[1:])

        self.hits = 0
        self.misses = 0

    def get_packed(self, slot: int) -> torch.Tensor:
        """Get the packed buffer for a slot (for single-copy operations)."""
        return self._packed[slot]

    def lookup(self, layer_idx: int, expert_idx: int) -> int | None:
        key = (layer_idx, expert_idx)
        if key in self._lru:
            self._lru.move_to_end(key)
            self.hits += 1
            return self._lru[key]
        self.misses += 1
        return None

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        return (layer_idx, expert_idx) in self._lru

    def allocate(self, layer_idx: int, expert_idx: int) -> int:
        key = (layer_idx, expert_idx)
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
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
        return available_bytes // EXPERT_BYTES
