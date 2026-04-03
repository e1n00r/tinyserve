"""GPU VRAM cache for expert weight buffers.

Separate from ExpertStore (CPU weight storage) to keep cache policy concerns
isolated from storage layout concerns.
"""

from __future__ import annotations

import torch

from .cache_policy import make_policy


class ExpertCache:
    """LRU cache for generic expert buffers in GPU VRAM."""

    def __init__(
        self,
        capacity: int,
        expert_bytes: int,
        device: torch.device,
        policy: str = "lru",
        num_layers: int = 1,
        num_experts: int = 1,
    ):
        self.capacity = capacity
        self.expert_bytes = expert_bytes
        self.device = device
        self._packed = torch.empty(capacity, expert_bytes, dtype=torch.uint8, device=device)
        self._policy = make_policy(policy, capacity)
        self._free_slots = list(range(capacity - 1, -1, -1))
        self.hits = 0
        self.misses = 0
        # Per-layer statistics
        self._layer_hits: dict[int, int] = {}
        self._layer_misses: dict[int, int] = {}
        self._layer_miss_latencies: dict[int, list[float]] = {}
        self._expert_access_count: dict[tuple[int, int], int] = {}
        # Per-step tracking
        self._step_experts: set[tuple[int, int]] | None = None
        self._step_lookups: int = 0
        # Slot map: CPU array is primary (written per-allocate, no CUDA kernel),
        # GPU tensor is synced lazily before lookup_slots reads it.
        import numpy as np
        self._slot_map: torch.Tensor | None = None
        self._slot_map_cpu: np.ndarray | None = None
        self._slot_map_dirty: bool = False
        self._slot_map_dims = (num_layers, num_experts)
        if num_layers > 1 or num_experts > 1:
            self._slot_map_cpu = np.full((num_layers, num_experts), -1, dtype=np.int32)
            self._slot_map = torch.from_numpy(self._slot_map_cpu).to(dtype=torch.int32, device=device)

    def lookup(self, layer_idx: int, expert_idx: int) -> int | None:
        slot = self._policy.lookup((layer_idx, expert_idx))
        key = (layer_idx, expert_idx)
        self._expert_access_count[key] = self._expert_access_count.get(key, 0) + 1
        if self._step_experts is not None:
            self._step_experts.add(key)
            self._step_lookups += 1
        if slot is not None:
            self.hits += 1
            self._layer_hits[layer_idx] = self._layer_hits.get(layer_idx, 0) + 1
        else:
            self.misses += 1
            self._layer_misses[layer_idx] = self._layer_misses.get(layer_idx, 0) + 1
        return slot

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        """Check if expert is in cache without updating policy state or stats."""
        return self._policy.contains((layer_idx, expert_idx))

    def begin_pass(self) -> None:
        """Notify the policy that a new token forward pass is starting.

        Only meaningful for LeastStalePolicy — rotates fresh→stale so experts
        loaded for the previous token become eviction candidates. No-op for
        all other policies.
        """
        if hasattr(self._policy, "begin_pass"):
            self._policy.begin_pass()

    def _ensure_slot_map(self, layer_idx: int, expert_idx: int):
        """Lazily create or grow the slot map to fit (layer_idx, expert_idx)."""
        import numpy as np
        nl = max(self._slot_map_dims[0], layer_idx + 1)
        ne = max(self._slot_map_dims[1], expert_idx + 1)
        if self._slot_map_cpu is None:
            self._slot_map_cpu = np.full((nl, ne), -1, dtype=np.int32)
            self._slot_map = torch.from_numpy(self._slot_map_cpu).to(dtype=torch.int32, device=self.device)
            self._slot_map_dims = (nl, ne)
        elif nl > self._slot_map_cpu.shape[0] or ne > self._slot_map_cpu.shape[1]:
            new_cpu = np.full((nl, ne), -1, dtype=np.int32)
            old = self._slot_map_cpu
            new_cpu[:old.shape[0], :old.shape[1]] = old
            self._slot_map_cpu = new_cpu
            self._slot_map = torch.from_numpy(new_cpu).to(dtype=torch.int32, device=self.device)
            self._slot_map_dims = (nl, ne)
            self._slot_map_dirty = False

    def allocate(self, layer_idx: int, expert_idx: int) -> int:
        key = (layer_idx, expert_idx)
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            evict_key, slot = self._policy.select_evict()
            self._policy.remove(evict_key)
            if self._slot_map is not None:
                self._slot_map_cpu[evict_key[0], evict_key[1]] = -1
                self._slot_map_dirty = True
        self._policy.insert(key, slot)
        self._ensure_slot_map(layer_idx, expert_idx)
        self._slot_map_cpu[layer_idx, expert_idx] = slot
        self._slot_map_dirty = True
        return slot

    def flush_slot_updates(self):
        """Sync CPU slot_map to GPU. Called automatically by lookup_slots."""
        if not self._slot_map_dirty or self._slot_map_cpu is None:
            return
        self._slot_map.copy_(torch.from_numpy(self._slot_map_cpu))
        self._slot_map_dirty = False

    def lookup_slots(self, layer_idx: int, expert_ids: torch.Tensor) -> torch.Tensor:
        """GPU tensor cache lookup — syncs from CPU if dirty.

        Args:
            layer_idx: which MoE layer
            expert_ids: [top_k] int tensor on GPU

        Returns:
            [top_k] int32 tensor on GPU. Values >= 0 are cache slot indices,
            -1 means cache miss.
        """
        if self._slot_map_dirty:
            self.flush_slot_updates()
        if self._slot_map is None:
            return torch.full_like(expert_ids, -1, dtype=torch.int32)
        if layer_idx >= self._slot_map.shape[0]:
            return torch.full_like(expert_ids, -1, dtype=torch.int32)
        ne = self._slot_map.shape[1]
        ids = expert_ids.long().to(self.device)
        safe = ids.clamp(max=ne - 1)
        result = self._slot_map[layer_idx, safe]
        result = torch.where(ids < ne, result,
            torch.tensor(-1, dtype=torch.int32, device=self.device))
        return result

    def store_from_buffer(self, slot: int, buf) -> None:
        self._packed[slot].copy_(buf.packed)

    def load_to_buffer(self, slot: int, buf) -> None:
        buf.packed.copy_(self._packed[slot])

    def get_packed(self, slot: int) -> torch.Tensor:
        return self._packed[slot]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_miss_latency(self, layer_idx: int, latency_ms: float):
        if layer_idx not in self._layer_miss_latencies:
            self._layer_miss_latencies[layer_idx] = []
        self._layer_miss_latencies[layer_idx].append(latency_ms)

    def get_layer_stats(self) -> dict[int, dict]:
        layers = set(self._layer_hits.keys()) | set(self._layer_misses.keys())
        layers |= set(range(self._slot_map_dims[0]))
        result = {}
        for li in sorted(layers):
            h = self._layer_hits.get(li, 0)
            m = self._layer_misses.get(li, 0)
            result[li] = {
                "hits": h,
                "misses": m,
                "hit_rate": h / (h + m) if (h + m) > 0 else 0.0,
                "miss_latency_ms": self._layer_miss_latencies.get(li, []),
            }
        return result

    def get_expert_frequencies(self) -> dict[tuple[int, int], int]:
        return dict(self._expert_access_count)

    def begin_step(self):
        self._step_experts = set()
        self._step_lookups = 0

    def end_step(self) -> dict:
        result = {
            "unique_experts_accessed": len(self._step_experts) if self._step_experts else 0,
            "total_lookups": self._step_lookups,
        }
        self._step_experts = None
        self._step_lookups = 0
        return result

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self._layer_hits.clear()
        self._layer_misses.clear()
        self._layer_miss_latencies.clear()
        self._expert_access_count.clear()

    def clear(self):
        """Evict all entries. Cache is empty after this call."""
        while len(self._policy) > 0:
            key, slot = self._policy.select_evict()
            self._policy.remove(key)
            self._free_slots.append(slot)
        if self._slot_map_cpu is not None:
            self._slot_map_cpu[:] = -1
            self._slot_map_dirty = True
        self.reset_stats()

    def shrink(self, n_slots: int) -> int:
        """Reduce capacity by n_slots. Evicts LRU experts as needed.

        Returns bytes freed (n_slots × expert_bytes).
        """
        if n_slots <= 0 or n_slots > self.capacity:
            return 0

        while len(self._free_slots) < n_slots and len(self._policy) > 0:
            key, slot = self._policy.select_evict()
            self._policy.remove(key)
            self._free_slots.append(slot)
            if self._slot_map_cpu is not None:
                self._slot_map_cpu[key[0], key[1]] = -1
                self._slot_map_dirty = True

        self._free_slots.sort(reverse=True)
        self._free_slots = self._free_slots[n_slots:]

        new_capacity = self.capacity - n_slots
        # Narrow without clone — avoids OOM from temporary double allocation.
        # The view keeps the original tensor alive, but we only access
        # [:new_capacity] positions going forward.
        self._packed = self._packed.narrow(0, 0, new_capacity)
        self.capacity = new_capacity

        if self._slot_map_dirty:
            self.flush_slot_updates()

        return n_slots * self.expert_bytes

    def grow(self, n_slots: int) -> None:
        """Increase capacity by n_slots."""
        if n_slots <= 0:
            return
        old_capacity = self.capacity
        new_capacity = old_capacity + n_slots
        new_packed = torch.empty(
            new_capacity, self.expert_bytes, dtype=torch.uint8, device=self.device
        )
        new_packed[:old_capacity] = self._packed
        self._packed = new_packed
        self.capacity = new_capacity
        self._free_slots.extend(range(old_capacity, new_capacity))
