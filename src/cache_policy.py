"""Pluggable eviction policies for the expert VRAM cache.

All policies share the same interface so GenericLRUCache and ExpertLRUCache
can swap them at construction time with no other code changes.

Policy responsibilities:
  - Track which (layer, expert) key maps to which slot
  - Decide which entry to evict when all slots are full
  - Update internal state on hit/miss/evict

Slot allocation (free list) stays in the cache class, not the policy.
"""

import heapq
from abc import ABC, abstractmethod
from collections import OrderedDict, deque


class CachePolicy(ABC):
    @abstractmethod
    def lookup(self, key: tuple) -> int | None:
        """Return slot if key is cached (update recency), else None."""

    @abstractmethod
    def insert(self, key: tuple, slot: int) -> None:
        """Register key→slot after a miss is resolved."""

    @abstractmethod
    def select_evict(self) -> tuple[tuple, int]:
        """Return (key, slot) to evict. Does NOT remove from state."""

    @abstractmethod
    def remove(self, key: tuple) -> int | None:
        """Remove key from policy state. Return its slot or None."""

    @abstractmethod
    def __len__(self) -> int: ...


class LRUPolicy(CachePolicy):
    def __init__(self) -> None:
        self._od: OrderedDict[tuple, int] = OrderedDict()

    def lookup(self, key: tuple) -> int | None:
        if key not in self._od:
            return None
        self._od.move_to_end(key)
        return self._od[key]

    def insert(self, key: tuple, slot: int) -> None:
        self._od[key] = slot
        self._od.move_to_end(key)

    def select_evict(self) -> tuple[tuple, int]:
        k, s = next(iter(self._od.items()))
        return k, s

    def remove(self, key: tuple) -> int | None:
        return self._od.pop(key, None)

    def __len__(self) -> int:
        return len(self._od)


class SLRUPolicy(CachePolicy):
    def __init__(self, capacity: int) -> None:
        self._n_protected = max(1, int(capacity * 0.8))
        self._n_probationary = capacity - self._n_protected
        self._protected: OrderedDict[tuple, int] = OrderedDict()
        self._probationary: OrderedDict[tuple, int] = OrderedDict()

    def lookup(self, key: tuple) -> int | None:
        if key in self._protected:
            self._protected.move_to_end(key)
            return self._protected[key]
        if key in self._probationary:
            slot = self._probationary.pop(key)
            if len(self._protected) >= self._n_protected:
                demote_key, demote_slot = next(iter(self._protected.items()))
                del self._protected[demote_key]
                self._probationary[demote_key] = demote_slot
                self._probationary.move_to_end(demote_key)
            self._protected[key] = slot
            self._protected.move_to_end(key)
            return slot
        return None

    def insert(self, key: tuple, slot: int) -> None:
        self._probationary[key] = slot
        self._probationary.move_to_end(key)

    def select_evict(self) -> tuple[tuple, int]:
        if self._probationary:
            k, s = next(iter(self._probationary.items()))
            return k, s
        k, s = next(iter(self._protected.items()))
        return k, s

    def remove(self, key: tuple) -> int | None:
        if key in self._probationary:
            return self._probationary.pop(key)
        if key in self._protected:
            return self._protected.pop(key)
        return None

    def __len__(self) -> int:
        return len(self._protected) + len(self._probationary)


class LFUPolicy(CachePolicy):
    def __init__(self) -> None:
        self._data: dict[tuple, tuple[int, int]] = {}
        self._heap: list[tuple[int, tuple]] = []
        self._counter = 0

    def lookup(self, key: tuple) -> int | None:
        if key not in self._data:
            return None
        slot, count = self._data[key]
        new_count = count + 1
        self._data[key] = (slot, new_count)
        heapq.heappush(self._heap, (new_count, key))
        return slot

    def insert(self, key: tuple, slot: int) -> None:
        self._data[key] = (slot, 1)
        heapq.heappush(self._heap, (1, key))

    def select_evict(self) -> tuple[tuple, int]:
        while self._heap:
            count, key = self._heap[0]
            if key in self._data and self._data[key][1] == count:
                slot = self._data[key][0]
                return key, slot
            heapq.heappop(self._heap)
        raise RuntimeError("select_evict called on empty LFUPolicy")

    def remove(self, key: tuple) -> int | None:
        if key not in self._data:
            return None
        slot, _ = self._data.pop(key)
        return slot

    def __len__(self) -> int:
        return len(self._data)


class FIFOPolicy(CachePolicy):
    def __init__(self) -> None:
        self._order: deque[tuple] = deque()
        self._slots: dict[tuple, int] = {}

    def lookup(self, key: tuple) -> int | None:
        return self._slots.get(key)

    def insert(self, key: tuple, slot: int) -> None:
        self._slots[key] = slot
        self._order.append(key)

    def select_evict(self) -> tuple[tuple, int]:
        key = self._order[0]
        return key, self._slots[key]

    def remove(self, key: tuple) -> int | None:
        if key not in self._slots:
            return None
        slot = self._slots.pop(key)
        try:
            self._order.remove(key)
        except ValueError:
            pass
        return slot

    def __len__(self) -> int:
        return len(self._slots)


def make_policy(name: str, capacity: int) -> CachePolicy:
    """Create a policy by name. name: 'lru' | 'slru' | 'lfu' | 'fifo'"""
    if name == "lru":
        return LRUPolicy()
    if name == "slru":
        return SLRUPolicy(capacity)
    if name == "lfu":
        return LFUPolicy()
    if name == "fifo":
        return FIFOPolicy()
    raise ValueError(f"Unknown cache policy: {name!r}. Choose from 'lru', 'slru', 'lfu', 'fifo'.")
