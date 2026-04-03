"""Pluggable eviction policies for the expert VRAM cache.

All policies share the same interface so ExpertCache and ExpertLRUCache
can swap them at construction time with no other code changes.

Policy responsibilities:
  - Track which (layer, expert) key maps to which slot
  - Decide which entry to evict when all slots are full
  - Update internal state on hit/miss/evict

Slot allocation (free list) stays in the cache class, not the policy.
"""

import heapq
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, deque

logger = logging.getLogger(__name__)


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
    def contains(self, key: tuple) -> bool:
        """Return True if key is cached. Does NOT update recency/frequency state."""

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

    def contains(self, key: tuple) -> bool:
        return key in self._od

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

    def contains(self, key: tuple) -> bool:
        return key in self._probationary or key in self._protected

    def __len__(self) -> int:
        return len(self._protected) + len(self._probationary)


class LFUPolicy(CachePolicy):
    def __init__(self) -> None:
        self._data: dict[tuple, tuple[int, int]] = {}
        self._heap: list[tuple[int, tuple]] = []

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

    def contains(self, key: tuple) -> bool:
        return key in self._data

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

    def contains(self, key: tuple) -> bool:
        return key in self._slots

    def __len__(self) -> int:
        return len(self._slots)


class LFRUPolicy(CachePolicy):
    """Frequency-Recency hybrid: evict the entry with the lowest freq/age ratio.

    Score = freq / age  (higher = more valuable to keep).
    Ties broken by recency.  Dominates both pure LRU (does not evict hot entries
    that were last used 2 tokens ago) and pure LFU (does not keep stale entries
    from early prefill).
    """

    def __init__(self) -> None:
        self._data: dict[tuple, list] = {}  # key -> [slot, freq, clock]
        self._clock = 0

    def lookup(self, key: tuple) -> int | None:
        if key not in self._data:
            return None
        entry = self._data[key]
        self._clock += 1
        entry[1] += 1  # freq
        entry[2] = self._clock  # last access
        return entry[0]

    def insert(self, key: tuple, slot: int) -> None:
        self._clock += 1
        self._data[key] = [slot, 1, self._clock]

    def select_evict(self) -> tuple[tuple, int]:
        try:
            from tinyserve._fast_cache import lfru_select_evict
            return lfru_select_evict(self._data, self._clock)
        except ImportError:
            logger.debug("Cython lfru_select_evict not available, using Python fallback")
        # Lowest score = freq / (clock - last_access + 1)
        # Evict entry where score is minimised.
        best_key = None
        best_score = float("inf")
        for k, (slot, freq, last) in self._data.items():
            age = self._clock - last + 1
            score = freq / age
            if score < best_score:
                best_score = score
                best_key = k
        slot = self._data[best_key][0]
        return best_key, slot

    def remove(self, key: tuple) -> int | None:
        entry = self._data.pop(key, None)
        return entry[0] if entry is not None else None

    def contains(self, key: tuple) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


class LeastStalePolicy(CachePolicy):
    """Least-Stale eviction: stale experts (accessed in a previous forward pass
    but not the current one) are evicted before fresh ones.

    Key insight (SpecMD, arxiv 2602.03921): MoE access within one token is
    sequential and deterministic. Once layer N's experts fire, they won't be
    re-accessed this token. LRU/LFRU keep them as "recently used" — wrong.
    Least-Stale marks them stale at pass end and evicts them first.

    Within the stale tier, FIFO by insertion order (oldest stale expert first).
    Within the fresh tier, also FIFO. This gives 1.6-1.9% collision rate at
    5% cache capacity vs LRU's 4.5-12.6%.

    Call begin_pass() at the start of each new token's forward sweep to rotate
    the fresh set into stale.
    """

    def __init__(self) -> None:
        # fresh: accessed in the CURRENT forward pass (keyed by insertion order)
        self._fresh: OrderedDict[tuple, int] = OrderedDict()
        # stale: accessed in a PREVIOUS pass but not yet evicted
        self._stale: OrderedDict[tuple, int] = OrderedDict()

    def begin_pass(self) -> None:
        """Call once per token before running the forward pass.

        Moves all current fresh entries to stale so that experts loaded for
        the previous token become eviction candidates.
        """
        for key, slot in self._fresh.items():
            self._stale[key] = slot
        self._fresh.clear()

    def lookup(self, key: tuple) -> int | None:
        if key in self._fresh:
            return self._fresh[key]
        if key in self._stale:
            slot = self._stale.pop(key)
            self._fresh[key] = slot
            return slot
        return None

    def insert(self, key: tuple, slot: int) -> None:
        self._fresh[key] = slot

    def select_evict(self) -> tuple[tuple, int]:
        # Evict stale-first (FIFO within stale), then fresh (FIFO within fresh).
        if self._stale:
            k, s = next(iter(self._stale.items()))
            return k, s
        k, s = next(iter(self._fresh.items()))
        return k, s

    def remove(self, key: tuple) -> int | None:
        if key in self._stale:
            return self._stale.pop(key)
        if key in self._fresh:
            return self._fresh.pop(key)
        return None

    def contains(self, key: tuple) -> bool:
        return key in self._fresh or key in self._stale

    def __len__(self) -> int:
        return len(self._fresh) + len(self._stale)


class DALIPolicy(CachePolicy):
    """Workload-aware sliding-window cache (DALI, arxiv 2602.03495).

    Tracks per-expert activation frequency over the last ``window`` token
    forward passes. Experts whose frequency exceeds ``hot_threshold * window``
    are "hot" and are never evicted while they remain above the threshold.
    Cold experts are managed by LRU in the remaining slots.

    Key property: the hot set self-adapts. An expert that was hot during
    prefill but inactive in decode naturally slides out of the protected tier
    as its window count decays, freeing its slot for the new hot set.

    This eliminates the competition between LFRU's frequency retention and
    FATE's cold-expert prefetch: FATE-predicted experts land in the LRU tier
    without displacing the hot set.
    """

    def __init__(self, capacity: int, window: int = 256, hot_threshold: float = 0.1) -> None:
        self._capacity = capacity
        self._min_hot_count = max(1, int(window * hot_threshold))
        self._slots: dict[tuple, int] = {}
        self._freq: dict[tuple, int] = {}
        self._history: deque[tuple] = deque(maxlen=window)
        self._lru: OrderedDict[tuple, int] = OrderedDict()  # cold experts only

    def _is_hot(self, key: tuple) -> bool:
        return self._freq.get(key, 0) >= self._min_hot_count

    def _record_access(self, key: tuple) -> None:
        """Slide the frequency window: increment key, decrement the oldest entry."""
        if len(self._history) == self._history.maxlen:
            old = self._history[0]  # will be dropped by deque maxlen
            old_count = self._freq.get(old, 0) - 1
            if old_count <= 0:
                self._freq.pop(old, None)
            else:
                self._freq[old] = old_count
        self._history.append(key)
        self._freq[key] = self._freq.get(key, 0) + 1

    def lookup(self, key: tuple) -> int | None:
        slot = self._slots.get(key)
        if slot is None:
            return None
        self._record_access(key)
        # Update hot/cold tier membership.
        if self._is_hot(key):
            self._lru.pop(key, None)  # promote out of LRU if it was cold
        else:
            self._lru[key] = slot
            self._lru.move_to_end(key)
        return slot

    def insert(self, key: tuple, slot: int) -> None:
        self._slots[key] = slot
        self._record_access(key)
        if not self._is_hot(key):
            self._lru[key] = slot
            self._lru.move_to_end(key)

    def select_evict(self) -> tuple[tuple, int]:
        # Prefer evicting the least-recently-used cold expert.
        for k in self._lru:
            return k, self._slots[k]
        # All slots hold hot experts (cache too small for hot set) — evict oldest.
        k = next(iter(self._slots))
        return k, self._slots[k]

    def remove(self, key: tuple) -> int | None:
        self._lru.pop(key, None)
        return self._slots.pop(key, None)

    def contains(self, key: tuple) -> bool:
        return key in self._slots

    def __len__(self) -> int:
        return len(self._slots)


def make_policy(name: str, capacity: int) -> CachePolicy:
    """Create a policy by name. name: 'lru' | 'slru' | 'lfu' | 'lfru' | 'fifo' | 'ls' | 'dali'"""
    if name == "lru":
        return LRUPolicy()
    if name == "slru":
        return SLRUPolicy(capacity)
    if name == "lfu":
        return LFUPolicy()
    if name == "lfru":
        return LFRUPolicy()
    if name == "fifo":
        return FIFOPolicy()
    if name == "ls":
        return LeastStalePolicy()
    if name == "dali":
        return DALIPolicy(capacity)
    raise ValueError(f"Unknown cache policy: {name!r}. Choose from 'lru', 'slru', 'lfu', 'lfru', 'fifo', 'ls', 'dali'.")
