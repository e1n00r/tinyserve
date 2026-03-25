"""Pinned-memory LRU cache for expert weights on CPU RAM.

Sits between SSD/disk and GPU VRAM. Expert blobs are loaded into pinned
host memory slots so that subsequent H2D transfers via cudaMemcpyAsync
can proceed at full PCIe bandwidth without OS page-fault overhead.

Thread-safe: all LRU state mutations are protected by a single lock.
Async prefetch uses a bounded ThreadPoolExecutor for SSD reads.
"""

import ctypes
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor

import torch

MADV_WILLNEED = 3
_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    return _libc


def madvise_willneed(src: torch.Tensor) -> None:
    """Call madvise(MADV_WILLNEED) on a tensor's memory region.

    Triggers kernel async readahead so page cache is primed before the
    actual copy. Safe to call on non-mmap memory (returns silently on error).
    """
    libc = _get_libc()
    addr = ctypes.c_void_p(src.data_ptr())
    length = ctypes.c_size_t(src.numel() * src.element_size())
    libc.madvise(addr, length, ctypes.c_int(MADV_WILLNEED))


class RAMCache:
    """Pinned-memory LRU cache for expert weight blobs.

    Args:
        num_slots: number of expert-sized slots in the pool.
        expert_bytes: size of one expert blob in bytes.
        max_workers: ThreadPoolExecutor size for async SSD reads.
    """

    def __init__(
        self,
        num_slots: int,
        expert_bytes: int,
        max_workers: int = 4,
    ):
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        if expert_bytes <= 0:
            raise ValueError(f"expert_bytes must be positive, got {expert_bytes}")

        self.num_slots = num_slots
        self.expert_bytes = expert_bytes
        pool = torch.empty(num_slots, expert_bytes, dtype=torch.uint8)
        if torch.cuda.is_available():
            pool = pool.pin_memory()
        self._pool = pool

        self._lock = threading.Lock()
        self._lru: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._free_slots: list[int] = list(range(num_slots - 1, -1, -1))
        self._pending: dict[tuple[int, int], Future] = {}

        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self.hits = 0
        self.misses = 0

    def _allocate_slot(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        evict_key, slot = self._lru.popitem(last=False)
        return slot

    def lookup(self, layer_idx: int, expert_idx: int) -> int | None:
        key = (layer_idx, expert_idx)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                self.hits += 1
                return self._lru[key]
            self.misses += 1
            return None

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        key = (layer_idx, expert_idx)
        with self._lock:
            return key in self._lru

    def get_slot_data(self, slot: int) -> torch.Tensor:
        return self._pool[slot]

    def load_sync(
        self,
        layer_idx: int,
        expert_idx: int,
        src: torch.Tensor,
    ) -> int:
        key = (layer_idx, expert_idx)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]
            slot = self._allocate_slot()
            self._lru[key] = slot
        self._pool[slot].copy_(src)
        return slot

    def prefetch_async(
        self,
        layer_idx: int,
        expert_idx: int,
        src: torch.Tensor,
    ) -> None:
        key = (layer_idx, expert_idx)
        with self._lock:
            if key in self._lru or key in self._pending:
                return
            slot = self._allocate_slot()
            self._lru[key] = slot
            # Prime kernel page cache before the threadpool copy.
            madvise_willneed(src)
            future = self._executor.submit(self._pool[slot].copy_, src)
            self._pending[key] = future

    def wait_pending(self, layer_idx: int, expert_idx: int) -> None:
        key = (layer_idx, expert_idx)
        with self._lock:
            future = self._pending.pop(key, None)
        if future is not None:
            future.result()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
