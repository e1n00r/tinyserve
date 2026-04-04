"""Pinned-memory LRU cache for expert weights on CPU RAM.

Sits between SSD/disk and GPU VRAM. Expert blobs are loaded into pinned
host memory slots so that subsequent H2D transfers via cudaMemcpyAsync
can proceed at full PCIe bandwidth without OS page-fault overhead.

Thread-safe: all LRU state mutations are protected by a single lock.
Async prefetch uses a bounded ThreadPoolExecutor for SSD reads.
"""

from __future__ import annotations

import ctypes
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .fast_io import FastExpertReader

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
        fast_reader: FastExpertReader | None = None,
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
        self._fast_reader = fast_reader

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
        src: torch.Tensor | None = None,
    ) -> int:
        key = (layer_idx, expert_idx)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]
            slot = self._allocate_slot()
            self._lru[key] = slot
        if self._fast_reader is not None:
            self._fast_reader.read_expert(layer_idx, expert_idx, self._pool[slot])
        else:
            self._pool[slot].copy_(src)
        return slot

    def prefetch_async(
        self,
        layer_idx: int,
        expert_idx: int,
        src: torch.Tensor | None = None,
    ) -> None:
        key = (layer_idx, expert_idx)
        with self._lock:
            if key in self._lru or key in self._pending:
                return
            slot = self._allocate_slot()
            self._lru[key] = slot
            if self._fast_reader is not None:
                future = self._executor.submit(
                    self._fast_reader.read_expert,
                    layer_idx,
                    expert_idx,
                    self._pool[slot],
                )
            else:
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

    def start_background_fill(
        self,
        mmap_data: torch.Tensor | None,
        num_layers: int,
        num_experts: int,
    ) -> threading.Thread:
        """Start daemon thread that loads ALL experts from mmap or pread into cache.

        Uses round-robin across layers (L0E0, L1E0, L2E0, ..., L0E1, L1E1, ...)
        so every layer gets some experts quickly.
        Non-blocking -- inference can proceed while fill runs.

        When a FastExpertReader is configured, mmap_data is ignored and reads
        go through pread (bypassing page fault overhead).
        """
        self._fill_done = threading.Event()

        def _fill():
            try:
                for expert_idx in range(num_experts):
                    for layer_idx in range(num_layers):
                        if (layer_idx, expert_idx) not in self._lru:
                            if self._fast_reader is not None:
                                self.load_sync(layer_idx, expert_idx)
                            else:
                                self.load_sync(layer_idx, expert_idx, mmap_data[layer_idx, expert_idx])
            finally:
                self._fill_done.set()

        thread = threading.Thread(target=_fill, daemon=True)
        thread.start()
        return thread

    @property
    def fill_complete(self) -> bool:
        """True when background fill has finished (or was never started)."""
        evt = getattr(self, "_fill_done", None)
        return evt is None or evt.is_set()

    def wait_for_fill(self, timeout: float = 0.05) -> bool:
        """Wait briefly for background fill to finish. Returns True if done."""
        evt = getattr(self, "_fill_done", None)
        if evt is None:
            return True
        return evt.wait(timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
