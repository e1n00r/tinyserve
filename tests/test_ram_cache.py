import threading

import pytest
import torch

from tinyserve.ram_cache import RAMCache


class TestRAMCacheInit:
    def test_zero_slots_raises(self):
        with pytest.raises(ValueError, match="num_slots"):
            RAMCache(num_slots=0, expert_bytes=64)

    def test_zero_bytes_raises(self):
        with pytest.raises(ValueError, match="expert_bytes"):
            RAMCache(num_slots=4, expert_bytes=0)

    def test_negative_slots_raises(self):
        with pytest.raises(ValueError, match="num_slots"):
            RAMCache(num_slots=-1, expert_bytes=64)

    def test_negative_bytes_raises(self):
        with pytest.raises(ValueError, match="expert_bytes"):
            RAMCache(num_slots=4, expert_bytes=-1)

    def test_pool_shape(self):
        cache = RAMCache(num_slots=8, expert_bytes=128)
        assert cache._pool.shape == (8, 128)
        assert cache._pool.dtype == torch.uint8
        cache.shutdown()


class TestLookup:
    def test_miss_returns_none(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        assert cache.lookup(0, 0) is None
        assert cache.misses == 1
        assert cache.hits == 0
        cache.shutdown()

    def test_hit_after_load(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        src = torch.ones(32, dtype=torch.uint8)
        cache.load_sync(0, 0, src)
        slot = cache.lookup(0, 0)
        assert slot is not None
        assert cache.hits == 1
        assert cache.misses == 0
        cache.shutdown()

    def test_hit_increments_stats(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        src = torch.ones(32, dtype=torch.uint8)
        cache.load_sync(0, 0, src)
        cache.lookup(0, 0)
        cache.lookup(0, 0)
        assert cache.hits == 2
        cache.shutdown()


class TestContains:
    def test_no_side_effects(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        src = torch.ones(32, dtype=torch.uint8)
        cache.load_sync(0, 0, src)
        assert cache.contains(0, 0) is True
        assert cache.contains(1, 1) is False
        assert cache.hits == 0
        assert cache.misses == 0
        cache.shutdown()


class TestLRUEviction:
    def test_eviction_order(self):
        cache = RAMCache(num_slots=2, expert_bytes=16)
        cache.load_sync(0, 0, torch.full((16,), 10, dtype=torch.uint8))
        cache.load_sync(0, 1, torch.full((16,), 20, dtype=torch.uint8))
        cache.load_sync(0, 2, torch.full((16,), 30, dtype=torch.uint8))
        assert cache.contains(0, 0) is False
        assert cache.contains(0, 1) is True
        assert cache.contains(0, 2) is True
        cache.shutdown()

    def test_reorder_on_lookup(self):
        cache = RAMCache(num_slots=2, expert_bytes=16)
        cache.load_sync(0, 0, torch.full((16,), 10, dtype=torch.uint8))
        cache.load_sync(0, 1, torch.full((16,), 20, dtype=torch.uint8))
        cache.lookup(0, 0)
        cache.load_sync(0, 2, torch.full((16,), 30, dtype=torch.uint8))
        assert cache.contains(0, 0) is True
        assert cache.contains(0, 1) is False
        assert cache.contains(0, 2) is True
        cache.shutdown()


class TestDataIntegrity:
    def test_load_sync_data(self):
        cache = RAMCache(num_slots=4, expert_bytes=64)
        src = torch.arange(64, dtype=torch.uint8)
        slot = cache.load_sync(0, 0, src)
        stored = cache.get_slot_data(slot)
        assert torch.equal(stored, src)
        cache.shutdown()

    def test_idempotent_load_sync(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        src = torch.ones(32, dtype=torch.uint8)
        slot_a = cache.load_sync(0, 0, src)
        slot_b = cache.load_sync(0, 0, src)
        assert slot_a == slot_b
        assert len(cache._lru) == 1
        cache.shutdown()


class TestPrefetchAsync:
    def test_round_trip(self):
        cache = RAMCache(num_slots=4, expert_bytes=64)
        src = torch.arange(64, dtype=torch.uint8)
        cache.prefetch_async(0, 0, src)
        cache.wait_pending(0, 0)
        slot = cache.lookup(0, 0)
        assert slot is not None
        assert torch.equal(cache.get_slot_data(slot), src)
        cache.shutdown()

    def test_dedup(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        src = torch.ones(32, dtype=torch.uint8)
        cache.prefetch_async(0, 0, src)
        cache.prefetch_async(0, 0, src)
        assert len(cache._lru) == 1
        cache.wait_pending(0, 0)
        cache.shutdown()

    def test_wait_pending_noop(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        cache.wait_pending(0, 0)
        cache.shutdown()


class TestHitRateAndStats:
    def test_hit_rate_empty(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        assert cache.hit_rate == 0.0
        cache.shutdown()

    def test_hit_rate_computed(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        src = torch.ones(32, dtype=torch.uint8)
        cache.load_sync(0, 0, src)
        cache.lookup(0, 0)
        cache.lookup(0, 1)
        assert cache.hit_rate == pytest.approx(0.5)
        cache.shutdown()

    def test_reset_stats(self):
        cache = RAMCache(num_slots=4, expert_bytes=32)
        src = torch.ones(32, dtype=torch.uint8)
        cache.load_sync(0, 0, src)
        cache.lookup(0, 0)
        cache.lookup(0, 1)
        cache.reset_stats()
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.hit_rate == 0.0
        cache.shutdown()


class TestConcurrentLoads:
    def test_16_threads(self):
        cache = RAMCache(num_slots=32, expert_bytes=64)
        barrier = threading.Barrier(16)
        errors: list[str] = []

        def worker(tid: int):
            try:
                barrier.wait(timeout=5)
                src = torch.full((64,), tid, dtype=torch.uint8)
                slot = cache.load_sync(0, tid, src)
                stored = cache.get_slot_data(slot)
                if not torch.equal(stored, src):
                    errors.append(f"thread {tid}: data mismatch")
            except Exception as exc:
                errors.append(f"thread {tid}: {exc}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, errors
        assert len(cache._lru) == 16
        cache.shutdown()
