import pytest
import torch

from src.lru_cache import ExpertLRUCache
from src.config import EXPERT_BYTES

from .conftest import requires_cuda


@requires_cuda
def test_hit_and_miss():
    cache = ExpertLRUCache(capacity=4, device=torch.device("cuda"))

    assert cache.lookup(0, 5) is None
    assert cache.misses == 1

    slot = cache.allocate(0, 5)
    assert cache.lookup(0, 5) == slot
    assert cache.hits == 1


@requires_cuda
def test_eviction_order():
    cache = ExpertLRUCache(capacity=3, device=torch.device("cuda"))

    cache.allocate(0, 0)
    evictable_slot = cache.allocate(0, 1)
    cache.allocate(0, 2)
    cache.lookup(0, 0)

    new_slot = cache.allocate(0, 3)
    assert new_slot == evictable_slot

    assert cache.lookup(0, 1) is None
    assert cache.lookup(0, 0) is not None
    assert cache.lookup(0, 2) is not None
    assert cache.lookup(0, 3) is not None


@requires_cuda
def test_hit_rate():
    cache = ExpertLRUCache(capacity=2, device=torch.device("cuda"))
    cache.allocate(0, 0)
    cache.allocate(0, 1)

    cache.lookup(0, 0)
    cache.lookup(0, 1)
    cache.lookup(0, 2)

    assert cache.hit_rate == pytest.approx(2 / 3)


def test_estimate_capacity():
    cap = ExpertLRUCache.estimate_capacity(1024 * 1024 * 1024)
    expected = 1024 * 1024 * 1024 // EXPERT_BYTES
    assert cap == expected
