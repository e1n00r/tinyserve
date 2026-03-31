"""Tests for per-layer cache statistics."""
import time
import torch
import pytest


def _make_cache(capacity=4, num_layers=3, num_experts=8, device="cpu"):
    from tinyserve.generic_store import GenericLRUCache
    expert_bytes = 64
    cache = GenericLRUCache(capacity, expert_bytes, torch.device(device),
                            num_layers=num_layers, num_experts=num_experts)
    return cache


def test_per_layer_hits_misses():
    cache = _make_cache()
    # Miss on layer 0, expert 0
    assert cache.lookup(0, 0) is None
    cache.allocate(0, 0)
    # Hit on layer 0, expert 0
    assert cache.lookup(0, 0) is not None
    # Miss on layer 1, expert 0
    assert cache.lookup(1, 0) is None

    stats = cache.get_layer_stats()
    assert stats[0]["hits"] == 1
    assert stats[0]["misses"] == 1
    assert stats[1]["hits"] == 0
    assert stats[1]["misses"] == 1


def test_per_layer_miss_latency():
    cache = _make_cache()
    # Simulate a miss — lookup returns None, record latency
    cache.lookup(0, 0)
    cache.record_miss_latency(0, 5.0)  # 5ms

    stats = cache.get_layer_stats()
    assert stats[0]["miss_latency_ms"] == [5.0]


def test_reset_layer_stats():
    cache = _make_cache()
    cache.lookup(0, 0)
    cache.reset_stats()
    stats = cache.get_layer_stats()
    assert stats[0]["hits"] == 0
    assert stats[0]["misses"] == 0


def test_expert_frequency_tracking():
    cache = _make_cache()
    # Access expert 3 on layer 0 three times
    for _ in range(3):
        if cache.lookup(0, 3) is None:
            cache.allocate(0, 3)

    freq = cache.get_expert_frequencies()
    assert freq[(0, 3)] == 3  # 1 miss + 2 hits = 3 accesses


def test_unique_experts_per_step():
    cache = _make_cache()
    cache.begin_step()
    cache.lookup(0, 1)
    cache.lookup(0, 2)
    cache.lookup(0, 1)  # duplicate in same step
    step_stats = cache.end_step()
    assert step_stats["unique_experts_accessed"] == 2
    assert step_stats["total_lookups"] == 3
