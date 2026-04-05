"""Tests for per-layer cache statistics."""

import torch


def _make_cache(capacity=4, num_layers=3, num_experts=8, device="cpu"):
    from tinyserve.expert_store import ExpertCache

    expert_bytes = 64
    cache = ExpertCache(capacity, expert_bytes, torch.device(device), num_layers=num_layers, num_experts=num_experts)
    return cache


def test_cache_tracks_per_layer_hit_miss_counts():
    cache = _make_cache()
    # Miss on layer 0, expert 0
    assert cache.locate((0, 0)) is None
    cache.claim_slot_for((0, 0))
    # Hit on layer 0, expert 0
    assert cache.locate((0, 0)) is not None
    # Miss on layer 1, expert 0
    assert cache.locate((1, 0)) is None

    stats = cache.get_layer_stats()
    assert stats[0]["hits"] == 1
    assert stats[0]["misses"] == 1
    assert stats[1]["hits"] == 0
    assert stats[1]["misses"] == 1


def test_cache_records_miss_latency_per_layer():
    cache = _make_cache()
    # Simulate a miss — locate returns None, record latency
    cache.locate((0, 0))
    cache.record_miss_latency(0, 5.0)  # 5ms

    stats = cache.get_layer_stats()
    assert stats[0]["miss_latency_ms"] == [5.0]


def test_stats_reset_zeros_all_per_layer_counters():
    cache = _make_cache()
    cache.locate((0, 0))
    cache.reset_stats()
    stats = cache.get_layer_stats()
    assert stats[0]["hits"] == 0
    assert stats[0]["misses"] == 0


def test_expert_frequency_counts_both_hits_and_misses():
    cache = _make_cache()
    # Access expert 3 on layer 0 three times
    for _ in range(3):
        if cache.locate((0, 3)) is None:
            cache.claim_slot_for((0, 3))

    freq = cache.get_expert_frequencies()
    assert freq[(0, 3)] == 3  # 1 miss + 2 hits = 3 accesses


def test_step_stats_deduplicate_experts_within_same_step():
    cache = _make_cache()
    cache.begin_step()
    cache.locate((0, 1))
    cache.locate((0, 2))
    cache.locate((0, 1))  # duplicate in same step
    step_stats = cache.end_step()
    assert step_stats["unique_experts_accessed"] == 2
    assert step_stats["total_lookups"] == 3


def test_slot_map_reflects_all_allocations_after_flush():
    cache = _make_cache(capacity=4, num_layers=2, num_experts=8)
    # Allocate several experts
    for eid in range(4):
        cache.claim_slot_for((0, eid))
    # Slot map should reflect all allocations after flush
    cache.flush_slot_updates()
    for eid in range(4):
        assert cache.locate((0, eid)) is not None
