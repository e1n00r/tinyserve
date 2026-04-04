"""Tests for Cython-accelerated cache operations."""

import time

import pytest

# Skip if Cython extension not compiled
try:
    from tinyserve._fast_cache import classify_hits_misses, group_tokens_by_expert, lfru_select_evict

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

pytestmark = pytest.mark.skipif(not HAS_CYTHON, reason="Cython extension not compiled")


def test_lfru_select_evict_correctness():
    """Cython eviction matches Python LFRU."""
    data = {
        (0, 0): [0, 10, 100],  # freq=10, last=100
        (0, 1): [1, 1, 99],  # freq=1, last=99
        (0, 2): [2, 5, 50],  # freq=5, last=50
    }
    clock = 100
    key, slot = lfru_select_evict(data, clock)
    # (0,2) has score 5/(100-50+1) = 0.098
    # (0,1) has score 1/(100-99+1) = 0.5
    # (0,0) has score 10/(100-100+1) = 10.0
    # (0,2) has lowest score
    assert key == (0, 2)
    assert slot == 2


def test_lfru_select_evict_speed():
    """Cython eviction should be >3x faster than Python on 238 entries."""
    data = {(0, i): [i, i % 10 + 1, 1000 - i] for i in range(238)}
    clock = 1000

    # Python baseline
    t0 = time.perf_counter()
    for _ in range(1000):
        best_key = None
        best_score = float("inf")
        for k, (slot, freq, last) in data.items():
            age = clock - last + 1
            score = freq / age
            if score < best_score:
                best_score = score
                best_key = k
    py_time = time.perf_counter() - t0

    # Cython
    t0 = time.perf_counter()
    for _ in range(1000):
        lfru_select_evict(data, clock)
    cy_time = time.perf_counter() - t0

    speedup = py_time / cy_time
    print(f"\nPython: {py_time * 1000:.1f}ms, Cython: {cy_time * 1000:.1f}ms, speedup: {speedup:.1f}x")
    assert speedup > 3.0, f"Expected >3x speedup, got {speedup:.1f}x"


def test_classify_hits_misses():
    eids = [0, 1, 2, 3]
    slots = [5, -1, 3, -1]
    hits, misses = classify_hits_misses(eids, slots)
    assert hits == [(0, 5), (2, 3)]
    assert misses == [1, 3]


def test_group_tokens_by_expert():
    eid_list = [[0, 1], [0, 2], [1, 2]]
    groups = group_tokens_by_expert(eid_list, 3, 2)
    assert 0 in groups
    assert len(groups[0]) == 2  # tokens 0 and 1 route to expert 0
    assert groups[0] == [(0, 0), (1, 0)]
