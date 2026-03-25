"""Unit tests for DALIPolicy — workload-aware sliding-window cache."""

from tinyserve.cache_policy import DALIPolicy


def _fill(policy, keys, slot_start=0):
    """Insert keys into policy, returns {key: slot}."""
    slots = {}
    for i, k in enumerate(keys):
        policy.insert(k, slot_start + i)
        slots[k] = slot_start + i
    return slots


def test_hot_expert_never_evicted():
    """Expert accessed > hot_threshold * window times is protected from eviction."""
    policy = DALIPolicy(capacity=3, window=10, hot_threshold=0.3)
    # Access key A 4 times (>= 3 = 0.3 * 10) to make it hot.
    policy.insert((0, 0), 0)
    for _ in range(3):
        policy.lookup((0, 0))
    assert policy._is_hot((0, 0))

    # Fill remaining slots with cold experts.
    policy.insert((0, 1), 1)
    policy.insert((0, 2), 2)

    # Adding a 4th expert must evict a cold one, not the hot (0, 0).
    evict_key, evict_slot = policy.select_evict()
    assert evict_key != (0, 0), "Hot expert should not be evicted"


def test_cold_expert_is_lru_evicted():
    """Among cold experts, the least-recently used is evicted first."""
    policy = DALIPolicy(capacity=3, window=100, hot_threshold=0.5)
    policy.insert((0, 0), 0)
    policy.insert((0, 1), 1)
    policy.insert((0, 2), 2)
    # Access (0, 0) and (0, 2) — (0, 1) is LRU.
    policy.lookup((0, 0))
    policy.lookup((0, 2))

    evict_key, evict_slot = policy.select_evict()
    assert evict_key == (0, 1)
    assert evict_slot == 1


def test_frequency_window_decays():
    """Expert falls below hot threshold after sliding out of the window."""
    window = 8
    policy = DALIPolicy(capacity=4, window=window, hot_threshold=0.5)
    # Make (0, 0) hot: access 5 times (>= 4 = 0.5 * 8).
    policy.insert((0, 0), 0)
    for _ in range(4):
        policy.lookup((0, 0))
    assert policy._is_hot((0, 0))

    # Push window forward by accessing a different key 8 times.
    policy.insert((0, 1), 1)
    for _ in range(window):
        policy.lookup((0, 1))

    # (0, 0)'s old accesses are now outside the window — should no longer be hot.
    assert not policy._is_hot((0, 0))


def test_lookup_miss_returns_none():
    policy = DALIPolicy(capacity=4, window=16, hot_threshold=0.25)
    assert policy.lookup((0, 99)) is None


def test_lookup_hit_returns_slot():
    policy = DALIPolicy(capacity=4, window=16, hot_threshold=0.25)
    policy.insert((1, 3), 7)
    assert policy.lookup((1, 3)) == 7


def test_contains():
    policy = DALIPolicy(capacity=4, window=16, hot_threshold=0.25)
    policy.insert((0, 0), 0)
    assert policy.contains((0, 0))
    assert not policy.contains((0, 1))


def test_remove():
    policy = DALIPolicy(capacity=4, window=16, hot_threshold=0.25)
    policy.insert((0, 0), 0)
    policy.insert((0, 1), 1)
    slot = policy.remove((0, 0))
    assert slot == 0
    assert not policy.contains((0, 0))
    assert len(policy) == 1


def test_hot_set_adapts_across_phases():
    """Hot experts from phase 1 lose protection in phase 2 as window slides."""
    window = 20
    policy = DALIPolicy(capacity=10, window=window, hot_threshold=0.3)

    # Phase 1: experts (0,0) and (0,1) are hot.
    policy.insert((0, 0), 0)
    policy.insert((0, 1), 1)
    for _ in range(7):  # 7 >= 0.3 * 20 = 6
        policy.lookup((0, 0))
        policy.lookup((0, 1))
    assert policy._is_hot((0, 0))
    assert policy._is_hot((0, 1))

    # Phase 2: only (0,2) is active for 20 tokens — window fully slides.
    policy.insert((0, 2), 2)
    for _ in range(window):
        policy.lookup((0, 2))

    # Phase-1 hot experts should have decayed below threshold.
    assert not policy._is_hot((0, 0))
    assert not policy._is_hot((0, 1))
    assert policy._is_hot((0, 2))
