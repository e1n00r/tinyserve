"""Tests for dynamic VRAM rebalancing."""

import torch


def _make_cache(capacity=10, num_layers=2, num_experts=8, device="cpu"):
    from tinyserve.expert_store import ExpertCache

    expert_bytes = 64
    cache = ExpertCache(capacity, expert_bytes, torch.device(device), num_layers=num_layers, num_experts=num_experts)
    return cache


def test_cache_shrink_evicts_experts_and_reduces_capacity():
    cache = _make_cache(capacity=10)
    # Fill 8 slots
    for i in range(8):
        cache.allocate(0, i)
    cache.flush_slot_updates()
    assert cache.capacity == 10

    freed = cache.shrink(4)  # shrink by 4 slots
    assert cache.capacity == 6
    assert freed == 4 * 64  # 4 slots × 64 bytes each
    assert len(cache._policy) <= 6


def test_cache_grow_adds_free_slots_and_increases_capacity():
    cache = _make_cache(capacity=6)
    for i in range(6):
        cache.allocate(0, i)
    cache.flush_slot_updates()

    cache.grow(4)  # grow by 4 slots
    assert cache.capacity == 10
    assert len(cache._free_slots) == 4  # 4 new free slots


def test_cache_shrink_below_used_capacity_evicts_lru_experts():
    cache = _make_cache(capacity=8)
    for i in range(8):
        cache.allocate(0, i)
    cache.flush_slot_updates()

    freed = cache.shrink(6)  # shrink by 6, only 0 free → must evict 6
    assert cache.capacity == 2
    assert len(cache._policy) == 2  # only 2 experts remain


def test_cache_shrink_returns_correct_freed_byte_count():
    cache = _make_cache(capacity=10)
    freed = cache.shrink(3)
    assert freed == 3 * 64


def test_cache_capacity_recovers_after_shrink_then_grow():
    cache = _make_cache(capacity=10)
    cache.shrink(5)
    assert cache.capacity == 5
    cache.grow(3)
    assert cache.capacity == 8


def test_kv_cache_extend_increases_max_seq_len():
    from tinyserve.static_kv_cache import StaticKVCache

    cache = StaticKVCache(max_seq_len=100, num_layers=2, num_kv_heads=4, head_dim=32, device=torch.device("cpu"))
    assert cache.max_seq_len == 100

    cache.extend(50)
    assert cache.max_seq_len == 150
    assert cache._k.shape[3] == 150


def test_kv_cache_extend_preserves_existing_key_value_data():
    from tinyserve.static_kv_cache import StaticKVCache

    cache = StaticKVCache(max_seq_len=100, num_layers=2, num_kv_heads=4, head_dim=32, device=torch.device("cpu"))
    # Write some data
    k = torch.randn(1, 4, 10, 32)
    v = torch.randn(1, 4, 10, 32)
    cache.update(k, v, 0)

    old_k = cache._k[0, 0, :, :10, :].clone()
    cache.extend(50)

    # Existing data preserved
    torch.testing.assert_close(cache._k[0, 0, :, :10, :], old_k)


def test_kv_cache_reports_correct_vram_byte_usage():
    from tinyserve.static_kv_cache import StaticKVCache

    cache = StaticKVCache(
        max_seq_len=100, num_layers=2, num_kv_heads=4, head_dim=32, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    expected = 2 * 2 * 1 * 4 * 100 * 32 * 2  # K+V × layers × batch × heads × seq × dim × bf16
    assert cache.vram_bytes == expected


def test_vram_budget_shrinks_experts_when_kv_pressure_is_high():
    from tinyserve.vram_budget import VRAMBudget

    cache = _make_cache(capacity=10)
    for i in range(8):
        cache.allocate(0, i)
    cache.flush_slot_updates()

    from tinyserve.static_kv_cache import StaticKVCache

    kv = StaticKVCache(max_seq_len=100, num_layers=2, num_kv_heads=4, head_dim=32, device=torch.device("cpu"))
    # Simulate KV at 90% capacity
    for li in range(2):
        kv._seq_lens[li] = 90

    budget = VRAMBudget(cache, kv, expert_bytes=64, kv_bytes_per_token=2 * 4 * 32 * 2 * 2)
    action = budget.check()

    assert action["should_rebalance"] is True
    assert action["direction"] == "shrink_experts"
    assert action["expert_slots_to_free"] > 0


def test_vram_budget_takes_no_action_when_usage_is_balanced():
    from tinyserve.vram_budget import VRAMBudget

    cache = _make_cache(capacity=10)
    for i in range(5):
        cache.allocate(0, i)
    cache.flush_slot_updates()

    from tinyserve.static_kv_cache import StaticKVCache

    kv = StaticKVCache(max_seq_len=100, num_layers=2, num_kv_heads=4, head_dim=32, device=torch.device("cpu"))
    kv._seq_lens[0] = 30  # 30% used
    kv._seq_lens[1] = 30

    budget = VRAMBudget(cache, kv, expert_bytes=64, kv_bytes_per_token=2 * 4 * 32 * 2 * 2)
    action = budget.check()

    assert action["should_rebalance"] is False


def test_vram_budget_grows_experts_when_kv_pressure_clears():
    from tinyserve.vram_budget import VRAMBudget

    cache = _make_cache(capacity=6)  # was shrunk from 10

    from tinyserve.static_kv_cache import StaticKVCache

    kv = StaticKVCache(max_seq_len=100, num_layers=2, num_kv_heads=4, head_dim=32, device=torch.device("cpu"))
    kv._seq_lens[0] = 0  # empty after request completion
    kv._seq_lens[1] = 0

    budget = VRAMBudget(cache, kv, expert_bytes=64, kv_bytes_per_token=2 * 4 * 32 * 2 * 2, max_expert_capacity=10)
    action = budget.check()

    assert action["should_rebalance"] is True
    assert action["direction"] == "grow_experts"
