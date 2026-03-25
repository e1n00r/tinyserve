"""Tests for StaticKVCache."""

import torch
import pytest
from tinyserve.static_kv_cache import StaticKVCache


def test_basic_update_and_read():
    cache = StaticKVCache(
        max_seq_len=64, num_layers=2, num_kv_heads=4,
        head_dim=8, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(1) == 0

    k = torch.randn(1, 4, 3, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 3, 8, dtype=torch.bfloat16)

    # Layer 0 update with cache_position
    k_out, v_out = cache.update(k, v, layer_idx=0,
                                 cache_kwargs={"cache_position": torch.tensor([0, 1, 2])})
    assert k_out.shape == (1, 4, 3, 8)
    assert torch.equal(k_out, k)

    # Layer 1 update
    cache.update(k, v, layer_idx=1,
                 cache_kwargs={"cache_position": torch.tensor([0, 1, 2])})
    assert cache.get_seq_length(1) == 3


def test_sequential_decode():
    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    # Prefill 5 tokens
    k = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    cache.update(k, v, 0, {"cache_position": torch.arange(5)})
    assert cache.get_seq_length(0) == 5

    # Decode token 6
    k1 = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
    v1 = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k1, v1, 0, {"cache_position": torch.tensor([5])})
    assert k_out.shape == (1, 2, 6, 4)
    assert cache.get_seq_length(0) == 6


def test_overflow_raises():
    cache = StaticKVCache(
        max_seq_len=4, num_layers=1, num_kv_heads=1,
        head_dim=2, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    k = torch.randn(1, 1, 5, 2, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 5, 2, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="overflow"):
        cache.update(k, v, 0, {"cache_position": torch.arange(5)})


def test_reset():
    cache = StaticKVCache(
        max_seq_len=16, num_layers=2, num_kv_heads=1,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    k = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    cache.update(k, v, 0, {"cache_position": torch.arange(3)})
    cache.update(k, v, 1, {"cache_position": torch.arange(3)})
    assert cache.get_seq_length(0) == 3
    cache.reset()
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(1) == 0


def test_fp8_quantization():
    cache = StaticKVCache(
        max_seq_len=16, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.float8_e4m3fn,
    )
    k = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, 0, {"cache_position": torch.arange(3)})
    # Output should be dequantized back to bf16
    assert k_out.dtype == torch.bfloat16
    # Internal storage should be fp8
    assert cache._k.dtype == torch.float8_e4m3fn
    # Values should be approximately equal (fp8 loses precision)
    assert torch.allclose(k_out, k, atol=0.2)


def test_bytes_per_token():
    # BF16: 2 bytes per element, 2 (K+V) * heads * head_dim * 2
    bpt = StaticKVCache.bytes_per_token(24, 8, 64, torch.bfloat16)
    assert bpt == 24 * 2 * 8 * 64 * 2  # 49152

    # FP8: 1 byte per element
    bpt_fp8 = StaticKVCache.bytes_per_token(24, 8, 64, torch.float8_e4m3fn)
    assert bpt_fp8 == 24 * 2 * 8 * 64 * 1  # 24576
    assert bpt_fp8 == bpt // 2


def test_iter_and_getitem():
    cache = StaticKVCache(
        max_seq_len=8, num_layers=2, num_kv_heads=1,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    k = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    cache.update(k, v, 0, {"cache_position": torch.arange(3)})
    cache.update(k, v, 1, {"cache_position": torch.arange(3)})

    # __len__
    assert len(cache) == 2

    # __getitem__
    k0, v0 = cache[0]
    assert k0.shape == (1, 1, 3, 4)

    # __iter__
    items = list(cache)
    assert len(items) == 2


def test_mask_sizes():
    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=1,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    # Before any tokens
    kv_len, offset = cache.get_mask_sizes(torch.tensor([0, 1, 2]), layer_idx=0)
    assert kv_len == 3  # just the new tokens
    assert offset == 0

    # After prefill
    k = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    cache.update(k, v, 0, {"cache_position": torch.arange(3)})

    kv_len, offset = cache.get_mask_sizes(torch.tensor([3]), layer_idx=0)
    assert kv_len == 4  # 3 past + 1 new


def test_static_shapes_returns_full_tensors():
    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
        static_shapes=True,
    )
    k = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, 0, {"cache_position": torch.arange(5)})
    # static_shapes=True: always returns max_seq_len-sized tensors
    assert k_out.shape == (1, 2, 32, 4)
    assert v_out.shape == (1, 2, 32, 4)
    assert cache.get_seq_length(0) == 5
    # Valid data matches input
    assert torch.equal(k_out[:, :, :5], k)
    # Padding is zero
    assert (k_out[:, :, 5:] == 0).all()

    # Decode token — shape stays constant
    k1 = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
    v1 = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
    k_out2, v_out2 = cache.update(k1, v1, 0, {"cache_position": torch.tensor([5])})
    assert k_out2.shape == (1, 2, 32, 4)
    assert cache.get_seq_length(0) == 6


def test_static_shapes_default_off():
    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    assert not cache.static_shapes
    k = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    k_out, _ = cache.update(k, v, 0, {"cache_position": torch.arange(5)})
    # Default: returns sliced tensors
    assert k_out.shape == (1, 2, 5, 4)
