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


def test_cpu_storage_gpu_return():
    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
        storage_device=torch.device("cpu"),
    )
    k = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, 0, {"cache_position": torch.arange(5)})
    assert cache._k.device.type == "cpu"
    assert cache._v.device.type == "cpu"
    assert k_out.device.type == "cpu"
    assert v_out.device.type == "cpu"
    assert torch.equal(k_out, k)


def test_cpu_storage_vram_bytes():
    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
        storage_device=torch.device("cpu"),
    )
    # vram_bytes reports actual bytes regardless of storage device
    expected = 2 * 1 * 1 * 2 * 32 * 4 * 2  # K+V × layers × batch × heads × seq × dim × bf16
    assert cache.vram_bytes == expected


def test_cpu_storage_sequential_decode():
    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
        storage_device=torch.device("cpu"),
    )
    k = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    cache.update(k, v, 0, {"cache_position": torch.arange(5)})
    assert cache.get_seq_length(0) == 5

    k1 = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
    v1 = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k1, v1, 0, {"cache_position": torch.tensor([5])})
    assert k_out.shape == (1, 2, 6, 4)
    assert cache.get_seq_length(0) == 6


def test_cpu_storage_fp8():
    cache = StaticKVCache(
        max_seq_len=16, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.float8_e4m3fn,
        storage_device=torch.device("cpu"),
    )
    k = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, 0, {"cache_position": torch.arange(3)})
    assert cache._k.dtype == torch.float8_e4m3fn
    assert k_out.dtype == torch.bfloat16
    assert torch.allclose(k_out, k, atol=0.2)


def test_cpu_storage_static_shapes():
    cache = StaticKVCache(
        max_seq_len=16, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
        storage_device=torch.device("cpu"), static_shapes=True,
    )
    k = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, 0, {"cache_position": torch.arange(3)})
    assert k_out.shape == (1, 2, 16, 4)
    assert torch.equal(k_out[:, :, :3], k)


def test_from_model_config_storage_device():
    class FakeConfig:
        num_hidden_layers = 2
        num_key_value_heads = 4
        num_attention_heads = 8
        hidden_size = 64
        head_dim = 8

    cache = StaticKVCache.from_model_config(
        FakeConfig(), max_seq_len=32, device="cpu", dtype=torch.bfloat16,
        storage_device="cpu",
    )
    assert cache._storage_device == torch.device("cpu")
    assert cache._compute_device == torch.device("cpu")
    expected = 2 * 2 * 1 * 4 * 32 * 8 * 2  # K+V × layers × batch × heads × seq × dim × bf16
    assert cache.vram_bytes == expected


def test_default_storage_device_matches_device():
    cache = StaticKVCache(
        max_seq_len=16, num_layers=1, num_kv_heads=1,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    assert cache._storage_device == torch.device("cpu")
    assert cache._compute_device == torch.device("cpu")
    expected = 2 * 1 * 1 * 1 * 16 * 4 * 2  # K+V × layers × batch × heads × seq × dim × bf16
    assert cache.vram_bytes == expected


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


def test_from_model_config_sliding_window_with_layer_types():
    """from_model_config populates is_sliding and _sliding_window from layer_types."""
    class FakeConfig:
        num_hidden_layers = 4
        num_key_value_heads = 2
        num_attention_heads = 4
        hidden_size = 32
        head_dim = 8
        sliding_window = 16
        layer_types = [
            "sliding_attention", "full_attention",
            "sliding_attention", "full_attention",
        ]

    cache = StaticKVCache.from_model_config(
        FakeConfig(), max_seq_len=64, device="cpu", dtype=torch.bfloat16,
    )
    assert cache._sliding_window == 16
    assert cache.is_sliding == [True, False, True, False]


def test_from_model_config_sliding_window_no_layer_types():
    """from_model_config marks all layers sliding when no layer_types present."""
    class FakeConfig:
        num_hidden_layers = 3
        num_key_value_heads = 2
        num_attention_heads = 4
        hidden_size = 32
        head_dim = 8
        sliding_window = 32

    cache = StaticKVCache.from_model_config(
        FakeConfig(), max_seq_len=64, device="cpu", dtype=torch.bfloat16,
    )
    assert cache._sliding_window == 32
    assert cache.is_sliding == [True, True, True]


def test_from_model_config_no_sliding_window():
    """from_model_config leaves is_sliding all False when no sliding_window."""
    class FakeConfig:
        num_hidden_layers = 2
        num_key_value_heads = 2
        num_attention_heads = 4
        hidden_size = 32
        head_dim = 8

    cache = StaticKVCache.from_model_config(
        FakeConfig(), max_seq_len=64, device="cpu", dtype=torch.bfloat16,
    )
    assert cache._sliding_window is None
    assert cache.is_sliding == [False, False]


def test_cpu_offload_sliding_layer_returns_window_tokens():
    """CPU-offload sliding layer returns only window tokens, not full sequence.

    Uses device='meta' as the compute device so that storage (cpu) != compute
    (meta), triggering the bandwidth-reduction path without a real GPU.
    The meta device accepts .to() calls and preserves shape, making it
    suitable for verifying slice dimensions.
    """
    window = 4
    cache = StaticKVCache(
        max_seq_len=32, num_layers=2, num_kv_heads=2,
        head_dim=4, device=torch.device("meta"), dtype=torch.bfloat16,
        storage_device=torch.device("cpu"),
    )
    cache._sliding_window = window
    cache.is_sliding = [True, False]

    k = torch.randn(1, 2, 10, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 10, 4, dtype=torch.bfloat16)

    # Sliding layer: must return only the last `window` tokens
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == (1, 2, window, 4), f"Expected window={window}, got {k_out.shape[2]}"
    assert v_out.shape == (1, 2, window, 4)

    # Full-attention layer: must return all 10 tokens
    k_out2, v_out2 = cache.update(k, v, layer_idx=1)
    assert k_out2.shape == (1, 2, 10, 4)


def test_cpu_offload_sliding_layer_seq_within_window_returns_all():
    """When seq_len <= window, sliding layer still returns all tokens."""
    window = 16
    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("meta"), dtype=torch.bfloat16,
        storage_device=torch.device("cpu"),
    )
    cache._sliding_window = window
    cache.is_sliding = [True]

    # Prefill 8 tokens (< window)
    k = torch.randn(1, 2, 8, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 8, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == (1, 2, 8, 4)


def test_no_offload_sliding_layer_returns_full_seq():
    """Without CPU offload, sliding layer always returns full sequence (SDPA slices later)."""
    window = 4
    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    cache._sliding_window = window
    cache.is_sliding = [True]

    k = torch.randn(1, 2, 10, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 10, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    # storage == compute (both cpu), so no bandwidth optimization — full seq returned
    assert k_out.shape == (1, 2, 10, 4)


# ---------------------------------------------------------------------------
# StreamingLLM eviction tests
# ---------------------------------------------------------------------------

def test_streaming_eviction_self_heals_on_overflow():
    """update() self-heals via streaming eviction when cache would overflow.

    Scenario: max_seq_len=256, window=200, sink=4.
    After the first 256 tokens, the next write would overflow.
    With streaming enabled, update() must evict and compact first,
    then write the new tokens without raising KVCacheOverflow.

    This reproduces the GPU KV + StreamingLLM index-out-of-bounds bug:
    previously _evict_streaming was dead code (never called from update()),
    so chunked prefill at >256 ctx raised KVCacheOverflow.
    """
    max_seq_len = 256
    sink = 4
    window = 200
    cache = StaticKVCache(
        max_seq_len=max_seq_len, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    cache.enable_streaming(sink_size=sink, window_size=window)

    # Fill the cache to capacity (256 tokens in two chunks)
    k_fill = torch.randn(1, 2, max_seq_len, 4, dtype=torch.bfloat16)
    v_fill = torch.randn(1, 2, max_seq_len, 4, dtype=torch.bfloat16)
    cache.update(k_fill, v_fill, layer_idx=0)
    assert cache.get_seq_length(0) == max_seq_len

    # This next write (50 more tokens) would overflow — but streaming must self-heal.
    k_new = torch.randn(1, 2, 50, 4, dtype=torch.bfloat16)
    v_new = torch.randn(1, 2, 50, 4, dtype=torch.bfloat16)
    # Must NOT raise KVCacheOverflow
    k_out, v_out = cache.update(k_new, v_new, layer_idx=0)

    # After eviction + write, seq_len = sink + window (compacted) + 50 new tokens
    expected_len = sink + window + 50
    assert cache.get_seq_length(0) == expected_len, (
        f"Expected seq_len={expected_len}, got {cache.get_seq_length(0)}"
    )
    # Returned KV must include all tokens written so far
    assert k_out.shape[2] == expected_len


def test_streaming_eviction_repeated_overflow_self_heals():
    """Multiple overflows in sequence all self-heal via streaming eviction."""
    max_seq_len = 256
    sink = 4
    window = 200
    cache = StaticKVCache(
        max_seq_len=max_seq_len, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    cache.enable_streaming(sink_size=sink, window_size=window)

    # Fill cache twice (each time eviction re-compacts to sink+window=204)
    chunk_size = 52  # sink(4) + window(200) + chunk(52) = 256 exactly fills max_seq_len
    for _ in range(5):
        k = torch.randn(1, 2, chunk_size, 4, dtype=torch.bfloat16)
        v = torch.randn(1, 2, chunk_size, 4, dtype=torch.bfloat16)
        cache.update(k, v, layer_idx=0)

    # After 5 chunks of 52: after first fills to 256, evicts to 204, adds 52 → 256,
    # evicts to 204, etc. Final seq_len depends on last evict + last chunk.
    # Crucially: no KVCacheOverflow raised during any of these.
    assert cache.get_seq_length(0) <= max_seq_len


def test_streaming_eviction_preserves_sink_tokens():
    """After eviction, sink tokens are still at positions 0..sink_size.

    Uses a small cache (max_seq_len=30) with sink=4, window=20 (max_kept=24).
    Fills 30 tokens then writes 5 more — must self-heal via eviction and
    sink tokens (first 4) must survive intact.
    """
    max_seq_len = 30
    sink = 4
    window = 20
    cache = StaticKVCache(
        max_seq_len=max_seq_len, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    cache.enable_streaming(sink_size=sink, window_size=window)

    # Fill cache with tokens 0..29, each token i has value i in all dims
    k_fill = torch.stack([
        torch.full((1, 2, 1, 4), float(i), dtype=torch.bfloat16)
        for i in range(max_seq_len)
    ], dim=2).squeeze(3)  # shape [1, 2, 30, 4]
    v_fill = torch.zeros(1, 2, max_seq_len, 4, dtype=torch.bfloat16)
    cache.update(k_fill, v_fill, layer_idx=0)
    assert cache.get_seq_length(0) == max_seq_len

    # Now write 5 more tokens — this must trigger streaming eviction
    k_new = torch.full((1, 2, 5, 4), 99.0, dtype=torch.bfloat16)
    v_new = torch.zeros(1, 2, 5, 4, dtype=torch.bfloat16)
    cache.update(k_new, v_new, layer_idx=0)  # must NOT raise

    # After eviction, sink tokens (positions 0..3) must still hold values 0..3.
    # cache._k shape: [layers, batch, heads, seq, head_dim]
    k_cached = cache._k[0, 0, 0, :sink, 0]  # first head_dim element for each sink pos
    for i in range(sink):
        assert float(k_cached[i]) == pytest.approx(float(i), abs=0.1), (
            f"Sink token {i} corrupted: expected {i}, got {float(k_cached[i])}"
        )


def test_streaming_disabled_still_raises_overflow():
    """Without streaming, overflow still raises KVCacheOverflow as before."""
    from tinyserve.static_kv_cache import KVCacheOverflow
    cache = StaticKVCache(
        max_seq_len=10, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    k = torch.randn(1, 2, 15, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 15, 4, dtype=torch.bfloat16)
    with pytest.raises(KVCacheOverflow):
        cache.update(k, v, layer_idx=0)


def test_streaming_no_eviction_when_below_capacity():
    """Streaming enabled but seq_len < max_kept: no eviction, normal behaviour."""
    max_seq_len = 256
    sink = 4
    window = 200
    cache = StaticKVCache(
        max_seq_len=max_seq_len, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    cache.enable_streaming(sink_size=sink, window_size=window)

    k = torch.randn(1, 2, 50, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 50, 4, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, layer_idx=0)

    # No eviction: seq_len == 50, returned shape == 50
    assert cache.get_seq_length(0) == 50
    assert k_out.shape == (1, 2, 50, 4)
