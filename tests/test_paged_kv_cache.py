"""Tests for PagedKVPool and PagedRequestKVCache."""

import torch
import pytest
from tinyserve.paged_kv_cache import PagedKVPool, PagedRequestKVCache, PAGE_SIZE

DEVICE = torch.device("cpu")
NUM_LAYERS = 2
NUM_KV_HEADS = 4
HEAD_DIM = 8


def _make_pool(num_pages=8, dtype=torch.bfloat16):
    return PagedKVPool(num_pages, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, DEVICE, dtype)


def test_page_allocation_and_free():
    pool = _make_pool(num_pages=4)
    assert pool.pages_free == 4
    assert pool.pages_used == 0

    p0 = pool.allocate_page()
    p1 = pool.allocate_page()
    assert pool.pages_free == 2
    assert pool.pages_used == 2

    pool.free_page(p0)
    assert pool.pages_free == 3

    pool.free_page(p1)
    assert pool.pages_free == 4


def test_write_read_single_page():
    pool = _make_pool(num_pages=2)
    pid = pool.allocate_page()

    k_data = torch.randn(NUM_KV_HEADS, 5, HEAD_DIM, dtype=torch.bfloat16)
    v_data = torch.randn(NUM_KV_HEADS, 5, HEAD_DIM, dtype=torch.bfloat16)

    pool.write(pid, layer_idx=0, kv_type=0, offset=0, data=k_data)
    pool.write(pid, layer_idx=0, kv_type=1, offset=0, data=v_data)

    k_out = pool.read([pid], layer_idx=0, kv_type=0, total_tokens=5)
    v_out = pool.read([pid], layer_idx=0, kv_type=1, total_tokens=5)

    assert k_out.shape == (1, NUM_KV_HEADS, 5, HEAD_DIM)
    assert torch.equal(k_out.squeeze(0), k_data)
    assert torch.equal(v_out.squeeze(0), v_data)


def test_multi_page_sequence():
    pool = _make_pool(num_pages=4)
    cache = PagedRequestKVCache(pool)

    total_tokens = PAGE_SIZE + 10
    k = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.bfloat16)

    k_out, v_out = cache.update(k, v, layer_idx=0,
                                 cache_kwargs={"cache_position": torch.arange(total_tokens)})

    assert cache.seq_len == total_tokens
    assert len(cache.page_ids) == 2
    assert k_out.shape == (1, NUM_KV_HEADS, total_tokens, HEAD_DIM)
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)

    cache.free()
    assert pool.pages_free == 4


def test_pool_exhaustion_raises():
    pool = _make_pool(num_pages=1)
    _ = pool.allocate_page()
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.allocate_page()


def test_request_kv_cache_update():
    pool = _make_pool(num_pages=4)
    cache = PagedRequestKVCache(pool)

    assert cache.get_seq_length() == 0
    assert not bool(cache)
    assert len(cache) == NUM_LAYERS

    # Prefill 5 tokens
    k = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k, v, 0, {"cache_position": torch.arange(5)})
    assert k_out.shape == (1, NUM_KV_HEADS, 5, HEAD_DIM)
    assert cache.get_seq_length() == 5
    assert bool(cache)

    # Decode one token
    k1 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    v1 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16)
    k_out, v_out = cache.update(k1, v1, 0, {"cache_position": torch.tensor([5])})
    assert k_out.shape == (1, NUM_KV_HEADS, 6, HEAD_DIM)
    assert cache.get_seq_length() == 6

    # Verify iter/getitem
    k0, v0 = cache[0]
    assert k0.shape == (1, NUM_KV_HEADS, 6, HEAD_DIM)
    items = list(cache)
    assert len(items) == NUM_LAYERS

    # Mask sizes
    kv_len, offset = cache.get_mask_sizes(torch.tensor([6]), layer_idx=0)
    assert kv_len == 7
    assert offset == 0

    # Reset
    cache.reset()
    assert cache.get_seq_length() == 0
    assert not bool(cache)


def test_multiple_requests_share_pool():
    pool = _make_pool(num_pages=8)
    c1 = PagedRequestKVCache(pool)
    c2 = PagedRequestKVCache(pool)

    k = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM, dtype=torch.bfloat16)

    c1.update(k, v, 0, {"cache_position": torch.arange(3)})
    c2.update(k, v, 0, {"cache_position": torch.arange(3)})

    # Each request uses 1 page (3 tokens < PAGE_SIZE)
    assert pool.pages_used == 2

    c1.free()
    assert pool.pages_used == 1

    c2.free()
    assert pool.pages_used == 0


def test_fp8_dtype():
    pool = _make_pool(num_pages=2, dtype=torch.float8_e4m3fn)
    cache = PagedRequestKVCache(pool)

    k = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, NUM_KV_HEADS, 3, HEAD_DIM, dtype=torch.bfloat16)

    k_out, v_out = cache.update(k, v, 0, {"cache_position": torch.arange(3)})
    assert k_out.dtype == torch.bfloat16
    assert pool._pool.dtype == torch.float8_e4m3fn
    assert torch.allclose(k_out, k, atol=0.2)

    cache.free()
