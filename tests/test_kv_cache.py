import pytest
import torch

from src.kv_cache import KVCache
from src.config import LAYER_TYPES, NUM_KV_HEADS, HEAD_DIM, SLIDING_WINDOW

from .conftest import requires_cuda


def _kv_pair(device="cuda"):
    return (
        torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device=device),
        torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device=device),
    )


def _sliding_layer():
    return next(i for i, t in enumerate(LAYER_TYPES) if t == "sliding_attention")


def _full_layer():
    return next(i for i, t in enumerate(LAYER_TYPES) if t == "full_attention")


@requires_cuda
def test_sliding_window_trims():
    cache = KVCache(torch.device("cuda"))
    layer = _sliding_layer()

    for _ in range(SLIDING_WINDOW + 10):
        k, v = _kv_pair()
        cache.update(layer, k, v)

    past_k, _ = cache.get_kv(layer)
    assert past_k.shape[2] == SLIDING_WINDOW
    assert past_k.device.type == "cuda"


@requires_cuda
def test_full_attn_accumulates_on_gpu():
    cache = KVCache(torch.device("cuda"))
    layer = _full_layer()
    n_tokens = 50

    for _ in range(n_tokens):
        k, v = _kv_pair()
        cache.update(layer, k, v)

    past_k, _ = cache.get_kv(layer)
    assert past_k.shape[2] == n_tokens
    assert past_k.device.type == "cuda"


@requires_cuda
def test_reset_clears_cache():
    cache = KVCache(torch.device("cuda"))
    k, v = _kv_pair()
    cache.update(0, k, v)
    cache.reset()
    assert cache.get_kv(0) is None


@requires_cuda
def test_overflow_raises():
    cache = KVCache(torch.device("cuda"), max_seq_len=5)
    layer = _full_layer()

    for _ in range(5):
        k, v = _kv_pair()
        cache.update(layer, k, v)

    with pytest.raises(RuntimeError, match="overflow"):
        k, v = _kv_pair()
        cache.update(layer, k, v)
