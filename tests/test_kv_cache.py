"""Test hybrid KV cache behavior."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.kv_cache import HybridKVCache
from src.config import LAYER_TYPES, NUM_KV_HEADS, HEAD_DIM, SLIDING_WINDOW


def test_sliding_window_stays_on_gpu():
    """Sliding-window layers keep KV on GPU and trim to window size."""
    if not torch.cuda.is_available():
        return

    cache = HybridKVCache(torch.device("cuda"))

    # Find a sliding-window layer
    sw_layer = next(i for i, t in enumerate(LAYER_TYPES) if t == "sliding_attention")

    # Add tokens beyond the window
    for step in range(SLIDING_WINDOW + 10):
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device="cuda")
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device="cuda")
        cache.update(sw_layer, k, v)

    kv = cache.get_kv(sw_layer)
    assert kv is not None
    past_k, past_v = kv
    assert past_k.shape[2] == SLIDING_WINDOW
    assert past_k.device.type == "cuda"


def test_full_attn_goes_to_cpu():
    """Full-attention layers store KV on CPU."""
    if not torch.cuda.is_available():
        return

    cache = HybridKVCache(torch.device("cuda"))

    # Find a full-attention layer
    fa_layer = next(i for i, t in enumerate(LAYER_TYPES) if t == "full_attention")

    k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device="cuda")
    v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device="cuda")
    cache.update(fa_layer, k, v)

    # Internal storage should be on CPU
    assert cache._k[fa_layer].device.type == "cpu"
    assert cache._v[fa_layer].device.type == "cpu"

    # get_kv should return GPU tensors
    kv = cache.get_kv(fa_layer)
    past_k, past_v = kv
    assert past_k.device.type == "cuda"


def test_full_attn_accumulates():
    """Full-attention layers keep all past tokens."""
    if not torch.cuda.is_available():
        return

    cache = HybridKVCache(torch.device("cuda"))
    fa_layer = next(i for i, t in enumerate(LAYER_TYPES) if t == "full_attention")

    n_tokens = 50
    for _ in range(n_tokens):
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device="cuda")
        v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, device="cuda")
        cache.update(fa_layer, k, v)

    kv = cache.get_kv(fa_layer)
    assert kv[0].shape[2] == n_tokens


def test_reset():
    cache = HybridKVCache(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
    v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
    cache.update(0, k, v)
    cache.reset()
    assert cache.get_kv(0) is None
