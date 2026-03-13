"""Test LRU expert cache."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lru_cache import ExpertLRUCache
from src.config import EXPERT_BYTES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hit_and_miss():
    cache = ExpertLRUCache(capacity=4, device=torch.device("cuda"))

    assert cache.lookup(0, 5) is None  # miss
    assert cache.misses == 1

    slot = cache.allocate(0, 5)
    assert slot is not None

    assert cache.lookup(0, 5) == slot  # hit
    assert cache.hits == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_eviction():
    cache = ExpertLRUCache(capacity=3, device=torch.device("cuda"))

    # Fill cache
    s0 = cache.allocate(0, 0)
    s1 = cache.allocate(0, 1)
    s2 = cache.allocate(0, 2)

    # Access 0 to make it recently used
    cache.lookup(0, 0)

    # Insert 4th — should evict expert 1 (LRU)
    s3 = cache.allocate(0, 3)
    assert s3 == s1  # reuses evicted slot

    assert cache.lookup(0, 1) is None  # evicted
    assert cache.lookup(0, 0) is not None  # still cached (was accessed)
    assert cache.lookup(0, 2) is not None  # still cached
    assert cache.lookup(0, 3) is not None  # just inserted


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hit_rate():
    cache = ExpertLRUCache(capacity=2, device=torch.device("cuda"))
    cache.allocate(0, 0)
    cache.allocate(0, 1)

    cache.lookup(0, 0)  # hit
    cache.lookup(0, 1)  # hit
    cache.lookup(0, 2)  # miss

    assert cache.hit_rate == pytest.approx(2 / 3)


def test_estimate_capacity():
    # 1 GB should hold ~80 experts (12.64 MB each)
    cap = ExpertLRUCache.estimate_capacity(1024 * 1024 * 1024)
    assert 75 <= cap <= 85


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_pipeline_matches_uncached(tmp_path):
    """Pipeline with cache produces same output as without cache."""
    from safetensors.torch import save_file
    from src.expert_store import ExpertBuffer, ExpertStore
    from src.experts import expert_forward
    from src.pipeline import ExpertPipeline
    from src.config import HIDDEN_SIZE, NUM_EXPERTS_PER_TOK
    import src.expert_store as es_mod

    original_num_layers = es_mod.NUM_LAYERS
    es_mod.NUM_LAYERS = 2

    try:
        num_experts = 8
        hidden = HIDDEN_SIZE
        intermediate = HIDDEN_SIZE

        for layer_idx in range(2):
            tensors = {
                "gate_up_proj_blocks": torch.randint(0, 256, (num_experts, 2*intermediate, hidden//32, 16), dtype=torch.uint8),
                "gate_up_proj_scales": torch.randint(120, 135, (num_experts, 2*intermediate, hidden//32), dtype=torch.uint8),
                "gate_up_proj_bias": torch.randn(num_experts, 2*intermediate, dtype=torch.float32) * 0.01,
                "down_proj_blocks": torch.randint(0, 256, (num_experts, hidden, intermediate//32, 16), dtype=torch.uint8),
                "down_proj_scales": torch.randint(120, 135, (num_experts, hidden, intermediate//32), dtype=torch.uint8),
                "down_proj_bias": torch.randn(num_experts, hidden, dtype=torch.float32) * 0.01,
            }
            save_file(tensors, str(tmp_path / f"experts_L{layer_idx:02d}.safetensors"))

        store = ExpertStore(str(tmp_path))
        store.load()
        device = torch.device("cuda")

        # Run without cache
        pipe_nocache = ExpertPipeline(store, device, cache_capacity=0)
        h = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        idx = torch.tensor([[0, 3, 5, 7]])
        w = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device, dtype=torch.bfloat16)
        out_nocache = pipe_nocache.execute_layer_experts(h, 0, idx, w)

        # Run with cache (all misses first time)
        pipe_cached = ExpertPipeline(store, device, cache_capacity=10)
        out_cached1 = pipe_cached.execute_layer_experts(h, 0, idx, w)
        torch.testing.assert_close(out_cached1, out_nocache, rtol=0, atol=0)

        # Run again — now all hits
        out_cached2 = pipe_cached.execute_layer_experts(h, 0, idx, w)
        torch.testing.assert_close(out_cached2, out_nocache, rtol=0, atol=0)

        assert pipe_cached.cache.hits == 4
        assert pipe_cached.cache.misses == 4

    finally:
        es_mod.NUM_LAYERS = original_num_layers
