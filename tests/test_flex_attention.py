"""Tests for FlexAttention with StaticKVCache static shapes."""

import torch
import pytest

from tests.conftest import requires_cuda


@requires_cuda
def test_flex_attention_static_shapes_forward():
    """FlexAttention with static_shapes=True produces valid output on a tiny Mixtral model."""
    from transformers import MixtralConfig, MixtralForCausalLM

    from tinyserve.offload import _register_flex_attention, offload_model

    _register_flex_attention()

    config = MixtralConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(42)
    model = MixtralForCausalLM(config).to(torch.bfloat16)
    model = offload_model(
        model,
        device="cuda",
        max_seq_len=64,
        fp8=False,
        attn_implementation="flex",
        gpu_memory_utilization=0.95,
    )
    assert hasattr(model, "_kv_cache")
    assert model._kv_cache.static_shapes is True
    assert model._kv_cache.max_seq_len == 64

    input_ids = torch.tensor([[1, 42, 100, 7]], device="cuda")
    with torch.no_grad():
        output = model(input_ids)
    logits = output.logits if hasattr(output, "logits") else output
    assert logits.shape == (1, 4, 256)
    assert torch.isfinite(logits).all()


@requires_cuda
def test_flex_attention_no_recompile_on_length_change():
    """Static shapes prevent torch.compile recompilation across different seq lengths."""
    from tinyserve.static_kv_cache import StaticKVCache

    cache = StaticKVCache(
        max_seq_len=32, num_layers=1, num_kv_heads=2,
        head_dim=4, device=torch.device("cuda"), dtype=torch.bfloat16,
        static_shapes=True,
    )

    # Prefill 5 tokens
    k = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16, device="cuda")
    k1, v1 = cache.update(k, v, 0, {"cache_position": torch.arange(5, device="cuda")})

    # Decode 1 token
    k2_in = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16, device="cuda")
    v2_in = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16, device="cuda")
    k2, v2 = cache.update(k2_in, v2_in, 0, {"cache_position": torch.tensor([5], device="cuda")})

    # Both returns must have identical shape (max_seq_len=32)
    assert k1.shape == k2.shape == (1, 2, 32, 4)
    assert v1.shape == v2.shape == (1, 2, 32, 4)
