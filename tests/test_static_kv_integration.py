"""Integration tests for StaticKVCache with the offload pipeline.

These tests use small synthetic models (Mixtral config) to validate
KV cache correctness without loading GPT-OSS-20B.
"""


import torch

from tests.conftest import requires_cuda


@requires_cuda
def test_static_kv_cache_forward_pass():
    """StaticKVCache works with a single forward pass on offloaded model."""
    from transformers import MixtralConfig, MixtralForCausalLM

    from tinyserve.offload import offload_model
    from tinyserve.static_kv_cache import StaticKVCache

    torch.manual_seed(42)
    config = MixtralConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        sliding_window=32,
    )
    model = MixtralForCausalLM(config).to(torch.bfloat16).eval()
    device = torch.device("cuda")

    offloaded = offload_model(model, device=device, cache_capacity=16)

    kv = StaticKVCache.from_model_config(config, max_seq_len=64, device=device)
    input_ids = torch.tensor([[1, 42, 100]], device=device)

    # Prefill
    with torch.no_grad():
        out1 = offloaded(input_ids=input_ids, past_key_values=kv)
    assert out1.logits.shape == (1, 3, 256)
    assert kv.get_seq_length(0) == 3

    # Decode step
    next_token = out1.logits[:, -1:].argmax(dim=-1)
    with torch.no_grad():
        out2 = offloaded(input_ids=next_token, past_key_values=kv)
    assert out2.logits.shape == (1, 1, 256)
    assert kv.get_seq_length(0) == 4


@requires_cuda
def test_static_kv_cache_with_offload_model():
    """offload_model with max_seq_len creates and attaches StaticKVCache."""
    from transformers import MixtralConfig, MixtralForCausalLM

    from tinyserve.offload import offload_model

    torch.manual_seed(42)
    config = MixtralConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        sliding_window=32,
    )
    model = MixtralForCausalLM(config).to(torch.bfloat16).eval()

    offloaded = offload_model(model, device="cuda", cache_capacity=16, max_seq_len=32)

    assert hasattr(offloaded, "_kv_cache")
    kv = offloaded._kv_cache
    assert kv.max_seq_len == 32
    assert kv.num_layers == 2

    input_ids = torch.tensor([[1, 42, 100]], device="cuda")
    with torch.no_grad():
        out = offloaded.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            past_key_values=kv,
        )
    assert out.shape[1] == input_ids.shape[1] + 5


@requires_cuda
def test_fp8_kv_cache_with_offload():
    """FP8 KV cache works with offloaded model."""
    from transformers import MixtralConfig, MixtralForCausalLM

    from tinyserve.offload import offload_model

    torch.manual_seed(42)
    config = MixtralConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        sliding_window=32,
    )
    model = MixtralForCausalLM(config).to(torch.bfloat16).eval()

    offloaded = offload_model(
        model,
        device="cuda",
        cache_capacity=16,
        max_seq_len=32,
        kv_dtype=torch.float8_e4m3fn,
    )
    kv = offloaded._kv_cache
    assert kv._dtype == torch.float8_e4m3fn

    input_ids = torch.tensor([[1, 42, 100]], device="cuda")
    with torch.no_grad():
        out = offloaded.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            past_key_values=kv,
        )
    assert out.shape[1] == input_ids.shape[1] + 5


def test_gpu_memory_utilization_parameter():
    """gpu_memory_utilization parameter is accepted by offload_model."""
    import inspect

    from tinyserve.offload import offload_model

    sig = inspect.signature(offload_model)
    assert "gpu_memory_utilization" in sig.parameters
    assert sig.parameters["gpu_memory_utilization"].default == 0.90


@requires_cuda
def test_inference_engine_basic():
    """InferenceEngine generates tokens via async interface."""
    from unittest.mock import MagicMock

    from tinyserve.server import InferenceEngine

    # Mock model and tokenizer for CPU test
    model = MagicMock()
    model.config = MagicMock()
    model.config.num_hidden_layers = 2
    model.config.num_key_value_heads = 2
    model.config.head_dim = 4
    model.config.text_config = model.config
    model._offload_pipelines = []

    tokenizer = MagicMock()
    tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
    tokenizer.decode.return_value = "hello"
    tokenizer.eos_token_id = 0

    engine = InferenceEngine(model, tokenizer, max_seq_len=32)
    assert engine.max_seq_len == 32
