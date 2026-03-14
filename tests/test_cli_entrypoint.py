"""Test the unified CLI entry point for offloaded inference."""

import torch

from tests.conftest import requires_cuda


@requires_cuda
def test_offload_from_config_mixtral():
    """offload_model() auto-detects Mixtral and offloads correctly."""
    from transformers import MixtralConfig, MixtralForCausalLM
    from src.offload import offload_model

    torch.manual_seed(42)
    config = MixtralConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64, sliding_window=32,
    )
    model = MixtralForCausalLM(config).to(torch.bfloat16).eval()
    device = torch.device("cuda")

    ref = MixtralForCausalLM(config).to(torch.bfloat16).eval()
    ref.load_state_dict(model.state_dict())
    ref = ref.to(device)

    input_ids = torch.tensor([[1, 42, 100]], device=device)

    with torch.no_grad():
        ref_logits = ref(input_ids).logits

    offloaded = offload_model(model, device=device, cache_capacity=16)

    with torch.no_grad():
        off_logits = offloaded(input_ids).logits

    assert off_logits[:, -1, :].argmax().item() == ref_logits[:, -1, :].argmax().item()


@requires_cuda
def test_offload_from_config_deepseek():
    """offload_model() auto-detects DeepSeek-V3 and offloads correctly."""
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
    from src.offload import offload_model

    torch.manual_seed(42)
    config = DeepseekV3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        n_routed_experts=8, num_experts_per_tok=2, max_position_embeddings=64,
        first_k_dense_replace=2, n_group=2, topk_group=1,
        n_shared_experts=1, moe_intermediate_size=64,
    )
    model = DeepseekV3ForCausalLM(config).to(torch.bfloat16).eval()
    device = torch.device("cuda")

    ref = DeepseekV3ForCausalLM(config).to(torch.bfloat16).eval()
    ref.load_state_dict(model.state_dict())
    ref = ref.to(device)

    input_ids = torch.tensor([[1, 42, 100]], device=device)

    with torch.no_grad():
        ref_tok = ref(input_ids).logits[:, -1, :].argmax().item()

    offloaded = offload_model(model, device=device, cache_capacity=16)

    with torch.no_grad():
        off_tok = offloaded(input_ids).logits[:, -1, :].argmax().item()

    assert off_tok == ref_tok
