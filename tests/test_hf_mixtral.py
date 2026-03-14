"""E2e test: offload a real HF Mixtral model (tiny config)."""

import torch

from tests.conftest import requires_cuda


def _make_tiny_mixtral():
    from transformers import MixtralConfig, MixtralForCausalLM

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
    return config, MixtralForCausalLM(config).to(torch.bfloat16).eval()


@requires_cuda
def test_mixtral_offloaded_matches_reference():
    """Offloaded tiny Mixtral produces identical logits to GPU reference."""
    from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP
    from src.offloaded_model import OffloadedModel

    torch.manual_seed(42)
    config, model = _make_tiny_mixtral()
    device = torch.device("cuda")

    from transformers import MixtralForCausalLM
    ref_model = MixtralForCausalLM(config).to(torch.bfloat16).eval()
    ref_model.load_state_dict(model.state_dict())
    ref_model = ref_model.to(device)

    input_ids = torch.tensor([[1, 42, 100, 7]], device=device)

    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits

    offloaded = OffloadedModel.from_module(
        model.model,
        expert_module_cls=MixtralBlockSparseTop2MLP,
        expert_cls_kwargs={"config": config},
        moe_block_attr="block_sparse_moe",
        expert_list_attr="experts",
        router_attr="gate",
        top_k=2,
        device=device,
        cache_capacity=16,
        returns_router_logits=True,
    )
    model.model = offloaded.model
    model = model.to(device)

    with torch.no_grad():
        offloaded_logits = model(input_ids).logits

    torch.testing.assert_close(offloaded_logits, ref_logits, rtol=0, atol=0)


@requires_cuda
def test_mixtral_autoregressive():
    """Offloaded tiny Mixtral generates identical greedy tokens."""
    from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP
    from src.offloaded_model import OffloadedModel

    torch.manual_seed(123)
    config, model = _make_tiny_mixtral()
    device = torch.device("cuda")

    from transformers import MixtralForCausalLM
    ref_model = MixtralForCausalLM(config).to(torch.bfloat16).eval()
    ref_model.load_state_dict(model.state_dict())
    ref_model = ref_model.to(device)

    input_ids = torch.tensor([[10, 20, 30]], device=device)

    with torch.no_grad():
        ref_tokens = []
        ids = input_ids.clone()
        for _ in range(5):
            next_tok = ref_model(ids).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ref_tokens.append(next_tok.item())
            ids = torch.cat([ids, next_tok], dim=1)

    offloaded = OffloadedModel.from_module(
        model.model,
        expert_module_cls=MixtralBlockSparseTop2MLP,
        expert_cls_kwargs={"config": config},
        moe_block_attr="block_sparse_moe",
        expert_list_attr="experts",
        router_attr="gate",
        top_k=2,
        device=device,
        cache_capacity=16,
        returns_router_logits=True,
    )
    model.model = offloaded.model
    model = model.to(device)

    with torch.no_grad():
        off_tokens = []
        ids = input_ids.clone()
        for _ in range(5):
            next_tok = model(ids).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            off_tokens.append(next_tok.item())
            ids = torch.cat([ids, next_tok], dim=1)

    assert off_tokens == ref_tokens
