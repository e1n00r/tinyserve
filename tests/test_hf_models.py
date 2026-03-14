"""E2e tests: offload real HF MoE models (tiny configs), verify exact match."""

import torch

from tests.conftest import requires_cuda


def _offload_and_compare(model_cls, config,
                         moe_block_attr, expert_list_attr, router_attr,
                         top_k, returns_router_logits, device,
                         softmax_order="topk_then_softmax", n_gen_tokens=5):
    """Shared helper: offload model, compare logits + autoregressive tokens."""
    from src.offloaded_model import OffloadedModel

    model = model_cls(config).to(torch.bfloat16).eval()

    ref_model = model_cls(config).to(torch.bfloat16).eval()
    ref_model.load_state_dict(model.state_dict())
    ref_model = ref_model.to(device)

    input_ids = torch.tensor([[1, 42, 100, 7]], device=device)

    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits

    offloaded = OffloadedModel.from_module(
        model.model,
        moe_block_attr=moe_block_attr,
        expert_list_attr=expert_list_attr,
        router_attr=router_attr,
        top_k=top_k,
        device=device,
        cache_capacity=32,
        returns_router_logits=returns_router_logits,
        softmax_order=softmax_order,
    )
    model.model = offloaded.model
    model = model.to(device)

    with torch.no_grad():
        offloaded_logits = model(input_ids).logits

    torch.testing.assert_close(offloaded_logits, ref_logits, rtol=0, atol=0)

    with torch.no_grad():
        ref_tokens = []
        ids = input_ids.clone()
        for _ in range(n_gen_tokens):
            next_tok = ref_model(ids).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ref_tokens.append(next_tok.item())
            ids = torch.cat([ids, next_tok], dim=1)

    ref_model_2 = model_cls(config).to(torch.bfloat16).eval()
    ref_model_2.load_state_dict(ref_model.state_dict())
    offloaded_2 = OffloadedModel.from_module(
        ref_model_2.model,
        moe_block_attr=moe_block_attr,
        expert_list_attr=expert_list_attr,
        router_attr=router_attr,
        top_k=top_k,
        device=device,
        cache_capacity=32,
        returns_router_logits=returns_router_logits,
        softmax_order=softmax_order,
    )
    ref_model_2.model = offloaded_2.model
    ref_model_2 = ref_model_2.to(device)

    with torch.no_grad():
        off_tokens = []
        ids = input_ids.clone()
        for _ in range(n_gen_tokens):
            next_tok = ref_model_2(ids).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            off_tokens.append(next_tok.item())
            ids = torch.cat([ids, next_tok], dim=1)

    assert off_tokens == ref_tokens


@requires_cuda
def test_mixtral():
    from transformers import MixtralConfig, MixtralForCausalLM

    torch.manual_seed(42)
    config = MixtralConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64, sliding_window=32,
    )
    _offload_and_compare(
        MixtralForCausalLM, config,
        "block_sparse_moe", "experts", "gate", top_k=2,
        returns_router_logits=True, device=torch.device("cuda"),
        softmax_order="softmax_then_topk",
    )


@requires_cuda
def test_qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_experts=8, num_experts_per_tok=2,
        max_position_embeddings=64, decoder_sparse_step=1,
    )
    _offload_and_compare(
        Qwen3MoeForCausalLM, config,
        "mlp", "experts", "gate", top_k=2,
        returns_router_logits=True, device=torch.device("cuda"),
        softmax_order="softmax_then_topk",
    )
