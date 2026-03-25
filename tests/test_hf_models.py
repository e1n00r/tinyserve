"""E2e tests: offload real HF MoE models (tiny configs), verify exact match."""

import torch

from tests.conftest import requires_cuda


def _offload_and_compare(
    model_cls,
    config,
    moe_block_attr,
    expert_list_attr,
    router_attr,
    top_k,
    returns_router_logits,
    device,
    softmax_order="topk_then_softmax",
    first_moe_layer=0,
    n_gen_tokens=5,
    atol=0.02,
):
    """Shared helper: offload model, compare logits + autoregressive tokens."""
    from tinyserve.offloaded_model import OffloadedModel

    model = model_cls(config).to(torch.bfloat16).eval()

    ref_model = model_cls(config).to(torch.bfloat16).eval()
    ref_model.load_state_dict(model.state_dict())
    ref_model = ref_model.to(device)

    input_ids = torch.tensor([[1, 42, 100, 7]], device=device)

    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits

    offloaded, *_ = OffloadedModel.from_module(
        model.model,
        moe_block_attr=moe_block_attr,
        expert_list_attr=expert_list_attr,
        router_attr=router_attr,
        top_k=top_k,
        device=device,
        cache_capacity=32,
        returns_router_logits=returns_router_logits,
        softmax_order=softmax_order,
        first_moe_layer=first_moe_layer,
    )
    model.model = offloaded.model
    model = model.to(device)

    with torch.no_grad():
        offloaded_logits = model(input_ids).logits

    torch.testing.assert_close(offloaded_logits, ref_logits, rtol=0, atol=atol)

    with torch.no_grad():
        ref_tokens = []
        ids = input_ids.clone()
        for _ in range(n_gen_tokens):
            next_tok = ref_model(ids).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ref_tokens.append(next_tok.item())
            ids = torch.cat([ids, next_tok], dim=1)

    ref_model_2 = model_cls(config).to(torch.bfloat16).eval()
    ref_model_2.load_state_dict(ref_model.state_dict())
    offloaded_2, *_ = OffloadedModel.from_module(
        ref_model_2.model,
        moe_block_attr=moe_block_attr,
        expert_list_attr=expert_list_attr,
        router_attr=router_attr,
        top_k=top_k,
        device=device,
        cache_capacity=32,
        returns_router_logits=returns_router_logits,
        softmax_order=softmax_order,
        first_moe_layer=first_moe_layer,
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
    _offload_and_compare(
        MixtralForCausalLM,
        config,
        "mlp",
        "experts",
        "gate",
        top_k=2,
        returns_router_logits=False,
        device=torch.device("cuda"),
        softmax_order="router_native",
    )


@requires_cuda
def test_qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        decoder_sparse_step=1,
    )
    _offload_and_compare(
        Qwen3MoeForCausalLM,
        config,
        "mlp",
        "experts",
        "gate",
        top_k=2,
        returns_router_logits=False,
        device=torch.device("cuda"),
        softmax_order="router_native",
    )


@requires_cuda
def test_deepseek_v3():
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    torch.manual_seed(42)
    config = DeepseekV3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        first_k_dense_replace=2,
        n_group=2,
        topk_group=1,
        n_shared_experts=1,
        moe_intermediate_size=64,
    )
    _offload_and_compare(
        DeepseekV3ForCausalLM,
        config,
        "mlp",
        "experts",
        "gate",
        top_k=2,
        returns_router_logits=False,
        device=torch.device("cuda"),
        softmax_order="router_native",
        first_moe_layer=2,
        n_gen_tokens=0,
    )


@requires_cuda
def test_qwen3_5_moe():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(42)
    config = Qwen3_5MoeTextConfig(
        num_hidden_layers=2,
        num_experts=4,
        hidden_size=64,
        moe_intermediate_size=32,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=64,
        num_experts_per_tok=2,
        full_attention_interval=1,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        mtp_num_hidden_layers=0,
        pad_token_id=None,
        layer_types=["linear_attention", "full_attention"],
    )

    model = Qwen3_5MoeForCausalLM(config).to(torch.bfloat16).eval()
    device = torch.device("cuda")

    ref = Qwen3_5MoeForCausalLM(config).to(torch.bfloat16).eval()
    ref.load_state_dict(model.state_dict())
    ref = ref.to(device)

    input_ids = torch.tensor([[1, 42, 100, 7]], device=device)

    with torch.no_grad():
        ref_tok = ref(input_ids).logits[:, -1, :].argmax().item()

    from tinyserve.offload import offload_model

    offloaded = offload_model(model, device=device, cache_capacity=16)

    with torch.no_grad():
        off_tok = offloaded(input_ids).logits[:, -1, :].argmax().item()

    assert off_tok == ref_tok


@requires_cuda
def test_gpt_oss():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b")
    config.num_hidden_layers = 2
    config.num_local_experts = 4
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.vocab_size = 256
    config.max_position_embeddings = 64
    config.head_dim = 16
    config.layer_types = ["sliding_attention", "full_attention"]
    config.pad_token_id = None

    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
    device = torch.device("cuda")

    ref = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
    ref.load_state_dict(model.state_dict())
    ref = ref.to(device)

    input_ids = torch.tensor([[1, 42, 100, 7]], device=device)
    with torch.no_grad():
        ref_tok = ref(input_ids).logits[:, -1, :].argmax().item()

    from tinyserve.offload import offload_model

    offloaded = offload_model(model, device=device, cache_capacity=16)

    with torch.no_grad():
        off_tok = offloaded(input_ids).logits[:, -1, :].argmax().item()

    assert off_tok == ref_tok
