"""Test the unified OffloadedModel that wraps any HF MoE model."""

import torch

from tests.conftest import requires_cuda
from tests.test_e2e_offload import TinyMoEModel


@requires_cuda
def test_offloaded_model_from_hf_module():
    """OffloadedModel wraps an existing nn.Module and offloads its experts."""
    from src.offloaded_model import OffloadedModel

    torch.manual_seed(42)
    hidden, intermediate = 64, 128
    num_layers, num_experts, top_k = 2, 8, 2
    vocab = 256

    model = TinyMoEModel(vocab, hidden, intermediate, num_layers, num_experts, top_k)
    model = model.to(torch.bfloat16)
    model.eval()

    device = torch.device("cuda")
    input_ids = torch.tensor([[1, 42, 100, 7]], device=device)

    model_gpu = TinyMoEModel(vocab, hidden, intermediate, num_layers, num_experts, top_k)
    model_gpu.load_state_dict(model.state_dict())
    model_gpu = model_gpu.to(device).to(torch.bfloat16)
    model_gpu.eval()

    with torch.no_grad():
        ref_logits = model_gpu(input_ids)

    offloaded = OffloadedModel.from_module(
        model,
        moe_block_attr="mlp",
        expert_list_attr="experts",
        router_attr="gate",
        top_k=top_k,
        device=device,
        cache_capacity=16,
    )

    with torch.no_grad():
        offloaded_logits = offloaded(input_ids)

    torch.testing.assert_close(offloaded_logits, ref_logits, rtol=0, atol=0)


@requires_cuda
def test_offloaded_model_generate():
    """OffloadedModel produces identical greedy tokens as reference."""
    from src.offloaded_model import OffloadedModel

    torch.manual_seed(99)
    hidden, intermediate = 64, 128
    num_layers, num_experts, top_k = 2, 8, 2
    vocab = 256

    model = TinyMoEModel(vocab, hidden, intermediate, num_layers, num_experts, top_k)
    model = model.to(torch.bfloat16)
    model.eval()

    device = torch.device("cuda")

    model_gpu = TinyMoEModel(vocab, hidden, intermediate, num_layers, num_experts, top_k)
    model_gpu.load_state_dict(model.state_dict())
    model_gpu = model_gpu.to(device).to(torch.bfloat16)
    model_gpu.eval()

    input_ids = torch.tensor([[10, 20, 30]], device=device)

    with torch.no_grad():
        ref_tokens = []
        ids = input_ids.clone()
        for _ in range(8):
            logits = model_gpu(ids)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ref_tokens.append(next_tok.item())
            ids = torch.cat([ids, next_tok], dim=1)

    offloaded = OffloadedModel.from_module(
        model,
        moe_block_attr="mlp",
        expert_list_attr="experts",
        router_attr="gate",
        top_k=top_k,
        device=device,
        cache_capacity=16,
    )

    with torch.no_grad():
        offloaded_tokens = []
        ids = input_ids.clone()
        for _ in range(8):
            logits = offloaded(ids)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            offloaded_tokens.append(next_tok.item())
            ids = torch.cat([ids, next_tok], dim=1)

    assert offloaded_tokens == ref_tokens


@requires_cuda
def test_cache_stats_accessible():
    """Cache hit/miss stats are accessible after inference."""
    from src.offloaded_model import OffloadedModel

    torch.manual_seed(7)
    model = TinyMoEModel(256, 32, 64, 1, 4, 2).to(torch.bfloat16)
    model.eval()

    offloaded = OffloadedModel.from_module(
        model,
        moe_block_attr="mlp",
        expert_list_attr="experts",
        router_attr="gate",
        top_k=2,
        device=torch.device("cuda"),
        cache_capacity=8,
    )

    input_ids = torch.tensor([[1, 2, 3]], device="cuda")
    with torch.no_grad():
        offloaded(input_ids)
        offloaded(input_ids)

    stats = offloaded.cache_stats()
    assert stats["total_hits"] >= 0
    assert stats["total_misses"] >= 0
