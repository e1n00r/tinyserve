import torch

from src.config import HIDDEN_SIZE, NUM_EXPERTS_PER_TOK
from src.expert_store import ExpertBuffer
from src.experts import expert_forward
from src.pipeline import ExpertPipeline

from .conftest import requires_cuda


@requires_cuda
def test_pipeline_matches_blocking(mock_expert_store):
    device = torch.device("cuda")
    pipeline = ExpertPipeline(mock_expert_store, device)
    buf = ExpertBuffer(device)

    h = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    expert_indices = torch.tensor([[0, 3, 5, 7]])
    weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device, dtype=torch.bfloat16)

    blocking_out = torch.zeros_like(h)
    for k in range(NUM_EXPERTS_PER_TOK):
        eidx = expert_indices[0, k].item()
        mock_expert_store.copy_to_buffer(buf, 0, eidx, non_blocking=False)
        out = expert_forward(
            h, buf.gate_up_blocks, buf.gate_up_scales, buf.gate_up_bias,
            buf.down_blocks, buf.down_scales, buf.down_bias,
        )
        blocking_out += weights[0, k] * out

    pipeline_out = pipeline.execute_layer_experts(h, 0, expert_indices, weights)
    torch.testing.assert_close(pipeline_out, blocking_out, rtol=0, atol=0)


@requires_cuda
def test_cached_pipeline_matches_uncached(mock_expert_store):
    device = torch.device("cuda")
    h = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    expert_indices = torch.tensor([[0, 3, 5, 7]])
    weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device, dtype=torch.bfloat16)

    uncached = ExpertPipeline(mock_expert_store, device, cache_capacity=0)
    out_uncached = uncached.execute_layer_experts(h, 0, expert_indices, weights)
    torch.cuda.synchronize()

    cached = ExpertPipeline(mock_expert_store, device, cache_capacity=10)
    out_miss = cached.execute_layer_experts(h, 0, expert_indices, weights)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_miss, out_uncached, rtol=0, atol=0)

    out_hit = cached.execute_layer_experts(h, 0, expert_indices, weights)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_hit, out_uncached, rtol=0, atol=0)

    assert cached.cache.hits == 4
    assert cached.cache.misses == 4


@requires_cuda
def test_pipeline_multi_token(mock_expert_store):
    device = torch.device("cuda")
    pipeline = ExpertPipeline(mock_expert_store, device)

    num_tokens = 3
    h = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    expert_indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]])
    weights = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1).to(torch.bfloat16)

    out = pipeline.execute_layer_experts(h, 0, expert_indices, weights)
    assert out.shape == (num_tokens, HIDDEN_SIZE)
    assert not torch.isnan(out).any()
