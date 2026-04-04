"""Tests for batched expert prefill — correctness vs token-by-token baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.conftest import requires_cuda


class TinyFusedExpert(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2 * intermediate, hidden))
        self.down_proj = nn.Parameter(torch.randn(hidden, intermediate))
        self._act_fn = nn.SiLU()
        self._has_bias = False
        self._is_mxfp4 = False
        self._param_names = ["gate_up_proj", "down_proj"]

    def forward(self, x):
        gate_up = F.linear(x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(self._act_fn(gate) * up, self.down_proj)


def _make_pipeline(num_experts=4, hidden=16, intermediate=32, device="cuda"):
    from tinyserve.expert_pipeline import ExpertPipeline
    from tinyserve.expert_store import ExpertCache, ExpertStore

    weights = {}
    for li in range(1):
        for ei in range(num_experts):
            weights[(li, ei)] = {
                "gate_up_proj": torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16),
                "down_proj": torch.randn(hidden, intermediate, dtype=torch.bfloat16),
            }
    store = ExpertStore.from_dict(weights, 1, num_experts)
    template = TinyFusedExpert(hidden, intermediate).to(device).to(torch.bfloat16)
    staging_buffer_a = store.allocate_buffer(torch.device(device))
    staging_buffer_b = store.allocate_buffer(torch.device(device))
    ts = torch.cuda.Stream(torch.device(device))
    cs = torch.cuda.Stream(torch.device(device))
    cache = ExpertCache(
        num_experts, store.buffer_expert_bytes, torch.device(device), num_layers=1, num_experts=num_experts
    )
    pipeline = ExpertPipeline(
        store, template, torch.device(device), staging_buffer_a, staging_buffer_b, ts, cs, cache=cache
    )
    return pipeline


@requires_cuda
def test_batched_dispatch_matches_sequential_for_single_expert():
    """4 tokens all routed to expert 0 — batched == sequential."""
    pipeline = _make_pipeline()
    seq_len, top_k = 4, 1
    h = torch.randn(seq_len, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.zeros(seq_len, top_k, device="cuda", dtype=torch.long)
    weights = torch.ones(seq_len, top_k, device="cuda", dtype=torch.bfloat16)

    ref = pipeline.execute_layer_experts(h, 0, expert_ids, weights)
    out = pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)


@requires_cuda
def test_batched_dispatch_matches_sequential_for_multiple_experts():
    """8 tokens with top_k=2, mixed experts — batched == sequential."""
    pipeline = _make_pipeline()
    seq_len, top_k = 8, 2
    h = torch.randn(seq_len, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.randint(0, 4, (seq_len, top_k), device="cuda")
    weights = torch.rand(seq_len, top_k, device="cuda", dtype=torch.bfloat16)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    ref = pipeline.execute_layer_experts(h, 0, expert_ids, weights)
    out = pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)


@requires_cuda
def test_batched_dispatch_loads_each_expert_once_per_step():
    """With 4 tokens all needing expert 2, cache should see 1 miss not 4."""
    pipeline = _make_pipeline()
    seq_len = 4
    h = torch.randn(seq_len, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.full((seq_len, 1), 2, device="cuda", dtype=torch.long)
    weights = torch.ones(seq_len, 1, device="cuda", dtype=torch.bfloat16)

    pipeline.cache.reset_stats()
    pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    assert pipeline.cache.misses <= 1, f"Expected 1 miss, got {pipeline.cache.misses}"


@requires_cuda
def test_batched_dispatch_returns_empty_output_for_zero_length_input():
    """Zero-length input returns zero-length output."""
    pipeline = _make_pipeline()
    h = torch.randn(0, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.zeros(0, 1, device="cuda", dtype=torch.long)
    weights = torch.zeros(0, 1, device="cuda", dtype=torch.bfloat16)
    out = pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    assert out.shape == (0, 16)


@requires_cuda
def test_batched_dispatch_handles_large_prefill_correctly():
    """512 tokens — stress test for batched path."""
    pipeline = _make_pipeline()
    seq_len, top_k = 512, 2
    h = torch.randn(seq_len, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.randint(0, 4, (seq_len, top_k), device="cuda")
    weights = torch.rand(seq_len, top_k, device="cuda", dtype=torch.bfloat16)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    ref = pipeline.execute_layer_experts(h, 0, expert_ids, weights)
    out = pipeline.execute_layer_experts_batched(h, 0, expert_ids, weights)
    # BF16 batched matmul accumulation order may differ from sequential
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)
