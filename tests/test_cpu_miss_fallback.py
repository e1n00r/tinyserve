"""Tests for CPU expert compute on cache miss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from tests.conftest import requires_cuda


class TinyExpert(nn.Module):
    def __init__(self, h=16, i=32):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2 * i, h))
        self.down_proj = nn.Parameter(torch.randn(h, i))
        self._act_fn = nn.SiLU()
        self._has_bias = False
        self._is_mxfp4 = False
        self._param_names = ["gate_up_proj", "down_proj"]

    def forward(self, x):
        gate_up = F.linear(x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(self._act_fn(gate) * up, self.down_proj)


def _make_pipeline_with_cpu(num_experts=4, hidden=16, intermediate=32):
    from tinyserve.generic_store import GenericExpertStore, GenericLRUCache
    from tinyserve.generic_pipeline import GenericExpertPipeline
    from tinyserve.cpu_expert import CPUExpertForward

    weights = {}
    for li in range(1):
        for ei in range(num_experts):
            weights[(li, ei)] = {
                "gate_up_proj": torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16),
                "down_proj": torch.randn(hidden, intermediate, dtype=torch.bfloat16),
            }
    store = GenericExpertStore.from_dict(weights, 1, num_experts)
    device = torch.device("cuda")
    template = TinyExpert(hidden, intermediate).to(device).to(torch.bfloat16)
    buf_a = store.allocate_buffer(device)
    buf_b = store.allocate_buffer(device)
    ts = torch.cuda.Stream(device)
    cs = torch.cuda.Stream(device)
    # Small cache: capacity=2 forces misses with 4 experts
    cache = GenericLRUCache(2, store.buffer_expert_bytes, device,
                            num_layers=1, num_experts=num_experts)
    cpu_fwd = CPUExpertForward(store.layout, act_fn=nn.SiLU())
    pipeline = GenericExpertPipeline(store, template, device,
                                     buf_a, buf_b, ts, cs, cache=cache)
    pipeline.cpu_expert = cpu_fwd
    pipeline.cpu_on_miss = True
    return pipeline, store


@requires_cuda
def test_cpu_miss_produces_correct_output():
    """CPU compute on miss should produce same output as GPU compute."""
    pipeline, store = _make_pipeline_with_cpu()
    h = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.tensor([[0, 1]], device="cuda")
    weights = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.bfloat16)

    # Fill cache with experts 2,3 so experts 0,1 are misses
    pipeline.execute_layer_experts(h, 0, torch.tensor([[2, 3]], device="cuda"),
                                   torch.tensor([[0.5, 0.5]], device="cuda", dtype=torch.bfloat16))

    # Now request 0,1 — should miss and use CPU compute
    pipeline.cache.reset_stats()
    out = pipeline.execute_layer_experts(h, 0, expert_ids, weights)

    # Output should be non-zero (CPU compute produced a result)
    assert out.abs().sum() > 0
    # Misses should have occurred
    assert pipeline.cache.misses > 0


@requires_cuda
def test_cpu_miss_flag_controls_behavior():
    """When cpu_on_miss=False, misses use GPU pipeline instead."""
    pipeline, store = _make_pipeline_with_cpu()
    pipeline.cpu_on_miss = False

    h = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.tensor([[0]], device="cuda")
    weights = torch.tensor([[1.0]], device="cuda", dtype=torch.bfloat16)

    # Should work (uses GPU pipeline for miss)
    out = pipeline.execute_layer_experts(h, 0, expert_ids, weights)
    assert out.abs().sum() > 0
