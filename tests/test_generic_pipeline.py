"""Test generic pipeline with template expert modules."""

import torch
import torch.nn as nn

from tests.conftest import requires_cuda


class TinySwiGLUExpert(nn.Module):
    """Minimal SwiGLU expert matching Qwen/Mixtral structure."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def _build_expert_weights(num_layers, num_experts, hidden, intermediate, dtype=torch.bfloat16):
    """Create random expert weights matching TinySwiGLUExpert layout."""
    weights = {}
    for li in range(num_layers):
        for ei in range(num_experts):
            weights[(li, ei)] = {
                "gate_proj.weight": torch.randn(intermediate, hidden, dtype=dtype),
                "up_proj.weight": torch.randn(intermediate, hidden, dtype=dtype),
                "down_proj.weight": torch.randn(hidden, intermediate, dtype=dtype),
            }
    return weights


@requires_cuda
def test_template_expert_weight_swap():
    """Swapping weights into a template expert produces correct output."""
    from src.generic_store import GenericExpertStore
    from src.generic_pipeline import swap_weights_and_forward

    hidden, intermediate = 32, 64
    expert_weights = _build_expert_weights(1, 2, hidden, intermediate)

    store = GenericExpertStore.from_dict(expert_weights, 1, 2)
    device = torch.device("cuda")
    buf = store.allocate_buffer(device)
    template = TinySwiGLUExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    h = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)

    for ei in range(2):
        store.copy_to_buffer(buf, 0, ei, non_blocking=False)
        torch.cuda.synchronize()
        out_offloaded = swap_weights_and_forward(template, buf, h)

        ref = TinySwiGLUExpert(hidden, intermediate).to(device).to(torch.bfloat16)
        with torch.no_grad():
            for name, tensor in expert_weights[(0, ei)].items():
                parts = name.split(".")
                param = ref
                for part in parts[:-1]:
                    param = getattr(param, part)
                getattr(param, parts[-1]).copy_(tensor.to(device))
        out_ref = ref(h)

        torch.testing.assert_close(out_offloaded, out_ref, rtol=0, atol=0)


@requires_cuda
def test_generic_pipeline_matches_direct():
    """Full pipeline (cache + double-buffer) matches direct expert computation."""
    from src.generic_store import GenericExpertStore
    from src.generic_pipeline import GenericExpertPipeline

    hidden, intermediate = 32, 64
    num_layers, num_experts, top_k = 2, 8, 2

    expert_weights = _build_expert_weights(num_layers, num_experts, hidden, intermediate)
    store = GenericExpertStore.from_dict(expert_weights, num_layers, num_experts)
    device = torch.device("cuda")
    template = TinySwiGLUExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    pipeline = GenericExpertPipeline(store, template, device, cache_capacity=4)

    h = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    expert_indices = torch.tensor([[2, 5]])
    routing_weights = torch.tensor([[0.6, 0.4]], device=device, dtype=torch.bfloat16)

    out_pipeline = pipeline.execute_layer_experts(h, 0, expert_indices, routing_weights)
    torch.cuda.synchronize()

    out_direct = torch.zeros_like(h)
    for k in range(top_k):
        eidx = expert_indices[0, k].item()
        ref = TinySwiGLUExpert(hidden, intermediate).to(device).to(torch.bfloat16)
        with torch.no_grad():
            for name, tensor in expert_weights[(0, eidx)].items():
                parts = name.split(".")
                mod = ref
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                getattr(mod, parts[-1]).copy_(tensor.to(device))
        out_direct += routing_weights[0, k] * ref(h)

    torch.testing.assert_close(out_pipeline, out_direct, rtol=0, atol=0)


@requires_cuda
def test_cache_hits_match_misses():
    """Second call (all cache hits) produces identical output to first call (all misses)."""
    from src.generic_store import GenericExpertStore
    from src.generic_pipeline import GenericExpertPipeline

    hidden, intermediate = 32, 64
    expert_weights = _build_expert_weights(1, 4, hidden, intermediate)
    store = GenericExpertStore.from_dict(expert_weights, 1, 4)
    device = torch.device("cuda")
    template = TinySwiGLUExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    pipeline = GenericExpertPipeline(store, template, device, cache_capacity=8)

    h = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    expert_indices = torch.tensor([[0, 3]])
    routing_weights = torch.tensor([[0.7, 0.3]], device=device, dtype=torch.bfloat16)

    out_miss = pipeline.execute_layer_experts(h, 0, expert_indices, routing_weights)
    torch.cuda.synchronize()
    out_hit = pipeline.execute_layer_experts(h, 0, expert_indices, routing_weights)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_hit, out_miss, rtol=0, atol=0)
    assert pipeline.cache.hits == 2
    assert pipeline.cache.misses == 2


@requires_cuda
def test_multi_token():
    """Pipeline handles multiple tokens correctly."""
    from src.generic_store import GenericExpertStore
    from src.generic_pipeline import GenericExpertPipeline

    hidden, intermediate = 32, 64
    expert_weights = _build_expert_weights(1, 8, hidden, intermediate)
    store = GenericExpertStore.from_dict(expert_weights, 1, 8)
    device = torch.device("cuda")
    template = TinySwiGLUExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    pipeline = GenericExpertPipeline(store, template, device, cache_capacity=0)

    num_tokens = 3
    h = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    expert_indices = torch.tensor([[0, 1], [2, 3], [4, 5]])
    routing_weights = torch.softmax(
        torch.randn(num_tokens, 2, device=device), dim=-1
    ).to(torch.bfloat16)

    out = pipeline.execute_layer_experts(h, 0, expert_indices, routing_weights)
    assert out.shape == (num_tokens, hidden)
    assert not torch.isnan(out).any()
