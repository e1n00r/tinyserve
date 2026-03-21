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

    from src.generic_store import GenericLRUCache
    pipeline = GenericExpertPipeline(
        store, template, device,
        buf_a=store.allocate_buffer(device),
        buf_b=store.allocate_buffer(device),
        transfer_stream=torch.cuda.Stream(device),
        compute_stream=torch.cuda.Stream(device),
        cache=GenericLRUCache(4, store.expert_bytes, device),
    )

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

    from src.generic_store import GenericLRUCache
    pipeline = GenericExpertPipeline(
        store, template, device,
        buf_a=store.allocate_buffer(device),
        buf_b=store.allocate_buffer(device),
        transfer_stream=torch.cuda.Stream(device),
        compute_stream=torch.cuda.Stream(device),
        cache=GenericLRUCache(8, store.expert_bytes, device),
    )

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
def test_least_stale_hits_after_begin_pass():
    """LeastStalePolicy: after begin_pass(), experts from the previous token are hits."""
    from src.generic_store import GenericExpertStore, GenericLRUCache
    from src.generic_pipeline import GenericExpertPipeline

    hidden, intermediate = 32, 64
    num_layers, num_experts = 4, 8
    expert_weights = _build_expert_weights(num_layers, num_experts, hidden, intermediate)
    store = GenericExpertStore.from_dict(expert_weights, num_layers, num_experts, fp8=True)
    device = torch.device("cuda")
    template = TinySwiGLUExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    cache = GenericLRUCache(32, store.buffer_expert_bytes, device, policy="ls")
    shared_transfer = torch.cuda.Stream(device)
    shared_compute = torch.cuda.Stream(device)

    pipelines = [
        GenericExpertPipeline(
            store, template, device,
            buf_a=store.allocate_buffer(device),
            buf_b=store.allocate_buffer(device),
            transfer_stream=shared_transfer,
            compute_stream=shared_compute,
            cache=cache,
        )
        for _ in range(num_layers)
    ]

    h = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    expert_indices = torch.tensor([[2, 5]])
    rw = torch.tensor([[0.6, 0.4]], device=device, dtype=torch.bfloat16)

    # Token 1: populate cache across all layers (all misses)
    for li, p in enumerate(pipelines):
        p.execute_layer_experts(h, li, expert_indices, rw)
    torch.cuda.synchronize()
    assert cache.hits == 0
    assert cache.misses == num_layers * 2
    cache.reset_stats()

    # begin_pass() simulates the start of token 2 (called on layer 0)
    cache.begin_pass()

    # Token 2: same experts → all hits
    for li, p in enumerate(pipelines):
        p.execute_layer_experts(h, li, expert_indices, rw)
    torch.cuda.synchronize()
    assert cache.hits == num_layers * 2, f"Expected {num_layers * 2} hits, got {cache.hits}"
    assert cache.misses == 0


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

    pipeline = GenericExpertPipeline(
        store, template, device,
        buf_a=store.allocate_buffer(device),
        buf_b=store.allocate_buffer(device),
        transfer_stream=torch.cuda.Stream(device),
        compute_stream=torch.cuda.Stream(device),
        cache=None,
    )

    num_tokens = 3
    h = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    expert_indices = torch.tensor([[0, 1], [2, 3], [4, 5]])
    routing_weights = torch.softmax(
        torch.randn(num_tokens, 2, device=device), dim=-1
    ).to(torch.bfloat16)

    out = pipeline.execute_layer_experts(h, 0, expert_indices, routing_weights)
    assert out.shape == (num_tokens, hidden)
    assert not torch.isnan(out).any()
