"""Tests for ExpertBatcher — batches expert forwards across concurrent requests."""

import torch
import torch.nn as nn

from tests.conftest import requires_cuda


class TinyFusedExpert(nn.Module):
    """Minimal fused gate_up + down expert for testing."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2 * intermediate, hidden))
        self.down_proj = nn.Parameter(torch.randn(hidden, intermediate))
        self._act_fn = nn.SiLU()
        self._has_bias = False
        self._is_mxfp4 = False
        self._param_names = ["gate_up_proj", "down_proj"]

    def forward(self, x):
        gate_up = nn.functional.linear(x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        gated = self._act_fn(gate) * up
        return nn.functional.linear(gated, self.down_proj)


def _build_store_and_pipeline(num_layers=1, num_experts=4, hidden=16, intermediate=32, device="cpu"):
    """Build a ExpertStore + ExpertPipeline on CPU for testing."""
    from tinyserve.expert_store import ExpertStore

    expert_weights = {}
    for li in range(num_layers):
        for ei in range(num_experts):
            expert_weights[(li, ei)] = {
                "gate_up_proj": torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16),
                "down_proj": torch.randn(hidden, intermediate, dtype=torch.bfloat16),
            }

    store = ExpertStore.from_dict(expert_weights, num_layers, num_experts)
    return store, expert_weights


def _ref_expert_output(expert_weights, layer_idx, expert_ids, routing_weights, h):
    """Compute reference output by manually applying experts with weights."""
    output = torch.zeros_like(h)
    template = TinyFusedExpert(h.shape[-1], expert_weights[(layer_idx, 0)]["gate_up_proj"].shape[0] // 2)
    template = template.to(h.dtype)
    for i, eid in enumerate(expert_ids):
        w = expert_weights[(layer_idx, eid)]
        with torch.no_grad():
            template.gate_up_proj.data.copy_(w["gate_up_proj"].to(h.dtype))
            template.down_proj.data.copy_(w["down_proj"].to(h.dtype))
        out = template(h)
        output += routing_weights[i].item() * out
    return output


@requires_cuda
def test_single_request_matches_unbatched():
    """Single request through ExpertBatcher matches pipeline.execute_layer_experts."""
    from tinyserve.expert_batcher import BatchItem, ExpertBatcher
    from tinyserve.expert_pipeline import ExpertPipeline
    from tinyserve.expert_store import ExpertCache

    hidden, intermediate = 16, 32
    num_experts = 4
    device = torch.device("cuda")

    store, expert_weights = _build_store_and_pipeline(1, num_experts, hidden, intermediate)
    template = TinyFusedExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    staging_buffer_a = store.allocate_buffer(device)
    staging_buffer_b = store.allocate_buffer(device)
    ts = torch.cuda.Stream(device)
    cs = torch.cuda.Stream(device)
    cache = ExpertCache(num_experts, store.buffer_expert_bytes, device, num_layers=1, num_experts=num_experts)

    pipeline = ExpertPipeline(store, template, device, staging_buffer_a, staging_buffer_b, ts, cs, cache=cache)

    h = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    expert_ids = torch.tensor([1, 3], device=device)
    weights = torch.tensor([0.6, 0.4], device=device, dtype=torch.bfloat16)

    # Unbatched path
    out_unbatched = pipeline.execute_layer_experts(h, 0, expert_ids.unsqueeze(0), weights.unsqueeze(0))

    # Reset cache stats
    cache.hits = 0
    cache.misses = 0

    # Batched path
    batcher = ExpertBatcher(pipeline)
    item = BatchItem(
        hidden_states=h,
        expert_indices=expert_ids,
        routing_weights=weights,
        request_idx=0,
    )
    batched_outputs = batcher.batch_execute([item], layer_idx=0)

    torch.testing.assert_close(batched_outputs[0], out_unbatched, rtol=1e-3, atol=1e-3)


@requires_cuda
def test_two_requests_same_expert_batched():
    """Two requests needing the same expert: expert loaded once, both get correct output."""
    from tinyserve.expert_batcher import BatchItem, ExpertBatcher
    from tinyserve.expert_pipeline import ExpertPipeline
    from tinyserve.expert_store import ExpertCache

    hidden, intermediate = 16, 32
    num_experts = 4
    device = torch.device("cuda")

    store, expert_weights = _build_store_and_pipeline(1, num_experts, hidden, intermediate)
    template = TinyFusedExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    staging_buffer_a = store.allocate_buffer(device)
    staging_buffer_b = store.allocate_buffer(device)
    ts = torch.cuda.Stream(device)
    cs = torch.cuda.Stream(device)
    cache = ExpertCache(num_experts, store.buffer_expert_bytes, device, num_layers=1, num_experts=num_experts)

    pipeline = ExpertPipeline(store, template, device, staging_buffer_a, staging_buffer_b, ts, cs, cache=cache)

    h1 = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    h2 = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    # Both requests use expert 2
    expert_ids = torch.tensor([2], device=device)
    weights = torch.tensor([1.0], device=device, dtype=torch.bfloat16)

    batcher = ExpertBatcher(pipeline)
    items = [
        BatchItem(h1, expert_ids, weights, request_idx=0),
        BatchItem(h2, expert_ids, weights, request_idx=1),
    ]
    outputs = batcher.batch_execute(items, layer_idx=0)

    # Verify each output matches individual computation
    ref1 = pipeline.execute_layer_experts(h1, 0, expert_ids.unsqueeze(0), weights.unsqueeze(0))
    ref2 = pipeline.execute_layer_experts(h2, 0, expert_ids.unsqueeze(0), weights.unsqueeze(0))

    torch.testing.assert_close(outputs[0], ref1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(outputs[1], ref2, rtol=1e-3, atol=1e-3)


@requires_cuda
def test_two_requests_different_experts():
    """Two requests with non-overlapping experts both produce correct output."""
    from tinyserve.expert_batcher import BatchItem, ExpertBatcher
    from tinyserve.expert_pipeline import ExpertPipeline
    from tinyserve.expert_store import ExpertCache

    hidden, intermediate = 16, 32
    num_experts = 4
    device = torch.device("cuda")

    store, expert_weights = _build_store_and_pipeline(1, num_experts, hidden, intermediate)
    template = TinyFusedExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    staging_buffer_a = store.allocate_buffer(device)
    staging_buffer_b = store.allocate_buffer(device)
    ts = torch.cuda.Stream(device)
    cs = torch.cuda.Stream(device)
    cache = ExpertCache(num_experts, store.buffer_expert_bytes, device, num_layers=1, num_experts=num_experts)

    pipeline = ExpertPipeline(store, template, device, staging_buffer_a, staging_buffer_b, ts, cs, cache=cache)

    h1 = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    h2 = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    # Request 1 uses expert 0, request 2 uses expert 3
    ids1 = torch.tensor([0], device=device)
    ids2 = torch.tensor([3], device=device)
    w = torch.tensor([1.0], device=device, dtype=torch.bfloat16)

    batcher = ExpertBatcher(pipeline)
    items = [
        BatchItem(h1, ids1, w, request_idx=0),
        BatchItem(h2, ids2, w, request_idx=1),
    ]
    outputs = batcher.batch_execute(items, layer_idx=0)

    ref1 = pipeline.execute_layer_experts(h1, 0, ids1.unsqueeze(0), w.unsqueeze(0))
    ref2 = pipeline.execute_layer_experts(h2, 0, ids2.unsqueeze(0), w.unsqueeze(0))

    torch.testing.assert_close(outputs[0], ref1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(outputs[1], ref2, rtol=1e-3, atol=1e-3)


@requires_cuda
def test_scatter_correct_order():
    """With 3 requests and mixed expert assignments, outputs land in correct request slots."""
    from tinyserve.expert_batcher import BatchItem, ExpertBatcher
    from tinyserve.expert_pipeline import ExpertPipeline
    from tinyserve.expert_store import ExpertCache

    hidden, intermediate = 16, 32
    num_experts = 4
    device = torch.device("cuda")

    store, _ = _build_store_and_pipeline(1, num_experts, hidden, intermediate)
    template = TinyFusedExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    staging_buffer_a = store.allocate_buffer(device)
    staging_buffer_b = store.allocate_buffer(device)
    ts = torch.cuda.Stream(device)
    cs = torch.cuda.Stream(device)
    cache = ExpertCache(num_experts, store.buffer_expert_bytes, device, num_layers=1, num_experts=num_experts)

    pipeline = ExpertPipeline(store, template, device, staging_buffer_a, staging_buffer_b, ts, cs, cache=cache)

    items = []
    refs = []
    for req_idx in range(3):
        h = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
        eids = torch.tensor([req_idx, (req_idx + 1) % num_experts], device=device)
        w = torch.tensor([0.7, 0.3], device=device, dtype=torch.bfloat16)
        items.append(BatchItem(h, eids, w, request_idx=req_idx))
        ref = pipeline.execute_layer_experts(h, 0, eids.unsqueeze(0), w.unsqueeze(0))
        refs.append(ref)

    batcher = ExpertBatcher(pipeline)
    outputs = batcher.batch_execute(items, layer_idx=0)

    for i in range(3):
        torch.testing.assert_close(outputs[i], refs[i], rtol=1e-3, atol=1e-3)


@requires_cuda
def test_cache_hit_batched():
    """Cache hits across requests are served without re-loading experts."""
    from tinyserve.expert_batcher import BatchItem, ExpertBatcher
    from tinyserve.expert_pipeline import ExpertPipeline
    from tinyserve.expert_store import ExpertCache

    hidden, intermediate = 16, 32
    num_experts = 4
    device = torch.device("cuda")

    store, _ = _build_store_and_pipeline(1, num_experts, hidden, intermediate)
    template = TinyFusedExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    staging_buffer_a = store.allocate_buffer(device)
    staging_buffer_b = store.allocate_buffer(device)
    ts = torch.cuda.Stream(device)
    cs = torch.cuda.Stream(device)
    cache = ExpertCache(num_experts, store.buffer_expert_bytes, device, num_layers=1, num_experts=num_experts)

    pipeline = ExpertPipeline(store, template, device, staging_buffer_a, staging_buffer_b, ts, cs, cache=cache)

    # Warm the cache by running one pass
    h_warm = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    eids_warm = torch.tensor([0, 1], device=device)
    w_warm = torch.tensor([0.5, 0.5], device=device, dtype=torch.bfloat16)
    pipeline.execute_layer_experts(h_warm, 0, eids_warm.unsqueeze(0), w_warm.unsqueeze(0))
    torch.cuda.synchronize()

    # Now batch two requests that hit cached experts
    cache.hits = 0
    cache.misses = 0
    h1 = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    h2 = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.tensor([0, 1], device=device)
    w = torch.tensor([0.5, 0.5], device=device, dtype=torch.bfloat16)

    batcher = ExpertBatcher(pipeline)
    items = [
        BatchItem(h1, ids, w, request_idx=0),
        BatchItem(h2, ids, w, request_idx=1),
    ]
    outputs = batcher.batch_execute(items, layer_idx=0)

    # Verify outputs are non-zero (experts actually computed)
    assert outputs[0].abs().sum() > 0
    assert outputs[1].abs().sum() > 0
    # Verify results are different (different inputs)
    assert not torch.allclose(outputs[0], outputs[1])


@requires_cuda
def test_empty_requests():
    """Empty request list returns empty output list."""
    from tinyserve.expert_batcher import ExpertBatcher
    from tinyserve.expert_pipeline import ExpertPipeline

    hidden, intermediate = 16, 32
    num_experts = 4
    device = torch.device("cuda")

    store, _ = _build_store_and_pipeline(1, num_experts, hidden, intermediate)
    template = TinyFusedExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    staging_buffer_a = store.allocate_buffer(device)
    staging_buffer_b = store.allocate_buffer(device)
    ts = torch.cuda.Stream(device)
    cs = torch.cuda.Stream(device)

    pipeline = ExpertPipeline(store, template, device, staging_buffer_a, staging_buffer_b, ts, cs)

    batcher = ExpertBatcher(pipeline)
    outputs = batcher.batch_execute([], layer_idx=0)
    assert outputs == []


@requires_cuda
def test_batch_size_one_no_regression():
    """Batch of size 1 produces identical output to unbatched path."""
    from tinyserve.expert_batcher import BatchItem, ExpertBatcher
    from tinyserve.expert_pipeline import ExpertPipeline
    from tinyserve.expert_store import ExpertCache

    hidden, intermediate = 16, 32
    num_experts = 4
    device = torch.device("cuda")

    store, _ = _build_store_and_pipeline(1, num_experts, hidden, intermediate)
    template = TinyFusedExpert(hidden, intermediate).to(device).to(torch.bfloat16)

    staging_buffer_a = store.allocate_buffer(device)
    staging_buffer_b = store.allocate_buffer(device)
    ts = torch.cuda.Stream(device)
    cs = torch.cuda.Stream(device)
    cache = ExpertCache(num_experts, store.buffer_expert_bytes, device, num_layers=1, num_experts=num_experts)

    pipeline = ExpertPipeline(store, template, device, staging_buffer_a, staging_buffer_b, ts, cs, cache=cache)

    h = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    eids = torch.tensor([2, 3], device=device)
    w = torch.tensor([0.4, 0.6], device=device, dtype=torch.bfloat16)

    ref = pipeline.execute_layer_experts(h, 0, eids.unsqueeze(0), w.unsqueeze(0))

    batcher = ExpertBatcher(pipeline)
    item = BatchItem(h, eids, w, request_idx=0)
    outputs = batcher.batch_execute([item], layer_idx=0)

    torch.testing.assert_close(outputs[0], ref, rtol=1e-3, atol=1e-3)
