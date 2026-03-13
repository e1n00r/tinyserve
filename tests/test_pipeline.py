"""Test that pipelined expert execution matches blocking execution."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import HIDDEN_SIZE, NUM_EXPERTS_PER_TOK
from src.expert_store import ExpertBuffer, ExpertStore
from src.experts import expert_forward
from src.pipeline import ExpertPipeline


@pytest.fixture
def mock_expert_store(tmp_path):
    """Create a tiny mock expert store with random weights for testing."""
    from safetensors.torch import save_file

    # Use small dimensions for fast tests
    num_experts = 8
    hidden = HIDDEN_SIZE
    intermediate = HIDDEN_SIZE

    for layer_idx in range(2):  # only 2 layers for testing
        tensors = {
            "gate_up_proj_blocks": torch.randint(
                0, 256, (num_experts, 2 * intermediate, hidden // 32, 16), dtype=torch.uint8
            ),
            "gate_up_proj_scales": torch.randint(
                120, 135, (num_experts, 2 * intermediate, hidden // 32), dtype=torch.uint8
            ),
            "gate_up_proj_bias": torch.randn(num_experts, 2 * intermediate, dtype=torch.float32) * 0.01,
            "down_proj_blocks": torch.randint(
                0, 256, (num_experts, hidden, intermediate // 32, 16), dtype=torch.uint8
            ),
            "down_proj_scales": torch.randint(
                120, 135, (num_experts, hidden, intermediate // 32), dtype=torch.uint8
            ),
            "down_proj_bias": torch.randn(num_experts, hidden, dtype=torch.float32) * 0.01,
        }
        save_file(tensors, str(tmp_path / f"experts_L{layer_idx:02d}.safetensors"))

    return tmp_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pipeline_matches_blocking(mock_expert_store):
    """Pipelined execution must produce identical output to blocking."""
    # Patch NUM_LAYERS for this test
    import src.expert_store as es_mod
    original_num_layers = es_mod.NUM_LAYERS
    es_mod.NUM_LAYERS = 2

    try:
        store = ExpertStore(str(mock_expert_store))
        store.load()

        device = torch.device("cuda")
        pipeline = ExpertPipeline(store, device)
        buf = ExpertBuffer(device)

        h = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        expert_indices = torch.tensor([[0, 3, 5, 7]])  # top-4 of 8 experts
        weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device, dtype=torch.bfloat16)
        layer_idx = 0

        # Blocking execution
        blocking_out = torch.zeros_like(h)
        for k in range(NUM_EXPERTS_PER_TOK):
            eidx = expert_indices[0, k].item()
            store.copy_to_buffer(buf, layer_idx, eidx, non_blocking=False)
            out = expert_forward(
                h, buf.gate_up_blocks, buf.gate_up_scales, buf.gate_up_bias,
                buf.down_blocks, buf.down_scales, buf.down_bias,
            )
            blocking_out += weights[0, k] * out

        # Pipelined execution
        pipeline_out = pipeline.execute_layer_experts(h, layer_idx, expert_indices, weights)

        torch.testing.assert_close(pipeline_out, blocking_out, rtol=0, atol=0)
    finally:
        es_mod.NUM_LAYERS = original_num_layers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pipeline_multiple_tokens(mock_expert_store):
    """Pipeline handles multiple tokens correctly."""
    import src.expert_store as es_mod
    original_num_layers = es_mod.NUM_LAYERS
    es_mod.NUM_LAYERS = 2

    try:
        store = ExpertStore(str(mock_expert_store))
        store.load()

        device = torch.device("cuda")
        pipeline = ExpertPipeline(store, device)

        num_tokens = 3
        h = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        expert_indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6]])
        weights = torch.softmax(torch.randn(num_tokens, 4, device=device), dim=-1).to(torch.bfloat16)

        out = pipeline.execute_layer_experts(h, 0, expert_indices, weights)
        assert out.shape == (num_tokens, HIDDEN_SIZE)
        assert not torch.isnan(out).any()
    finally:
        es_mod.NUM_LAYERS = original_num_layers
