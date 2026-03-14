import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import HIDDEN_SIZE


@pytest.fixture
def mock_expert_store(tmp_path, monkeypatch):
    """Create a tiny mock expert store with 2 layers, 8 experts."""
    from safetensors.torch import save_file
    import src.expert_store as es_mod
    from src.expert_store import ExpertStore

    monkeypatch.setattr(es_mod, "NUM_LAYERS", 2)

    num_experts = 8
    for layer_idx in range(2):
        tensors = {
            "gate_up_proj_blocks": torch.randint(0, 256, (num_experts, 2 * HIDDEN_SIZE, HIDDEN_SIZE // 32, 16), dtype=torch.uint8),
            "gate_up_proj_scales": torch.randint(120, 135, (num_experts, 2 * HIDDEN_SIZE, HIDDEN_SIZE // 32), dtype=torch.uint8),
            "gate_up_proj_bias": torch.randn(num_experts, 2 * HIDDEN_SIZE, dtype=torch.float32) * 0.01,
            "down_proj_blocks": torch.randint(0, 256, (num_experts, HIDDEN_SIZE, HIDDEN_SIZE // 32, 16), dtype=torch.uint8),
            "down_proj_scales": torch.randint(120, 135, (num_experts, HIDDEN_SIZE, HIDDEN_SIZE // 32), dtype=torch.uint8),
            "down_proj_bias": torch.randn(num_experts, HIDDEN_SIZE, dtype=torch.float32) * 0.01,
        }
        save_file(tensors, str(tmp_path / f"experts_L{layer_idx:02d}.safetensors"))

    store = ExpertStore(str(tmp_path))
    store.load()
    return store


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
