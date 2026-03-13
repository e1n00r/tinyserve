"""Test MXFP4 dequantization matches HuggingFace reference."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mxfp4 import dequant_mxfp4


def test_dequant_known_values():
    """Test with hand-crafted inputs where we know the expected output."""
    # Single row: 1 group of 16 bytes = 32 FP4 values
    # FP4 LUT: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    #           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    #
    # Byte 0x21 = low nibble 1 (0.5), high nibble 2 (1.0)
    # Scale = 127 means exponent = 0, so no scaling (2^0 = 1)

    blocks = torch.tensor([[[0x21, 0x43, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00]]], dtype=torch.uint8)
    scales = torch.tensor([[127]], dtype=torch.uint8)

    out = dequant_mxfp4(blocks, scales, dtype=torch.float32)
    # blocks shape: [1, 1, 16], scales shape: [1, 1]
    # output shape should be: transposed from [1, 32] -> [32, 1]

    # Low nibble of 0x21 = 1 -> 0.5, high nibble = 2 -> 1.0
    # Low nibble of 0x43 = 3 -> 1.5, high nibble = 4 -> 2.0
    # Rest are 0x00 -> low=0 (0.0), high=0 (0.0)
    # Interleaved: [0.5, 1.0, 1.5, 2.0, 0.0, 0.0, ...]
    assert out[0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert out[1, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert out[2, 0].item() == pytest.approx(1.5, abs=1e-6)
    assert out[3, 0].item() == pytest.approx(2.0, abs=1e-6)


def test_dequant_with_scaling():
    """Test that E8M0 scale exponent works correctly."""
    # Scale = 128 means exponent = 1, so multiply by 2^1 = 2
    blocks = torch.tensor([[[0x02, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00]]], dtype=torch.uint8)
    scales = torch.tensor([[128]], dtype=torch.uint8)

    out = dequant_mxfp4(blocks, scales, dtype=torch.float32)
    # Low nibble of 0x02 = 2 -> LUT[2] = 1.0, scaled by 2^1 = 2.0
    assert out[0, 0].item() == pytest.approx(2.0, abs=1e-6)


def test_dequant_negative_values():
    """Test negative FP4 values (indices 8-15)."""
    # 0x0A = low nibble A=10 -> LUT[10] = -1.0, high nibble 0 -> 0.0
    blocks = torch.tensor([[[0x0A, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00]]], dtype=torch.uint8)
    scales = torch.tensor([[127]], dtype=torch.uint8)

    out = dequant_mxfp4(blocks, scales, dtype=torch.float32)
    assert out[0, 0].item() == pytest.approx(-1.0, abs=1e-6)


def test_dequant_output_shape():
    """Test output shape with multi-row input."""
    # Shape: [out_features=4, groups=2, 16] -> after dequant and transpose
    # Each group produces 32 values, so 2 groups = 64 values per row
    # 4 rows of 64 values, transposed -> [64, 4]
    blocks = torch.zeros(4, 2, 16, dtype=torch.uint8)
    scales = torch.full((4, 2), 127, dtype=torch.uint8)

    out = dequant_mxfp4(blocks, scales, dtype=torch.float32)
    assert out.shape == (64, 4)


def test_dequant_matches_hf_reference():
    """Compare our dequant against HF's convert_moe_packed_tensors."""
    try:
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    except ImportError:
        pytest.skip("transformers not available")

    torch.manual_seed(42)
    # Simulate a small expert weight shape — HF expects [num_experts, out_features, groups, 16]
    num_experts = 2
    out_features = 32
    groups = 4  # in_features // 32
    blocks = torch.randint(0, 256, (num_experts, out_features, groups, 16), dtype=torch.uint8)
    scales = torch.randint(120, 135, (num_experts, out_features, groups), dtype=torch.uint8)

    hf_out = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)

    # Our function works per-expert, so test each expert separately
    for i in range(num_experts):
        our_out = dequant_mxfp4(blocks[i], scales[i], dtype=torch.bfloat16)
        torch.testing.assert_close(our_out.to(hf_out.device), hf_out[i], rtol=0, atol=0)
