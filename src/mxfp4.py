"""MXFP4 dequantization for GPT-OSS expert weights.

Ported from transformers.integrations.mxfp4.convert_moe_packed_tensors.
"""

import math

import torch

from .config import FP4_LUT


def dequant_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize MXFP4 packed tensors to dense float.

    Args:
        blocks: uint8 tensor, shape [..., G, 16] where each byte holds 2 FP4 values
        scales: uint8 tensor, shape [..., G] with E8M0 block scales
        dtype: output dtype

    Returns:
        Dequantized tensor, shape [..., G*32] with the last two dims merged and
        transposed as [in_features, out_features] (ready for matmul).
    """
    device = blocks.device
    lut = FP4_LUT.to(dtype=dtype, device=device)
    scales_int = scales.to(torch.int32) - 127

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks_flat = blocks.reshape(rows_total, B)
    scales_flat = scales_int.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=device)

    # Process all rows — single expert fits in VRAM easily
    idx_lo = (blocks_flat & 0x0F).to(torch.int32)
    idx_hi = (blocks_flat >> 4).to(torch.int32)

    out[:, 0::2] = lut[idx_lo]
    out[:, 1::2] = lut[idx_hi]
    torch.ldexp(out, scales_flat, out=out)
    del idx_lo, idx_hi

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    return out.transpose(-2, -1).contiguous()


def dequant_single_expert(
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a single expert's weights.

    Args:
        gate_up_blocks: [2*intermediate, hidden//32, 16] uint8
        gate_up_scales: [2*intermediate, hidden//32] uint8
        down_blocks: [hidden, intermediate//32, 16] uint8
        down_scales: [hidden, intermediate//32] uint8

    Returns:
        (gate_up_weight, down_weight) both in dtype, shapes:
        gate_up: [hidden_size, 2*intermediate_size]
        down:    [intermediate_size, hidden_size]
    """
    gate_up_w = dequant_mxfp4(gate_up_blocks, gate_up_scales, dtype)
    down_w = dequant_mxfp4(down_blocks, down_scales, dtype)
    return gate_up_w, down_w
