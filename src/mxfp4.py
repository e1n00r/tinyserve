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

    # Chunk processing to control memory (matches HF implementation)
    chunk = 32768 * 1024
    for r0 in range(0, rows_total, chunk):
        r1 = min(r0 + chunk, rows_total)
        blk = blocks_flat[r0:r1]
        exp = scales_flat[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]
        torch.ldexp(sub, exp, out=sub)

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
