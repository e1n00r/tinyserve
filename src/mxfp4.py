"""MXFP4 dequantization for GPT-OSS expert weights."""

import torch

from .config import FP4_LUT

_lut_cache: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}


def _get_lut(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dtype, device)
    if key not in _lut_cache:
        _lut_cache[key] = FP4_LUT.to(dtype=dtype, device=device)
    return _lut_cache[key]


def dequant_mxfp4_no_transpose(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize MXFP4 to [out_features, in_features] for F.linear.

    If `out` is provided, writes into it (must be [out_features, G*B*2] in dtype).
    """
    lut = _get_lut(dtype, blocks.device)

    out_features, G, B = blocks.shape
    rows = out_features * G

    bf = blocks.reshape(rows, B)
    si = (scales.to(torch.int32) - 127).reshape(rows, 1)

    if out is None:
        out = torch.empty(rows, B * 2, dtype=dtype, device=blocks.device)
    else:
        out = out.view(rows, B * 2)

    out[:, 0::2] = lut[(bf & 0x0F).to(torch.int32)]
    out[:, 1::2] = lut[(bf >> 4).to(torch.int32)]
    torch.ldexp(out, si, out=out)

    return out.view(out_features, G * B * 2)


# Backward-compat for tests
def dequant_mxfp4(blocks, scales, dtype=torch.bfloat16):
    w = dequant_mxfp4_no_transpose(blocks, scales, dtype)
    return w.T.contiguous()


def dequant_single_expert(gate_up_blocks, gate_up_scales, down_blocks, down_scales, dtype=torch.bfloat16):
    gate_up_w = dequant_mxfp4(gate_up_blocks, gate_up_scales, dtype)
    down_w = dequant_mxfp4(down_blocks, down_scales, dtype)
    return gate_up_w, down_w
