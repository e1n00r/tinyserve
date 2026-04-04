"""MXFP4 dequantization utilities."""

import torch

FP4_LUT = torch.tensor(
    [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.bfloat16,
)

_lut_cache: dict[tuple, torch.Tensor] = {}


def get_lut(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
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
    """Dequantize MXFP4 to [out_features, in_features] for F.linear."""
    lut = get_lut(dtype, blocks.device)

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


def dequant_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize MXFP4, returning [in_features, out_features] (transposed for F.linear)."""
    return dequant_mxfp4_no_transpose(blocks, scales, dtype).T.contiguous()
