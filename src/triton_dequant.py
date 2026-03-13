"""Triton kernel for fused MXFP4 dequant + vector-matrix multiply.

For single-token decode: computes y = x @ dequant(W).T where W is in MXFP4 format.
Fuses nibble unpack, LUT lookup, E8M0 scaling, and dot product in a single kernel,
avoiding materialization of the full bf16 weight matrix.
"""

import torch
import triton
import triton.language as tl

# FP4 E2M1 values as bf16 raw bits, indexed by nibble value 0-15
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@triton.jit
def _dequant_vecmat_kernel(
    x_ptr,         # [K] bf16 input vector
    blocks_ptr,    # [N, G, B] uint8 packed FP4
    scales_ptr,    # [N, G] uint8 E8M0
    bias_ptr,      # [N] f32 bias (or null)
    out_ptr,       # [N] bf16 output
    N,             # number of output features
    G: tl.constexpr,  # number of groups per output feature
    B: tl.constexpr,  # bytes per group (16)
    K: tl.constexpr,  # input features = G * B * 2
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,  # number of output features per program
):
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offs < N

    # Load input vector x [K] - fits in registers for K=2880
    # Process in chunks of 32 (one group at a time)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for g in range(G):
        # Load scale for each output in this block
        scale_offs = n_offs * G + g
        scale_bytes = tl.load(scales_ptr + scale_offs, mask=mask_n, other=127)
        scale_exp = scale_bytes.to(tl.int32) - 127
        # ldexp(1.0, exp) = 2^exp
        scale = tl.math.exp2(scale_exp.to(tl.float32))

        # Process B bytes per group (each byte = 2 FP4 values)
        for b in range(B):
            block_offs = n_offs * (G * B) + g * B + b
            packed = tl.load(blocks_ptr + block_offs, mask=mask_n, other=0)

            lo_nib = (packed & 0x0F).to(tl.int32)
            hi_nib = (packed >> 4).to(tl.int32)

            # FP4 LUT - inline the 16 values using select chain
            lo_val = _fp4_lookup(lo_nib)
            hi_val = _fp4_lookup(hi_nib)

            # Scale
            lo_scaled = lo_val * scale
            hi_scaled = hi_val * scale

            # Load corresponding x values
            k_lo = g * B * 2 + b * 2
            k_hi = k_lo + 1
            x_lo = tl.load(x_ptr + k_lo).to(tl.float32)
            x_hi = tl.load(x_ptr + k_hi).to(tl.float32)

            acc += lo_scaled * x_lo + hi_scaled * x_hi

    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_offs, mask=mask_n, other=0.0)
        acc += bias

    tl.store(out_ptr + n_offs, acc.to(tl.bfloat16), mask=mask_n)


@triton.jit
def _fp4_lookup(nibble):
    """FP4 E2M1 lookup: nibble (0-15) -> float value."""
    # Using bit manipulation instead of LUT:
    # nibble format: [sign(1), exp(2), mantissa(1)]
    # sign = bit 3, exp = bits 2:1, mant = bit 0
    sign = (nibble >> 3) & 1
    exp = (nibble >> 1) & 3
    mant = nibble & 1

    # Value = (-1)^sign * 2^(exp-1) * (1 + mant*0.5)  for exp > 0
    # Value = (-1)^sign * 0.5 * mant                    for exp = 0 (subnormal)
    # Special: exp=0, mant=0 -> 0.0

    # Normal: 2^(exp-1) * (1 + 0.5*mant) = 2^(exp-1) + 2^(exp-2)*mant
    is_zero = (exp == 0) & (mant == 0)
    is_subnormal = (exp == 0) & (mant == 1)

    val = tl.where(is_zero, 0.0,
          tl.where(is_subnormal, 0.5,
          tl.math.exp2((exp - 1).to(tl.float32)) * (1.0 + 0.5 * mant.to(tl.float32))))

    return tl.where(sign == 1, -val, val)


def fused_dequant_vecmat(
    x: torch.Tensor,       # [1, K] or [K] bf16
    blocks: torch.Tensor,  # [N, G, B] uint8
    scales: torch.Tensor,  # [N, G] uint8
    bias: torch.Tensor | None = None,  # [N] f32
) -> torch.Tensor:
    """Fused MXFP4 dequant + vector-matrix multiply.

    Computes: F.linear(x, dequant(blocks, scales), bias) without materializing W.
    """
    if x.dim() == 2:
        x = x.squeeze(0)

    N, G, B = blocks.shape
    K = G * B * 2

    out = torch.empty(N, dtype=torch.bfloat16, device=x.device)

    BLOCK_N = 64
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    _dequant_vecmat_kernel[grid](
        x, blocks.reshape(N, G * B), scales, bias, out,
        N=N, G=G, B=B, K=K,
        HAS_BIAS=bias is not None,
        BLOCK_N=BLOCK_N,
    )
    return out.unsqueeze(0)  # [1, N]
