"""Native FP4 Tensor Core matmul via tl.dot_scaled on Blackwell (SM 12.0).

Replaces software MXFP4 dequant with hardware-accelerated block-scaled matmul.
bf16 activations × e2m1 weights, E8M0 block scales (block_size=32).

Layout:
  lhs: [1, K] bf16 activations
  rhs: [K_HALF, N] uint8 packed FP4 (loaded transposed from [N, K_HALF] storage)
  rhs_scale: [N, K//32] uint8 E8M0
  Output: [1, N] bf16
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dot_scaled_vecmat(
    x_ptr,
    w_ptr,
    w_scale_ptr,
    bias_ptr,
    out_ptr,
    N,
    K_HALF,
    K_GROUPS,
    NUM_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K_HALF: tl.constexpr,
    BLOCK_K_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offs < N

    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    for tile in range(NUM_TILES):
        k_start = tile * BLOCK_K
        k_half_start = tile * BLOCK_K_HALF
        k_group_start = tile * BLOCK_K_GROUPS

        # lhs: [1, BLOCK_K] bf16
        x_block = tl.load(x_ptr + k_start + tl.arange(0, BLOCK_K)[None, :])

        # rhs: [BLOCK_K_HALF, BLOCK_N] — transposed load from [N, K_HALF] storage
        w_k = (k_half_start + tl.arange(0, BLOCK_K_HALF))[:, None]
        w_n = n_offs[None, :]
        w_block = tl.load(w_ptr + w_n * K_HALF + w_k, mask=mask_n[None, :])

        # rhs_scale: [BLOCK_N, BLOCK_K_GROUPS]
        ws_k = tl.arange(0, BLOCK_K_GROUPS)[None, :]
        w_scale = tl.load(w_scale_ptr + n_offs[:, None] * K_GROUPS + k_group_start + ws_k, mask=mask_n[:, None])

        acc = tl.dot_scaled(x_block, None, "bf16", w_block, w_scale, "e2m1", acc)

    acc_1d = tl.reshape(acc, (BLOCK_N,))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_offs, mask=mask_n)
        acc_1d = acc_1d + bias
    tl.store(out_ptr + n_offs, acc_1d.to(tl.bfloat16), mask=mask_n)


def dot_scaled_vecmat(
    x: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Native FP4 TC vector-matrix multiply.

    Args:
        x: [1, K] or [K] bf16 input vector
        blocks: [N, G, B] or [N, K_HALF] uint8 packed FP4
        scales: [N, G] uint8 E8M0 block scales
        bias: [N] f32 bias (optional)

    Returns:
        [1, N] bf16 output
    """
    if x.dim() == 2:
        x = x.squeeze(0)

    if blocks.dim() == 3:
        N, G, B = blocks.shape
        K = G * B * 2
        blocks_flat = blocks.reshape(N, G * B)
    else:
        N, K_HALF = blocks.shape
        K = K_HALF * 2
        blocks_flat = blocks

    K_HALF = K // 2
    K_GROUPS = K // 32
    BLOCK_K = 64
    BLOCK_N = 64
    NUM_TILES = K // BLOCK_K

    out = torch.empty(N, dtype=torch.bfloat16, device=x.device)
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    _dot_scaled_vecmat[grid](
        x,
        blocks_flat,
        scales,
        bias,
        out,
        N=N,
        K_HALF=K_HALF,
        K_GROUPS=K_GROUPS,
        NUM_TILES=NUM_TILES,
        BLOCK_K=BLOCK_K,
        BLOCK_K_HALF=BLOCK_K // 2,
        BLOCK_K_GROUPS=BLOCK_K // 32,
        BLOCK_N=BLOCK_N,
        HAS_BIAS=bias is not None,
    )
    return out.unsqueeze(0)  # [1, N]
