"""SwiGLU expert forward pass for GPT-OSS-120B."""

import torch

from .config import SWIGLU_ALPHA, SWIGLU_LIMIT
from .mxfp4 import dequant_single_expert


def expert_forward(
    hidden_states: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute a single expert's output using the GPT-OSS custom SwiGLU.

    Args:
        hidden_states: [num_tokens, hidden_size] input
        gate_up_blocks/scales: MXFP4 packed gate+up projection weights
        gate_up_bias: [2*intermediate_size] float32
        down_blocks/scales: MXFP4 packed down projection weights
        down_bias: [hidden_size] float32

    Returns:
        [num_tokens, hidden_size] expert output
    """
    gate_up_w, down_w = dequant_single_expert(
        gate_up_blocks, gate_up_scales, down_blocks, down_scales, dtype
    )

    # gate_up_w: [hidden_size, 2*intermediate_size]
    # down_w: [intermediate_size, hidden_size]
    gate_up = hidden_states @ gate_up_w + gate_up_bias.to(dtype)

    # Interleaved split (NOT halved)
    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]

    # Custom activation with clamps
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)

    # gate * sigmoid(gate * alpha), NOT standard silu
    glu = gate * torch.sigmoid(gate * SWIGLU_ALPHA)

    # Residual +1 inside expert
    gated_output = (up + 1) * glu

    out = gated_output @ down_w + down_bias.to(dtype)
    return out
