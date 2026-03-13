"""SwiGLU expert forward pass for GPT-OSS-120B."""

import torch
import torch.nn.functional as F

from .config import SWIGLU_ALPHA, SWIGLU_LIMIT
from .mxfp4 import dequant_mxfp4_no_transpose

# Try native FP4 TC (Blackwell), then software Triton, then PyTorch
_BACKEND = "pytorch"
try:
    from .triton_dot_scaled import dot_scaled_vecmat
    _BACKEND = "dot_scaled"
except Exception:
    try:
        from .triton_dequant import fused_dequant_vecmat
        _BACKEND = "triton_sw"
    except ImportError:
        pass


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
    """Compute a single expert's output using the GPT-OSS custom SwiGLU."""
    if hidden_states.shape[0] == 1:
        if _BACKEND == "dot_scaled":
            return _expert_forward_dot_scaled(
                hidden_states, gate_up_blocks, gate_up_scales, gate_up_bias,
                down_blocks, down_scales, down_bias,
            )
        if _BACKEND == "triton_sw":
            return _expert_forward_triton(
                hidden_states, gate_up_blocks, gate_up_scales, gate_up_bias,
                down_blocks, down_scales, down_bias,
            )

    gate_up_w = dequant_mxfp4_no_transpose(gate_up_blocks, gate_up_scales, dtype)
    down_w = dequant_mxfp4_no_transpose(down_blocks, down_scales, dtype)

    gate_up = F.linear(hidden_states, gate_up_w, gate_up_bias.to(dtype))
    gate = gate_up[..., ::2].clamp(max=SWIGLU_LIMIT)
    up = gate_up[..., 1::2].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    glu = gate * torch.sigmoid(gate * SWIGLU_ALPHA)
    gated_output = (up + 1) * glu
    return F.linear(gated_output, down_w, down_bias.to(dtype))


def _expert_forward_dot_scaled(
    hidden_states: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    """Native FP4 Tensor Core path (Blackwell): ~5x faster than software dequant."""
    gate_up = dot_scaled_vecmat(hidden_states, gate_up_blocks, gate_up_scales, gate_up_bias)

    gate = gate_up[..., ::2].clamp(max=SWIGLU_LIMIT)
    up = gate_up[..., 1::2].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    glu = gate * torch.sigmoid(gate * SWIGLU_ALPHA)
    gated_output = (up + 1) * glu

    return dot_scaled_vecmat(gated_output, down_blocks, down_scales, down_bias)


def _expert_forward_triton(
    hidden_states: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    """Software Triton path: fused dequant+matmul."""
    from .triton_dequant import fused_dequant_vecmat
    gate_up = fused_dequant_vecmat(hidden_states, gate_up_blocks, gate_up_scales, gate_up_bias)

    gate = gate_up[..., ::2].clamp(max=SWIGLU_LIMIT)
    up = gate_up[..., 1::2].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    glu = gate * torch.sigmoid(gate * SWIGLU_ALPHA)
    gated_output = (up + 1) * glu

    return fused_dequant_vecmat(gated_output, down_blocks, down_scales, down_bias)
