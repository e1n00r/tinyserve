"""SwiGLU expert forward pass for GPT-OSS-120B."""

import torch
import torch.nn.functional as F

from .config import SWIGLU_ALPHA, SWIGLU_LIMIT
from .mxfp4 import dequant_mxfp4_no_transpose

# Try to use Triton fused kernel, fall back to PyTorch
try:
    from .triton_dequant import fused_dequant_vecmat
    _USE_TRITON = True
except ImportError:
    _USE_TRITON = False


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
    if _USE_TRITON and hidden_states.shape[0] == 1:
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


def _expert_forward_triton(
    hidden_states: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    """Triton path: fused dequant+matmul, no intermediate weight materialization."""
    gate_up = fused_dequant_vecmat(hidden_states, gate_up_blocks, gate_up_scales, gate_up_bias)

    gate = gate_up[..., ::2].clamp(max=SWIGLU_LIMIT)
    up = gate_up[..., 1::2].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    glu = gate * torch.sigmoid(gate * SWIGLU_ALPHA)
    gated_output = (up + 1) * glu

    return fused_dequant_vecmat(gated_output, down_blocks, down_scales, down_bias)
