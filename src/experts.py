"""SwiGLU expert forward pass for GPT-OSS-120B."""

import torch
import torch.nn.functional as F

from .config import SWIGLU_ALPHA, SWIGLU_LIMIT
from .mxfp4 import dequant_mxfp4_no_transpose

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


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate = gate_up[..., ::2].clamp(max=SWIGLU_LIMIT)
    up = gate_up[..., 1::2].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (up + 1) * gate * torch.sigmoid(gate * SWIGLU_ALPHA)


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
    if hidden_states.shape[0] == 1:
        if _BACKEND == "dot_scaled":
            gate_up = dot_scaled_vecmat(hidden_states, gate_up_blocks, gate_up_scales, gate_up_bias)
            return dot_scaled_vecmat(_swiglu(gate_up), down_blocks, down_scales, down_bias)
        if _BACKEND == "triton_sw":
            gate_up = fused_dequant_vecmat(hidden_states, gate_up_blocks, gate_up_scales, gate_up_bias)
            return fused_dequant_vecmat(_swiglu(gate_up), down_blocks, down_scales, down_bias)

    gate_up_w = dequant_mxfp4_no_transpose(gate_up_blocks, gate_up_scales, dtype)
    down_w = dequant_mxfp4_no_transpose(down_blocks, down_scales, dtype)
    gate_up = F.linear(hidden_states, gate_up_w, gate_up_bias.to(dtype))
    return F.linear(_swiglu(gate_up), down_w, down_bias.to(dtype))
