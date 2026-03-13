"""CPU-side expert forward for deferred expert computation.

Used for deferred experts that overlap with the next layer's GPU compute
(~30ms budget). Not used for regular cache misses (PCIe pipeline is faster).

Uses PyTorch CPU ops with float32 dequant + matmul. Releases GIL via MKL.
"""

import torch
import torch.nn.functional as F

from .config import FP4_LUT, SWIGLU_ALPHA, SWIGLU_LIMIT

_cpu_lut = FP4_LUT.to(dtype=torch.float32, device="cpu")


def _dequant_mxfp4_cpu(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """MXFP4 dequant on CPU to float32."""
    out_features, G, B = blocks.shape
    rows = out_features * G

    bf = blocks.reshape(rows, B)
    si = (scales.to(torch.int32) - 127).reshape(rows, 1)

    out = torch.empty(rows, B * 2, dtype=torch.float32)
    out[:, 0::2] = _cpu_lut[(bf & 0x0F).to(torch.int32)]
    out[:, 1::2] = _cpu_lut[(bf >> 4).to(torch.int32)]
    torch.ldexp(out, si, out=out)

    return out.view(out_features, G * B * 2)


def expert_forward_cpu(
    hidden_states: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    """Full SwiGLU expert forward on CPU.

    Returns [1, hidden_size] float32 on CPU.
    Budget: must complete within ~30ms (overlaps with next layer's GPU compute).
    """
    h = hidden_states.float()
    if h.is_cuda:
        h = h.cpu()

    gate_up_w = _dequant_mxfp4_cpu(gate_up_blocks, gate_up_scales)
    gate_up = F.linear(h, gate_up_w, gate_up_bias.float())
    del gate_up_w

    gate = gate_up[..., ::2].clamp(max=SWIGLU_LIMIT)
    up = gate_up[..., 1::2].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    glu = gate * torch.sigmoid(gate * SWIGLU_ALPHA)
    gated_output = (up + 1) * glu
    del gate_up

    down_w = _dequant_mxfp4_cpu(down_blocks, down_scales)
    out = F.linear(gated_output, down_w, down_bias.float())

    return out
