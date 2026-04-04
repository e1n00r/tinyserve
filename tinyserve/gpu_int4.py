"""GPU-side INT4 expert forward using torch._weight_int4pack_mm.

Converts MXFP4 expert weights to INT4 packed format and runs matmul on GPU.
5x+ faster than BF16 F.linear for batch=1 decode (memory-bandwidth bound).

GPU packing format differs from CPU:
  - Input: uint8 where each byte = (even_col << 4) | odd_col
  - Uses torch._convert_weight_to_int4pack (not the _for_cpu variant)

Graceful fallback: HAS_INT4_GPU is False if ops or CUDA unavailable.
"""

import torch

from .expert_forward import _QUICK_GELU_COEFF
from .expert_store import TensorLayout

HAS_INT4_GPU = (
    hasattr(torch, "_weight_int4pack_mm")
    and hasattr(torch, "_convert_weight_to_int4pack")
    and torch.cuda.is_available()
)

# MXFP4 nibble look-up table: index [0..15] -> float value
_FP4_LUT = torch.tensor(
    [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def mxfp4_to_int4pack_gpu(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert MXFP4 blocks+scales to GPU INT4 packed format.

    Args:
        blocks: [out_features, in_features//32, 16] uint8 (MXFP4 nibble blocks)
        scales: [out_features, in_features//32] uint8 (E8M0 scales)
        group_size: group size for INT4 quantization (must divide in_features)
        device: target CUDA device

    Returns:
        w_int4packed: torch GPU INT4 packed tensor
        scales_and_zeros: [in_features//group_size, out_features, 2] bfloat16
    """
    out_features, G, B = blocks.shape
    in_features = G * B * 2

    # Step 1: Dequant MXFP4 to float32
    lut = _FP4_LUT
    bf = blocks.reshape(out_features * G, B)
    lo_nibbles = (bf & 0x0F).to(torch.int64)
    hi_nibbles = (bf >> 4).to(torch.int64)

    w_float = torch.empty(out_features * G, B * 2, dtype=torch.float32)
    w_float[:, 0::2] = lut[lo_nibbles]
    w_float[:, 1::2] = lut[hi_nibbles]

    si = (scales.to(torch.int32) - 127).reshape(out_features * G, 1)
    torch.ldexp(w_float, si, out=w_float)
    w_float = w_float.view(out_features, in_features)

    # Step 2: Per-group symmetric INT4 quantization
    n_groups = in_features // group_size
    w_grouped = w_float.view(out_features, n_groups, group_size)
    amax = w_grouped.abs().amax(dim=-1)
    scale = (amax / 7.0).clamp(min=1e-10)

    w_q = torch.round(w_grouped / scale.unsqueeze(-1)) + 8
    w_q = w_q.clamp(0, 15).to(torch.int32)
    w_q = w_q.view(out_features, in_features)

    # Step 3: Pack for GPU — uint8 with (even_col << 4) | odd_col
    w_uint8 = (w_q[:, 1::2] | (w_q[:, 0::2] << 4)).to(torch.uint8).to(device)
    w_int4packed = torch._convert_weight_to_int4pack(w_uint8, 2)

    # Step 4: Build scales_and_zeros [n_groups, out_features, 2]
    scales_and_zeros = torch.zeros(n_groups, out_features, 2, dtype=torch.bfloat16, device=device)
    scales_and_zeros[:, :, 0] = scale.T.to(torch.bfloat16)

    return w_int4packed, scales_and_zeros


class GPUINT4Forward:
    """GPU-side INT4 expert forward using torch._weight_int4pack_mm.

    Converts MXFP4 cache slot data to GPU INT4 packed format on first use,
    then runs matmul entirely on GPU. ~5x faster than BF16 F.linear for
    batch=1 decode.
    """

    def __init__(self, layout: TensorLayout, group_size: int = 32, act_fn=None):
        self.group_size = group_size
        self._act_fn = act_fn
        self._layout = layout
        self._int4_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _convert_expert(
        self, expert_packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = self._layout
        device = expert_packed.device

        gu_off = layout.offsets["gate_up_proj"]
        gu_sz = layout.sizes["gate_up_proj"]
        gu_shape = layout.specs["gate_up_proj"][0]
        gu_blocks = expert_packed[gu_off : gu_off + gu_sz].view(torch.uint8).view(gu_shape)

        gus_off = layout.offsets["gate_up_proj_scales"]
        gus_sz = layout.sizes["gate_up_proj_scales"]
        gus_shape = layout.specs["gate_up_proj_scales"][0]
        gu_scales = expert_packed[gus_off : gus_off + gus_sz].view(torch.uint8).view(gus_shape)

        dn_off = layout.offsets["down_proj"]
        dn_sz = layout.sizes["down_proj"]
        dn_shape = layout.specs["down_proj"][0]
        dn_blocks = expert_packed[dn_off : dn_off + dn_sz].view(torch.uint8).view(dn_shape)

        dns_off = layout.offsets["down_proj_scales"]
        dns_sz = layout.sizes["down_proj_scales"]
        dns_shape = layout.specs["down_proj_scales"][0]
        dn_scales = expert_packed[dns_off : dns_off + dns_sz].view(torch.uint8).view(dns_shape)

        # Move to CPU for dequant, then pack for GPU
        gu_packed, gu_sz_tensor = mxfp4_to_int4pack_gpu(gu_blocks.cpu(), gu_scales.cpu(), self.group_size, device)
        dn_packed, dn_sz_tensor = mxfp4_to_int4pack_gpu(dn_blocks.cpu(), dn_scales.cpu(), self.group_size, device)

        return gu_packed, gu_sz_tensor, dn_packed, dn_sz_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_packed: torch.Tensor,
    ) -> torch.Tensor:
        cache_key = expert_packed.data_ptr()
        if cache_key not in self._int4_cache:
            self._int4_cache[cache_key] = self._convert_expert(expert_packed)
        gu_packed, gu_sz, dn_packed, dn_sz = self._int4_cache[cache_key]

        h = hidden_states.to(torch.bfloat16)

        gate_up = torch._weight_int4pack_mm(h, gu_packed, self.group_size, gu_sz)

        if self._act_fn is not None:
            gate, up = gate_up.chunk(2, dim=-1)
            gated = self._act_fn(gate) * up
        else:
            gate = gate_up[..., ::2].clamp(max=7.0)
            up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
            gated = (up + 1) * gate * torch.sigmoid(gate * _QUICK_GELU_COEFF)

        out = torch._weight_int4pack_mm(gated, dn_packed, self.group_size, dn_sz)
        return out

    def clear_cache(self):
        self._int4_cache.clear()
