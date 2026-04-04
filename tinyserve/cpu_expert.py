"""CPU-side expert forward pass using F.linear (OneDNN auto-dispatches to AMX/AVX-512/AVX2).

No custom kernels — PyTorch's F.linear on CPU uses OneDNN which selects the best
available instruction set (AMX on Sapphire Rapids+, AVX-512, AVX2) automatically.

Supports two layout variants:
  1. "fused": gate_up_proj + down_proj (GPT-OSS, Qwen3.5)
  2. "separate": gate_proj + up_proj + down_proj (Mixtral, Qwen3, DeepSeek)

MXFP4 layouts (gate_up_proj_scales) are handled via CPUINT4Forward which
converts MXFP4 blocks+scales to torch's native INT4 packed format and uses
_weight_int4pack_mm_for_cpu for fast CPU inference.
"""

import torch
import torch.nn.functional as F

from .expert_forward import _QUICK_GELU_COEFF
from .expert_store import TensorLayout

HAS_INT4_CPU = hasattr(torch.ops.aten, "_weight_int4pack_mm_for_cpu") and hasattr(
    torch.ops.aten, "_convert_weight_to_int4pack_for_cpu"
)

# MXFP4 nibble look-up table: index [0..15] -> float value
_FP4_LUT = torch.tensor(
    [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def mxfp4_to_int4pack(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert MXFP4 blocks+scales to torch INT4 packed format.

    Args:
        blocks: [out_features, in_features//32, 16] uint8 (MXFP4 nibble blocks)
        scales: [out_features, in_features//32] uint8 (E8M0 scales)
        group_size: group size for INT4 quantization (must divide in_features)

    Returns:
        w_int4packed: torch INT4 packed tensor (from _convert_weight_to_int4pack_for_cpu)
        scales_and_zeros: [in_features//group_size, out_features, 2] bfloat16
            where [..., 0] = scale, [..., 1] = zero
            dequant formula: val = (w_int4 - 8) * scale + zero
    """
    out_features, G, B = blocks.shape
    in_features = G * B * 2  # each uint8 byte holds 2 FP4 nibbles

    # Step 1: Dequant MXFP4 to float32 (one-time conversion cost)
    lut = _FP4_LUT
    bf = blocks.reshape(out_features * G, B)
    lo_nibbles = (bf & 0x0F).to(torch.int64)
    hi_nibbles = (bf >> 4).to(torch.int64)

    w_float = torch.empty(out_features * G, B * 2, dtype=torch.float32)
    w_float[:, 0::2] = lut[lo_nibbles]
    w_float[:, 1::2] = lut[hi_nibbles]

    # Apply E8M0 scales: 2^(scale_uint8 - 127)
    si = (scales.to(torch.int32) - 127).reshape(out_features * G, 1)
    torch.ldexp(w_float, si, out=w_float)
    w_float = w_float.view(out_features, in_features)

    # Step 2: Per-group symmetric INT4 quantization
    # INT4 range [0, 15], zero point at 8, so effective range is [-8, 7] mapped to [-8s, 7s]
    n_groups = in_features // group_size
    w_grouped = w_float.view(out_features, n_groups, group_size)

    # Compute per-group scale: map max(abs) to 7 (the positive range from 8)
    amax = w_grouped.abs().amax(dim=-1)  # [out_features, n_groups]
    scale = (amax / 7.0).clamp(min=1e-10)  # [out_features, n_groups]

    # Quantize: w_int4 = round(w / scale) + 8, clamped to [0, 15]
    w_q = torch.round(w_grouped / scale.unsqueeze(-1)) + 8
    w_q = w_q.clamp(0, 15).to(torch.int32)
    w_q = w_q.view(out_features, in_features)

    # Step 3: Pack via torch op (requires N divisible by 16)
    w_int4packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(w_q, 2)

    # Step 4: Build scales_and_zeros in [n_groups, out_features, 2] layout
    # dequant formula: val = (w_int4 - 8) * scale + zero
    # For symmetric quant, zero = 0 (we centered at 8 in integer space)
    scales_and_zeros = torch.zeros(n_groups, out_features, 2, dtype=torch.bfloat16)
    scales_and_zeros[:, :, 0] = scale.T.to(torch.bfloat16)  # [n_groups, out_features]
    scales_and_zeros[:, :, 1] = 0.0  # symmetric => zero offset is 0

    return w_int4packed, scales_and_zeros.contiguous()


class CPUINT4Forward:
    """Expert forward on CPU using native INT4 packed matmul.

    Accepts pre-converted INT4 packed weights (from mxfp4_to_int4pack) and
    computes the MoE expert forward pass using torch's _weight_int4pack_mm_for_cpu.
    """

    def __init__(self, layout: TensorLayout, group_size: int = 32, act_fn=None, num_threads: int = 4):
        self.group_size = group_size
        self._act_fn = act_fn
        self._num_threads = num_threads
        self._layout = layout
        self._int4_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _convert_expert(
        self, expert_packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert MXFP4 expert data to INT4 packed format.

        Returns: (gate_up_packed, gate_up_sz, down_packed, down_sz)
        """
        layout = self._layout

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

        gu_packed, gu_sz_tensor = mxfp4_to_int4pack(gu_blocks, gu_scales, self.group_size)
        dn_packed, dn_sz_tensor = mxfp4_to_int4pack(dn_blocks, dn_scales, self.group_size)

        return gu_packed, gu_sz_tensor, dn_packed, dn_sz_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_packed: torch.Tensor,
    ) -> torch.Tensor:
        """Run expert forward using INT4 packed matmul.

        Lazily converts MXFP4 data to INT4 on first call per expert (keyed by
        data_ptr). Subsequent calls with the same buffer reuse cached INT4 data.
        """
        input_device = hidden_states.device

        # Lazy INT4 conversion with caching by data_ptr
        cache_key = expert_packed.data_ptr()
        if cache_key not in self._int4_cache:
            self._int4_cache[cache_key] = self._convert_expert(expert_packed)
        gu_packed, gu_sz, dn_packed, dn_sz = self._int4_cache[cache_key]

        h = hidden_states.to("cpu").to(torch.bfloat16)

        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._num_threads)
            gate_up = torch.ops.aten._weight_int4pack_mm_for_cpu(h, gu_packed, self.group_size, gu_sz)

            if self._act_fn is not None:
                gate, up = gate_up.chunk(2, dim=-1)
                gated = self._act_fn(gate) * up
            else:
                gate = gate_up[..., ::2].clamp(max=7.0)
                up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
                gated = (up + 1) * gate * torch.sigmoid(gate * _QUICK_GELU_COEFF)

            out = torch.ops.aten._weight_int4pack_mm_for_cpu(gated, dn_packed, self.group_size, dn_sz)
        finally:
            torch.set_num_threads(old_threads)

        return out.to(input_device)

    def clear_cache(self):
        """Clear the INT4 conversion cache."""
        self._int4_cache.clear()


class CPUExpertForward:
    """Pre-baked CPU expert forward with tensor offsets computed at init."""

    def __init__(
        self,
        layout: TensorLayout,
        act_fn=None,
        num_threads: int = 4,
    ):
        specs = layout.specs

        if "gate_up_proj_scales" in specs:
            if not HAS_INT4_CPU:
                raise ValueError(
                    "MXFP4 layouts (gate_up_proj_scales) require INT4 CPU ops "
                    "(_weight_int4pack_mm_for_cpu) which are not available in this "
                    "PyTorch build. Upgrade to PyTorch >= 2.10."
                )
            self._variant = "mxfp4_int4"
            self._int4_fwd = CPUINT4Forward(layout, group_size=32, act_fn=act_fn, num_threads=num_threads)
            self._num_threads = num_threads
            self._act_fn = act_fn
            return

        self._num_threads = num_threads
        self._act_fn = act_fn

        if "gate_up_proj" in specs and "down_proj" in specs:
            self._variant = "fused"
            self._init_fused(layout)
        elif "gate_proj" in specs and "up_proj" in specs and "down_proj" in specs:
            self._variant = "separate"
            self._init_separate(layout)
        else:
            raise ValueError(
                f"Unknown expert layout. Expected fused (gate_up_proj, down_proj) "
                f"or separate (gate_proj, up_proj, down_proj), got: {list(specs.keys())}"
            )

    def _init_fused(self, layout: TensorLayout):
        specs = layout.specs
        self._gu_off = layout.offsets["gate_up_proj"]
        self._gu_sz = layout.sizes["gate_up_proj"]
        self._gu_shape, self._gu_dtype = specs["gate_up_proj"]
        self._dn_off = layout.offsets["down_proj"]
        self._dn_sz = layout.sizes["down_proj"]
        self._dn_shape, self._dn_dtype = specs["down_proj"]

        hidden_dim = min(self._gu_shape)
        self._gu_needs_t = self._gu_shape[0] == hidden_dim
        gated_dim = max(self._gu_shape) // 2
        self._dn_needs_t = self._dn_shape[0] == gated_dim

    def _init_separate(self, layout: TensorLayout):
        specs = layout.specs
        for prefix in ("gate_proj", "up_proj", "down_proj"):
            key = prefix + ".weight" if prefix + ".weight" in specs else prefix
            off = layout.offsets[key]
            sz = layout.sizes[key]
            shape, dtype = specs[key]
            setattr(self, f"_{prefix}_off", off)
            setattr(self, f"_{prefix}_sz", sz)
            setattr(self, f"_{prefix}_shape", shape)
            setattr(self, f"_{prefix}_dtype", dtype)

        gate_shape = self._gate_proj_shape
        hidden_dim = min(gate_shape)
        self._gate_needs_t = gate_shape[0] == hidden_dim
        self._up_needs_t = self._up_proj_shape[0] == hidden_dim
        intermediate_dim = max(gate_shape)
        if gate_shape[0] == hidden_dim:
            intermediate_dim = gate_shape[1]
        else:
            intermediate_dim = gate_shape[0]
        self._dn_needs_t = self._down_proj_shape[0] == intermediate_dim

    def forward(self, hidden_states: torch.Tensor, expert_packed: torch.Tensor) -> torch.Tensor:
        if self._variant == "mxfp4_int4":
            return self._int4_fwd.forward(hidden_states, expert_packed)

        input_device = hidden_states.device
        h = hidden_states.to("cpu")
        packed = expert_packed

        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._num_threads)
            if self._variant == "fused":
                result = self._forward_fused(h, packed)
            else:
                result = self._forward_separate(h, packed)
        finally:
            torch.set_num_threads(old_threads)

        return result.to(input_device)

    def _forward_fused(self, h: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        w_gu = packed[self._gu_off : self._gu_off + self._gu_sz].view(self._gu_dtype).view(self._gu_shape)
        if self._gu_needs_t:
            w_gu = w_gu.t()
        gate_up = F.linear(h, w_gu)

        if self._act_fn is not None:
            gate, up = gate_up.chunk(2, dim=-1)
            gated = self._act_fn(gate) * up
        else:
            gate = gate_up[..., ::2].clamp(max=7.0)
            up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
            gated = (up + 1) * gate * torch.sigmoid(gate * _QUICK_GELU_COEFF)

        w_dn = packed[self._dn_off : self._dn_off + self._dn_sz].view(self._dn_dtype).view(self._dn_shape)
        if self._dn_needs_t:
            w_dn = w_dn.t()
        return F.linear(gated, w_dn)

    def _forward_separate(self, h: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        w_gate = (
            packed[self._gate_proj_off : self._gate_proj_off + self._gate_proj_sz]
            .view(self._gate_proj_dtype)
            .view(self._gate_proj_shape)
        )
        if self._gate_needs_t:
            w_gate = w_gate.t()

        w_up = (
            packed[self._up_proj_off : self._up_proj_off + self._up_proj_sz]
            .view(self._up_proj_dtype)
            .view(self._up_proj_shape)
        )
        if self._up_needs_t:
            w_up = w_up.t()

        if self._act_fn is not None:
            gated = self._act_fn(F.linear(h, w_gate)) * F.linear(h, w_up)
        else:
            gated = F.silu(F.linear(h, w_gate)) * F.linear(h, w_up)

        w_dn = (
            packed[self._down_proj_off : self._down_proj_off + self._down_proj_sz]
            .view(self._down_proj_dtype)
            .view(self._down_proj_shape)
        )
        if self._dn_needs_t:
            w_dn = w_dn.t()
        return F.linear(gated, w_dn)
