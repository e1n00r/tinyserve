"""Expert store backed by GGUF quantized weights.

Loads expert weights from GGUF, converts Q4_K to torch INT4 pack
at construction time (one-time). Stores INT4 packed data for
direct compute via GGUFINT4Forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .expert_store import TensorLayout, _pack_tensors
from .gguf_quant import q4k_expert_to_int4pack
from .gguf_reader import GGUFReader


class GGUFINT4Forward:
    """Expert forward on CPU using pre-converted INT4 packed weights from GGUF.

    Unlike CPUINT4Forward (which converts MXFP4 on first use), this class
    reads INT4 packed tensors that were already converted during store construction.
    Layout keys: gate_packed, gate_sz, up_packed, up_sz, down_packed, down_sz.
    """

    def __init__(self, layout: TensorLayout, group_size: int = 32, act_fn=None, num_threads: int = 4):
        self.group_size = group_size
        self._act_fn = act_fn
        self._num_threads = num_threads
        self._layout = layout

    def forward(self, hidden_states: torch.Tensor, expert_packed: torch.Tensor) -> torch.Tensor:
        input_device = hidden_states.device
        layout = self._layout

        def _load(name: str):
            off = layout.offsets[name]
            sz = layout.sizes[name]
            shape, dtype = layout.specs[name]
            return expert_packed[off : off + sz].view(dtype).view(shape)

        gate_packed = _load("gate_packed")
        gate_sz = _load("gate_sz")
        up_packed = _load("up_packed")
        up_sz = _load("up_sz")
        down_packed = _load("down_packed")
        down_sz = _load("down_sz")

        hidden_states_cpu = hidden_states.to("cpu").to(torch.bfloat16)

        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._num_threads)

            gate_out = torch.ops.aten._weight_int4pack_mm_for_cpu(hidden_states_cpu, gate_packed, self.group_size, gate_sz)
            up_out = torch.ops.aten._weight_int4pack_mm_for_cpu(hidden_states_cpu, up_packed, self.group_size, up_sz)

            if self._act_fn is not None:
                gated = self._act_fn(gate_out) * up_out
            else:
                gated = F.silu(gate_out) * up_out

            out = torch.ops.aten._weight_int4pack_mm_for_cpu(gated, down_packed, self.group_size, down_sz)
        finally:
            torch.set_num_threads(old_threads)

        return out.to(input_device)


class GGUFExpertStore:
    """Expert store backed by GGUF quantized weights.

    Loads expert weights from GGUF, converts Q4_K to torch INT4 pack
    at construction time. Stores INT4 packed data for direct compute
    via GGUFINT4Forward.
    """

    def __init__(
        self,
        data: torch.Tensor,
        layout: TensorLayout,
        num_layers: int,
        num_experts: int,
    ):
        self._data = data
        self.layout = layout
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_bytes = layout.total_bytes
        self.buffer_expert_bytes = layout.total_bytes

    @classmethod
    def from_gguf(cls, gguf_path: str, num_threads: int = 4) -> GGUFExpertStore:
        """Load expert weights from GGUF file.

        1. Parse GGUF header via GGUFReader
        2. Group expert tensors by (layer, expert)
        3. Convert each expert's Q4_K weights to INT4 pack
        4. Store in flat buffer
        """
        reader = GGUFReader(gguf_path)
        groups = reader.list_expert_tensors()

        if not groups:
            reader.close()
            raise ValueError(f"No expert tensors found in {gguf_path}")

        layers = sorted({k[0] for k in groups})
        experts_per_layer = sorted({k[1] for k in groups})
        num_layers = len(layers)
        num_experts = len(experts_per_layer)

        # Convert first expert to determine layout
        first_key = (layers[0], experts_per_layer[0])
        first_projs = groups[first_key]

        gate_info = first_projs["gate"]
        up_info = first_projs["up"]
        down_info = first_projs["down"]

        gate_data = reader.get_tensor_data(gate_info)
        up_data = reader.get_tensor_data(up_info)
        down_data = reader.get_tensor_data(down_info)

        gate_shape = (gate_info.shape[0], gate_info.shape[1])
        up_shape = (up_info.shape[0], up_info.shape[1])
        down_shape = (down_info.shape[0], down_info.shape[1])

        g_packed, g_sz, u_packed, u_sz, d_packed, d_sz = q4k_expert_to_int4pack(
            gate_data,
            up_data,
            down_data,
            gate_shape,
            up_shape,
            down_shape,
        )

        # Store pre-converted INT4 packed tensors as raw bytes
        specs = {
            "gate_packed": (tuple(g_packed.shape), g_packed.dtype),
            "gate_sz": (tuple(g_sz.shape), g_sz.dtype),
            "up_packed": (tuple(u_packed.shape), u_packed.dtype),
            "up_sz": (tuple(u_sz.shape), u_sz.dtype),
            "down_packed": (tuple(d_packed.shape), d_packed.dtype),
            "down_sz": (tuple(d_sz.shape), d_sz.dtype),
        }
        layout = TensorLayout(specs)

        # Allocate and pack all experts
        data = torch.empty(
            num_layers,
            num_experts,
            layout.total_bytes,
            dtype=torch.uint8,
        )
        if torch.cuda.is_available():
            data = data.pin_memory()

        for li, layer_idx in enumerate(layers):
            for ei, expert_idx in enumerate(experts_per_layer):
                key = (layer_idx, expert_idx)
                projs = groups[key]

                g_data = reader.get_tensor_data(projs["gate"])
                u_data = reader.get_tensor_data(projs["up"])
                d_data = reader.get_tensor_data(projs["down"])

                g_s = (projs["gate"].shape[0], projs["gate"].shape[1])
                u_s = (projs["up"].shape[0], projs["up"].shape[1])
                d_s = (projs["down"].shape[0], projs["down"].shape[1])

                gp, gsz, up, usz, dp, dsz = q4k_expert_to_int4pack(
                    g_data,
                    u_data,
                    d_data,
                    g_s,
                    u_s,
                    d_s,
                )

                tensors = {
                    "gate_packed": gp,
                    "gate_sz": gsz,
                    "up_packed": up,
                    "up_sz": usz,
                    "down_packed": dp,
                    "down_sz": dsz,
                }
                _pack_tensors(data[li, ei], layout, tensors)

        reader.close()
        return cls(data, layout, num_layers, num_experts)
