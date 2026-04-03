"""GGUF weight loading helpers: param navigation and expert store construction.

Extracted from gguf_loader.py. Handles _set_param, _get_param, _find_tensor_info,
_build_expert_store_from_reader, and _build_expert_store_from_fused_reader.
"""

from __future__ import annotations

import logging

import torch

from .gguf_reader import GGUFTensorInfo
from .gguf_dequant import _dequant_fused_tensor

logger = logging.getLogger(__name__)


def _find_tensor_info(reader: GGUFReader, name: str) -> GGUFTensorInfo:
    """Find a GGUFTensorInfo by name in a single-shard reader."""
    for t in reader.tensors:
        if t.name == name:
            return t
    raise KeyError(f"Tensor '{name}' not found in GGUF file")


def _get_param(model: torch.nn.Module, hf_name: str) -> torch.nn.Parameter | None:
    """Navigate dotted path to find a parameter in the model."""
    parts = hf_name.split(".")
    obj = model
    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif part.isdigit() and hasattr(obj, "__getitem__"):
            obj = obj[int(part)]
        else:
            return None
    final = parts[-1]
    if hasattr(obj, final):
        attr = getattr(obj, final)
        if isinstance(attr, (torch.nn.Parameter, torch.Tensor)):
            return attr
    return None


def _set_param(model: torch.nn.Module, hf_name: str, tensor: torch.Tensor):
    """Set a parameter in the model by dotted path."""
    parts = hf_name.split(".")
    obj = model
    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif part.isdigit() and hasattr(obj, "__getitem__"):
            obj = obj[int(part)]
        else:
            raise AttributeError(f"Cannot navigate to {hf_name}: missing {part}")
    final = parts[-1]
    if hasattr(obj, final):
        attr = getattr(obj, final)
        if isinstance(attr, torch.nn.Parameter):
            attr.data = tensor.to(dtype=attr.data.dtype, device=attr.data.device)
        else:
            setattr(obj, final, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(obj, final, torch.nn.Parameter(tensor, requires_grad=False))


def _build_expert_store_from_reader(
    reader,
    expert_groups: dict[tuple[int, int], dict[str, GGUFTensorInfo]],
    is_multi: bool,
):
    """Build a GGUFExpertStore from parsed expert tensor groups."""
    from .gguf_quant import q4k_expert_to_int4pack
    from .gguf_store import GGUFExpertStore
    from .expert_store import TensorLayout, _pack_tensors

    layers = sorted({k[0] for k in expert_groups})
    experts_per_layer = sorted({k[1] for k in expert_groups})
    num_layers = len(layers)
    num_experts = len(experts_per_layer)

    # Convert first expert to determine layout
    first_key = (layers[0], experts_per_layer[0])
    first_projs = expert_groups[first_key]

    def _read_data(info: GGUFTensorInfo, name: str) -> bytes:
        if is_multi:
            return reader.get_tensor_data(name)
        return reader.get_tensor_data(info)

    gate_info = first_projs["gate"]
    up_info = first_projs["up"]
    down_info = first_projs["down"]

    gate_name = f"blk.{layers[0]}.ffn_gate.{experts_per_layer[0]}.weight"
    up_name = f"blk.{layers[0]}.ffn_up.{experts_per_layer[0]}.weight"
    down_name = f"blk.{layers[0]}.ffn_down.{experts_per_layer[0]}.weight"

    gate_data = _read_data(gate_info, gate_name)
    up_data = _read_data(up_info, up_name)
    down_data = _read_data(down_info, down_name)

    gate_shape = (gate_info.shape[0], gate_info.shape[1])
    up_shape = (up_info.shape[0], up_info.shape[1])
    down_shape = (down_info.shape[0], down_info.shape[1])

    g_packed, g_sz, u_packed, u_sz, d_packed, d_sz = q4k_expert_to_int4pack(
        gate_data, up_data, down_data,
        gate_shape, up_shape, down_shape,
    )

    specs = {
        "gate_packed": (tuple(g_packed.shape), g_packed.dtype),
        "gate_sz": (tuple(g_sz.shape), g_sz.dtype),
        "up_packed": (tuple(u_packed.shape), u_packed.dtype),
        "up_sz": (tuple(u_sz.shape), u_sz.dtype),
        "down_packed": (tuple(d_packed.shape), d_packed.dtype),
        "down_sz": (tuple(d_sz.shape), d_sz.dtype),
    }
    layout = TensorLayout(specs)

    data = torch.empty(
        num_layers, num_experts, layout.total_bytes,
        dtype=torch.uint8,
    )
    if torch.cuda.is_available():
        data = data.pin_memory()

    for li, layer_idx in enumerate(layers):
        for ei, expert_idx in enumerate(experts_per_layer):
            key = (layer_idx, expert_idx)
            projs = expert_groups[key]

            g_name = f"blk.{layer_idx}.ffn_gate.{expert_idx}.weight"
            u_name = f"blk.{layer_idx}.ffn_up.{expert_idx}.weight"
            d_name = f"blk.{layer_idx}.ffn_down.{expert_idx}.weight"

            g_data = _read_data(projs["gate"], g_name)
            u_data = _read_data(projs["up"], u_name)
            d_data = _read_data(projs["down"], d_name)

            g_s = (projs["gate"].shape[0], projs["gate"].shape[1])
            u_s = (projs["up"].shape[0], projs["up"].shape[1])
            d_s = (projs["down"].shape[0], projs["down"].shape[1])

            gp, gsz, up, usz, dp, dsz = q4k_expert_to_int4pack(
                g_data, u_data, d_data, g_s, u_s, d_s,
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

    return GGUFExpertStore(data, layout, num_layers, num_experts)


def _build_expert_store_from_fused_reader(
    reader,
    num_layers: int,
    num_experts: int,
    device: str | torch.device,
):
    """Build a ExpertStore from fused expert tensors (Qwen 3.5 style).

    Fused expert tensors store all experts in a single tensor per projection:
    ``blk.<L>.ffn_gate_exps.weight`` with shape ``(out_dim, in_dim, n_experts)``.

    This function dequants one layer at a time (memory-efficient) and slices
    per expert, then packs into a ``ExpertStore``.

    Returns ``None`` when no fused expert tensors are present.
    """
    from .expert_store import ExpertStore

    fused_layers = reader.list_fused_expert_tensors()
    if not fused_layers:
        return None

    layers = sorted(fused_layers.keys())
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    for layer_idx in layers:
        layer_tensors = fused_layers[layer_idx]

        gate_name = f"blk.{layer_idx}.ffn_gate_exps.weight"
        up_name = f"blk.{layer_idx}.ffn_up_exps.weight"
        down_name = f"blk.{layer_idx}.ffn_down_exps.weight"

        # Dequant full fused tensors on CPU first (shape: [out, in, n_experts])
        gate_bf16 = _dequant_fused_tensor(reader, layer_tensors["gate"], gate_name, "cpu")
        up_bf16 = _dequant_fused_tensor(reader, layer_tensors["up"], up_name, "cpu")
        down_bf16 = _dequant_fused_tensor(reader, layer_tensors["down"], down_name, "cpu")

        n_exp = gate_bf16.shape[2]

        for expert_idx in range(n_exp):
            gate_e = gate_bf16[:, :, expert_idx]  # [intermediate, hidden]
            up_e = up_bf16[:, :, expert_idx]      # [intermediate, hidden]
            down_e = down_bf16[:, :, expert_idx]  # [hidden, intermediate]

            gate_up = torch.cat([gate_e, up_e], dim=0)  # [2*intermediate, hidden]

            expert_weights[(layer_idx, expert_idx)] = {
                "gate_up_proj": gate_up.to(torch.bfloat16),
                "down_proj": down_e.to(torch.bfloat16),
            }

        # Release fused tensors immediately to bound peak RAM
        del gate_bf16, up_bf16, down_bf16

    return ExpertStore.from_dict(expert_weights, num_layers, num_experts)
