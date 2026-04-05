"""Zero-copy GGUF expert storage via mmap.

Parses GGUF per-expert tensors, builds an offset table for each
(layer, expert) pair, and provides copy_to_buffer via a pinned
staging buffer for async H2D transfers.

RAM cost: one pinned staging buffer of expert_bytes. All expert
data stays in the mmap'd file (OS page cache), never in pinned RAM.
"""

from __future__ import annotations

import json
import logging
import mmap
import os
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

import torch

from .expert_cache import ExpertCache
from .expert_store import ExpertBuffer, TensorLayout
from .gguf_dequant import _dequant_fused_tensor
from .gguf_model_loader import open_gguf
from .gguf_reader import GGML_TYPES, GGUFTensorInfo


class MmapExpertStore:
    """Expert store backed by mmap'd GGUF files.

    Conforms to the ExpertStore interface so ExpertPipeline works
    with it unchanged.

    Attributes:
        num_layers: number of MoE layers.
        num_experts: number of experts per layer.
        expert_bytes: bytes per expert (native quant, gate+up+down contiguous).
        buffer_expert_bytes: same as expert_bytes (no BF16 expansion needed).
        layout: TensorLayout with specs {"gate": ..., "up": ..., "down": ...},
            each projection stored as raw uint8 bytes.
        _bf16_layout: same object as layout (native-quant path, no dequant).
        ggml_types: {"gate": int, "up": int, "down": int} GGML type codes.
        proj_shapes: {"gate": (N, K), "up": (N, K), "down": (K, N)}.
    """

    def __init__(self, path: str | Path):
        self._reader = open_gguf(path)
        groups = self._reader.list_expert_tensors()

        if not groups:
            self._reader.close()
            raise ValueError(f"No per-expert tensors found in {path}")

        layers = sorted({k[0] for k in groups})
        experts = sorted({k[1] for k in groups})
        self.num_layers = len(layers)
        self.num_experts = len(experts)

        # Build layout from the first expert
        first_key = (layers[0], experts[0])
        first_projs = groups[first_key]

        gate_info: GGUFTensorInfo = first_projs["gate"]
        up_info: GGUFTensorInfo = first_projs["up"]
        down_info: GGUFTensorInfo = first_projs["down"]

        gate_nbytes = gate_info.nbytes
        up_nbytes = up_info.nbytes
        down_nbytes = down_info.nbytes

        self.ggml_types: dict[str, int] = {
            "gate": gate_info.ggml_type,
            "up": up_info.ggml_type,
            "down": down_info.ggml_type,
        }

        self.proj_shapes: dict[str, tuple[int, int]] = {
            "gate": (gate_info.shape[0], gate_info.shape[1]),
            "up": (up_info.shape[0], up_info.shape[1]),
            "down": (down_info.shape[0], down_info.shape[1]),
        }

        specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "gate": ((gate_nbytes,), torch.uint8),
            "up": ((up_nbytes,), torch.uint8),
            "down": ((down_nbytes,), torch.uint8),
        }
        self.layout = TensorLayout(specs)
        self._bf16_layout = self.layout

        self.expert_bytes = self.layout.total_bytes
        self.buffer_expert_bytes = self.expert_bytes

        # Build (layer_store_idx, expert_store_idx) -> {proj: TensorInfo} table
        self._groups = groups
        self._layer_map: dict[int, int] = {layer: i for i, layer in enumerate(layers)}
        self._expert_map: dict[int, int] = {expert: i for i, expert in enumerate(experts)}

        # Pinned staging buffer: mmap pages are not pinned, so non_blocking
        # with mmap is silently synchronous. This buffer bridges the gap.
        self._pinned_staging = torch.empty(self.expert_bytes, dtype=torch.uint8).pin_memory()

    @property
    def _fp8(self) -> bool:
        return False

    def _read_expert(self, layer_idx: int, expert_idx: int) -> bytes:
        """Read raw bytes for one expert (gate+up+down concatenated), padded to expert_bytes."""
        projs = self._groups[(layer_idx, expert_idx)]
        reader = self._reader
        parts = []
        for proj in ("gate", "up", "down"):
            info = projs[proj]
            # Read actual bytes
            if hasattr(reader, 'tensor_names'):
                if info.name in reader.tensor_names:
                    raw = reader.get_tensor_data(info.name)
                else:
                    raw = reader.get_tensor_data_by_offset(info.offset, info.nbytes)
            else:
                raw = reader.get_tensor_data(info)
            # Pad to layout size (mixed quant: some layers have smaller projections)
            expected = self.layout.sizes[proj]
            if len(raw) < expected:
                raw = raw + b"\x00" * (expected - len(raw))
            parts.append(raw)
        return b"".join(parts)

    def get_expert_data(self, layer_idx: int, expert_idx: int) -> torch.Tensor:
        """Return packed expert data as a CPU uint8 tensor."""
        raw = self._read_expert(layer_idx, expert_idx)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    def allocate_buffer(self, device: torch.device) -> ExpertBuffer:
        return ExpertBuffer(self.layout, device)

    def copy_packed_weights(
        self,
        buf: ExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ) -> None:
        """Copy packed expert weights from mmap into a GPU buffer."""
        raw = self._read_expert(layer_idx, expert_idx)
        self._pinned_staging[: len(raw)].copy_(
            torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        )
        buf.packed.copy_(self._pinned_staging, non_blocking=non_blocking)

    def copy_to_buffer(
        self,
        buf: ExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ) -> None:
        self.copy_packed_weights(buf, layer_idx, expert_idx, non_blocking=non_blocking)

    def copy_to_buffer_slot(
        self,
        cache: ExpertCache,
        slot: int,
        layer_idx: int,
        expert_idx: int,
    ) -> None:
        raw = self._read_expert(layer_idx, expert_idx)
        self._pinned_staging[: len(raw)].copy_(
            torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        )
        cache._packed[slot].copy_(self._pinned_staging)

    @classmethod
    def from_fused(cls, path: str | Path) -> MmapExpertStore:
        """Create MmapExpertStore from fused GGUF (Qwen-style).

        Fused tensors have shape (out_dim, in_dim, n_experts). In ggml's layout,
        the expert dimension is outermost — each expert is a contiguous byte range.
        No conversion or dequantization needed.

        Expert i's data starts at: tensor_offset + i * bytes_per_expert
        where bytes_per_expert = (out_dim * in_dim // block_size) * bytes_per_block.
        """
        reader = open_gguf(path)
        fused = reader.list_fused_expert_tensors()
        if not fused:
            reader.close()
            raise ValueError(f"No fused expert tensors in {path}")

        layers = sorted(fused.keys())
        first = fused[layers[0]]
        n_experts = first["gate"].shape[2]

        # Per-expert byte sizes from fused tensor dimensions
        def _expert_bytes(info: GGUFTensorInfo) -> int:
            elements = info.shape[0] * info.shape[1]
            _, bpb, bs = GGML_TYPES[info.ggml_type]
            return (elements // bs) * bpb

        # Mixed quantization (e.g., Q4_K_M) means different layers can have
        # different quant types for the same projection. Use max across all layers
        # so cache slots and buffers fit the largest expert.
        gate_nbytes = max(_expert_bytes(fused[l]["gate"]) for l in layers)
        up_nbytes = max(_expert_bytes(fused[l]["up"]) for l in layers)
        down_nbytes = max(_expert_bytes(fused[l]["down"]) for l in layers)

        # Build synthetic per-expert groups by computing byte offsets
        # within each fused tensor. Expert i starts at tensor_offset + i * expert_proj_bytes.
        groups: dict[tuple[int, int], dict[str, GGUFTensorInfo]] = {}
        for layer_idx in layers:
            layer_fused = fused[layer_idx]
            for expert_idx in range(n_experts):
                expert_projs = {}
                for proj in ("gate", "up", "down"):
                    fused_info = layer_fused[proj]
                    proj_expert_bytes = _expert_bytes(fused_info)
                    # Create a synthetic TensorInfo pointing to this expert's slice
                    expert_projs[proj] = GGUFTensorInfo(
                        name=f"blk.{layer_idx}.ffn_{proj}.{expert_idx}.weight",
                        shape=(fused_info.shape[0], fused_info.shape[1]),
                        ggml_type=fused_info.ggml_type,
                        ggml_type_name=fused_info.ggml_type_name,
                        offset=fused_info.offset + expert_idx * proj_expert_bytes,
                        nbytes=proj_expert_bytes,
                        block_size=fused_info.block_size,
                    )
                groups[(layer_idx, expert_idx)] = expert_projs

        # Build the store using the synthetic per-expert groups
        store = cls.__new__(cls)
        store._reader = reader
        store.num_layers = len(layers)
        store.num_experts = n_experts

        # Default ggml_types from first layer. For mixed quant (_M suffix),
        # actual types vary per layer — stored in _layer_ggml_types.
        store.ggml_types = {
            "gate": first["gate"].ggml_type,
            "up": first["up"].ggml_type,
            "down": first["down"].ggml_type,
        }
        store._layer_ggml_types = {}
        store._layer_proj_nbytes = {}
        for layer_idx in layers:
            layer_types = {}
            layer_nbytes = {}
            for proj in ("gate", "up", "down"):
                layer_types[proj] = fused[layer_idx][proj].ggml_type
                layer_nbytes[proj] = _expert_bytes(fused[layer_idx][proj])
            if layer_types != store.ggml_types:
                store._layer_ggml_types[layer_idx] = layer_types
            store._layer_proj_nbytes[layer_idx] = layer_nbytes
        store.proj_shapes = {
            "gate": (first["gate"].shape[0], first["gate"].shape[1]),
            "up": (first["up"].shape[0], first["up"].shape[1]),
            "down": (first["down"].shape[0], first["down"].shape[1]),
        }

        specs = {
            "gate": ((gate_nbytes,), torch.uint8),
            "up": ((up_nbytes,), torch.uint8),
            "down": ((down_nbytes,), torch.uint8),
        }
        store.layout = TensorLayout(specs)
        store._bf16_layout = store.layout
        store.expert_bytes = store.layout.total_bytes
        store.buffer_expert_bytes = store.expert_bytes

        store._groups = groups
        store._layer_map = {layer: i for i, layer in enumerate(layers)}
        store._expert_map = {expert: i for i, expert in enumerate(range(n_experts))}
        store._pinned_staging = torch.empty(store.expert_bytes, dtype=torch.uint8).pin_memory()

        logger.info(
            "MmapExpertStore (fused, zero-copy): %d layers, %d experts, "
            "%.1f MB/expert (%s gate, %s down)",
            store.num_layers, store.num_experts,
            store.expert_bytes / 1e6,
            first["gate"].ggml_type_name, first["down"].ggml_type_name,
        )
        return store

    def close(self) -> None:
        self._reader.close()


def quantize_to_q8_0(tensor: torch.Tensor) -> bytes:
    """Quantize a float tensor to Q8_0 format.

    Q8_0 format: 34 bytes per 32 elements.
    Layout per block: float16 scale (2 bytes) + int8[32] quants (32 bytes).
    Scale is amax/127 so max absolute value maps to +/-127.

    Args:
        tensor: Any shape float tensor. Trailing elements (< 32) are dropped.

    Returns:
        Raw bytes in Q8_0 format, n_blocks * 34 bytes.
    """
    flat = tensor.flatten().float()
    n_blocks = flat.shape[0] // 32
    blocks = flat[: n_blocks * 32].reshape(n_blocks, 32)
    amax = blocks.abs().amax(dim=1)
    scales = (amax / 127.0).clamp(min=1e-10)
    quants = torch.round(blocks / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    scales_f16 = scales.to(torch.float16)
    scales_bytes = scales_f16.numpy().tobytes()
    quants_bytes = quants.numpy().tobytes()
    result = bytearray(n_blocks * 34)
    for i in range(n_blocks):
        result[i * 34 : i * 34 + 2] = scales_bytes[i * 2 : i * 2 + 2]
        result[i * 34 + 2 : i * 34 + 34] = quants_bytes[i * 32 : i * 32 + 32]
    return bytes(result)


def quantize_to_q4_0(tensor: torch.Tensor) -> bytes:
    """Quantize a float tensor to Q4_0 format.

    Q4_0 format: 18 bytes per 32 elements.
    Layout per block: float16 scale (2 bytes) + nibble-packed uint8[16] (16 bytes).
    Symmetric quantization: scale = amax/8, q = round(x/scale) + 8, clamp [0,15].

    ~1.9x smaller than Q8_0.
    """
    flat = tensor.flatten().float()
    n_blocks = flat.shape[0] // 32
    blocks = flat[: n_blocks * 32].reshape(n_blocks, 32)
    amax = blocks.abs().amax(dim=1)
    scales = (amax / 8.0).clamp(min=1e-10)
    quants = torch.round(blocks / scales.unsqueeze(1) + 8).clamp(0, 15).to(torch.uint8)
    scales_f16 = scales.to(torch.float16)
    # Pack pairs of 4-bit values into bytes: low nibble = even index, high nibble = odd
    even = quants[:, 0::2]  # [n_blocks, 16]
    odd = quants[:, 1::2]   # [n_blocks, 16]
    packed = (even | (odd << 4)).to(torch.uint8)  # [n_blocks, 16]
    scales_bytes = scales_f16.numpy().tobytes()
    packed_bytes = packed.numpy().tobytes()
    result = bytearray(n_blocks * 18)
    for i in range(n_blocks):
        result[i * 18 : i * 18 + 2] = scales_bytes[i * 2 : i * 2 + 2]
        result[i * 18 + 2 : i * 18 + 18] = packed_bytes[i * 16 : i * 16 + 16]
    return bytes(result)


class FusedMmapExpertStore:
    """Expert store backed by a pre-converted .experts file.

    The .experts file contains Q8_0 data for every (layer, expert) pair,
    laid out in layer-major order. The file header is a JSON blob prefixed
    by a 4-byte little-endian length.

    Attributes:
        num_layers: number of MoE layers.
        num_experts: number of experts per layer.
        expert_bytes: bytes per expert (Q8_0, gate+up+down contiguous).
        buffer_expert_bytes: same as expert_bytes.
        layout: TensorLayout with {"gate", "up", "down"} as uint8.
        _bf16_layout: same object as layout.
        ggml_types: {"gate": 8, "up": 8, "down": 8} (Q8_0 = 8).
        proj_shapes: {"gate": (N, K), "up": (N, K), "down": (K, N)}.
    """

    def __init__(
        self,
        experts_path: Path,
        header: dict,
        data_offset: int,
        mm: mmap.mmap,
        f,
    ):
        self._experts_path = experts_path
        self._header = header
        self._data_offset = data_offset
        self._mm = mm
        self._f = f

        self.num_layers: int = header["num_layers"]
        self.num_experts: int = header["num_experts"]

        gate_bytes: int = header["gate_bytes"]
        up_bytes: int = header["up_bytes"]
        down_bytes: int = header["down_bytes"]

        ggml_type = header.get("ggml_type", 2)
        self.ggml_types: dict[str, int] = {"gate": ggml_type, "up": ggml_type, "down": ggml_type}
        self.proj_shapes: dict[str, tuple[int, int]] = {
            "gate": tuple(header["gate_shape"]),
            "up": tuple(header["up_shape"]),
            "down": tuple(header["down_shape"]),
        }

        specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "gate": ((gate_bytes,), torch.uint8),
            "up": ((up_bytes,), torch.uint8),
            "down": ((down_bytes,), torch.uint8),
        }
        self.layout = TensorLayout(specs)
        self._bf16_layout = self.layout
        self.expert_bytes: int = header["expert_bytes"]
        self.buffer_expert_bytes: int = self.expert_bytes

        self._pinned_staging = torch.empty(self.expert_bytes, dtype=torch.uint8).pin_memory()

    @property
    def _fp8(self) -> bool:
        return False

    def _expert_offset(self, layer_idx: int, expert_idx: int) -> int:
        return self._data_offset + (layer_idx * self.num_experts + expert_idx) * self.expert_bytes

    def _read_expert(self, layer_idx: int, expert_idx: int) -> bytes:
        off = self._expert_offset(layer_idx, expert_idx)
        return self._mm[off : off + self.expert_bytes]

    def get_expert_data(self, layer_idx: int, expert_idx: int) -> torch.Tensor:
        raw = self._read_expert(layer_idx, expert_idx)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    def allocate_buffer(self, device: torch.device) -> ExpertBuffer:
        return ExpertBuffer(self.layout, device)

    def copy_packed_weights(
        self,
        buf: ExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ) -> None:
        """Copy packed expert weights from mmap into a GPU buffer."""
        raw = self._read_expert(layer_idx, expert_idx)
        self._pinned_staging[: len(raw)].copy_(
            torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        )
        buf.packed.copy_(self._pinned_staging, non_blocking=non_blocking)

    def copy_to_buffer(
        self,
        buf: ExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ) -> None:
        # Backward-compat shim — removed in Task 12
        return self.copy_packed_weights(buf, layer_idx, expert_idx, non_blocking=non_blocking)

    def copy_to_buffer_slot(
        self,
        cache: ExpertCache,
        slot: int,
        layer_idx: int,
        expert_idx: int,
    ) -> None:
        raw = self._read_expert(layer_idx, expert_idx)
        self._pinned_staging[: len(raw)].copy_(
            torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        )
        cache._packed[slot].copy_(self._pinned_staging)

    def close(self) -> None:
        self._mm.close()
        self._f.close()


def _convert_fused_to_per_expert(gguf_path: Path, experts_path: Path) -> None:
    """Convert fused GGUF expert tensors to per-expert Q4_0 .experts file.

    Q4_0 (18 bytes/32 elements) is ~1.9x smaller than Q8_0 and keeps the
    .experts file at ~65 GB for Qwen 122B (vs 123 GB for Q8_0).

    Writes to a .tmp file first, then atomically renames to experts_path.
    Processes one layer at a time to bound peak RAM usage (~3 GB peak).
    """
    tmp_path = Path(str(experts_path) + ".tmp")
    reader = open_gguf(gguf_path)
    fused = reader.list_fused_expert_tensors()

    if not fused:
        reader.close()
        raise ValueError(f"No fused expert tensors found in {gguf_path}")

    layers = sorted(fused.keys())
    first_layer_info = fused[layers[0]]
    gate_info = first_layer_info["gate"]
    up_info = first_layer_info["up"]
    down_info = first_layer_info["down"]

    n_experts = gate_info.shape[2]
    gate_shape = (gate_info.shape[0], gate_info.shape[1])
    up_shape = (up_info.shape[0], up_info.shape[1])
    down_shape = (down_info.shape[0], down_info.shape[1])

    # Compute Q4_0 byte sizes (18 bytes per 32 elements — compact, fast to quantize)
    gate_elements = gate_shape[0] * gate_shape[1]
    up_elements = up_shape[0] * up_shape[1]
    down_elements = down_shape[0] * down_shape[1]
    gate_bytes = (gate_elements // 32) * 18
    up_bytes = (up_elements // 32) * 18
    down_bytes = (down_elements // 32) * 18
    expert_bytes = gate_bytes + up_bytes + down_bytes

    header = {
        "num_layers": len(layers),
        "num_experts": n_experts,
        "ggml_type": 2,  # Q4_0
        "gate_shape": list(gate_shape),
        "up_shape": list(up_shape),
        "down_shape": list(down_shape),
        "gate_bytes": gate_bytes,
        "up_bytes": up_bytes,
        "down_bytes": down_bytes,
        "expert_bytes": expert_bytes,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with open(tmp_path, "wb") as f:
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)

        for li, layer_idx in enumerate(layers):
            info = fused[layer_idx]
            logger.info("Converting layer %d/%d...", li + 1, len(layers))

            # Use vectorized city96 dequant (100x faster than loop-based _dequant_fused_tensor)
            from .gguf_dequant import dequant_tensor

            def _dequant_fused(tensor_info):
                raw = reader.get_tensor_data(tensor_info.name if hasattr(tensor_info, 'name') else tensor_info)
                shape_3d = tuple(tensor_info.shape)
                n_elements = 1
                for d in shape_3d:
                    n_elements *= d
                return dequant_tensor(raw, tensor_info.ggml_type, (n_elements,)).reshape(shape_3d).float()

            gate_f32 = _dequant_fused(info["gate"])
            up_f32 = _dequant_fused(info["up"])
            down_f32 = _dequant_fused(info["down"])

            for expert_idx in range(n_experts):
                gate_e = gate_f32[:, :, expert_idx]
                up_e = up_f32[:, :, expert_idx]
                down_e = down_f32[:, :, expert_idx]
                f.write(quantize_to_q4_0(gate_e))
                f.write(quantize_to_q4_0(up_e))
                f.write(quantize_to_q4_0(down_e))

            del gate_f32, up_f32, down_f32

    reader.close()
    os.rename(tmp_path, experts_path)


def _from_experts_file(experts_path: Path) -> FusedMmapExpertStore:
    """Open an existing .experts file and return a FusedMmapExpertStore."""
    f = open(experts_path, "rb")
    header_len = struct.unpack("<I", f.read(4))[0]
    header = json.loads(f.read(header_len).decode("utf-8"))
    data_offset = 4 + header_len
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return FusedMmapExpertStore(experts_path, header, data_offset, mm, f)


def from_fused_gguf(path: str | Path) -> FusedMmapExpertStore:
    """Load fused GGUF expert tensors, converting to .experts file on first call.

    On the first call, dequantizes all fused expert tensors to float32,
    slices per expert, requantizes to Q8_0, and writes a .experts file
    adjacent to the .gguf file. Subsequent calls reuse the .experts file
    directly via mmap.

    Args:
        path: Path to a GGUF file containing fused expert tensors
              (blk.<L>.ffn_{gate,up,down}_exps.weight).

    Returns:
        FusedMmapExpertStore backed by the .experts file.
    """
    path = Path(path)
    experts_path = Path(str(path) + ".experts")

    if not experts_path.exists():
        _convert_fused_to_per_expert(path, experts_path)

    return _from_experts_file(experts_path)
