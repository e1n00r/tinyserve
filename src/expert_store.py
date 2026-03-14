"""Expert storage with mmap + async GPU transfer.

Supports two storage formats:
1. Packed binary (.bin): contiguous per-expert, single sequential read
2. Safetensors (legacy): 6 separate tensors per layer, scattered layout
"""

import ctypes
import ctypes.util
import mmap
import os
from pathlib import Path

import torch
from safetensors import safe_open

from .config import (
    DOWN_BIAS_SHAPE,
    DOWN_BLOCKS_SHAPE,
    DOWN_SCALES_SHAPE,
    EXPERT_BYTES,
    GATE_UP_BIAS_SHAPE,
    GATE_UP_BLOCKS_SHAPE,
    GATE_UP_SCALES_SHAPE,
    NUM_EXPERTS,
    NUM_LAYERS,
    PACK_DN_BIAS_OFF,
    PACK_DN_BLOCKS_OFF,
    PACK_DN_SCALES_OFF,
    PACK_GU_BIAS_OFF,
    PACK_GU_BLOCKS_OFF,
    PACK_GU_SCALES_OFF,
)

MADV_WILLNEED = 3
_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    return _libc


def _madvise_willneed(ptr: int, length: int):
    libc = _get_libc()
    libc.madvise(ctypes.c_void_p(ptr), ctypes.c_size_t(length), ctypes.c_int(MADV_WILLNEED))


def _byte_size(shape: tuple, dtype_bytes: int = 1) -> int:
    result = dtype_bytes
    for dim in shape[1:]:
        result *= dim
    return result


_GU_B = _byte_size(GATE_UP_BLOCKS_SHAPE)
_GU_S = _byte_size(GATE_UP_SCALES_SHAPE)
_GU_BIAS = _byte_size(GATE_UP_BIAS_SHAPE, 4)
_DN_B = _byte_size(DOWN_BLOCKS_SHAPE)
_DN_S = _byte_size(DOWN_SCALES_SHAPE)
_DN_BIAS = _byte_size(DOWN_BIAS_SHAPE, 4)


def _create_expert_views(packed: torch.Tensor, prefix_shape: tuple = ()):
    """Create typed views into a packed uint8 buffer.

    Works for both single-expert (ExpertBuffer) and batched (LRU cache).
    prefix_shape is () for single, (capacity,) for batched.
    """
    def view(off, nbytes, shape):
        raw = packed[..., off:off + nbytes]
        return raw.view(*prefix_shape, *shape)

    def view_f32(off, nbytes, shape):
        raw = packed[..., off:off + nbytes]
        if prefix_shape:
            raw = raw.reshape(*prefix_shape, nbytes)
        return raw.view(torch.float32).view(*prefix_shape, *shape)

    return (
        view(PACK_GU_BLOCKS_OFF, _GU_B, GATE_UP_BLOCKS_SHAPE[1:]),
        view(PACK_GU_SCALES_OFF, _GU_S, GATE_UP_SCALES_SHAPE[1:]),
        view_f32(PACK_GU_BIAS_OFF, _GU_BIAS, GATE_UP_BIAS_SHAPE[1:]),
        view(PACK_DN_BLOCKS_OFF, _DN_B, DOWN_BLOCKS_SHAPE[1:]),
        view(PACK_DN_SCALES_OFF, _DN_S, DOWN_SCALES_SHAPE[1:]),
        view_f32(PACK_DN_BIAS_OFF, _DN_BIAS, DOWN_BIAS_SHAPE[1:]),
    )


class ExpertBuffer:
    """Pre-allocated GPU buffer for a single expert's MXFP4 data."""

    def __init__(self, device: torch.device):
        self.packed = torch.empty(EXPERT_BYTES, dtype=torch.uint8, device=device)
        (self.gate_up_blocks, self.gate_up_scales, self.gate_up_bias,
         self.down_blocks, self.down_scales, self.down_bias) = _create_expert_views(self.packed)


class ExpertStore:
    """Holds all experts in mmap'd CPU memory, supports async copy to GPU buffers."""

    def __init__(self, weights_dir: str):
        self.weights_dir = Path(weights_dir)
        self._packed = False
        self.gate_up_blocks: list[torch.Tensor] = []
        self.gate_up_scales: list[torch.Tensor] = []
        self.gate_up_bias: list[torch.Tensor] = []
        self.down_blocks: list[torch.Tensor] = []
        self.down_scales: list[torch.Tensor] = []
        self.down_bias: list[torch.Tensor] = []
        self._mmap_tensors: list[torch.Tensor] = []
        self._mmap_files: list[mmap.mmap] = []
        self._file_handles: list = []

    def load(self):
        bin_path = self.weights_dir / "experts_L00.bin"
        if bin_path.exists():
            self._load_packed()
        else:
            self._load_safetensors()

    def _load_packed(self):
        self._packed = True
        print(f"Loading experts from {self.weights_dir} (packed binary, mmap)...")

        for layer_idx in range(NUM_LAYERS):
            path = self.weights_dir / f"experts_L{layer_idx:02d}.bin"
            fd = os.open(str(path), os.O_RDONLY)
            self._file_handles.append(fd)
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            self._mmap_files.append(mm)
            self._mmap_tensors.append(
                torch.frombuffer(mm, dtype=torch.uint8).view(NUM_EXPERTS, EXPERT_BYTES)
            )
            if (layer_idx + 1) % 12 == 0 or layer_idx == NUM_LAYERS - 1:
                print(f"  Mapped {layer_idx + 1}/{NUM_LAYERS} layers")

    def _load_safetensors(self):
        print(f"Loading experts from {self.weights_dir} (safetensors, mmap)...")
        self._handles: list = []
        for layer_idx in range(NUM_LAYERS):
            path = self.weights_dir / f"experts_L{layer_idx:02d}.safetensors"
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._handles.append(handle)
            self.gate_up_blocks.append(handle.get_tensor("gate_up_proj_blocks"))
            self.gate_up_scales.append(handle.get_tensor("gate_up_proj_scales"))
            self.gate_up_bias.append(handle.get_tensor("gate_up_proj_bias"))
            self.down_blocks.append(handle.get_tensor("down_proj_blocks"))
            self.down_scales.append(handle.get_tensor("down_proj_scales"))
            self.down_bias.append(handle.get_tensor("down_proj_bias"))
            if (layer_idx + 1) % 12 == 0 or layer_idx == NUM_LAYERS - 1:
                print(f"  Mapped {layer_idx + 1}/{NUM_LAYERS} layers")

    def prefetch(self, layer_idx: int, expert_idx: int):
        if self._packed:
            tensor = self._mmap_tensors[layer_idx][expert_idx]
            _madvise_willneed(tensor.data_ptr(), EXPERT_BYTES)
        else:
            tensor = self.gate_up_blocks[layer_idx][expert_idx]
            _madvise_willneed(tensor.data_ptr(), tensor.nbytes)

    def copy_to_buffer(
        self,
        buf: ExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ):
        if self._packed:
            buf.packed.copy_(self._mmap_tensors[layer_idx][expert_idx], non_blocking=non_blocking)
        else:
            buf.gate_up_blocks.copy_(self.gate_up_blocks[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.gate_up_scales.copy_(self.gate_up_scales[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.gate_up_bias.copy_(self.gate_up_bias[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.down_blocks.copy_(self.down_blocks[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.down_scales.copy_(self.down_scales[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.down_bias.copy_(self.down_bias[layer_idx][expert_idx], non_blocking=non_blocking)
