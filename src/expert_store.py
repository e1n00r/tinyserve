"""Expert storage with mmap + async GPU transfer.

Supports two storage formats:
1. Safetensors (original): 6 separate tensors per layer, scattered layout
2. Packed binary (.bin): contiguous per-expert, single sequential read
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

_gu_b = GATE_UP_BLOCKS_SHAPE[1] * GATE_UP_BLOCKS_SHAPE[2] * GATE_UP_BLOCKS_SHAPE[3]
_gu_s = GATE_UP_SCALES_SHAPE[1] * GATE_UP_SCALES_SHAPE[2]
_gu_bias = GATE_UP_BIAS_SHAPE[1] * 4
_dn_b = DOWN_BLOCKS_SHAPE[1] * DOWN_BLOCKS_SHAPE[2] * DOWN_BLOCKS_SHAPE[3]
_dn_s = DOWN_SCALES_SHAPE[1] * DOWN_SCALES_SHAPE[2]
_dn_bias = DOWN_BIAS_SHAPE[1] * 4

# madvise constants
MADV_WILLNEED = 3
_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    return _libc


def _madvise_willneed(ptr: int, length: int):
    """Issue madvise(MADV_WILLNEED) to trigger kernel readahead."""
    libc = _get_libc()
    libc.madvise(ctypes.c_void_p(ptr), ctypes.c_size_t(length), ctypes.c_int(MADV_WILLNEED))


class ExpertBuffer:
    """Pre-allocated GPU buffer for a single expert's MXFP4 data.

    Uses a contiguous packed buffer with views for each tensor.
    """

    def __init__(self, device: torch.device):
        self.packed = torch.empty(EXPERT_BYTES, dtype=torch.uint8, device=device)
        self._create_views()

    def _create_views(self):
        p = self.packed
        self.gate_up_blocks = p[PACK_GU_BLOCKS_OFF:PACK_GU_BLOCKS_OFF + _gu_b].view(
            *GATE_UP_BLOCKS_SHAPE[1:])
        self.gate_up_scales = p[PACK_GU_SCALES_OFF:PACK_GU_SCALES_OFF + _gu_s].view(
            *GATE_UP_SCALES_SHAPE[1:])
        self.gate_up_bias = p[PACK_GU_BIAS_OFF:PACK_GU_BIAS_OFF + _gu_bias].view(
            torch.float32).view(*GATE_UP_BIAS_SHAPE[1:])
        self.down_blocks = p[PACK_DN_BLOCKS_OFF:PACK_DN_BLOCKS_OFF + _dn_b].view(
            *DOWN_BLOCKS_SHAPE[1:])
        self.down_scales = p[PACK_DN_SCALES_OFF:PACK_DN_SCALES_OFF + _dn_s].view(
            *DOWN_SCALES_SHAPE[1:])
        self.down_bias = p[PACK_DN_BIAS_OFF:PACK_DN_BIAS_OFF + _dn_bias].view(
            torch.float32).view(*DOWN_BIAS_SHAPE[1:])


class ExpertStore:
    """Holds all experts in mmap'd CPU memory, supports async copy to GPU buffers.

    Auto-detects packed binary format (.bin) vs safetensors.
    """

    def __init__(self, weights_dir: str):
        self.weights_dir = Path(weights_dir)
        self._packed = False
        # Safetensors format
        self.gate_up_blocks: list[torch.Tensor] = []
        self.gate_up_scales: list[torch.Tensor] = []
        self.gate_up_bias: list[torch.Tensor] = []
        self.down_blocks: list[torch.Tensor] = []
        self.down_scales: list[torch.Tensor] = []
        self.down_bias: list[torch.Tensor] = []
        # Packed binary format
        self._mmap_tensors: list[torch.Tensor] = []
        self._mmap_files: list[mmap.mmap] = []
        self._file_handles: list = []

    def load(self):
        """Load all expert weights. Auto-detects packed binary vs safetensors."""
        bin_path = self.weights_dir / "experts_L00.bin"
        if bin_path.exists():
            self._load_packed()
        else:
            self._load_safetensors()

    def _load_packed(self):
        """Load contiguous binary expert files via mmap."""
        self._packed = True
        print(f"Loading experts from {self.weights_dir} (packed binary, mmap)...")

        for layer_idx in range(NUM_LAYERS):
            path = self.weights_dir / f"experts_L{layer_idx:02d}.bin"
            fd = os.open(str(path), os.O_RDONLY)
            self._file_handles.append(fd)

            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            self._mmap_files.append(mm)

            # Create a tensor view over the mmap'd region
            # [NUM_EXPERTS, EXPERT_BYTES] uint8
            t = torch.frombuffer(mm, dtype=torch.uint8).view(NUM_EXPERTS, EXPERT_BYTES)
            self._mmap_tensors.append(t)

            if (layer_idx + 1) % 12 == 0 or layer_idx == NUM_LAYERS - 1:
                print(f"  Mapped {layer_idx + 1}/{NUM_LAYERS} layers (packed)")

    def _load_safetensors(self):
        """Load safetensors format (original, scattered layout)."""
        print(f"Loading experts from {self.weights_dir} (safetensors, mmap)...")
        self._handles: list = []
        for layer_idx in range(NUM_LAYERS):
            path = self.weights_dir / f"experts_L{layer_idx:02d}.safetensors"
            f = safe_open(str(path), framework="pt", device="cpu")
            self._handles.append(f)

            self.gate_up_blocks.append(f.get_tensor("gate_up_proj_blocks"))
            self.gate_up_scales.append(f.get_tensor("gate_up_proj_scales"))
            self.gate_up_bias.append(f.get_tensor("gate_up_proj_bias"))
            self.down_blocks.append(f.get_tensor("down_proj_blocks"))
            self.down_scales.append(f.get_tensor("down_proj_scales"))
            self.down_bias.append(f.get_tensor("down_proj_bias"))

            if (layer_idx + 1) % 12 == 0 or layer_idx == NUM_LAYERS - 1:
                print(f"  Mapped {layer_idx + 1}/{NUM_LAYERS} layers")

    def prefetch(self, layer_idx: int, expert_idx: int):
        """Issue madvise(WILLNEED) to pre-fault expert pages from SSD."""
        if self._packed:
            t = self._mmap_tensors[layer_idx][expert_idx]
            _madvise_willneed(t.data_ptr(), EXPERT_BYTES)
        else:
            # Prefetch the largest tensor (gate_up_blocks) to trigger readahead
            t = self.gate_up_blocks[layer_idx][expert_idx]
            _madvise_willneed(t.data_ptr(), t.nbytes)

    def copy_to_buffer(
        self,
        buf: ExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ):
        """Copy a single expert's data from mmap to GPU buffer."""
        if self._packed:
            # Single contiguous copy
            buf.packed.copy_(self._mmap_tensors[layer_idx][expert_idx], non_blocking=non_blocking)
        else:
            # 6 separate copies (scattered layout)
            buf.gate_up_blocks.copy_(self.gate_up_blocks[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.gate_up_scales.copy_(self.gate_up_scales[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.gate_up_bias.copy_(self.gate_up_bias[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.down_blocks.copy_(self.down_blocks[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.down_scales.copy_(self.down_scales[layer_idx][expert_idx], non_blocking=non_blocking)
            buf.down_bias.copy_(self.down_bias[layer_idx][expert_idx], non_blocking=non_blocking)
