"""Expert storage with mmap + async GPU transfer."""

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
    """Holds all experts in mmap'd CPU memory, supports async copy to GPU buffers."""

    def __init__(self, weights_dir: str):
        self.weights_dir = Path(weights_dir)
        self.gate_up_blocks: list[torch.Tensor] = []
        self.gate_up_scales: list[torch.Tensor] = []
        self.gate_up_bias: list[torch.Tensor] = []
        self.down_blocks: list[torch.Tensor] = []
        self.down_scales: list[torch.Tensor] = []
        self.down_bias: list[torch.Tensor] = []

    def load(self):
        """Load all expert weights via mmap (zero-copy from disk)."""
        print(f"Loading experts from {self.weights_dir} (mmap)...")
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

    def copy_to_buffer(
        self,
        buf: ExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ):
        """Copy a single expert's data from mmap to GPU buffer."""
        buf.gate_up_blocks.copy_(self.gate_up_blocks[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.gate_up_scales.copy_(self.gate_up_scales[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.gate_up_bias.copy_(self.gate_up_bias[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.down_blocks.copy_(self.down_blocks[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.down_scales.copy_(self.down_scales[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.down_bias.copy_(self.down_bias[layer_idx][expert_idx], non_blocking=non_blocking)
