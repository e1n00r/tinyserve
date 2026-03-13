"""Pinned CPU expert storage with async GPU transfer."""

from pathlib import Path

import torch
from safetensors import safe_open

from .config import (
    DOWN_BIAS_SHAPE,
    DOWN_BLOCKS_SHAPE,
    DOWN_SCALES_SHAPE,
    GATE_UP_BIAS_SHAPE,
    GATE_UP_BLOCKS_SHAPE,
    GATE_UP_SCALES_SHAPE,
    NUM_EXPERTS,
    NUM_LAYERS,
)


class ExpertBuffer:
    """Pre-allocated GPU buffer for a single expert's MXFP4 data."""

    def __init__(self, device: torch.device):
        # Shapes for a single expert (no batch dim)
        self.gate_up_blocks = torch.empty(
            GATE_UP_BLOCKS_SHAPE[1:], dtype=torch.uint8, device=device
        )
        self.gate_up_scales = torch.empty(
            GATE_UP_SCALES_SHAPE[1:], dtype=torch.uint8, device=device
        )
        self.gate_up_bias = torch.empty(
            GATE_UP_BIAS_SHAPE[1:], dtype=torch.float32, device=device
        )
        self.down_blocks = torch.empty(
            DOWN_BLOCKS_SHAPE[1:], dtype=torch.uint8, device=device
        )
        self.down_scales = torch.empty(
            DOWN_SCALES_SHAPE[1:], dtype=torch.uint8, device=device
        )
        self.down_bias = torch.empty(
            DOWN_BIAS_SHAPE[1:], dtype=torch.float32, device=device
        )


class ExpertStore:
    """Holds all experts in pinned CPU memory, supports async copy to GPU buffers."""

    def __init__(self, weights_dir: str):
        self.weights_dir = Path(weights_dir)
        # Per-layer tensors: [NUM_EXPERTS, ...]
        self.gate_up_blocks: list[torch.Tensor] = []
        self.gate_up_scales: list[torch.Tensor] = []
        self.gate_up_bias: list[torch.Tensor] = []
        self.down_blocks: list[torch.Tensor] = []
        self.down_scales: list[torch.Tensor] = []
        self.down_bias: list[torch.Tensor] = []

    def load(self):
        """Load all expert weights into pinned CPU memory."""
        print(f"Loading experts from {self.weights_dir}...")
        for layer_idx in range(NUM_LAYERS):
            path = self.weights_dir / f"experts_L{layer_idx:02d}.safetensors"
            with safe_open(str(path), framework="pt", device="cpu") as f:
                gu_b = f.get_tensor("gate_up_proj_blocks").pin_memory()
                gu_s = f.get_tensor("gate_up_proj_scales").pin_memory()
                gu_bias = f.get_tensor("gate_up_proj_bias").pin_memory()
                dn_b = f.get_tensor("down_proj_blocks").pin_memory()
                dn_s = f.get_tensor("down_proj_scales").pin_memory()
                dn_bias = f.get_tensor("down_proj_bias").pin_memory()

            self.gate_up_blocks.append(gu_b)
            self.gate_up_scales.append(gu_s)
            self.gate_up_bias.append(gu_bias)
            self.down_blocks.append(dn_b)
            self.down_scales.append(dn_s)
            self.down_bias.append(dn_bias)

            if (layer_idx + 1) % 12 == 0 or layer_idx == NUM_LAYERS - 1:
                loaded_gb = sum(
                    t.nbytes for tensors in [
                        self.gate_up_blocks, self.gate_up_scales, self.gate_up_bias,
                        self.down_blocks, self.down_scales, self.down_bias,
                    ] for t in tensors
                ) / 1024**3
                print(f"  Loaded {layer_idx + 1}/{NUM_LAYERS} layers ({loaded_gb:.1f} GB pinned)")

    def copy_to_buffer(
        self,
        buf: ExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ):
        """Copy a single expert's data from CPU pinned memory to GPU buffer."""
        buf.gate_up_blocks.copy_(self.gate_up_blocks[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.gate_up_scales.copy_(self.gate_up_scales[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.gate_up_bias.copy_(self.gate_up_bias[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.down_blocks.copy_(self.down_blocks[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.down_scales.copy_(self.down_scales[layer_idx][expert_idx], non_blocking=non_blocking)
        buf.down_bias.copy_(self.down_bias[layer_idx][expert_idx], non_blocking=non_blocking)
