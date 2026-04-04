"""Zero-copy GGUF expert storage via mmap.

Parses GGUF per-expert tensors, builds an offset table for each
(layer, expert) pair, and provides copy_to_buffer via a pinned
staging buffer for async H2D transfers.

RAM cost: one pinned staging buffer of expert_bytes. All expert
data stays in the mmap'd file (OS page cache), never in pinned RAM.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .expert_cache import ExpertCache
from .expert_store import ExpertBuffer, TensorLayout
from .gguf_loader import open_gguf
from .gguf_reader import GGUFTensorInfo


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
        """Read raw bytes for one expert (gate+up+down concatenated)."""
        projs = self._groups[(layer_idx, expert_idx)]
        gate_bytes = self._reader.get_tensor_data(projs["gate"])
        up_bytes = self._reader.get_tensor_data(projs["up"])
        down_bytes = self._reader.get_tensor_data(projs["down"])
        return gate_bytes + up_bytes + down_bytes

    def get_expert_data(self, layer_idx: int, expert_idx: int) -> torch.Tensor:
        """Return packed expert data as a CPU uint8 tensor."""
        raw = self._read_expert(layer_idx, expert_idx)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    def allocate_buffer(self, device: torch.device) -> ExpertBuffer:
        return ExpertBuffer(self.layout, device)

    def copy_to_buffer(
        self,
        buf: ExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ) -> None:
        raw = self._read_expert(layer_idx, expert_idx)
        self._pinned_staging[: len(raw)].copy_(
            torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        )
        buf.packed.copy_(self._pinned_staging, non_blocking=non_blocking)

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
        self._reader.close()
