# SPDX-License-Identifier: MIT
"""GGMLExpertForward — native quant expert compute with ggml MMVQ kernels.

batch=1: uses ggml CUDA kernels (3 calls: gate, up, down).
batch>1 or no ggml: city96 dequant + F.linear fallback.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .expert_store import TensorLayout


def _check_ggml() -> bool:
    try:
        return (
            hasattr(torch.ops, "tinyserve_ggml_ops")
            and hasattr(torch.ops.tinyserve_ggml_ops, "ggml_mul_mat_vec")
        )
    except Exception:
        return False


class GGMLExpertForward:
    """Expert forward that uses ggml MMVQ kernels for batch=1.

    Falls back to city96 dequant + F.linear for batch>1 or when the
    ggml CUDA extension is unavailable.

    Attributes:
        _has_ggml: whether the ggml CUDA extension is loaded.
    """

    def __init__(
        self,
        layout: TensorLayout,
        ggml_types: dict[str, int],
        act_fn,
        proj_shapes: dict[str, tuple[int, int]],
    ) -> None:
        """
        Args:
            layout: TensorLayout with specs like
                {"gate": ((nbytes,), torch.uint8), ...}.
            ggml_types: GGML type IDs per projection, e.g.
                {"gate": 12, "up": 12, "down": 13}.
            act_fn: activation function (e.g. torch.nn.SiLU()).
            proj_shapes: (N, K) per projection,
                {"gate": (N, K), "up": (N, K), "down": (K, N)}.
        """
        self._ggml_types = ggml_types
        self._act_fn = act_fn
        self._proj_shapes = proj_shapes

        # Bake offsets once — not per call
        self._gate_off: int = layout.offsets["gate"]
        self._gate_end: int = self._gate_off + layout.sizes["gate"]
        self._up_off: int = layout.offsets["up"]
        self._up_end: int = self._up_off + layout.sizes["up"]
        self._down_off: int = layout.offsets["down"]
        self._down_end: int = self._down_off + layout.sizes["down"]

        self._has_ggml: bool = _check_ggml()

    def forward(self, packed: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forward on native quant expert data.

        Args:
            packed: [expert_bytes] uint8 tensor — raw GGUF quant bytes.
            h: [batch, hidden] BF16 tensor.

        Returns:
            [batch, hidden] BF16 tensor.
        """
        if h.shape[0] == 1 and self._has_ggml:
            return self._ggml_forward(packed, h)
        return self._fallback_forward(packed, h)

    def _ggml_forward(self, packed: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gate_data = packed[self._gate_off : self._gate_end]
        up_data = packed[self._up_off : self._up_end]
        down_data = packed[self._down_off : self._down_end]

        op = torch.ops.tinyserve_ggml_ops.ggml_mul_mat_vec

        gate_out = op(
            h,
            gate_data,
            self._ggml_types["gate"],
            self._proj_shapes["gate"][0],
            self._proj_shapes["gate"][1],
        )
        up_out = op(
            h,
            up_data,
            self._ggml_types["up"],
            self._proj_shapes["up"][0],
            self._proj_shapes["up"][1],
        )

        hidden = self._act_fn(gate_out) * up_out

        return op(
            hidden,
            down_data,
            self._ggml_types["down"],
            self._proj_shapes["down"][0],
            self._proj_shapes["down"][1],
        )

    def _fallback_forward(self, packed: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        from .gguf_dequant_torch import dequant_tensor

        gate_bytes = packed[self._gate_off : self._gate_end]
        up_bytes = packed[self._up_off : self._up_end]
        down_bytes = packed[self._down_off : self._down_end]

        gate_w = dequant_tensor(gate_bytes.cpu(), self._ggml_types["gate"], self._proj_shapes["gate"])
        up_w = dequant_tensor(up_bytes.cpu(), self._ggml_types["up"], self._proj_shapes["up"])
        down_w = dequant_tensor(down_bytes.cpu(), self._ggml_types["down"], self._proj_shapes["down"])

        gate_w = gate_w.to(device=h.device, dtype=h.dtype)
        up_w = up_w.to(device=h.device, dtype=h.dtype)
        down_w = down_w.to(device=h.device, dtype=h.dtype)

        gate_out = F.linear(h, gate_w)
        up_out = F.linear(h, up_w)
        hidden = self._act_fn(gate_out) * up_out
        return F.linear(hidden, down_w)
