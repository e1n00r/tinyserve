"""Drop-in replacement for nn.Linear that stores weights in native GGUF quant format.

Uses ggml MMVQ kernel for batch=1, city96 dequant+F.linear for batch>1.
Saves ~2.5x VRAM vs BF16 for non-expert weights (attention, embeddings, etc.).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HAS_GGML = None


def _check_ggml() -> bool:
    global _HAS_GGML
    if _HAS_GGML is None:
        try:
            torch.ops.tinyserve_ggml_ops.ggml_mul_mat_vec
            _HAS_GGML = True
        except (AttributeError, RuntimeError):
            _HAS_GGML = False
    return _HAS_GGML


class GGMLLinear(nn.Module):
    """Linear layer backed by native GGUF quantized weights.

    Stores raw Q4_K/Q5_K/Q6_K/Q8_0 bytes as a uint8 buffer on GPU.
    Forward uses ggml MMVQ kernel (batch=1) or dequant fallback (batch>1).
    """

    def __init__(
        self,
        raw_bytes: bytes | torch.Tensor,
        ggml_type: int,
        out_features: int,
        in_features: int,
        bias: torch.Tensor | None = None,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        if isinstance(raw_bytes, bytes):
            self._qweight = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8).to(device)
        else:
            self._qweight = raw_bytes.to(device)
        self._ggml_type = ggml_type
        self.out_features = out_features
        self.in_features = in_features
        self.bias = nn.Parameter(bias.to(device)) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0] if x.dim() == 2 else x.shape[:-1].numel()

        if batch == 1 and _check_ggml():
            x_2d = x.reshape(1, -1)
            out = torch.ops.tinyserve_ggml_ops.ggml_mul_mat_vec(
                x_2d, self._qweight, self._ggml_type, self.out_features, self.in_features,
            )
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*x.shape[:-1], self.out_features)

        # Fallback: dequant to BF16, F.linear
        from .gguf_dequant_torch import dequant_tensor
        w_bytes = self._qweight.cpu().numpy().tobytes()
        w = dequant_tensor(w_bytes, self._ggml_type, (self.out_features, self.in_features))
        w = w.to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        from .gguf_reader import GGML_TYPES
        type_name = GGML_TYPES.get(self._ggml_type, ("?",))[0]
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"ggml_type={type_name}, qweight={self._qweight.shape[0]} bytes, "
            f"bias={self.bias is not None}"
        )


def replace_linear_with_ggml(
    model: nn.Module,
    raw_weights: dict[str, tuple[bytes, int, tuple[int, int]]],
    device: torch.device | str = "cuda",
) -> int:
    """Replace nn.Linear modules with GGMLLinear using raw quantized weight data.

    Args:
        model: HF model to modify in-place.
        raw_weights: mapping from HF parameter path (e.g. "model.layers.0.self_attn.q_proj")
            to (raw_bytes, ggml_type, (out_features, in_features)).
        device: target GPU device.

    Returns:
        Number of modules replaced.
    """
    replaced = 0
    for name, (raw_bytes, ggml_type, shape) in raw_weights.items():
        # Navigate to parent module
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_path, attr_name = parts
            parent = model
            for p in parent_path.split("."):
                parent = getattr(parent, p, None)
                if parent is None:
                    break
            if parent is None:
                continue
        else:
            parent = model
            attr_name = parts[0]

        old_module = getattr(parent, attr_name, None)
        if old_module is None or not isinstance(old_module, nn.Linear):
            continue

        bias = old_module.bias
        if bias is not None and bias.device.type == "meta":
            bias = None  # skip meta bias

        new_module = GGMLLinear(
            raw_bytes=raw_bytes,
            ggml_type=ggml_type,
            out_features=shape[0],
            in_features=shape[1],
            bias=bias,
            device=device,
        )
        setattr(parent, attr_name, new_module)
        replaced += 1

    logger.info("Replaced %d nn.Linear modules with GGMLLinear", replaced)
    return replaced
