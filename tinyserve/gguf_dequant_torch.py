# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Ported from https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py
# Pure-PyTorch GGUF dequantization fallback — no CUDA compilation required.
# Used when ggml CUDA kernels are unavailable and for batch>1 prefill.
from __future__ import annotations

import torch

# GGML quantization type IDs for the supported types
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14

# (block_elements, type_size_bytes) for each supported type
_QUANT_SIZES: dict[int, tuple[int, int]] = {
    GGML_Q4_0: (32, 18),
    GGML_Q4_1: (32, 20),
    GGML_Q5_0: (32, 22),
    GGML_Q5_1: (32, 24),
    GGML_Q8_0: (32, 34),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
}

QK_K = 256
K_SCALE_SIZE = 12


def dequant_tensor(
    data: bytes | torch.Tensor,
    ggml_type: int,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Dequantize GGUF quantized data to a float32 tensor.

    Args:
        data: Raw quantized bytes or uint8 tensor.
        ggml_type: GGML quantization type ID (e.g. 12 for Q4_K).
        shape: Output tensor shape. Product must equal n_blocks * block_elements.

    Returns:
        Float32 tensor of the given shape.

    Raises:
        ValueError: If ggml_type is not supported.
    """
    # F32 and F16: trivial, just reinterpret bytes
    if ggml_type == 0:  # F32
        raw = data if isinstance(data, (bytes, bytearray)) else data.numpy().tobytes()
        return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
    if ggml_type == 1:  # F16
        raw = data if isinstance(data, (bytes, bytearray)) else data.numpy().tobytes()
        return torch.frombuffer(bytearray(raw), dtype=torch.float16).float().reshape(shape)

    if ggml_type not in _QUANT_SIZES:
        raise ValueError(
            f"Unsupported GGML quantization type {ggml_type}. "
            f"Supported: [0, 1] + {sorted(_QUANT_SIZES)}"
        )

    if isinstance(data, (bytes, bytearray)):
        raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    else:
        raw = data.view(torch.uint8).reshape(-1)

    block_size, type_size = _QUANT_SIZES[ggml_type]
    n_blocks = raw.numel() // type_size
    blocks = raw.reshape(n_blocks, type_size)

    dequant_fn = _DEQUANT_FUNCTIONS[ggml_type]
    out = dequant_fn(blocks, block_size, type_size)
    return out.reshape(shape).to(torch.float32)


# ---------------------------------------------------------------------------
# Internal helpers (ported verbatim from city96/ComfyUI-GGUF dequant.py)
# ---------------------------------------------------------------------------

def _to_uint32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _to_uint16(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


def _split_block_dims(blocks: torch.Tensor, *args: int) -> list[torch.Tensor]:
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode 6-bit packed sub-block scales and minimums for K-Quants."""
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8).reshape(n_blocks, 3, 4)

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return sc.reshape(n_blocks, 8), mn.reshape(n_blocks, 8)


# ---------------------------------------------------------------------------
# Legacy quant dequantizers
# ---------------------------------------------------------------------------

def _dequant_q8_0(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    d, x = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(torch.float32)
    x = x.view(torch.int8)
    return d * x


def _dequant_q4_0(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(torch.float32)
    qs = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1).to(torch.int8) - 8
    return d * qs


def _dequant_q4_1(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(torch.float32)
    m = m.view(torch.float16).to(torch.float32)
    qs = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return d * qs + m


def _dequant_q5_0(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(torch.float32)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def _dequant_q5_1(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(torch.float32)
    m = m.view(torch.float16).to(torch.float32)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = ql | (qh << 4)
    return d * qs + m


# ---------------------------------------------------------------------------
# K-Quant dequantizers
# ---------------------------------------------------------------------------

def _dequant_q4_k(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(torch.float32)
    dmin = dmin.view(torch.float16).to(torch.float32)

    sc, m = _get_scale_min(scales)

    d = (d * sc).reshape(n_blocks, -1, 1)
    dm = (dmin * m).reshape(n_blocks, -1, 1)

    qs = qs.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1, 32)

    return (d * qs - dm).reshape(n_blocks, QK_K)


def _dequant_q5_k(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(torch.float32)
    dmin = dmin.view(torch.float16).to(torch.float32)

    sc, m = _get_scale_min(scales)

    d = (d * sc).reshape(n_blocks, -1, 1)
    dm = (dmin * m).reshape(n_blocks, -1, 1)

    ql = qs.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = qh.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        list(range(8)), device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 8, 1)
    ql = (ql & 0x0F).reshape(n_blocks, -1, 32)
    qh = (qh & 0x01).reshape(n_blocks, -1, 32)
    q = ql | (qh << 4)

    return (d * q - dm).reshape(n_blocks, QK_K)


def _dequant_q6_k(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(torch.float32)
    d = d.view(torch.float16).to(torch.float32)
    d = (d * scales).reshape(n_blocks, QK_K // 16, 1)

    ql = ql.reshape(n_blocks, -1, 1, 64) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    ql = (ql & 0x0F).reshape(n_blocks, -1, 32)
    qh = qh.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 4, 1)
    qh = (qh & 0x03).reshape(n_blocks, -1, 32)
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape(n_blocks, QK_K // 16, -1)

    return (d * q).reshape(n_blocks, QK_K)


_DEQUANT_FUNCTIONS = {
    GGML_Q4_0: _dequant_q4_0,
    GGML_Q4_1: _dequant_q4_1,
    GGML_Q5_0: _dequant_q5_0,
    GGML_Q5_1: _dequant_q5_1,
    GGML_Q8_0: _dequant_q8_0,
    GGML_Q4_K: _dequant_q4_k,
    GGML_Q5_K: _dequant_q5_k,
    GGML_Q6_K: _dequant_q6_k,
}
