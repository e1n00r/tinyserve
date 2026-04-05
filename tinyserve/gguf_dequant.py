"""GGUF tensor dequantization: Q4_K, Q6_K, Q8_0, F16, F32, plus pure-PyTorch vectorized dequant.

Used by gguf_model_loader for non-expert and fused expert tensor dequantization.
Also absorbs gguf_dequant_torch.py (pure-PyTorch GGUF dequantization fallback — no CUDA compilation required).
"""

from __future__ import annotations

import logging

import torch

from .gguf_reader import GGML_TYPES, GGUFTensorInfo

logger = logging.getLogger(__name__)


def _dequant_tensor(
    reader,
    info: GGUFTensorInfo,
    name: str,
    device: str | torch.device,
) -> torch.Tensor:
    """Dequantize a GGUF tensor to BF16 on the target device.

    Supports F32, F16, Q4_K, Q6_K, Q8_0, and other common GGUF types.
    """
    import numpy as np

    if hasattr(reader, "tensor_names"):
        raw = reader.get_tensor_data(name)
    else:
        raw = reader.get_tensor_data(info)

    ggml_type = info.ggml_type

    if ggml_type == 0:  # F32
        t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(info.shape)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 1:  # F16
        t = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(info.shape)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 12:  # Q4_K
        from .gguf_quant import parse_q4k_blocks

        w_float = parse_q4k_blocks(raw, (info.shape[0], info.shape[1]))
        t = torch.from_numpy(w_float)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 8:  # Q8_0
        # Q8_0: 34 bytes per block of 32 elements
        # Layout: float16 scale (2 bytes) + int8[32] (32 bytes) = 34 bytes
        n_elements = 1
        for d in info.shape:
            n_elements *= d
        n_blocks = n_elements // 32
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 34 : (b + 1) * 34]
            scale = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
            quants = np.frombuffer(block[2:34], dtype=np.int8).astype(np.float32)
            values[b * 32 : (b + 1) * 32] = scale * quants
        t = torch.from_numpy(values.reshape(info.shape))
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 13:  # Q5_K — 176 bytes per 256 elements
        n_elements = 1
        for d in info.shape:
            n_elements *= d
        n_blocks = n_elements // 256
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 176 : (b + 1) * 176]
            # Q5_K layout: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128]
            d_val = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
            dmin = np.frombuffer(block[2:4], dtype=np.float16).astype(np.float32)[0]
            scales_raw = block[4:16]
            qh = np.frombuffer(block[16:48], dtype=np.uint8)
            qs = np.frombuffer(block[48:176], dtype=np.uint8)
            # Decode 6-bit scales and mins (same packing as Q4_K)
            sc = np.zeros(8, dtype=np.float32)
            mn = np.zeros(8, dtype=np.float32)
            for i in range(8):
                s_lo = scales_raw[i] & 0x0F
                m_lo = (scales_raw[i] >> 4) & 0x0F
                byte_idx = 8 + (i // 2)
                shift = (i % 2) * 4
                packed_hi = (scales_raw[byte_idx] >> shift) & 0x0F
                s_hi = packed_hi & 0x03
                m_hi = (packed_hi >> 2) & 0x03
                sc[i] = d_val * (s_lo | (s_hi << 4))
                mn[i] = dmin * (m_lo | (m_hi << 4))
            # Decode 5-bit values: lower 4 bits from qs, 5th bit from qh
            for i in range(8):
                for j in range(32):
                    idx = i * 32 + j
                    q_lo = int(qs[idx // 2] >> (4 * (idx % 2))) & 0x0F
                    q_hi = int(qh[idx // 8] >> (idx % 8)) & 1
                    q = q_lo | (q_hi << 4)
                    values[b * 256 + idx] = sc[i] * q - mn[i]
        t = torch.from_numpy(values.reshape(info.shape))
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 14:  # Q6_K — 210 bytes per 256 elements
        n_elements = 1
        for d in info.shape:
            n_elements *= d
        n_blocks = n_elements // 256
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 210 : (b + 1) * 210]
            # Q6_K layout: ql[128] + qh[64] + scales[16] + d(float16)
            ql = np.frombuffer(block[:128], dtype=np.uint8)
            qh = np.frombuffer(block[128:192], dtype=np.uint8)
            scales = np.frombuffer(block[192:208], dtype=np.int8).astype(np.float32)
            d = np.frombuffer(block[208:210], dtype=np.float16).astype(np.float32)[0]
            # Decode 6-bit values: lower 4 bits from ql, upper 2 bits from qh
            for j in range(256):
                q_lo = int(ql[j // 2] >> (4 * (j % 2))) & 0xF
                q_hi = int(qh[j // 4] >> (2 * (j % 4))) & 0x3
                q = q_lo | (q_hi << 4)
                sc_idx = j // 16
                values[b * 256 + j] = d * scales[sc_idx] * (q - 32)
        t = torch.from_numpy(values.reshape(info.shape))
        return t.to(torch.bfloat16).to(device)

    raise ValueError(
        f"Unsupported GGML type {ggml_type} ({GGML_TYPES.get(ggml_type, ('UNKNOWN',))[0]}) "
        f"for non-expert tensor '{name}'. Only F32, F16, Q4_K, Q6_K, Q8_0 are supported."
    )


def _dequant_fused_tensor(
    reader,
    info: GGUFTensorInfo,
    name: str,
    device: str | torch.device,
) -> torch.Tensor:
    """Dequantize a fused 3-D GGUF expert tensor to BF16.

    Fused expert tensors have shape ``(out_dim, in_dim, n_experts)``.  Q4_K
    quantisation applies to the flat element buffer, so we dequant into a 2-D
    array of ``n_elements`` elements and then reshape.
    """
    import numpy as np

    if hasattr(reader, "tensor_names"):
        raw = reader.get_tensor_data(name)
    else:
        raw = reader.get_tensor_data(info)

    ggml_type = info.ggml_type
    shape_3d = tuple(info.shape)  # (out_dim, in_dim, n_experts)
    n_elements = 1
    for d in shape_3d:
        n_elements *= d

    if ggml_type == 0:  # F32
        t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape_3d)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 1:  # F16
        t = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape_3d)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 12:  # Q4_K — 256-element blocks, 144 bytes each
        from .gguf_quant import parse_q4k_block

        n_blocks = n_elements // 256
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 144 : (b + 1) * 144]
            vals, _, _ = parse_q4k_block(block)
            values[b * 256 : (b + 1) * 256] = vals
        t = torch.from_numpy(values.reshape(shape_3d))
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 8:  # Q8_0 — 34 bytes per block of 32 elements
        n_blocks = n_elements // 32
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 34 : (b + 1) * 34]
            scale = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
            quants = np.frombuffer(block[2:34], dtype=np.int8).astype(np.float32)
            values[b * 32 : (b + 1) * 32] = scale * quants
        t = torch.from_numpy(values.reshape(shape_3d))
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 13:  # Q5_K — 176 bytes per 256 elements
        n_blocks = n_elements // 256
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 176 : (b + 1) * 176]
            d_val = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
            dmin = np.frombuffer(block[2:4], dtype=np.float16).astype(np.float32)[0]
            scales_raw = block[4:16]
            qh = np.frombuffer(block[16:48], dtype=np.uint8)
            qs = np.frombuffer(block[48:176], dtype=np.uint8)
            sc = np.zeros(8, dtype=np.float32)
            mn = np.zeros(8, dtype=np.float32)
            for i in range(8):
                s_lo = scales_raw[i] & 0x0F
                m_lo = (scales_raw[i] >> 4) & 0x0F
                byte_idx = 8 + (i // 2)
                shift = (i % 2) * 4
                packed_hi = (scales_raw[byte_idx] >> shift) & 0x0F
                s_hi = packed_hi & 0x03
                m_hi = (packed_hi >> 2) & 0x03
                sc[i] = d_val * (s_lo | (s_hi << 4))
                mn[i] = dmin * (m_lo | (m_hi << 4))
            for i in range(8):
                for j in range(32):
                    idx = i * 32 + j
                    q_lo = int(qs[idx // 2] >> (4 * (idx % 2))) & 0x0F
                    q_hi = int(qh[idx // 8] >> (idx % 8)) & 1
                    q = q_lo | (q_hi << 4)
                    values[b * 256 + idx] = sc[i] * q - mn[i]
        t = torch.from_numpy(values.reshape(shape_3d))
        return t.to(torch.bfloat16).to(device)

    raise ValueError(
        f"Unsupported GGML type {ggml_type} ({GGML_TYPES.get(ggml_type, ('UNKNOWN',))[0]}) "
        f"for fused expert tensor '{name}'. Only F32, F16, Q4_K, Q5_K, Q8_0 are supported."
    )


# ---------------------------------------------------------------------------
# Pure-PyTorch vectorized dequantization (absorbed from gguf_dequant_torch.py)
# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Ported from https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py
# Pure-PyTorch GGUF dequantization fallback — no CUDA compilation required.
# Used when ggml CUDA kernels are unavailable and for batch>1 prefill.
# ---------------------------------------------------------------------------

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
