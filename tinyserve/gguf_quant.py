"""Q4_K block parser and INT4 converter for GGUF quantized weights.

Parses llama.cpp Q4_K blocks (144 bytes -> 256 float32 values) and converts
expert weight matrices from Q4_K to torch INT4 packed format for CPU inference
via _weight_int4pack_mm_for_cpu.
"""

from __future__ import annotations

import struct

import numpy as np
import torch


def parse_q4k_block(block_bytes: bytes) -> tuple[np.ndarray, float, float]:
    """Parse one Q4_K block (144 bytes -> 256 float32 values).

    Q4_K layout:
      d:         float16 at offset 0    (super-block scale)
      dmin:      float16 at offset 2    (super-block minimum)
      scales[12]: bytes 4:16            (6-bit packed sub-block scales/mins)
      qs[128]:   bytes 16:144           (4-bit packed values, 2 per byte)

    Returns (values_f32[256], d, dmin).
    """
    d = struct.unpack_from("<e", block_bytes, 0)[0]
    dmin = struct.unpack_from("<e", block_bytes, 2)[0]
    scales_raw = block_bytes[4:16]
    qs = block_bytes[16:144]

    # Decode 6-bit sub-block scales and mins
    sc = np.zeros(8, dtype=np.float32)
    mn = np.zeros(8, dtype=np.float32)

    for i in range(8):
        if i < 4:
            s_lo = scales_raw[i] & 0x0F
            m_lo = (scales_raw[i] >> 4) & 0x0F
        else:
            s_lo = scales_raw[i] & 0x0F
            m_lo = (scales_raw[i] >> 4) & 0x0F

        byte_idx = 8 + (i // 2)
        shift = (i % 2) * 4
        packed_hi = (scales_raw[byte_idx] >> shift) & 0x0F
        s_hi = packed_hi & 0x03
        m_hi = (packed_hi >> 2) & 0x03

        raw_s = s_lo | (s_hi << 4)
        raw_m = m_lo | (m_hi << 4)

        sc[i] = d * raw_s
        mn[i] = dmin * raw_m

    # Dequantize 4-bit values
    values = np.empty(256, dtype=np.float32)
    for i in range(8):
        for j in range(32):
            idx = i * 32 + j
            byte_idx = idx // 2
            if idx % 2 == 0:
                q = qs[byte_idx] & 0x0F
            else:
                q = (qs[byte_idx] >> 4) & 0x0F
            values[idx] = sc[i] * q - mn[i]

    return values, float(d), float(dmin)


def parse_q4k_blocks(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Parse all Q4_K blocks for a weight matrix, returning float32 array.

    Args:
        data: raw Q4_K bytes
        shape: (out_features, in_features) weight matrix shape

    Returns:
        float32 array of shape ``shape``
    """
    n_elements = shape[0] * shape[1]
    n_blocks = n_elements // 256
    values = np.empty(n_elements, dtype=np.float32)

    for b in range(n_blocks):
        block = data[b * 144 : (b + 1) * 144]
        vals, _, _ = parse_q4k_block(block)
        values[b * 256 : (b + 1) * 256] = vals

    return values.reshape(shape)


def q4k_expert_to_int4pack(
    gate_data: bytes,
    up_data: bytes,
    down_data: bytes,
    gate_shape: tuple[int, int],
    up_shape: tuple[int, int],
    down_shape: tuple[int, int],
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Q4_K expert weights to torch INT4 packed format.

    One-time format conversion: Q4_K (4-bit asymmetric) -> torch INT4 (4-bit symmetric).
    The float32 intermediate exists only during this conversion.

    Returns: (gate_packed, gate_sz, up_packed, up_sz, down_packed, down_sz)
    """
    results = []
    for data, shape in [(gate_data, gate_shape), (up_data, up_shape), (down_data, down_shape)]:
        w_float = parse_q4k_blocks(data, shape)
        w_t = torch.from_numpy(w_float)
        packed, sz = _float_to_int4pack(w_t, group_size)
        results.append(packed)
        results.append(sz)

    return tuple(results)


def _float_to_int4pack(
    w_float: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a float32 weight matrix to torch INT4 packed format.

    Args:
        w_float: [out_features, in_features] float32 weight matrix
        group_size: group size for symmetric quantization

    Returns:
        (w_int4packed, scales_and_zeros)
    """
    out_features, in_features = w_float.shape
    n_groups = in_features // group_size
    w_grouped = w_float.view(out_features, n_groups, group_size)

    amax = w_grouped.abs().amax(dim=-1)
    scale = (amax / 7.0).clamp(min=1e-10)

    w_q = torch.round(w_grouped / scale.unsqueeze(-1)) + 8
    w_q = w_q.clamp(0, 15).to(torch.int32)
    w_q = w_q.view(out_features, in_features)

    w_int4packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(w_q, 2)

    scales_and_zeros = torch.zeros(n_groups, out_features, 2, dtype=torch.bfloat16)
    scales_and_zeros[:, :, 0] = scale.T.to(torch.bfloat16)
    scales_and_zeros[:, :, 1] = 0.0

    return w_int4packed, scales_and_zeros.contiguous()
