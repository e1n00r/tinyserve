"""GGUF tensor dequantization: Q4_K, Q6_K, Q8_0, F16, F32.

Extracted from gguf_loader.py. Used by gguf_loader and gguf_weights.
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

    raise ValueError(
        f"Unsupported GGML type {ggml_type} ({GGML_TYPES.get(ggml_type, ('UNKNOWN',))[0]}) "
        f"for fused expert tensor '{name}'. Only F32, F16, Q4_K, Q8_0 are supported."
    )
