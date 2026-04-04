# SPDX-License-Identifier: MIT
"""Tests for the ggml_ops CUDA extension (fused dequant + matvec)."""
from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
import torch

from tests.conftest import requires_cuda


def _load_ggml_ext():
    try:
        from torch.utils.cpp_extension import load

        ext = load(
            name="tinyserve_ggml_ops",
            sources=["tinyserve/csrc/ggml_ops.cu"],
            extra_cuda_cflags=["-O2", "--use_fast_math",
                               "-gencode=arch=compute_90,code=compute_90"],
            verbose=False,
        )
        return ext
    except Exception:
        return None


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = _load_ggml_ext()
    return _ext


def _has_ggml_ops():
    return _get_ext() is not None


needs_ggml = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ggml type IDs
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14


def _ref_matvec(weight_bytes: torch.Tensor, act: torch.Tensor,
                ggml_type: int, N: int, K: int) -> torch.Tensor:
    """Reference matvec via PyTorch dequant fallback."""
    from tinyserve.gguf_dequant_torch import dequant_tensor

    w_f32 = dequant_tensor(weight_bytes.cpu(), ggml_type, (N, K))
    act_f32 = act.float().cpu().view(1, K)
    return (act_f32 @ w_f32.T).squeeze(0)


def _make_q8_0_data(w_f32: torch.Tensor) -> torch.Tensor:
    """Quantize float32 weight matrix to Q8_0 format.

    Args:
        w_f32: [N, K] float32 weight matrix (K must be divisible by 32).

    Returns:
        uint8 tensor of raw Q8_0 bytes.
    """
    N, K = w_f32.shape
    assert K % 32 == 0
    n_blocks = K // 32
    blocks = w_f32.reshape(N * n_blocks, 32)
    amax = blocks.abs().max(dim=1).values
    d = amax / 127.0
    d_safe = d.clone()
    d_safe[d_safe == 0] = 1.0
    qs = torch.round(blocks / d_safe.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    d_f16 = d.to(torch.float16)

    result = bytearray()
    for i in range(N * n_blocks):
        result += d_f16[i].numpy().tobytes()
        result += qs[i].numpy().tobytes()

    return torch.tensor(list(result), dtype=torch.uint8)


# -------------------------------------------------------------------------
# Q8_0 tests
# -------------------------------------------------------------------------

@needs_ggml
def test_q8_0_shape_and_dtype():
    ext = _get_ext()
    if ext is None:
        pytest.skip("ggml extension not built")

    K, N = 64, 32
    w_f32 = torch.randn(N, K)
    weight = _make_q8_0_data(w_f32).cuda()
    act = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")

    out = ext.ggml_mul_mat_vec(act, weight, GGML_Q8_0, N, K)
    assert out.shape == (1, N)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()


@needs_ggml
def test_q8_0_correctness():
    ext = _get_ext()
    if ext is None:
        pytest.skip("ggml extension not built")

    K, N = 256, 64
    w_f32 = torch.randn(N, K) * 0.5
    weight = _make_q8_0_data(w_f32).cuda()
    act = torch.randn(1, K, dtype=torch.float32, device="cuda")

    out = ext.ggml_mul_mat_vec(act, weight, GGML_Q8_0, N, K)
    ref = _ref_matvec(weight.cpu(), act, GGML_Q8_0, N, K)

    torch.testing.assert_close(
        out.float().cpu().squeeze(0), ref, atol=1e-3, rtol=1e-3
    )


# -------------------------------------------------------------------------
# K-quant tests using real quantized data from dequant_torch as reference
# -------------------------------------------------------------------------

def _make_valid_quant_data(ggml_type: int, N: int, K: int) -> torch.Tensor:
    """Create structurally valid quantized weight data with safe FP16 scales.

    Produces bytes that won't create NaN/Inf when interpreted as FP16 scales.
    """
    rng = np.random.default_rng(42 + ggml_type * 100 + K + N)

    if ggml_type == GGML_Q4_K:
        # block_q4_K: d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes
        n_blocks = (N * K) // 256
        result = bytearray()
        for _ in range(n_blocks):
            d = np.float16(rng.uniform(0.001, 0.1))
            dmin = np.float16(rng.uniform(0.001, 0.1))
            scales = rng.integers(0, 64, size=12, dtype=np.uint8).tobytes()
            qs = rng.integers(0, 256, size=128, dtype=np.uint8).tobytes()
            result += d.tobytes() + dmin.tobytes() + scales + qs
        return torch.tensor(list(result), dtype=torch.uint8)

    elif ggml_type == GGML_Q5_K:
        # block_q5_K: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 bytes
        n_blocks = (N * K) // 256
        result = bytearray()
        for _ in range(n_blocks):
            d = np.float16(rng.uniform(0.001, 0.1))
            dmin = np.float16(rng.uniform(0.001, 0.1))
            scales = rng.integers(0, 64, size=12, dtype=np.uint8).tobytes()
            qh = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
            qs = rng.integers(0, 256, size=128, dtype=np.uint8).tobytes()
            result += d.tobytes() + dmin.tobytes() + scales + qh + qs
        return torch.tensor(list(result), dtype=torch.uint8)

    elif ggml_type == GGML_Q6_K:
        # block_q6_K: ql(128) + qh(64) + scales(16) + d(2) = 210 bytes
        n_blocks = (N * K) // 256
        result = bytearray()
        for _ in range(n_blocks):
            ql = rng.integers(0, 256, size=128, dtype=np.uint8).tobytes()
            qh = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()
            scales = rng.integers(-64, 64, size=16, dtype=np.int8).tobytes()
            d = np.float16(rng.uniform(0.001, 0.1))
            result += ql + qh + scales + d.tobytes()
        return torch.tensor(list(result), dtype=torch.uint8)

    else:
        raise ValueError(f"Unsupported type {ggml_type}")


@needs_ggml
@pytest.mark.parametrize("ggml_type,K,N", [
    (GGML_Q4_K, 256, 32),
    (GGML_Q4_K, 512, 64),
    (GGML_Q5_K, 256, 32),
    (GGML_Q5_K, 512, 64),
    (GGML_Q6_K, 256, 32),
    (GGML_Q6_K, 512, 64),
])
def test_kquant_correctness(ggml_type, K, N):
    ext = _get_ext()
    if ext is None:
        pytest.skip("ggml extension not built")

    weight = _make_valid_quant_data(ggml_type, N, K)
    act = torch.randn(1, K, dtype=torch.float32, device="cuda")

    out = ext.ggml_mul_mat_vec(act, weight.cuda(), ggml_type, N, K)
    ref = _ref_matvec(weight, act, ggml_type, N, K)

    assert out.shape == (1, N)
    assert not torch.isnan(out).any()

    torch.testing.assert_close(
        out.float().cpu().squeeze(0), ref, atol=0.5, rtol=0.05
    )


@needs_ggml
def test_unsupported_type_raises():
    ext = _get_ext()
    if ext is None:
        pytest.skip("ggml extension not built")

    act = torch.randn(1, 32, dtype=torch.float32, device="cuda")
    weight = torch.zeros(100, dtype=torch.uint8, device="cuda")

    with pytest.raises(RuntimeError, match="Unsupported ggml_type"):
        ext.ggml_mul_mat_vec(act, weight, 99, 10, 32)


@needs_ggml
def test_bf16_input():
    ext = _get_ext()
    if ext is None:
        pytest.skip("ggml extension not built")

    K, N = 256, 16
    weight = _make_valid_quant_data(GGML_Q4_K, N, K)
    act = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")

    out = ext.ggml_mul_mat_vec(act, weight.cuda(), GGML_Q4_K, N, K)
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, N)
    assert not torch.isnan(out).any()
