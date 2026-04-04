# SPDX-License-Identifier: MIT
"""Tests for GGMLExpertForward — native quant expert forward."""
from __future__ import annotations

import torch
import pytest

from tests.conftest import requires_cuda
from tinyserve.expert_store import TensorLayout

# Q8_0: 34 bytes per 32 elements
GGML_Q8_0 = 8


def _make_q8_0_bytes(w_f32: torch.Tensor) -> bytes:
    """Quantize [N, K] float32 to Q8_0 bytes (CPU, no CUDA required)."""
    N, K = w_f32.shape
    assert K % 32 == 0
    n_blocks = K // 32
    blocks = w_f32.reshape(N * n_blocks, 32)
    amax = blocks.abs().amax(dim=1)
    d = (amax / 127.0).clamp(min=1e-10)
    qs = torch.round(blocks / d.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    d_f16 = d.to(torch.float16)
    result = bytearray()
    for i in range(N * n_blocks):
        result += d_f16[i].numpy().tobytes()
        result += qs[i].numpy().tobytes()
    return bytes(result)


def _build_packed_q8_0(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, TensorLayout, dict[str, int], dict[str, tuple[int, int]]]:
    """Return (packed_uint8, layout, ggml_types, proj_shapes) for 3 Q8_0 projections."""
    gate_bytes = _make_q8_0_bytes(gate_w)
    up_bytes = _make_q8_0_bytes(up_w)
    down_bytes = _make_q8_0_bytes(down_w)

    gate_n, gate_k = gate_w.shape
    up_n, up_k = up_w.shape
    down_n, down_k = down_w.shape

    specs = {
        "gate": ((len(gate_bytes),), torch.uint8),
        "up": ((len(up_bytes),), torch.uint8),
        "down": ((len(down_bytes),), torch.uint8),
    }
    layout = TensorLayout(specs)

    total = len(gate_bytes) + len(up_bytes) + len(down_bytes)
    packed = torch.zeros(total, dtype=torch.uint8, device=device)
    packed[: len(gate_bytes)].copy_(
        torch.frombuffer(bytearray(gate_bytes), dtype=torch.uint8)
    )
    packed[len(gate_bytes) : len(gate_bytes) + len(up_bytes)].copy_(
        torch.frombuffer(bytearray(up_bytes), dtype=torch.uint8)
    )
    packed[len(gate_bytes) + len(up_bytes) :].copy_(
        torch.frombuffer(bytearray(down_bytes), dtype=torch.uint8)
    )

    ggml_types = {"gate": GGML_Q8_0, "up": GGML_Q8_0, "down": GGML_Q8_0}
    proj_shapes = {
        "gate": (gate_n, gate_k),
        "up": (up_n, up_k),
        "down": (down_n, down_k),
    }
    return packed, layout, ggml_types, proj_shapes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_ggml_forward_batch_1_produces_valid_output():
    """GGMLExpertForward batch=1: 3 projections, SiLU, non-NaN output."""
    from tinyserve.ggml_forward import GGMLExpertForward

    hidden = 64
    inter = 128
    device = torch.device("cuda")

    gate_w = torch.randn(inter, hidden) * 0.02
    up_w = torch.randn(inter, hidden) * 0.02
    down_w = torch.randn(hidden, inter) * 0.02

    packed, layout, ggml_types, proj_shapes = _build_packed_q8_0(
        gate_w, up_w, down_w, device
    )

    act_fn = torch.nn.SiLU()
    fwd = GGMLExpertForward(layout, ggml_types, act_fn, proj_shapes)

    h = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)
    out = fwd.forward(packed, h)

    assert out.shape == (1, hidden), f"Expected (1, {hidden}), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


def test_ggml_forward_fallback_produces_valid_output():
    """GGMLExpertForward fallback: batch=4 uses dequant+F.linear, CPU-only."""
    from tinyserve.ggml_forward import GGMLExpertForward

    hidden = 64
    inter = 128
    device = torch.device("cpu")

    gate_w = torch.randn(inter, hidden) * 0.02
    up_w = torch.randn(inter, hidden) * 0.02
    down_w = torch.randn(hidden, inter) * 0.02

    packed, layout, ggml_types, proj_shapes = _build_packed_q8_0(
        gate_w, up_w, down_w, device
    )

    act_fn = torch.nn.SiLU()
    fwd = GGMLExpertForward(layout, ggml_types, act_fn, proj_shapes)

    h = torch.randn(4, hidden, dtype=torch.bfloat16, device=device)
    out = fwd.forward(packed, h)

    assert out.shape == (4, hidden), f"Expected (4, {hidden}), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


@requires_cuda
def test_ggml_forward_matches_fallback():
    """ggml kernel output ≈ dequant+F.linear output within quant noise (Q8_0)."""
    from tinyserve.ggml_forward import GGMLExpertForward

    hidden = 64
    inter = 128
    device = torch.device("cuda")

    torch.manual_seed(0)
    gate_w = torch.randn(inter, hidden) * 0.1
    up_w = torch.randn(inter, hidden) * 0.1
    down_w = torch.randn(hidden, inter) * 0.1

    packed, layout, ggml_types, proj_shapes = _build_packed_q8_0(
        gate_w, up_w, down_w, device
    )

    act_fn = torch.nn.SiLU()
    fwd = GGMLExpertForward(layout, ggml_types, act_fn, proj_shapes)

    h = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)

    # Force fallback by patching _has_ggml
    original = fwd._has_ggml
    fwd._has_ggml = False
    fallback_out = fwd.forward(packed, h)
    fwd._has_ggml = original

    if fwd._has_ggml:
        ggml_out = fwd._ggml_forward(packed, h)
        torch.testing.assert_close(
            ggml_out.float(), fallback_out.float(), atol=0.05, rtol=0.05
        )
    else:
        # No ggml extension — only verify fallback non-NaN
        assert not torch.isnan(fallback_out).any()


def test_ggml_forward_batch_gt1_uses_fallback_path():
    """batch>1 always routes through fallback regardless of ggml availability."""
    from tinyserve.ggml_forward import GGMLExpertForward

    hidden = 64
    inter = 128
    device = torch.device("cpu")

    gate_w = torch.randn(inter, hidden) * 0.02
    up_w = torch.randn(inter, hidden) * 0.02
    down_w = torch.randn(hidden, inter) * 0.02

    packed, layout, ggml_types, proj_shapes = _build_packed_q8_0(
        gate_w, up_w, down_w, device
    )

    act_fn = torch.nn.SiLU()
    fwd = GGMLExpertForward(layout, ggml_types, act_fn, proj_shapes)
    fwd._has_ggml = True  # pretend ggml is available — should still use fallback

    h = torch.randn(3, hidden, dtype=torch.bfloat16, device=device)
    out = fwd.forward(packed, h)

    assert out.shape == (3, hidden)
    assert not torch.isnan(out).any()


def test_ggml_forward_offsets_baked_in_init():
    """Offsets are computed once in __init__, not per forward call."""
    from tinyserve.ggml_forward import GGMLExpertForward

    hidden = 32
    inter = 64

    gate_w = torch.randn(inter, hidden) * 0.02
    up_w = torch.randn(inter, hidden) * 0.02
    down_w = torch.randn(hidden, inter) * 0.02

    gate_bytes = _make_q8_0_bytes(gate_w)
    up_bytes = _make_q8_0_bytes(up_w)
    down_bytes = _make_q8_0_bytes(down_w)

    specs = {
        "gate": ((len(gate_bytes),), torch.uint8),
        "up": ((len(up_bytes),), torch.uint8),
        "down": ((len(down_bytes),), torch.uint8),
    }
    layout = TensorLayout(specs)
    ggml_types = {"gate": GGML_Q8_0, "up": GGML_Q8_0, "down": GGML_Q8_0}
    proj_shapes = {
        "gate": (inter, hidden),
        "up": (inter, hidden),
        "down": (hidden, inter),
    }

    fwd = GGMLExpertForward(layout, ggml_types, torch.nn.SiLU(), proj_shapes)

    assert fwd._gate_off == 0
    assert fwd._gate_end == len(gate_bytes)
    assert fwd._up_off == len(gate_bytes)
    assert fwd._up_end == len(gate_bytes) + len(up_bytes)
    assert fwd._down_off == len(gate_bytes) + len(up_bytes)
    assert fwd._down_end == len(gate_bytes) + len(up_bytes) + len(down_bytes)
