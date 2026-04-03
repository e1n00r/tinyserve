"""Tests for CPUExpertForward — CPU-side expert computation via F.linear."""

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import requires_cuda
from tinyserve.cpu_expert import CPUExpertForward
from tinyserve.expert_store import TensorLayout, _pack_tensors


HIDDEN = 64
INTERMEDIATE = 128


def _make_fused_layout(transposed: bool = False):
    if transposed:
        gu_shape = (HIDDEN, 2 * INTERMEDIATE)
        dn_shape = (INTERMEDIATE, HIDDEN)
    else:
        gu_shape = (2 * INTERMEDIATE, HIDDEN)
        dn_shape = (HIDDEN, INTERMEDIATE)
    specs = {
        "gate_up_proj": (gu_shape, torch.float32),
        "down_proj": (dn_shape, torch.float32),
    }
    return TensorLayout(specs)


def _make_separate_layout(transposed: bool = False):
    if transposed:
        gate_shape = (HIDDEN, INTERMEDIATE)
        up_shape = (HIDDEN, INTERMEDIATE)
        dn_shape = (INTERMEDIATE, HIDDEN)
    else:
        gate_shape = (INTERMEDIATE, HIDDEN)
        up_shape = (INTERMEDIATE, HIDDEN)
        dn_shape = (HIDDEN, INTERMEDIATE)
    specs = {
        "gate_proj": (gate_shape, torch.float32),
        "up_proj": (up_shape, torch.float32),
        "down_proj": (dn_shape, torch.float32),
    }
    return TensorLayout(specs)


def _pack(layout, tensors):
    packed = torch.empty(layout.total_bytes, dtype=torch.uint8)
    _pack_tensors(packed, layout, tensors)
    return packed


def _ref_fused_silu(h, w_gu, w_dn):
    gate_up = F.linear(h, w_gu)
    gate, up = gate_up.chunk(2, dim=-1)
    gated = F.silu(gate) * up
    return F.linear(gated, w_dn)


def _ref_fused_swiglu(h, w_gu, w_dn):
    gate_up = F.linear(h, w_gu)
    gate = gate_up[..., ::2].clamp(max=7.0)
    up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
    gated = (up + 1) * gate * torch.sigmoid(gate * 1.702)
    return F.linear(gated, w_dn)


def _ref_separate_silu(h, w_gate, w_up, w_dn):
    gated = F.silu(F.linear(h, w_gate)) * F.linear(h, w_up)
    return F.linear(gated, w_dn)


class TestInit:
    def test_fused_layout(self):
        layout = _make_fused_layout()
        fwd = CPUExpertForward(layout, act_fn=F.silu)
        assert fwd._variant == "fused"

    def test_separate_layout(self):
        layout = _make_separate_layout()
        fwd = CPUExpertForward(layout, act_fn=F.silu)
        assert fwd._variant == "separate"

    def test_mxfp4_accepted_when_int4_available(self):
        from tinyserve.cpu_expert import HAS_INT4_CPU

        specs = {
            "gate_up_proj": ((256, 4, 16), torch.uint8),
            "gate_up_proj_scales": ((256, 4), torch.uint8),
            "down_proj": ((128, 4, 16), torch.uint8),
            "down_proj_scales": ((128, 4), torch.uint8),
        }
        layout = TensorLayout(specs)
        if HAS_INT4_CPU:
            fwd = CPUExpertForward(layout)
            assert fwd._variant == "mxfp4_int4"
        else:
            with pytest.raises(ValueError, match="INT4 CPU ops"):
                CPUExpertForward(layout)

    def test_unknown_layout_raises(self):
        specs = {
            "some_weight": ((64, 64), torch.float32),
        }
        layout = TensorLayout(specs)
        with pytest.raises(ValueError, match="Unknown expert layout"):
            CPUExpertForward(layout)


class TestFusedSiLU:
    def test_matches_reference(self):
        torch.manual_seed(42)
        layout = _make_fused_layout(transposed=False)
        w_gu = torch.randn(2 * INTERMEDIATE, HIDDEN)
        w_dn = torch.randn(HIDDEN, INTERMEDIATE)
        packed = _pack(layout, {"gate_up_proj": w_gu, "down_proj": w_dn})

        h = torch.randn(1, HIDDEN)
        fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=1)
        result = fwd.forward(h, packed)
        expected = _ref_fused_silu(h, w_gu, w_dn)
        torch.testing.assert_close(result, expected)


class TestFusedSwiGLU:
    def test_matches_reference(self):
        torch.manual_seed(42)
        layout = _make_fused_layout(transposed=False)
        w_gu = torch.randn(2 * INTERMEDIATE, HIDDEN)
        w_dn = torch.randn(HIDDEN, INTERMEDIATE)
        packed = _pack(layout, {"gate_up_proj": w_gu, "down_proj": w_dn})

        h = torch.randn(1, HIDDEN)
        fwd = CPUExpertForward(layout, act_fn=None, num_threads=1)
        result = fwd.forward(h, packed)
        expected = _ref_fused_swiglu(h, w_gu, w_dn)
        torch.testing.assert_close(result, expected)


class TestTranspose:
    def test_transposed_fused_matches_reference(self):
        torch.manual_seed(42)
        w_gu_standard = torch.randn(2 * INTERMEDIATE, HIDDEN)
        w_dn_standard = torch.randn(HIDDEN, INTERMEDIATE)

        layout_t = _make_fused_layout(transposed=True)
        packed = _pack(layout_t, {"gate_up_proj": w_gu_standard.t(), "down_proj": w_dn_standard.t()})

        h = torch.randn(1, HIDDEN)
        fwd = CPUExpertForward(layout_t, act_fn=F.silu, num_threads=1)
        result = fwd.forward(h, packed)
        expected = _ref_fused_silu(h, w_gu_standard, w_dn_standard)
        torch.testing.assert_close(result, expected, atol=2e-4, rtol=1e-5)

    def test_transposed_separate_matches_reference(self):
        torch.manual_seed(42)
        w_gate = torch.randn(INTERMEDIATE, HIDDEN)
        w_up = torch.randn(INTERMEDIATE, HIDDEN)
        w_dn = torch.randn(HIDDEN, INTERMEDIATE)

        layout_t = _make_separate_layout(transposed=True)
        packed = _pack(layout_t, {
            "gate_proj": w_gate.t(),
            "up_proj": w_up.t(),
            "down_proj": w_dn.t(),
        })

        h = torch.randn(1, HIDDEN)
        fwd = CPUExpertForward(layout_t, act_fn=F.silu, num_threads=1)
        result = fwd.forward(h, packed)
        expected = _ref_separate_silu(h, w_gate, w_up, w_dn)
        torch.testing.assert_close(result, expected, atol=2e-4, rtol=1e-5)


class TestSeparate:
    def test_matches_reference(self):
        torch.manual_seed(42)
        layout = _make_separate_layout(transposed=False)
        w_gate = torch.randn(INTERMEDIATE, HIDDEN)
        w_up = torch.randn(INTERMEDIATE, HIDDEN)
        w_dn = torch.randn(HIDDEN, INTERMEDIATE)
        packed = _pack(layout, {"gate_proj": w_gate, "up_proj": w_up, "down_proj": w_dn})

        h = torch.randn(1, HIDDEN)
        fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=1)
        result = fwd.forward(h, packed)
        expected = _ref_separate_silu(h, w_gate, w_up, w_dn)
        torch.testing.assert_close(result, expected)

    def test_default_act_fn_is_silu(self):
        torch.manual_seed(42)
        layout = _make_separate_layout(transposed=False)
        w_gate = torch.randn(INTERMEDIATE, HIDDEN)
        w_up = torch.randn(INTERMEDIATE, HIDDEN)
        w_dn = torch.randn(HIDDEN, INTERMEDIATE)
        packed = _pack(layout, {"gate_proj": w_gate, "up_proj": w_up, "down_proj": w_dn})

        h = torch.randn(1, HIDDEN)
        fwd_explicit = CPUExpertForward(layout, act_fn=F.silu, num_threads=1)
        fwd_default = CPUExpertForward(layout, act_fn=None, num_threads=1)
        result_explicit = fwd_explicit.forward(h, packed)
        result_default = fwd_default.forward(h, packed)
        torch.testing.assert_close(result_explicit, result_default)


class TestDeviceHandling:
    @requires_cuda
    def test_gpu_input_returns_gpu(self):
        torch.manual_seed(42)
        layout = _make_fused_layout(transposed=False)
        w_gu = torch.randn(2 * INTERMEDIATE, HIDDEN)
        w_dn = torch.randn(HIDDEN, INTERMEDIATE)
        packed = _pack(layout, {"gate_up_proj": w_gu, "down_proj": w_dn})

        h = torch.randn(1, HIDDEN, device="cuda")
        fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=1)
        result = fwd.forward(h, packed)
        assert result.device.type == "cuda"

    def test_cpu_input_returns_cpu(self):
        torch.manual_seed(42)
        layout = _make_fused_layout(transposed=False)
        w_gu = torch.randn(2 * INTERMEDIATE, HIDDEN)
        w_dn = torch.randn(HIDDEN, INTERMEDIATE)
        packed = _pack(layout, {"gate_up_proj": w_gu, "down_proj": w_dn})

        h = torch.randn(1, HIDDEN)
        fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=1)
        result = fwd.forward(h, packed)
        assert result.device.type == "cpu"


class TestThreadRestore:
    def test_thread_count_restored(self):
        layout = _make_fused_layout(transposed=False)
        w_gu = torch.randn(2 * INTERMEDIATE, HIDDEN)
        w_dn = torch.randn(HIDDEN, INTERMEDIATE)
        packed = _pack(layout, {"gate_up_proj": w_gu, "down_proj": w_dn})

        original = torch.get_num_threads()
        fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=2)
        h = torch.randn(1, HIDDEN)
        fwd.forward(h, packed)
        assert torch.get_num_threads() == original

    def test_thread_count_restored_on_error(self):
        layout = _make_fused_layout(transposed=False)
        packed = torch.empty(0, dtype=torch.uint8)

        original = torch.get_num_threads()
        fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=2)
        h = torch.randn(1, HIDDEN)
        with pytest.raises(RuntimeError):
            fwd.forward(h, packed)
        assert torch.get_num_threads() == original
