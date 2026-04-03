"""Tests for CPU INT4 expert forward pass (MXFP4 -> INT4 packed)."""

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import requires_cuda
from tinyserve.cpu_expert import (
    CPUINT4Forward,
    CPUExpertForward,
    HAS_INT4_CPU,
    mxfp4_to_int4pack,
)
from tinyserve.expert_store import TensorLayout, _pack_tensors
from tinyserve.mxfp4 import dequant_mxfp4_no_transpose

requires_int4 = pytest.mark.skipif(not HAS_INT4_CPU, reason="INT4 CPU ops not available")

# Dimensions must satisfy: out_features % 16 == 0, in_features % 32 == 0 (group_size)
OUT_FEATURES = 64
IN_FEATURES = 128
GROUP_SIZE = 32


def _make_mxfp4_weights(out_features: int, in_features: int, seed: int = 42):
    """Create synthetic MXFP4 blocks and scales.

    blocks: [out_features, in_features//32, 16] uint8 — each byte holds 2 FP4 nibbles
    scales: [out_features, in_features//32] uint8 — E8M0 exponent (biased by 127)
    """
    torch.manual_seed(seed)
    n_groups = in_features // 32
    blocks = torch.randint(0, 256, (out_features, n_groups, 16), dtype=torch.uint8)
    # Scales in E8M0: use values around 127 (=2^0=1.0) for reasonable magnitudes
    scales = torch.randint(120, 134, (out_features, n_groups), dtype=torch.uint8)
    return blocks, scales


def _make_mxfp4_layout(out_features_gu: int, in_features: int, out_features_dn: int):
    """Create a TensorLayout matching MXFP4 fused expert weights."""
    n_groups = in_features // 32
    dn_groups = out_features_gu // 2 // 32  # down_proj input = intermediate = gate_up output / 2
    specs = {
        "gate_up_proj": ((out_features_gu, n_groups, 16), torch.uint8),
        "gate_up_proj_scales": ((out_features_gu, n_groups), torch.uint8),
        "down_proj": ((out_features_dn, dn_groups, 16), torch.uint8),
        "down_proj_scales": ((out_features_dn, dn_groups), torch.uint8),
    }
    return TensorLayout(specs)


def _pack(layout, tensors):
    packed = torch.empty(layout.total_bytes, dtype=torch.uint8)
    _pack_tensors(packed, layout, tensors)
    return packed


class TestINT4PackMMAvailable:
    @requires_int4
    def test_convert_weight_exists(self):
        assert hasattr(torch.ops.aten, "_convert_weight_to_int4pack_for_cpu")

    @requires_int4
    def test_weight_int4pack_mm_exists(self):
        assert hasattr(torch.ops.aten, "_weight_int4pack_mm_for_cpu")

    @requires_int4
    def test_basic_int4_roundtrip(self):
        N, K = 16, 32
        w = torch.randint(0, 16, (N, K), dtype=torch.int32)
        packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(w, 2)
        sz = torch.zeros(K // 32, N, 2, dtype=torch.bfloat16)
        sz[..., 0] = 1.0
        sz[..., 1] = 0.0
        x = torch.randn(1, K, dtype=torch.bfloat16)
        out = torch.ops.aten._weight_int4pack_mm_for_cpu(x, packed, 32, sz)
        assert out.shape == (1, N)
        assert out.dtype == torch.bfloat16


class TestMXFP4ToINT4Conversion:
    @requires_int4
    def test_output_shapes(self):
        blocks, scales = _make_mxfp4_weights(OUT_FEATURES, IN_FEATURES)
        w_packed, sz = mxfp4_to_int4pack(blocks, scales, GROUP_SIZE)
        n_groups = IN_FEATURES // GROUP_SIZE
        assert sz.shape == (n_groups, OUT_FEATURES, 2)
        assert sz.dtype == torch.bfloat16

    @requires_int4
    def test_roundtrip_quality(self):
        """INT4 requantization of MXFP4 weights should closely match dequanted values."""
        blocks, scales = _make_mxfp4_weights(OUT_FEATURES, IN_FEATURES)
        w_packed, sz = mxfp4_to_int4pack(blocks, scales, GROUP_SIZE)

        # Reference: dequant MXFP4 to float
        w_ref = dequant_mxfp4_no_transpose(blocks, scales, dtype=torch.float32)

        # Forward through INT4 and reference
        x = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16)
        out_int4 = torch.ops.aten._weight_int4pack_mm_for_cpu(x, w_packed, GROUP_SIZE, sz)

        # Reference: x @ w_ref.T using bfloat16
        out_ref = (x.float() @ w_ref.T).to(torch.bfloat16)

        # INT4 requantization introduces some error, but should be well-correlated
        cos_sim = F.cosine_similarity(out_int4.float(), out_ref.float(), dim=-1)
        assert cos_sim.item() > 0.90, f"Cosine similarity {cos_sim.item():.4f} too low"

    @requires_int4
    def test_deterministic(self):
        blocks, scales = _make_mxfp4_weights(OUT_FEATURES, IN_FEATURES)
        p1, sz1 = mxfp4_to_int4pack(blocks, scales, GROUP_SIZE)
        p2, sz2 = mxfp4_to_int4pack(blocks, scales, GROUP_SIZE)
        torch.testing.assert_close(sz1, sz2)

    @requires_int4
    def test_scales_and_zeros_contiguous(self):
        blocks, scales = _make_mxfp4_weights(OUT_FEATURES, IN_FEATURES)
        _, sz = mxfp4_to_int4pack(blocks, scales, GROUP_SIZE)
        assert sz.is_contiguous()

    @requires_int4
    def test_symmetric_zero_offset(self):
        """Symmetric quantization should have zero = 0."""
        blocks, scales = _make_mxfp4_weights(OUT_FEATURES, IN_FEATURES)
        _, sz = mxfp4_to_int4pack(blocks, scales, GROUP_SIZE)
        torch.testing.assert_close(
            sz[..., 1], torch.zeros_like(sz[..., 1]),
        )


class TestINT4ForwardMatchesReference:
    @requires_int4
    def test_silu_activation(self):
        """INT4 forward with SiLU should closely match dequant + F.linear reference."""
        torch.manual_seed(42)
        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES, seed=42)
        intermediate = OUT_FEATURES  # gate_up output / 2
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        })

        h = torch.randn(1, IN_FEATURES)
        fwd = CPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu, num_threads=1)
        out_int4 = fwd.forward(h, packed)

        # Reference: dequant both weight matrices, apply SiLU activation
        w_gu = dequant_mxfp4_no_transpose(gu_blocks, gu_scales, dtype=torch.bfloat16)
        w_dn = dequant_mxfp4_no_transpose(dn_blocks, dn_scales, dtype=torch.bfloat16)
        h_bf16 = h.to(torch.bfloat16)
        gate_up = h_bf16 @ w_gu.T
        gate, up = gate_up.chunk(2, dim=-1)
        gated = F.silu(gate) * up
        out_ref = gated @ w_dn.T

        cos_sim = F.cosine_similarity(out_int4.float(), out_ref.float(), dim=-1)
        assert cos_sim.item() > 0.85, f"Cosine similarity {cos_sim.item():.4f} too low"

    @requires_int4
    def test_output_shape(self):
        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        })

        h = torch.randn(1, IN_FEATURES)
        fwd = CPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu, num_threads=1)
        out = fwd.forward(h, packed)
        assert out.shape == (1, IN_FEATURES)
        assert out.dtype == torch.bfloat16


class TestINT4ForwardGPURoundtrip:
    @requires_cuda
    @requires_int4
    def test_gpu_input_returns_gpu(self):
        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        })

        h = torch.randn(1, IN_FEATURES, device="cuda")
        fwd = CPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu, num_threads=1)
        out = fwd.forward(h, packed)
        assert out.device.type == "cuda"
        assert out.shape == (1, IN_FEATURES)


class TestINT4ForwardSwiGLU:
    @requires_int4
    def test_swiglu_runs(self):
        """GPT-OSS SwiGLU activation path should produce valid output."""
        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        })

        h = torch.randn(1, IN_FEATURES)
        fwd = CPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=None, num_threads=1)
        out = fwd.forward(h, packed)
        assert out.shape == (1, IN_FEATURES)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @requires_int4
    def test_swiglu_matches_reference(self):
        """SwiGLU via INT4 should correlate with dequant reference."""
        torch.manual_seed(42)
        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES, seed=42)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        })

        h = torch.randn(1, IN_FEATURES)
        fwd = CPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=None, num_threads=1)
        out_int4 = fwd.forward(h, packed)

        # Reference
        w_gu = dequant_mxfp4_no_transpose(gu_blocks, gu_scales, dtype=torch.bfloat16)
        w_dn = dequant_mxfp4_no_transpose(dn_blocks, dn_scales, dtype=torch.bfloat16)
        h_bf16 = h.to(torch.bfloat16)
        gate_up = h_bf16 @ w_gu.T
        gate = gate_up[..., ::2].clamp(max=7.0)
        up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
        gated = (up + 1) * gate * torch.sigmoid(gate * 1.702)
        out_ref = gated @ w_dn.T

        cos_sim = F.cosine_similarity(out_int4.float(), out_ref.float(), dim=-1)
        assert cos_sim.item() > 0.85, f"Cosine similarity {cos_sim.item():.4f} too low"


class TestCPUExpertForwardMXFP4Integration:
    @requires_int4
    def test_mxfp4_no_longer_raises(self):
        """CPUExpertForward should accept MXFP4 layouts when INT4 ops are available."""
        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        fwd = CPUExpertForward(layout, act_fn=F.silu)
        assert fwd._variant == "mxfp4_int4"

    @requires_int4
    def test_mxfp4_forward_runs(self):
        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        })

        h = torch.randn(1, IN_FEATURES)
        fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=1)
        out = fwd.forward(h, packed)
        assert out.shape == (1, IN_FEATURES)
        assert not torch.isnan(out).any()

    @requires_int4
    def test_int4_cache_reuse(self):
        """Second forward with same buffer should reuse cached INT4 data."""
        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        })

        h = torch.randn(1, IN_FEATURES)
        fwd = CPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu, num_threads=1)
        out1 = fwd.forward(h, packed)
        assert len(fwd._int4_cache) == 1
        out2 = fwd.forward(h, packed)
        assert len(fwd._int4_cache) == 1  # reused, not re-converted
        torch.testing.assert_close(out1, out2)

    @requires_int4
    def test_thread_count_restored(self):
        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        })

        original = torch.get_num_threads()
        fwd = CPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu, num_threads=2)
        h = torch.randn(1, IN_FEATURES)
        fwd.forward(h, packed)
        assert torch.get_num_threads() == original
