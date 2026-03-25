"""Tests for GPU INT4 expert forward pass using torch._weight_int4pack_mm."""

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import requires_cuda
from tinyserve.cpu_expert import HAS_INT4_CPU, mxfp4_to_int4pack
from tinyserve.generic_store import TensorLayout, _pack_tensors
from tinyserve.mxfp4 import dequant_mxfp4_no_transpose

HAS_INT4_GPU = (
    hasattr(torch, "_weight_int4pack_mm")
    and hasattr(torch, "_convert_weight_to_int4pack")
    and torch.cuda.is_available()
)

requires_gpu_int4 = pytest.mark.skipif(
    not HAS_INT4_GPU, reason="GPU INT4 ops or CUDA not available"
)

OUT_FEATURES = 64
IN_FEATURES = 128
GROUP_SIZE = 32


def _make_mxfp4_weights(out_features: int, in_features: int, seed: int = 42):
    torch.manual_seed(seed)
    n_groups = in_features // 32
    blocks = torch.randint(0, 256, (out_features, n_groups, 16), dtype=torch.uint8)
    scales = torch.randint(120, 134, (out_features, n_groups), dtype=torch.uint8)
    return blocks, scales


def _make_mxfp4_layout(out_features_gu: int, in_features: int, out_features_dn: int):
    n_groups = in_features // 32
    dn_groups = out_features_gu // 2 // 32
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


class TestGPUINT4Available:
    @requires_gpu_int4
    def test_convert_weight_exists(self):
        assert hasattr(torch, "_convert_weight_to_int4pack")

    @requires_gpu_int4
    def test_weight_int4pack_mm_exists(self):
        assert hasattr(torch, "_weight_int4pack_mm")

    @requires_gpu_int4
    def test_basic_gpu_int4_roundtrip(self):
        N, K = 32, 64
        torch.manual_seed(42)
        w_q = torch.randint(0, 16, (N, K), dtype=torch.int32)
        w_uint8 = (w_q[:, 1::2] | (w_q[:, 0::2] << 4)).to(torch.uint8).cuda()
        packed = torch._convert_weight_to_int4pack(w_uint8, 2)

        sz = torch.zeros(K // 32, N, 2, dtype=torch.bfloat16, device="cuda")
        sz[..., 0] = 1.0
        x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
        out = torch._weight_int4pack_mm(x, packed, 32, sz)
        assert out.shape == (1, N)
        assert out.dtype == torch.bfloat16


class TestGPUINT4Forward:
    @requires_gpu_int4
    def test_forward_output_shape(self):
        from tinyserve.gpu_int4 import GPUINT4Forward

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

        h = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16, device="cuda")
        fwd = GPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu)
        out = fwd.forward(h, packed.cuda())
        assert out.shape == (1, IN_FEATURES)
        assert out.device.type == "cuda"
        assert out.dtype == torch.bfloat16

    @requires_gpu_int4
    def test_forward_no_nan_inf(self):
        from tinyserve.gpu_int4 import GPUINT4Forward

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

        h = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16, device="cuda")
        fwd = GPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu)
        out = fwd.forward(h, packed.cuda())
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @requires_gpu_int4
    def test_forward_swiglu(self):
        from tinyserve.gpu_int4 import GPUINT4Forward

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

        h = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16, device="cuda")
        fwd = GPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=None)
        out = fwd.forward(h, packed.cuda())
        assert out.shape == (1, IN_FEATURES)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestGPUINT4MatchesCPU:
    @requires_gpu_int4
    @pytest.mark.skipif(not HAS_INT4_CPU, reason="CPU INT4 ops not available")
    def test_gpu_matches_cpu_silu(self):
        """GPU INT4 forward should produce outputs close to CPU INT4 forward."""
        from tinyserve.cpu_expert import CPUINT4Forward
        from tinyserve.gpu_int4 import GPUINT4Forward

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

        h = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16)

        cpu_fwd = CPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu, num_threads=1)
        out_cpu = cpu_fwd.forward(h, packed)

        gpu_fwd = GPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu)
        out_gpu = gpu_fwd.forward(h.cuda(), packed.cuda())

        cos_sim = F.cosine_similarity(out_cpu.float(), out_gpu.cpu().float(), dim=-1)
        assert cos_sim.item() > 0.85, f"GPU vs CPU cosine similarity {cos_sim.item():.4f} too low"

    @requires_gpu_int4
    def test_gpu_matches_dequant_reference(self):
        """GPU INT4 forward should correlate with dequant + F.linear reference."""
        from tinyserve.gpu_int4 import GPUINT4Forward

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

        h = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16)

        gpu_fwd = GPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu)
        out_gpu = gpu_fwd.forward(h.cuda(), packed.cuda())

        # Reference: dequant MXFP4 -> float, standard matmul
        w_gu = dequant_mxfp4_no_transpose(gu_blocks, gu_scales, dtype=torch.bfloat16)
        w_dn = dequant_mxfp4_no_transpose(dn_blocks, dn_scales, dtype=torch.bfloat16)
        h_bf16 = h.to(torch.bfloat16)
        gate_up = h_bf16 @ w_gu.T
        gate, up = gate_up.chunk(2, dim=-1)
        gated = F.silu(gate) * up
        out_ref = gated @ w_dn.T

        cos_sim = F.cosine_similarity(out_gpu.cpu().float(), out_ref.float(), dim=-1)
        assert cos_sim.item() > 0.80, f"GPU INT4 vs dequant ref cosine similarity {cos_sim.item():.4f} too low"


class TestGPUINT4ConversionCache:
    @requires_gpu_int4
    def test_cache_reuse(self):
        """Second forward with same buffer should reuse cached INT4 data."""
        from tinyserve.gpu_int4 import GPUINT4Forward

        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        }).cuda()

        h = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16, device="cuda")
        fwd = GPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu)
        out1 = fwd.forward(h, packed)
        assert len(fwd._int4_cache) == 1
        out2 = fwd.forward(h, packed)
        assert len(fwd._int4_cache) == 1  # reused
        torch.testing.assert_close(out1, out2)

    @requires_gpu_int4
    def test_clear_cache(self):
        from tinyserve.gpu_int4 import GPUINT4Forward

        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        }).cuda()

        h = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16, device="cuda")
        fwd = GPUINT4Forward(layout, group_size=GROUP_SIZE, act_fn=F.silu)
        fwd.forward(h, packed)
        assert len(fwd._int4_cache) == 1
        fwd.clear_cache()
        assert len(fwd._int4_cache) == 0


class TestGPUINT4Availability:
    def test_has_int4_gpu_flag(self):
        from tinyserve.gpu_int4 import HAS_INT4_GPU as flag
        # Should be True on CUDA machines with modern PyTorch
        if torch.cuda.is_available():
            assert flag is True

    @requires_gpu_int4
    def test_graceful_import(self):
        """Module should import cleanly even when testing availability."""
        from tinyserve.gpu_int4 import GPUINT4Forward, HAS_INT4_GPU, mxfp4_to_int4pack_gpu
        assert HAS_INT4_GPU is True


class TestGPUINT4PipelineIntegration:
    @requires_gpu_int4
    def test_pipeline_uses_gpu_int4_for_mxfp4(self):
        """GenericExpertPipeline should use GPU INT4 forward for MXFP4 layouts."""
        from tinyserve.gpu_int4 import GPUINT4Forward

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        # _build_inline_forward returns None for MXFP4
        from tinyserve.generic_pipeline import _build_inline_forward
        assert _build_inline_forward(layout, F.silu) is None

        # _build_gpu_int4_forward should return a GPUINT4Forward-based callable
        from tinyserve.generic_pipeline import _build_gpu_int4_forward
        gpu_fwd = _build_gpu_int4_forward(layout, F.silu)
        assert gpu_fwd is not None

    @requires_gpu_int4
    def test_gpu_int4_inline_produces_valid_output(self):
        """GPU INT4 inline forward from pipeline should produce valid output."""
        from tinyserve.generic_pipeline import _build_gpu_int4_forward

        gu_blocks, gu_scales = _make_mxfp4_weights(2 * OUT_FEATURES, IN_FEATURES, seed=42)
        intermediate = OUT_FEATURES
        dn_blocks, dn_scales = _make_mxfp4_weights(IN_FEATURES, intermediate, seed=43)

        layout = _make_mxfp4_layout(2 * OUT_FEATURES, IN_FEATURES, IN_FEATURES)
        packed = _pack(layout, {
            "gate_up_proj": gu_blocks,
            "gate_up_proj_scales": gu_scales,
            "down_proj": dn_blocks,
            "down_proj_scales": dn_scales,
        }).cuda()

        gpu_fwd = _build_gpu_int4_forward(layout, F.silu)
        h = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16, device="cuda")
        out = gpu_fwd(packed, h)
        assert out.shape == (1, IN_FEATURES)
        assert out.device.type == "cuda"
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
