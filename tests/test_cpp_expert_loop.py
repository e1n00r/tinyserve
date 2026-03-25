"""Tests for the C++ expert_loop extension."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyserve.generic_store import TensorLayout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_cpp_extension():
    try:
        from tinyserve.csrc import get_expert_loop
        ext = get_expert_loop()
        return ext
    except Exception:
        return None


def _make_expert_weights(hidden_dim, intermediate_dim, dtype=torch.bfloat16):
    """Create gate_up_proj [2*intermediate, hidden] and down_proj [hidden, intermediate]."""
    w_gu = torch.randn(2 * intermediate_dim, hidden_dim, dtype=dtype)
    w_dn = torch.randn(hidden_dim, intermediate_dim, dtype=dtype)
    return w_gu, w_dn


def _pack_expert(w_gu, w_dn, layout):
    """Pack two weight tensors into a flat uint8 buffer per layout."""
    packed = torch.zeros(layout.total_bytes, dtype=torch.uint8)
    gu_raw = w_gu.contiguous().view(-1).view(torch.uint8)
    packed[layout.offsets["gate_up_proj"]:layout.offsets["gate_up_proj"] + layout.sizes["gate_up_proj"]] = gu_raw
    dn_raw = w_dn.contiguous().view(-1).view(torch.uint8)
    packed[layout.offsets["down_proj"]:layout.offsets["down_proj"] + layout.sizes["down_proj"]] = dn_raw
    return packed


def _python_silu_forward(h, w_gu, w_dn):
    """Reference SiLU-gated forward (matches _build_inline_forward with act_fn=silu)."""
    gate_up = F.linear(h, w_gu)
    gate, up = gate_up.chunk(2, dim=-1)
    gated = F.silu(gate) * up
    return F.linear(gated, w_dn)


def _python_swiglu_forward(h, w_gu, w_dn):
    """Reference GPT-OSS SwiGLU forward."""
    gate_up = F.linear(h, w_gu)
    gate = gate_up[..., ::2].clamp(max=7.0)
    up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
    gated = (up + 1) * gate * torch.sigmoid(gate * 1.702)
    return F.linear(gated, w_dn)


_DTYPE_TO_INT = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.bfloat16: 15,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpp_ext():
    ext = _try_load_cpp_extension()
    if ext is None:
        pytest.skip("C++ expert_loop extension could not be compiled (no compiler?)")
    return ext


class TestCppExtensionCompiles:
    def test_cpp_extension_compiles(self, cpp_ext):
        assert cpp_ext is not None
        assert hasattr(cpp_ext, "fast_expert_forward")


class TestFastForwardMatchesPython:

    @pytest.mark.parametrize("activation", ["silu", "swiglu"])
    def test_single_expert(self, cpp_ext, activation):
        hidden_dim = 64
        intermediate_dim = 128
        w_gu, w_dn = _make_expert_weights(hidden_dim, intermediate_dim)
        layout = TensorLayout({
            "gate_up_proj": (w_gu.shape, w_gu.dtype),
            "down_proj": (w_dn.shape, w_dn.dtype),
        })
        packed = _pack_expert(w_gu, w_dn, layout)
        cache_packed = packed.unsqueeze(0)  # [1, expert_bytes]

        h = torch.randn(1, hidden_dim, dtype=torch.bfloat16)
        slots = torch.tensor([0], dtype=torch.int32)
        weights = torch.tensor([1.0], dtype=torch.float32)

        # gu_shape[0] = 2*intermediate > hidden = gu_shape[1], so no transpose
        gu_needs_t = w_gu.shape[0] == hidden_dim
        gated_dim = max(w_gu.shape) // 2
        dn_needs_t = w_dn.shape[0] == gated_dim

        result = cpp_ext.fast_expert_forward(
            h, slots, weights, cache_packed,
            layout.offsets["gate_up_proj"], layout.sizes["gate_up_proj"],
            list(w_gu.shape), _DTYPE_TO_INT[w_gu.dtype], gu_needs_t,
            layout.offsets["down_proj"], layout.sizes["down_proj"],
            list(w_dn.shape), _DTYPE_TO_INT[w_dn.dtype], dn_needs_t,
            False,  # has_bias
            0, 0, [1], _DTYPE_TO_INT[torch.bfloat16],
            0, 0, [1], _DTYPE_TO_INT[torch.bfloat16],
            activation,
        )

        if activation == "silu":
            expected = _python_silu_forward(h, w_gu, w_dn)
        else:
            expected = _python_swiglu_forward(h, w_gu, w_dn)

        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-2)

    def test_multiple_experts_weighted(self, cpp_ext):
        hidden_dim = 32
        intermediate_dim = 64
        top_k = 4

        experts = [_make_expert_weights(hidden_dim, intermediate_dim) for _ in range(top_k)]
        layout = TensorLayout({
            "gate_up_proj": (experts[0][0].shape, experts[0][0].dtype),
            "down_proj": (experts[0][1].shape, experts[0][1].dtype),
        })

        packed_list = [_pack_expert(w_gu, w_dn, layout) for w_gu, w_dn in experts]
        cache_packed = torch.stack(packed_list)  # [top_k, expert_bytes]

        h = torch.randn(1, hidden_dim, dtype=torch.bfloat16)
        slots = torch.arange(top_k, dtype=torch.int32)
        routing = torch.softmax(torch.randn(top_k), dim=0).float()

        gu_needs_t = experts[0][0].shape[0] == hidden_dim
        gated_dim = max(experts[0][0].shape) // 2
        dn_needs_t = experts[0][1].shape[0] == gated_dim

        result = cpp_ext.fast_expert_forward(
            h, slots, routing, cache_packed,
            layout.offsets["gate_up_proj"], layout.sizes["gate_up_proj"],
            list(experts[0][0].shape), _DTYPE_TO_INT[torch.bfloat16], gu_needs_t,
            layout.offsets["down_proj"], layout.sizes["down_proj"],
            list(experts[0][1].shape), _DTYPE_TO_INT[torch.bfloat16], dn_needs_t,
            False,
            0, 0, [1], _DTYPE_TO_INT[torch.bfloat16],
            0, 0, [1], _DTYPE_TO_INT[torch.bfloat16],
            "silu",
        )

        expected = torch.zeros_like(h)
        for i, (w_gu, w_dn) in enumerate(experts):
            expected += routing[i].item() * _python_silu_forward(h, w_gu, w_dn)

        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_negative_slot_skipped(self, cpp_ext):
        hidden_dim = 32
        intermediate_dim = 64
        w_gu, w_dn = _make_expert_weights(hidden_dim, intermediate_dim)
        layout = TensorLayout({
            "gate_up_proj": (w_gu.shape, w_gu.dtype),
            "down_proj": (w_dn.shape, w_dn.dtype),
        })
        packed = _pack_expert(w_gu, w_dn, layout)
        cache_packed = packed.unsqueeze(0)

        h = torch.randn(1, hidden_dim, dtype=torch.bfloat16)
        # All slots are -1 (cache miss) — should return zeros
        slots = torch.tensor([-1, -1], dtype=torch.int32)
        weights = torch.tensor([0.5, 0.5], dtype=torch.float32)

        gu_needs_t = w_gu.shape[0] == hidden_dim
        gated_dim = max(w_gu.shape) // 2
        dn_needs_t = w_dn.shape[0] == gated_dim

        result = cpp_ext.fast_expert_forward(
            h, slots, weights, cache_packed,
            layout.offsets["gate_up_proj"], layout.sizes["gate_up_proj"],
            list(w_gu.shape), _DTYPE_TO_INT[w_gu.dtype], gu_needs_t,
            layout.offsets["down_proj"], layout.sizes["down_proj"],
            list(w_dn.shape), _DTYPE_TO_INT[w_dn.dtype], dn_needs_t,
            False,
            0, 0, [1], _DTYPE_TO_INT[torch.bfloat16],
            0, 0, [1], _DTYPE_TO_INT[torch.bfloat16],
            "silu",
        )

        assert torch.allclose(result, torch.zeros_like(result))

    def test_build_cpp_layout_args(self):
        from tinyserve.generic_pipeline import _build_cpp_layout_args

        layout = TensorLayout({
            "gate_up_proj": ((256, 64), torch.bfloat16),
            "down_proj": ((64, 128), torch.bfloat16),
        })
        args = _build_cpp_layout_args(layout, act_fn=torch.nn.functional.silu)
        assert args is not None
        assert args["activation"] == "silu"
        assert args["gu_dtype_int"] == 15  # bfloat16

        args_swiglu = _build_cpp_layout_args(layout, act_fn=None)
        assert args_swiglu["activation"] == "swiglu"

    def test_build_cpp_layout_args_returns_none_for_mxfp4(self):
        from tinyserve.generic_pipeline import _build_cpp_layout_args

        layout = TensorLayout({
            "gate_up_proj": ((256, 64), torch.uint8),
            "gate_up_proj_scales": ((16, 64), torch.uint8),
            "down_proj": ((64, 128), torch.uint8),
        })
        assert _build_cpp_layout_args(layout, act_fn=None) is None

    def test_transposed_weights(self, cpp_ext):
        hidden_dim = 64
        intermediate_dim = 128
        # Create weights in [hidden, 2*intermediate] layout (transposed)
        w_gu = torch.randn(hidden_dim, 2 * intermediate_dim, dtype=torch.bfloat16)
        w_dn = torch.randn(intermediate_dim, hidden_dim, dtype=torch.bfloat16)

        layout = TensorLayout({
            "gate_up_proj": (w_gu.shape, w_gu.dtype),
            "down_proj": (w_dn.shape, w_dn.dtype),
        })
        packed = _pack_expert(w_gu, w_dn, layout)
        cache_packed = packed.unsqueeze(0)

        h = torch.randn(1, hidden_dim, dtype=torch.bfloat16)
        slots = torch.tensor([0], dtype=torch.int32)
        weights = torch.tensor([1.0], dtype=torch.float32)

        # With [hidden, 2*intermediate], shape[0]=hidden=min, so gu_needs_t=True
        gu_needs_t = w_gu.shape[0] == min(w_gu.shape)
        gated_dim = max(w_gu.shape) // 2
        dn_needs_t = w_dn.shape[0] == gated_dim

        result = cpp_ext.fast_expert_forward(
            h, slots, weights, cache_packed,
            layout.offsets["gate_up_proj"], layout.sizes["gate_up_proj"],
            list(w_gu.shape), _DTYPE_TO_INT[w_gu.dtype], gu_needs_t,
            layout.offsets["down_proj"], layout.sizes["down_proj"],
            list(w_dn.shape), _DTYPE_TO_INT[w_dn.dtype], dn_needs_t,
            False,
            0, 0, [1], _DTYPE_TO_INT[torch.bfloat16],
            0, 0, [1], _DTYPE_TO_INT[torch.bfloat16],
            "silu",
        )

        # Python reference with same transpose logic
        w_gu_t = w_gu.t() if gu_needs_t else w_gu
        w_dn_t = w_dn.t() if dn_needs_t else w_dn
        expected = _python_silu_forward(h, w_gu_t, w_dn_t)

        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-2)
