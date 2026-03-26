"""Tests for SDPA attention with sliding window decode optimization."""

import torch
import torch.nn.functional as F
import pytest


class FakeModule:
    pass


def _get_sdpa_fn():
    """Extract the sdpa_attention_with_sinks function via registration."""
    from tinyserve.offload import _register_sdpa_attention

    _register_sdpa_attention()
    import transformers

    return transformers.AttentionInterface._global_mapping["sdpa"]


class TestSdpaSlidingWindowDecode:
    """Sliding window optimization for decode (L=1) in SDPA path."""

    def test_decode_no_sliding_window_uses_all_kv(self):
        """Without sliding_window, decode attends to all S positions."""
        sdpa_fn = _get_sdpa_fn()
        module = FakeModule()
        N, H, G, S, E = 1, 4, 2, 32, 8
        q = torch.randn(N, H, 1, E, dtype=torch.bfloat16)
        k = torch.randn(N, G, S, E, dtype=torch.bfloat16)
        v = torch.randn(N, G, S, E, dtype=torch.bfloat16)

        out, _ = sdpa_fn(module, q, k, v, attention_mask=None, scaling=1.0 / (E ** 0.5))
        assert out.shape == (N, 1, H, E)
        assert torch.isfinite(out).all()

    def test_decode_sliding_window_slices_kv(self):
        """With sliding_window < S, decode only attends to last W positions."""
        sdpa_fn = _get_sdpa_fn()
        module = FakeModule()
        N, H, G, S, E = 1, 4, 2, 64, 8
        window = 16

        torch.manual_seed(42)
        q = torch.randn(N, H, 1, E, dtype=torch.bfloat16)
        k = torch.randn(N, G, S, E, dtype=torch.bfloat16)
        v = torch.randn(N, G, S, E, dtype=torch.bfloat16)

        out_full, _ = sdpa_fn(module, q, k, v, attention_mask=None,
                              scaling=1.0 / (E ** 0.5), sliding_window=None)
        out_windowed, _ = sdpa_fn(module, q, k, v, attention_mask=None,
                                  scaling=1.0 / (E ** 0.5), sliding_window=window)

        assert out_windowed.shape == (N, 1, H, E)
        assert torch.isfinite(out_windowed).all()
        # Windowed output should differ from full since it ignores early positions
        assert not torch.allclose(out_full, out_windowed, atol=1e-2)

    def test_decode_sliding_window_matches_manual_slice(self):
        """Sliding window decode matches manually slicing KV to last W entries."""
        sdpa_fn = _get_sdpa_fn()
        module = FakeModule()
        N, H, G, S, E = 1, 8, 2, 128, 16
        window = 32

        torch.manual_seed(123)
        q = torch.randn(N, H, 1, E, dtype=torch.bfloat16)
        k = torch.randn(N, G, S, E, dtype=torch.bfloat16)
        v = torch.randn(N, G, S, E, dtype=torch.bfloat16)

        out_sw, _ = sdpa_fn(module, q, k, v, attention_mask=None,
                            scaling=1.0 / (E ** 0.5), sliding_window=window)

        # Manual: expand GQA and slice to last W
        k_exp = k.repeat_interleave(H // G, dim=1)
        v_exp = v.repeat_interleave(H // G, dim=1)
        out_manual = F.scaled_dot_product_attention(
            q, k_exp[:, :, -window:], v_exp[:, :, -window:],
            attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=1.0 / (E ** 0.5),
        )
        out_manual = out_manual.transpose(1, 2).contiguous()

        assert torch.allclose(out_sw, out_manual, atol=1e-3)

    def test_decode_sliding_window_noop_when_s_le_window(self):
        """When S <= sliding_window, all KV is used (no slice)."""
        sdpa_fn = _get_sdpa_fn()
        module = FakeModule()
        N, H, G, S, E = 1, 4, 2, 16, 8
        window = 32  # window larger than S

        torch.manual_seed(7)
        q = torch.randn(N, H, 1, E, dtype=torch.bfloat16)
        k = torch.randn(N, G, S, E, dtype=torch.bfloat16)
        v = torch.randn(N, G, S, E, dtype=torch.bfloat16)

        out_full, _ = sdpa_fn(module, q, k, v, attention_mask=None,
                              scaling=1.0 / (E ** 0.5), sliding_window=None)
        out_windowed, _ = sdpa_fn(module, q, k, v, attention_mask=None,
                                  scaling=1.0 / (E ** 0.5), sliding_window=window)

        # Should be identical since window >= S
        assert torch.allclose(out_full, out_windowed, atol=1e-3)

    def test_prefill_not_affected_by_sliding_window(self):
        """Prefill path (L > 1) is unchanged regardless of sliding_window arg."""
        sdpa_fn = _get_sdpa_fn()
        module = FakeModule()
        N, H, G, L, S, E = 1, 4, 2, 8, 8, 8

        torch.manual_seed(99)
        q = torch.randn(N, H, L, E, dtype=torch.bfloat16)
        k = torch.randn(N, G, S, E, dtype=torch.bfloat16)
        v = torch.randn(N, G, S, E, dtype=torch.bfloat16)

        out_none, _ = sdpa_fn(module, q, k, v, attention_mask=None,
                              scaling=1.0 / (E ** 0.5), sliding_window=None)
        out_sw, _ = sdpa_fn(module, q, k, v, attention_mask=None,
                            scaling=1.0 / (E ** 0.5), sliding_window=16)

        # Both should be identical since prefill uses is_causal=True, no window slicing
        assert torch.allclose(out_none, out_sw, atol=1e-3)

    def test_gqa_8x_sliding_window(self):
        """GPT-OSS-like config: 64 Q heads, 8 KV heads, window=128."""
        sdpa_fn = _get_sdpa_fn()
        module = FakeModule()
        N, H, G, S, E = 1, 64, 8, 256, 32
        window = 128

        torch.manual_seed(42)
        q = torch.randn(N, H, 1, E, dtype=torch.bfloat16)
        k = torch.randn(N, G, S, E, dtype=torch.bfloat16)
        v = torch.randn(N, G, S, E, dtype=torch.bfloat16)

        out, _ = sdpa_fn(module, q, k, v, attention_mask=None,
                         scaling=1.0 / (E ** 0.5), sliding_window=window)

        assert out.shape == (N, 1, H, E)
        assert torch.isfinite(out).all()
