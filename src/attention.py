"""Attention, RMSNorm, and RoPE for GPT-OSS-120B."""

import torch
import torch.nn.functional as F

from .config import (
    HEAD_DIM,
    NUM_ATTENTION_HEADS,
    NUM_KV_HEADS,
    RMS_NORM_EPS,
)

NUM_KV_GROUPS = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # 8


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + RMS_NORM_EPS)
    return (weight * x).to(input_dtype)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """GPT-OSS uses chunk-based rotation (first_half, second_half), NOT interleave."""
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_, second_), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _fp8_linear(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Linear with FP8 weight using _scaled_mm."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])

    # _scaled_mm requires fp8 inputs; quantize activation on the fly
    x_amax = x_2d.abs().amax()
    x_scale = (x_amax / torch.finfo(torch.float8_e4m3fn).max).clamp(min=1e-12)
    x_fp8 = (x_2d / x_scale).to(torch.float8_e4m3fn)

    out = torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=x_scale.float(),
        scale_b=scale,
        out_dtype=torch.bfloat16,
    )

    if bias is not None:
        out = out + bias

    return out.reshape(*orig_shape[:-1], out.shape[-1])


def attention_forward_fp8(
    hidden_states: torch.Tensor,
    q_proj: tuple[torch.Tensor, torch.Tensor],
    q_proj_b: torch.Tensor,
    k_proj: tuple[torch.Tensor, torch.Tensor],
    k_proj_b: torch.Tensor,
    v_proj: tuple[torch.Tensor, torch.Tensor],
    v_proj_b: torch.Tensor,
    o_proj: tuple[torch.Tensor, torch.Tensor],
    o_proj_b: torch.Tensor,
    sinks: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None,
    sliding_window: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention forward with FP8 weight projections."""
    batch, seq_len, _ = hidden_states.shape

    q = _fp8_linear(hidden_states, q_proj[0], q_proj[1], q_proj_b)
    k = _fp8_linear(hidden_states, k_proj[0], k_proj[1], k_proj_b)
    v = _fp8_linear(hidden_states, v_proj[0], v_proj[1], v_proj_b)

    q = q.view(batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    cos_unsq = cos.unsqueeze(1)
    sin_unsq = sin.unsqueeze(1)
    q = apply_rotary_emb(q, cos_unsq, sin_unsq)
    k = apply_rotary_emb(k, cos_unsq, sin_unsq)

    new_k, new_v = k, v

    if past_kv is not None:
        past_k, past_v = past_kv
        k = torch.cat([past_k.to(k.dtype), k], dim=2)
        v = torch.cat([past_v.to(v.dtype), v], dim=2)

    kv_len = k.shape[2]

    # Decode path (seq_len=1): skip mask allocation, trim KV for sliding window
    if seq_len == 1:
        if sliding_window is not None and kv_len > sliding_window:
            k = k[:, :, -sliding_window:, :]
            v = v[:, :, -sliding_window:, :]
            kv_len = sliding_window

        # GQA expand (no copy needed — expand is a view)
        k_expanded = k[:, :, None, :, :].expand(
            batch, NUM_KV_HEADS, NUM_KV_GROUPS, kv_len, HEAD_DIM
        ).reshape(batch, NUM_ATTENTION_HEADS, kv_len, HEAD_DIM)
        v_expanded = v[:, :, None, :, :].expand(
            batch, NUM_KV_HEADS, NUM_KV_GROUPS, kv_len, HEAD_DIM
        ).reshape(batch, NUM_ATTENTION_HEADS, kv_len, HEAD_DIM)

        scores = torch.matmul(q, k_expanded.transpose(2, 3)) * (HEAD_DIM ** -0.5)

        # Sink: append per-head sink logit, softmax, strip sink probability
        sink_col = sinks.reshape(1, -1, 1, 1).expand(batch, NUM_ATTENTION_HEADS, 1, 1)
        combined = torch.cat([scores, sink_col], dim=-1)
        combined = combined - combined.amax(dim=-1, keepdim=True)
        probs = torch.softmax(combined, dim=-1)
        attn_weights = probs[..., :-1].to(v_expanded.dtype)

        attn_output = torch.matmul(attn_weights, v_expanded)
    else:
        # Prefill path: full mask computation
        k_expanded = repeat_kv(k, NUM_KV_GROUPS)
        v_expanded = repeat_kv(v, NUM_KV_GROUPS)

        scaling = HEAD_DIM ** -0.5
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * scaling

        causal_mask = torch.triu(
            torch.full((seq_len, kv_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=kv_len - seq_len + 1,
        )

        if sliding_window is not None:
            sw_mask = torch.tril(
                torch.full((seq_len, kv_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
                diagonal=-(sliding_window + 1) + (kv_len - seq_len),
            )
            causal_mask = causal_mask + sw_mask

        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        sink_logits = sinks.reshape(1, -1, 1, 1).expand(batch, -1, seq_len, 1)
        combined = torch.cat([attn_weights, sink_logits], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
        attn_weights = probs[..., :-1].to(v_expanded.dtype)

        attn_output = torch.matmul(attn_weights, v_expanded)

    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    output = _fp8_linear(attn_output, o_proj[0], o_proj[1], o_proj_b)
    return output, new_k, new_v
