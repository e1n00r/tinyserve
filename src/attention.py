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


def attention_forward(
    hidden_states: torch.Tensor,
    q_proj_w: torch.Tensor,
    q_proj_b: torch.Tensor,
    k_proj_w: torch.Tensor,
    k_proj_b: torch.Tensor,
    v_proj_w: torch.Tensor,
    v_proj_b: torch.Tensor,
    o_proj_w: torch.Tensor,
    o_proj_b: torch.Tensor,
    sinks: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None,
    sliding_window: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full attention forward with sinks, GQA, and optional sliding window.

    Returns:
        (output, new_k, new_v) where new_k/new_v are the NEW tokens' KV only
        (not concatenated with past). Caller handles caching.
    """
    batch, seq_len, _ = hidden_states.shape

    q = F.linear(hidden_states, q_proj_w, q_proj_b)
    k = F.linear(hidden_states, k_proj_w, k_proj_b)
    v = F.linear(hidden_states, v_proj_w, v_proj_b)

    q = q.view(batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    cos_unsq = cos.unsqueeze(1)
    sin_unsq = sin.unsqueeze(1)
    q = apply_rotary_emb(q, cos_unsq, sin_unsq)
    k = apply_rotary_emb(k, cos_unsq, sin_unsq)

    # Save new K, V before concatenating with past (caller stores these)
    new_k, new_v = k, v

    # Concatenate with past KV for attention computation
    if past_kv is not None:
        past_k, past_v = past_kv
        # Upcast past KV from cache dtype (FP8) to compute dtype
        k = torch.cat([past_k.to(k.dtype), k], dim=2)
        v = torch.cat([past_v.to(v.dtype), v], dim=2)

    # GQA
    k_expanded = repeat_kv(k, NUM_KV_GROUPS)
    v_expanded = repeat_kv(v, NUM_KV_GROUPS)

    scaling = HEAD_DIM ** -0.5
    attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * scaling

    # Causal mask
    kv_len = k_expanded.shape[2]
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

    # Learned sinks
    sink_logits = sinks.reshape(1, -1, 1, 1).expand(batch, -1, seq_len, 1)
    combined = torch.cat([attn_weights, sink_logits], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
    attn_weights = probs[..., :-1]

    attn_output = torch.matmul(attn_weights, v_expanded)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

    output = F.linear(attn_output, o_proj_w, o_proj_b)
    return output, new_k, new_v
