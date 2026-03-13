"""Attention, RMSNorm, and RoPE for GPT-OSS-120B."""

import torch
import torch.nn.functional as F

from .config import (
    HEAD_DIM,
    HIDDEN_SIZE,
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
    kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
    sliding_window: int | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Full attention forward with sinks, GQA, and optional sliding window.

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        q/k/v/o_proj_w: projection weights
        q/k/v/o_proj_b: projection biases
        sinks: [num_attention_heads] learned sink logits
        cos, sin: [batch, seq_len, head_dim//2] or broadcastable
        kv_cache: (past_k, past_v) each [batch, num_kv_heads, past_len, head_dim] or None
        sliding_window: if not None, limit attention to this many past tokens

    Returns:
        (output, (new_k_cache, new_v_cache))
    """
    batch, seq_len, _ = hidden_states.shape

    # Project Q, K, V
    q = F.linear(hidden_states, q_proj_w, q_proj_b)
    k = F.linear(hidden_states, k_proj_w, k_proj_b)
    v = F.linear(hidden_states, v_proj_w, v_proj_b)

    # Reshape to [batch, heads, seq_len, head_dim]
    q = q.view(batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    # RoPE
    cos_unsq = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim//2]
    sin_unsq = sin.unsqueeze(1)
    q = apply_rotary_emb(q, cos_unsq, sin_unsq)
    k = apply_rotary_emb(k, cos_unsq, sin_unsq)

    # Update KV cache
    if kv_cache is not None:
        past_k, past_v = kv_cache
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
    new_kv_cache = (k, v)

    # GQA: repeat KV heads
    k_expanded = repeat_kv(k, NUM_KV_GROUPS)
    v_expanded = repeat_kv(v, NUM_KV_GROUPS)

    # Attention scores
    scaling = HEAD_DIM ** -0.5
    attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * scaling

    # Causal mask
    kv_len = k_expanded.shape[2]
    causal_mask = torch.triu(
        torch.full((seq_len, kv_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
        diagonal=kv_len - seq_len + 1,
    )

    # Sliding window mask
    if sliding_window is not None:
        sw_mask = torch.tril(
            torch.full((seq_len, kv_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=-(sliding_window + 1) + (kv_len - seq_len),
        )
        causal_mask = causal_mask + sw_mask

    attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

    # Learned sinks: concatenate sink logit, softmax, then drop
    sink_logits = sinks.reshape(1, -1, 1, 1).expand(batch, -1, seq_len, 1)
    combined = torch.cat([attn_weights, sink_logits], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
    attn_weights = probs[..., :-1]  # drop sink column

    # Weighted sum
    attn_output = torch.matmul(attn_weights, v_expanded)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

    # Output projection
    output = F.linear(attn_output, o_proj_w, o_proj_b)
    return output, new_kv_cache
