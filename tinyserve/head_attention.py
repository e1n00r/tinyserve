"""Head-wise attention for CPU KV offload.

When KV cache lives on CPU, transferring all heads to GPU for SDPA
creates a VRAM bottleneck at long context. Head-wise attention processes
one KV head at a time, keeping peak VRAM at O(seq_len × head_dim)
instead of O(seq_len × num_heads × head_dim).

Based on HeadInfer (arxiv 2502.12574).
"""

import torch
import torch.nn.functional as F


def head_wise_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    sliding_window: int | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """SDPA with head-wise KV streaming from CPU.

    Processes one GQA group at a time to minimize peak GPU memory.
    K/V can be on CPU — only one head's data is on GPU at any time.

    Args:
        query: [1, num_q_heads, seq_len, head_dim] on GPU
        key: [1, num_kv_heads, kv_len, head_dim] on CPU or GPU
        value: [1, num_kv_heads, kv_len, head_dim] on CPU or GPU
        scaling: attention scale factor
        sliding_window: if set, only attend to last N tokens
        is_causal: causal masking for prefill

    Returns:
        [1, seq_len, num_q_heads, head_dim] on GPU (transposed for HF compat)
    """
    N, H, L, E = query.shape
    _, G, S, _ = key.shape
    heads_per_group = H // G
    device = query.device

    # Apply sliding window to key/value
    if sliding_window is not None and sliding_window < S:
        key = key[:, :, -sliding_window:]
        value = value[:, :, -sliding_window:]
        S = key.shape[2]

    output = torch.empty(N, H, L, E, device=device, dtype=query.dtype)

    # Process one KV head (GQA group) at a time
    for g in range(G):
        # Transfer one head's KV to GPU
        k_head = key[:, g : g + 1, :, :].to(device=device, non_blocking=True)
        v_head = value[:, g : g + 1, :, :].to(device=device, non_blocking=True)

        # Get the query heads for this GQA group
        q_start = g * heads_per_group
        q_end = q_start + heads_per_group
        q_group = query[:, q_start:q_end, :, :]  # [1, heads_per_group, L, E]

        # Expand KV to match query heads (GQA)
        k_expanded = k_head.expand(-1, heads_per_group, -1, -1)
        v_expanded = v_head.expand(-1, heads_per_group, -1, -1)

        # SDPA for this group
        out_group = F.scaled_dot_product_attention(
            q_group,
            k_expanded,
            v_expanded,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scaling,
        )
        output[:, q_start:q_end, :, :] = out_group

    # Transpose to [N, L, H, E] for HF compatibility
    return output.transpose(1, 2).contiguous(), None
