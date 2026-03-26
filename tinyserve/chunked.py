"""Chunked prefill for long-context generation on small GPUs.

Splits the prefill phase into chunks of ``chunk_size`` tokens so the
attention score matrix never exceeds O(chunk_size^2), preventing OOM
on 8 GB GPUs at 2K+ tokens.
"""

import torch


def chunked_prefill(model, input_ids, kv_cache, chunk_size=512):
    """Process prefill in chunks to cap attention VRAM at O(chunk_size^2).

    Each chunk is forwarded through the model with the KV cache accumulating
    context. Only the output from the last chunk (containing the logits for
    the next token prediction) is returned.

    Args:
        model: HuggingFace CausalLM model (may be offloaded).
        input_ids: (1, seq_len) token ids tensor.
        kv_cache: A StaticKVCache or compatible past_key_values object.
        chunk_size: Maximum tokens per prefill chunk.

    Returns:
        Model output from the last chunk.
    """
    seq_len = input_ids.shape[1]
    outputs = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_ids = input_ids[:, start:end]
        with torch.inference_mode():
            outputs = model(input_ids=chunk_ids, past_key_values=kv_cache)
    return outputs


def generate_chunked(
    model,
    input_ids,
    max_new_tokens,
    kv_cache=None,
    chunk_size=512,
    eos_token_id=None,
):
    """Generate with chunked prefill for long inputs on small GPUs.

    Phase 1 splits the prompt into ``chunk_size`` windows so peak attention
    VRAM is O(chunk_size^2) instead of O(seq_len^2).

    Phase 2 is standard autoregressive decoding (one token at a time).

    Args:
        model: HuggingFace CausalLM model.
        input_ids: (1, seq_len) prompt token ids.
        max_new_tokens: Number of tokens to generate.
        kv_cache: Pre-allocated KV cache (StaticKVCache or compatible).
        chunk_size: Prefill chunk size in tokens.
        eos_token_id: Stop generation when this token is produced.

    Returns:
        (1, seq_len + generated) tensor of token ids.
    """
    # Phase 1: chunked prefill
    out = chunked_prefill(model, input_ids, kv_cache, chunk_size)

    # Phase 2: autoregressive decode
    next_token = out.logits[:, -1:].argmax(dim=-1)
    generated = [next_token]

    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=kv_cache)
        next_token = out.logits[:, -1:].argmax(dim=-1)
        generated.append(next_token)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return torch.cat([input_ids] + generated, dim=1)
