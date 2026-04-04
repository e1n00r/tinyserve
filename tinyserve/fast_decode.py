"""Minimal decode loop that bypasses HuggingFace generate() overhead.

HF generate() adds ~3ms/tok from stopping criteria, score tracking,
logit processors, and unfinished_sequences tensor checks. This loop
does greedy argmax decode with zero framework overhead.
"""

import torch


def fast_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    past_key_values=None,
) -> torch.Tensor:
    """Greedy decode without HF generate() overhead.

    Args:
        model: the inner HF model (not OffloadedLM — pass model._model)
        input_ids: [1, seq_len] prompt tokens on GPU
        max_new_tokens: number of tokens to generate
        eos_token_id: stop on this token (None = never stop)
        past_key_values: KV cache (StaticKVCache or None)

    Returns:
        [1, seq_len + generated] tensor of all token IDs
    """
    generated = []
    kw = {}
    if past_key_values is not None:
        kw["past_key_values"] = past_key_values

    # Prefill
    with torch.inference_mode():
        out = model(input_ids=input_ids, **kw)

    next_token = out.logits[:, -1:].argmax(dim=-1)
    generated.append(next_token)

    # Decode loop — zero overhead
    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            out = model(input_ids=next_token, **kw)
        next_token = out.logits[:, -1:].argmax(dim=-1)
        generated.append(next_token)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return torch.cat([input_ids] + generated, dim=-1)
