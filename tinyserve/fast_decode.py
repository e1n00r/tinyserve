"""Minimal decode loop that bypasses HuggingFace generate() overhead.

HF generate() adds ~3ms/tok from stopping criteria, score tracking,
logit processors, and unfinished_sequences tensor checks. This loop
does greedy argmax decode with zero framework overhead.

fast_decode_step() additionally bypasses GptOssModel.forward() to skip
the causal_mask_mapping dict construction (~0.7ms/tok upper bound per
benchmarks/manual_layer_test_20260404.txt).  The real system bottleneck
is expert I/O mmap page faults (>90% of token time); this is a minor
win kept here for completeness.
"""

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


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


def fast_decode_step(
    model,
    next_token: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    past_key_values=None,
) -> torch.Tensor:
    """Single decode step bypassing GptOssModel.forward() overhead.

    Calls embed -> rotary_emb -> layers -> norm -> lm_head directly,
    skipping HF's per-step causal_mask_mapping dict construction and
    MoeModelOutputWithPast allocation.

    Expected gain: ~0.7 ms/tok upper bound (measured in
    benchmarks/manual_layer_test_20260404.txt).  The dominant cost is
    expert I/O mmap page faults, not this path.

    Args:
        model: GptOssForCausalLM (the inner HF model, e.g. offloaded._model)
        next_token: [1, 1] int64 token id on GPU
        position_ids: [1, 1] int64 position on GPU.  None = derive from
            past_key_values seq length (matches HF default behavior).
        past_key_values: HF Cache object or None

    Returns:
        logits: [1, 1, vocab_size] — last-position logits only
    """
    inner = model.model  # GptOssModel

    hidden = inner.embed_tokens(next_token)  # [1, 1, D]

    if position_ids is None:
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(
            past_len,
            past_len + hidden.shape[1],
            dtype=torch.long,
            device=hidden.device,
        ).unsqueeze(0)

    position_embeddings = inner.rotary_emb(hidden, position_ids)  # (cos, sin)

    # Pre-compute causal mask once for all layers.
    # For a single decode token with no padding, create_causal_mask returns
    # None (sdpa handles causal masking internally), so this is a no-op dict.
    mask_kwargs = {
        "config": inner.config,
        "inputs_embeds": hidden,
        "attention_mask": None,
        "past_key_values": past_key_values,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

    for i, layer in enumerate(inner.layers):
        layer_type = inner.config.layer_types[i]
        hidden = layer.forward(
            hidden,
            attention_mask=causal_mask_mapping[layer_type],
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
        )

    hidden = inner.norm(hidden)
    return model.lm_head(hidden[:, -1:])  # [1, 1, vocab_size]


def fast_generate_layerloop(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    past_key_values=None,
) -> torch.Tensor:
    """Greedy decode using fast_decode_step() for the decode phase.

    Prefill uses model.forward() (same as fast_generate).  Decode tokens
    use the direct layer loop, skipping GptOssModel.forward() overhead.

    Args:
        model: GptOssForCausalLM (inner HF model, e.g. offloaded._model)
        input_ids: [1, seq_len] prompt tokens on GPU
        max_new_tokens: number of tokens to generate
        eos_token_id: stop on this token (None = never stop)
        past_key_values: KV cache or None

    Returns:
        [1, seq_len + generated] tensor of all token IDs
    """
    generated = []
    kw = {}
    if past_key_values is not None:
        kw["past_key_values"] = past_key_values

    # Prefill — keep using HF forward (seq > 1, position handling is identical)
    with torch.inference_mode():
        out = model(input_ids=input_ids, **kw)

    next_token = out.logits[:, -1:].argmax(dim=-1)
    generated.append(next_token)

    for step in range(max_new_tokens - 1):
        with torch.inference_mode():
            # Pass position_ids=None so fast_decode_step derives position from
            # past_key_values (or defaults to 0 without KV cache), matching
            # HF model.forward() default behavior exactly.
            logits = fast_decode_step(model, next_token, None, past_key_values)
        next_token = logits.argmax(dim=-1)
        generated.append(next_token)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return torch.cat([input_ids] + generated, dim=-1)
