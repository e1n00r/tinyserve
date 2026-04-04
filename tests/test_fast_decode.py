"""Tests for fast_decode.fast_decode_step and fast_generate_layerloop.

Verifies that the direct layer-loop path produces the same token
predictions as the reference HF model.forward() path.  Uses a tiny
synthetic GptOss config so no weights need to be downloaded.
"""

import torch

from tests.conftest import requires_cuda


def _make_tiny_gpt_oss(device: torch.device):
    """Return (model, input_ids) for a minimal GptOss config on `device`."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b")
    config.num_hidden_layers = 2
    config.num_local_experts = 4
    config.hidden_size = 64
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.vocab_size = 256
    config.max_position_embeddings = 64
    config.head_dim = 16
    config.layer_types = ["sliding_attention", "full_attention"]
    config.pad_token_id = None

    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval().to(device)
    input_ids = torch.tensor([[1, 42, 100, 7]], device=device)
    return model, input_ids


@requires_cuda
def test_fast_decode_step_matches_hf_forward():
    """fast_decode_step() must produce the same argmax token as model.forward().

    Both use position_ids=None (defaults to 0), so they should be identical.
    """
    from tinyserve.fast_decode import fast_decode_step

    device = torch.device("cuda")
    model, input_ids = _make_tiny_gpt_oss(device)

    # Prefill via HF to get the first decode token
    with torch.inference_mode():
        out_hf = model(input_ids=input_ids)
    next_token = out_hf.logits[:, -1:].argmax(dim=-1)  # [1, 1]

    # HF reference: one more step — position defaults to 0 (no past_key_values)
    with torch.inference_mode():
        ref_out = model(input_ids=next_token)
    ref_logits = ref_out.logits[:, -1:]  # [1, 1, vocab]

    # fast_decode_step with position_ids=None matches HF default (position 0)
    with torch.inference_mode():
        step_logits = fast_decode_step(model, next_token)

    assert step_logits.shape == (1, 1, model.config.vocab_size)
    assert step_logits.argmax(dim=-1).item() == ref_logits.argmax(dim=-1).item()


@requires_cuda
def test_fast_decode_step_logits_close_to_hf():
    """fast_decode_step logit values must be numerically close to HF forward."""
    from tinyserve.fast_decode import fast_decode_step

    device = torch.device("cuda")
    model, input_ids = _make_tiny_gpt_oss(device)

    with torch.inference_mode():
        out_hf = model(input_ids=input_ids)
    next_token = out_hf.logits[:, -1:].argmax(dim=-1)

    with torch.inference_mode():
        ref_logits = model(input_ids=next_token).logits[:, -1:]  # [1, 1, V]

    with torch.inference_mode():
        step_logits = fast_decode_step(model, next_token)

    # bfloat16 — allow generous tolerance; exact match is not expected due to
    # bfloat16 non-associativity but top-1 must agree.
    assert torch.allclose(step_logits.float(), ref_logits.float(), atol=1e-2), (
        f"max diff: {(step_logits.float() - ref_logits.float()).abs().max()}"
    )


@requires_cuda
def test_fast_generate_layerloop_matches_fast_generate():
    """fast_generate_layerloop produces the same token sequence as fast_generate."""
    from tinyserve.fast_decode import fast_generate, fast_generate_layerloop

    device = torch.device("cuda")
    model, input_ids = _make_tiny_gpt_oss(device)

    with torch.inference_mode():
        ref_ids = fast_generate(model, input_ids, max_new_tokens=4)

    with torch.inference_mode():
        loop_ids = fast_generate_layerloop(model, input_ids, max_new_tokens=4)

    assert ref_ids.shape == loop_ids.shape, f"shape mismatch: ref {ref_ids.shape} vs loop {loop_ids.shape}"
    assert torch.equal(ref_ids, loop_ids), f"token mismatch:\n  ref:  {ref_ids.tolist()}\n  loop: {loop_ids.tolist()}"


@requires_cuda
def test_fast_generate_layerloop_eos_stops_early():
    """fast_generate_layerloop respects eos_token_id."""
    from tinyserve.fast_decode import fast_generate, fast_generate_layerloop

    device = torch.device("cuda")
    model, input_ids = _make_tiny_gpt_oss(device)

    # Use fast_generate to determine the second generated token (both functions
    # are identical at this point, so this gives a valid eos target).
    with torch.inference_mode():
        ref_ids = fast_generate(model, input_ids, max_new_tokens=4)
    second_generated_token = ref_ids[0, input_ids.shape[1] + 1].item()

    # eos = second generated token; loop must stop after 2 total tokens
    with torch.inference_mode():
        ids = fast_generate_layerloop(model, input_ids, max_new_tokens=10, eos_token_id=second_generated_token)

    generated_len = ids.shape[1] - input_ids.shape[1]
    assert generated_len == 2, f"expected 2 tokens generated (stopped at eos), got {generated_len}"
