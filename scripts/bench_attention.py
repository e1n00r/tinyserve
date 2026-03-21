"""Benchmark attention implementations: eager vs SDPA vs FlexAttention."""
import torch
import time
import sys

PYTHONPATH = "/home/elnur/gpt-oss-offload"
if PYTHONPATH not in sys.path:
    sys.path.insert(0, PYTHONPATH)

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def sdpa_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0., **_
):
    N, H, L, E = query.shape
    _, G, *_ = key.shape
    if attention_mask is None:
        attention_mask = torch.zeros(N, 1, L, key.shape[2], device=query.device, dtype=query.dtype)
    key_expanded = key.repeat_interleave(H // G, dim=1)
    value_expanded = value.repeat_interleave(H // G, dim=1)
    sink_k = torch.zeros(N, H, 1, E, device=key.device, dtype=key.dtype)
    sink_v = torch.zeros(N, H, 1, E, device=value.device, dtype=value.dtype)
    k_with_sink = torch.cat([key_expanded, sink_k], dim=2)
    v_with_sink = torch.cat([value_expanded, sink_v], dim=2)
    sink_bias = module.sinks.reshape(1, H, 1, 1).expand(N, H, L, 1)
    mask_expanded = attention_mask.expand(N, H, L, -1).clone()
    mask_with_sink = torch.cat([mask_expanded, sink_bias], dim=3)
    out = torch.nn.functional.scaled_dot_product_attention(
        query, k_with_sink, v_with_sink, mask_with_sink,
        dropout_p=0.0, scale=scaling
    )
    return out.transpose(1, 2).contiguous(), None


def register_sdpa():
    transformers.AttentionInterface.register("sdpa", sdpa_attention_forward)
    try:
        transformers.AttentionMaskInterface.register(
            "sdpa", transformers.masking_utils.eager_mask
        )
    except Exception:
        pass
    transformers.models.gpt_oss.modeling_gpt_oss.GptOssPreTrainedModel._supports_sdpa = True


def bench(model, tok, name, n_warmup=10, n_measure=40):
    inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=n_warmup, do_sample=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=n_measure, do_sample=False)
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    n = out.shape[1] - inputs["input_ids"].shape[1]
    print(f"  {name}: {n} tok in {el:.2f}s = {n/el:.1f} tok/s")
    return n / el


def bench_long_context(model, tok, name, ctx_sizes=[50, 100, 200]):
    base = "In the field of artificial intelligence, " * 20
    for ctx in ctx_sizes:
        prompt = base[:ctx * 4]  # rough chars-to-tokens
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, do_sample=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        n = out.shape[1] - plen
        print(f"  {name} ctx={plen:4d}: {n} tok in {el:.2f}s = {n/el:.1f} tok/s")


def main():
    model_id = "openai/gpt-oss-20b"
    tok = AutoTokenizer.from_pretrained(model_id)

    # === Option 1: SDPA ===
    print("=== Option 1: SDPA with virtual sink tokens ===")
    register_sdpa()
    try:
        from src.offload import offload_model
        raw = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cpu"
        )
        model_sdpa = offload_model(raw, device="cuda", cache_capacity=0, model_id=model_id)
        sdpa_speed = bench(model_sdpa, tok, "SDPA")
        bench_long_context(model_sdpa, tok, "SDPA")
        del model_sdpa, raw
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
        sdpa_speed = 0

    # === Baseline: Eager ===
    print("\n=== Baseline: Eager attention ===")
    try:
        from src.offload import load_and_offload
        model_eager = load_and_offload(model_id, cache_capacity=0, cache_policy="lfru")
        eager_speed = bench(model_eager, tok, "Eager")
        bench_long_context(model_eager, tok, "Eager")
        del model_eager
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
        eager_speed = 0

    # === Option 2: FA3 kernels ===
    print("\n=== Option 2: FlashAttention-3 with native sinks ===")
    try:
        raw2 = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            attn_implementation="kernels-community/vllm-flash-attn3",
            device_map="cpu"
        )
        model_fa3 = offload_model(raw2, device="cuda", cache_capacity=0, model_id=model_id)
        fa3_speed = bench(model_fa3, tok, "FA3")
        bench_long_context(model_fa3, tok, "FA3")
        del model_fa3, raw2
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
        fa3_speed = 0

    # === Option 3: FlexAttention ===
    print("\n=== Option 3: FlexAttention with score_mod ===")
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask

        def flex_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=0., **_
        ):
            N, H, L, E = query.shape
            _, G, *_ = key.shape
            key_expanded = key.repeat_interleave(H // G, dim=1)
            value_expanded = value.repeat_interleave(H // G, dim=1)
            sinks = module.sinks  # [H]

            def score_mod(score, b, h, q_idx, kv_idx):
                return score * scaling

            # FlexAttention doesn't directly support sinks — would need custom score_mod
            # For now, test basic FlexAttention without sinks
            out = flex_attention(query, key_expanded, value_expanded, scale=scaling)
            return out.transpose(1, 2).contiguous(), None

        transformers.AttentionInterface.register("flex", flex_attention_forward)
        try:
            transformers.AttentionMaskInterface.register(
                "flex", transformers.masking_utils.eager_mask
            )
        except Exception:
            pass
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssPreTrainedModel._supports_flex = True

        raw3 = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flex", device_map="cpu"
        )
        model_flex = offload_model(raw3, device="cuda", cache_capacity=0, model_id=model_id)
        flex_speed = bench(model_flex, tok, "FlexAttn")
        bench_long_context(model_flex, tok, "FlexAttn")
        del model_flex, raw3
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")
        flex_speed = 0

    print("\n=== SUMMARY ===")
    print(f"  Eager:    {eager_speed:.1f} tok/s")
    print(f"  SDPA:     {sdpa_speed:.1f} tok/s")
    print(f"  FA3:      {fa3_speed:.1f} tok/s")
    print(f"  FlexAttn: {flex_speed:.1f} tok/s")


if __name__ == "__main__":
    main()
