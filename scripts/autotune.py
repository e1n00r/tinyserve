#!/usr/bin/env python3
"""Auto-tune expert cache vs KV cache trade-off for a given workload.

Sweeps different (max_seq_len, expert_cache_capacity) configurations,
measures tok/s and hit rate, and reports the Pareto frontier — configs
where you can't improve context size without losing throughput.

Usage:
    # Default workload (mixed short/medium/long prompts):
    python scripts/autotune.py --model openai/gpt-oss-20b

    # Custom workload from file (one prompt per line):
    python scripts/autotune.py --model openai/gpt-oss-20b --workload prompts.txt

    # Specify max tokens to generate per prompt:
    python scripts/autotune.py --model openai/gpt-oss-20b --gen-tokens 60

    # Use FP8 KV cache for 2x context:
    python scripts/autotune.py --model openai/gpt-oss-20b --kv-fp8
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

_DEFAULT_PROMPTS = [
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a short recipe for pasta:",
    "def fibonacci(n):\n    \"\"\"Return nth fibonacci number.\"\"\"",
    "Столица России — город",
    "Solve step by step: If 3x + 7 = 22, what is x?",
]


def run_workload(model, tokenizer, prompts, gen_tokens, kv_cache_factory):
    results = []
    for prompt in prompts:
        kv = kv_cache_factory() if kv_cache_factory else None
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        plen = inputs["input_ids"].shape[1]

        # Reset expert cache stats
        for p in model._offload_pipelines:
            if p.cache:
                p.cache.reset_stats()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            kw = {"past_key_values": kv} if kv else {}
            out = model.generate(**inputs, max_new_tokens=gen_tokens, do_sample=False, **kw)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        n = out.shape[1] - plen
        hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
        misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
        hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
        results.append({"tokens": n, "elapsed": elapsed, "tok_s": n / elapsed, "hit_rate": hr})

    avg_tok_s = sum(r["tok_s"] for r in results) / len(results)
    avg_hr = sum(r["hit_rate"] for r in results) / len(results)
    return avg_tok_s, avg_hr


def main():
    parser = argparse.ArgumentParser(description="Auto-tune expert/KV cache trade-off")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--workload", help="File with one prompt per line")
    parser.add_argument("--gen-tokens", type=int, default=40)
    parser.add_argument("--kv-fp8", action="store_true", help="Use FP8 KV cache (2x context)")
    parser.add_argument("--cache-policy", default="lfru")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoConfig
    from src.static_kv_cache import StaticKVCache

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    effective = getattr(config, "text_config", config)
    kv_dtype = torch.float8_e4m3fn if args.kv_fp8 else torch.bfloat16

    bpt = StaticKVCache.bytes_per_token(
        effective.num_hidden_layers, effective.num_key_value_heads,
        effective.head_dim, kv_dtype,
    )

    prompts = _DEFAULT_PROMPTS
    if args.workload:
        with open(args.workload) as f:
            prompts = [line.strip() for line in f if line.strip()]

    # Sweep configs: different max_seq_len values
    seq_lens = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    print(f"Model: {args.model}")
    print(f"KV dtype: {kv_dtype}, bytes/token: {bpt}")
    print(f"Workload: {len(prompts)} prompts × {args.gen_tokens} tokens")
    print(f"{'max_seq_len':>12s} {'KV MB':>8s} {'Expert slots':>13s} {'tok/s':>8s} {'HR%':>6s} {'Pareto':>7s}")
    print("─" * 60)

    results = []
    from src.offload import load_and_offload

    for seq_len in seq_lens:
        kv_mb = seq_len * bpt / 1e6
        # Skip configs that would use more than 3GB for KV
        if kv_mb > 3000:
            continue

        try:
            model = load_and_offload(
                args.model, cache_capacity=0, cache_policy=args.cache_policy,
                max_seq_len=seq_len, kv_dtype=kv_dtype,
            )
        except Exception as e:
            print(f"{seq_len:>12d} {'FAILED':>8s} — {e}")
            continue

        expert_slots = model._offload_pipelines[0].cache.capacity if model._offload_pipelines[0].cache else 0

        def kv_factory():
            if seq_len == 0:
                return None
            return StaticKVCache.from_model_config(config, max_seq_len=seq_len,
                                                    device="cuda", dtype=kv_dtype)

        # Warmup
        kv = kv_factory()
        inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
        with torch.no_grad():
            kw = {"past_key_values": kv} if kv else {}
            model.generate(**inputs, max_new_tokens=10, do_sample=False, **kw)

        avg_tok_s, avg_hr = run_workload(model, tokenizer, prompts, args.gen_tokens, kv_factory)
        results.append({
            "seq_len": seq_len, "kv_mb": kv_mb, "expert_slots": expert_slots,
            "tok_s": avg_tok_s, "hr": avg_hr,
        })

        del model
        torch.cuda.empty_cache()

    # Mark Pareto-optimal configs
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, s in enumerate(results):
            if i == j:
                continue
            # s dominates r if s has >= context AND >= throughput
            if s["seq_len"] >= r["seq_len"] and s["tok_s"] >= r["tok_s"] and (
                s["seq_len"] > r["seq_len"] or s["tok_s"] > r["tok_s"]
            ):
                dominated = True
                break
        pareto.append(not dominated)

    for r, is_pareto in zip(results, pareto):
        marker = "  ★" if is_pareto else ""
        print(f"{r['seq_len']:>12d} {r['kv_mb']:>7.0f}M {r['expert_slots']:>13d} {r['tok_s']:>7.1f} {r['hr']:>5.1f}%{marker}")

    print()
    print("★ = Pareto-optimal (can't improve context without losing throughput)")
    print()
    best = max((r for r, p in zip(results, pareto) if p), key=lambda r: r["tok_s"])
    print(f"Recommended: max_seq_len={best['seq_len']}, {best['expert_slots']} expert slots")
    print(f"  → {best['tok_s']:.1f} tok/s, {best['hr']:.0f}% hit rate, {best['kv_mb']:.0f}MB KV cache")


if __name__ == "__main__":
    main()
