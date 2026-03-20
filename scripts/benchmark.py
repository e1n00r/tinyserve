"""Benchmark offloaded MoE model decode throughput.

Primary targets:
    openai/gpt-oss-20b          GPT-OSS-20B  (fast iteration)
    openai/gpt-oss-120b         GPT-OSS-120B (production target)
    Qwen/Qwen3.5-35B-A3B        Qwen3.5-MoE 35B
    Qwen/Qwen3.5-122B-A10B      Qwen3.5-MoE 122B

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --model Qwen/Qwen3.5-35B-A3B
    python -m scripts.benchmark --cache-policy slru
    python -m scripts.benchmark --compare
    python -m scripts.benchmark --no-cache
    python -m scripts.benchmark --both-families    # GPT-OSS-20B + Qwen3.5-35B-A3B
"""
import argparse
import json
import time

import torch

DEFAULT_PROMPT = (
    "Explain the theory of relativity in simple terms. "
    "Albert Einstein developed two theories of relativity that "
    "fundamentally changed our understanding of space, time, and gravity."
)

_POLICIES = ("lru", "slru", "lfu", "fifo")

_FAMILY_MODELS = {
    "gpt-oss": "openai/gpt-oss-20b",
    "qwen35":  "Qwen/Qwen3.5-35B-A3B",
}


def _collect_cache_stats(model) -> tuple[int, int]:
    hits = 0
    misses = 0
    pipelines = getattr(model, "_offload_pipelines", [])
    for p in pipelines:
        if p.cache is not None:
            hits += p.cache.hits
            misses += p.cache.misses
    return hits, misses


def _reset_cache_stats(model):
    for p in getattr(model, "_offload_pipelines", []):
        if p.cache is not None:
            p.cache.reset_stats()


def _has_cache(model) -> bool:
    for p in getattr(model, "_offload_pipelines", []):
        if p.cache is not None:
            return True
    return False


def run_benchmark(
    model_id: str = "openai/gpt-oss-20b",
    prompt: str = DEFAULT_PROMPT,
    n_warmup: int = 40,
    n_measure: int = 60,
    no_cache: bool = False,
    cache_capacity: int | None = None,
    cache_policy: str = "lru",
    fp8: bool = True,
) -> dict:
    from transformers import AutoTokenizer

    from src.offload import load_and_offload

    cap = 0 if no_cache else cache_capacity
    model = load_and_offload(
        model_id,
        device="cuda",
        cache_capacity=cap if cap is not None else 0,
        cache_policy=cache_policy,
        fp8=fp8,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    input_ids = tok.encode(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(input_ids=next_token, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    _reset_cache_stats(model)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        with torch.no_grad():
            out = model(input_ids=next_token, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps = n_measure / elapsed
    result: dict = {
        "model": model_id,
        "policy": cache_policy,
        "tok_s": round(tps, 2),
        "ms_per_tok": round(elapsed * 1000 / n_measure, 1),
        "n_warmup": n_warmup,
        "n_measure": n_measure,
    }

    if _has_cache(model):
        hits, misses = _collect_cache_stats(model)
        total = hits + misses
        result["cache_hits"] = hits
        result["cache_misses"] = misses
        result["cache_hit_rate"] = round(hits / total if total > 0 else 0.0, 4)

    return result


def _print_result(result: dict, cache_capacity: int | None = None):
    sep = "\u2500" * 38
    cap_str = f"{cache_capacity} slots" if cache_capacity else "auto"
    print(f"\nModel: {result['model']} | Policy: {result['policy']} | Cache: {cap_str}")
    print(f"Warmup: {result['n_warmup']} tokens | Measure: {result['n_measure']} tokens")
    print(sep)
    print(f"  tok/s       {result['tok_s']}")
    print(f"  ms/tok      {result['ms_per_tok']}")
    if "cache_hit_rate" in result:
        hit_pct = result["cache_hit_rate"] * 100
        hits = result["cache_hits"]
        misses = result["cache_misses"]
        print(f"  hit rate    {hit_pct:.1f}%")
        print(f"  hits/misses {hits}/{misses}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Benchmark offloaded MoE model decode")
    parser.add_argument("--model", default="openai/gpt-oss-20b",
                        help="HuggingFace model id or local path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--measure", type=int, default=60)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-fp8", action="store_true",
                        help="Disable FP8 expert compression (default: FP8 on)")
    parser.add_argument("--cache-capacity", type=int, default=None)
    parser.add_argument("--cache-policy", default="lru", choices=list(_POLICIES))
    parser.add_argument("--compare", action="store_true",
                        help="Run LRU then SLRU back-to-back and print comparison table")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--both-families", action="store_true",
                        help="Run GPT-OSS-20B then Qwen3.5-35B-A3B and print side-by-side")
    args = parser.parse_args()

    if args.both_families:
        results = []
        for name, mid in _FAMILY_MODELS.items():
            r = run_benchmark(
                model_id=mid,
                prompt=args.prompt,
                n_warmup=args.warmup,
                n_measure=args.measure,
                no_cache=args.no_cache, fp8=not args.no_fp8,
                cache_capacity=args.cache_capacity,
                cache_policy=args.cache_policy,
            )
            results.append((name, r))
            _print_result(r, args.cache_capacity)
        sep = "\u2500" * 44
        print(f"\nSummary — policy: {args.cache_policy}")
        print(sep)
        print(f"  {'model':<20}{'tok/s':>8}{'ms/tok':>8}{'hit%':>8}")
        print(sep)
        for name, r in results:
            hr = f"{r['cache_hit_rate']*100:.1f}%" if "cache_hit_rate" in r else "—"
            print(f"  {name:<20}{r['tok_s']:>8}{r['ms_per_tok']:>8}{hr:>8}")
        print(sep)
        return

    if args.compare:
        results = []
        for policy in ("lru", "slru"):
            r = run_benchmark(
                model_id=args.model,
                prompt=args.prompt,
                n_warmup=args.warmup,
                n_measure=args.measure,
                no_cache=args.no_cache, fp8=not args.no_fp8,
                cache_capacity=args.cache_capacity,
                cache_policy=policy,
            )
            results.append(r)
            if args.json:
                print(json.dumps(r))
            else:
                _print_result(r, args.cache_capacity)

        if not args.json:
            sep = "\u2500" * 38
            lru, slru = results[0], results[1]
            print(f"\nComparison: LRU vs SLRU  (model: {args.model})")
            print(sep)
            print(f"  {'metric':<16}{'LRU':>10}{'SLRU':>10}")
            print(sep)
            print(f"  {'tok/s':<16}{lru['tok_s']:>10}{slru['tok_s']:>10}")
            print(f"  {'ms/tok':<16}{lru['ms_per_tok']:>10}{slru['ms_per_tok']:>10}")
            if "cache_hit_rate" in lru and "cache_hit_rate" in slru:
                lru_hr = f"{lru['cache_hit_rate']*100:.1f}%"
                slru_hr = f"{slru['cache_hit_rate']*100:.1f}%"
                print(f"  {'hit rate':<16}{lru_hr:>10}{slru_hr:>10}")
            print(sep)
        return

    result = run_benchmark(
        model_id=args.model,
        prompt=args.prompt,
        n_warmup=args.warmup,
        n_measure=args.measure,
        no_cache=args.no_cache, fp8=not args.no_fp8,
        cache_capacity=args.cache_capacity,
        cache_policy=args.cache_policy,
    )

    if args.json:
        print(json.dumps(result))
    else:
        _print_result(result, args.cache_capacity)


if __name__ == "__main__":
    main()
