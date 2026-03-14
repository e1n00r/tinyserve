"""Benchmark for offloaded GPT-OSS-120B decode throughput.

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --warmup 80 --measure 120
    python -m scripts.benchmark --no-cache
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


def run_benchmark(
    weights_dir: str = "./weights",
    prompt: str = DEFAULT_PROMPT,
    n_warmup: int = 40,
    n_measure: int = 60,
    no_cache: bool = False,
    cache_capacity: int | None = None,
) -> dict:
    from transformers import AutoTokenizer

    from src.model import OffloadedGptOss

    cap = 0 if no_cache else cache_capacity
    model = OffloadedGptOss(weights_dir, cache_capacity=cap)
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    input_ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    pl = input_ids.shape[1]

    model.reset()
    pos = torch.arange(pl, device=model.device).unsqueeze(0)
    logits, _ = model.forward(input_ids, pos)
    next_token = logits[:, -1, :].argmax(dim=-1)

    for step in range(n_warmup):
        pos = torch.tensor([[pl + step]], device=model.device)
        logits, _ = model.forward(next_token.unsqueeze(0), pos)
        next_token = logits[:, -1, :].argmax(dim=-1)

    cache = model.expert_pipeline.cache
    warmup_hits = cache.hits if cache else 0
    warmup_misses = cache.misses if cache else 0
    if cache:
        cache.reset_stats()

    all_timings: dict[str, float] = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(n_measure):
        pos = torch.tensor([[pl + n_warmup + step]], device=model.device)
        logits, timings = model.forward(next_token.unsqueeze(0), pos)
        next_token = logits[:, -1, :].argmax(dim=-1)
        for k, v in timings.items():
            all_timings[k] = all_timings.get(k, 0.0) + v
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps = n_measure / elapsed
    result = {
        "tok_s": round(tps, 2),
        "ms_per_tok": round(elapsed * 1000 / n_measure, 1),
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "prompt_len": pl,
    }

    if cache:
        result["cache_hits"] = cache.hits
        result["cache_misses"] = cache.misses
        result["cache_hit_rate"] = round(cache.hit_rate, 4)
        result["cache_capacity"] = cache.capacity
        result["warmup_hits"] = warmup_hits
        result["warmup_misses"] = warmup_misses

    for k, v in all_timings.items():
        result[f"time_{k}_s"] = round(v, 3)

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark offloaded GPT-OSS decode")
    parser.add_argument("--weights-dir", default="./weights")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--measure", type=int, default=60)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-capacity", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_benchmark(
        weights_dir=args.weights_dir,
        prompt=args.prompt,
        n_warmup=args.warmup,
        n_measure=args.measure,
        no_cache=args.no_cache,
        cache_capacity=args.cache_capacity,
    )

    if args.json:
        print(json.dumps(result))
    else:
        print(f"\n{'='*60}")
        print(f"{result['tok_s']} tok/s | {result['ms_per_tok']} ms/tok")
        if "cache_hit_rate" in result:
            print(f"  Cache: {result['cache_hit_rate']:.1%} hit rate "
                  f"(h={result['cache_hits']} m={result['cache_misses']}, "
                  f"cap={result['cache_capacity']})")
            print(f"  Warmup: h={result['warmup_hits']} m={result['warmup_misses']}")
        for k, v in result.items():
            if k.startswith("time_"):
                print(f"  {k[5:]}: {v:.3f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
