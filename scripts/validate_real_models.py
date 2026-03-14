"""Full-weight validation on real models.

Usage:
    python -m scripts.validate_real_models --model openai/gpt-oss-20b
    python -m scripts.validate_real_models --model Qwen/Qwen3.5-35B-A3B
    python -m scripts.validate_real_models --model Qwen/Qwen3.5-122B-A10B
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.offload import offload_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The theory of relativity states that")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--cache-capacity", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda")

    print(f"Loading {args.model} to CPU...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.0f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"  Prompt: {input_ids.shape[1]} tokens")

    print("Offloading experts...")
    t0 = time.perf_counter()
    model = offload_model(model, device=device, cache_capacity=args.cache_capacity)
    print(f"  Offloaded in {time.perf_counter() - t0:.0f}s")

    print(f"Generating ({args.max_tokens} tokens, greedy)...")
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(input_ids, max_new_tokens=args.max_tokens, do_sample=False)
        elapsed = time.perf_counter() - t0

    n_new = out.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  {n_new} tokens in {elapsed:.1f}s ({n_new / elapsed:.1f} tok/s)")
    print(f"  Output: {text}")

    if hasattr(model, "_offload_pipelines"):
        total_hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
        total_misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
        total = total_hits + total_misses
        rate = total_hits / total if total > 0 else 0
        print(f"  Cache: {total_hits} hits, {total_misses} misses ({rate:.1%} hit rate)")

    print("\nVALIDATION COMPLETE")


if __name__ == "__main__":
    main()
