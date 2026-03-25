#!/usr/bin/env python3
"""FlexAttention-only benchmark. Run separately from eager bench to avoid OOM.

Usage: python3 scripts/bench_flex_only.py
Results: cat /tmp/flex_bench.log
"""
import gc
import sys
import time
import traceback

sys.path.insert(0, ".")
import torch

LOG = "/tmp/flex_bench.log"
BASE = "The history of artificial intelligence began in antiquity. " * 200


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    with open(LOG, "w") as f:
        f.write("")

    log("=== FlexAttention Benchmark ===")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    from transformers import AutoTokenizer
    from tinyserve.offload import load_and_offload

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    log("Loading model with FlexAttention + FP8 KV + max_seq_len=4096...")
    try:
        model = load_and_offload(
            "openai/gpt-oss-20b",
            cache_capacity=0,
            max_seq_len=4096,
            kv_dtype=torch.float8_e4m3fn,
            gpu_memory_utilization=0.80,
            attn_implementation="flex",
        )
    except Exception:
        log(f"LOAD FAILED:\n{traceback.format_exc()}")
        return

    kv = getattr(model, "_kv_cache", None)
    slots = 0
    for p in model._offload_pipelines:
        if p.cache:
            slots = p.cache.capacity
            break
    kv_mb = kv.vram_bytes / 1e6 if kv else 0
    log(f"KV={kv_mb:.0f}MB, experts={slots}, static_shapes={getattr(kv, 'static_shapes', 'N/A')}")
    log(f"VRAM free: {torch.cuda.mem_get_info(0)[0]/1e6:.0f}MB")

    log("Compiling (warmup)...")
    try:
        inp = tok("Hello world", return_tensors="pt").to("cuda")
        if kv:
            kv.reset()
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=10, do_sample=False, past_key_values=kv)
        log("Compiled OK.")
    except Exception:
        log(f"COMPILE FAILED:\n{traceback.format_exc()}")
        return

    log(f"{'ctx':>7s} {'gen':>4s} {'tok/s':>7s} {'ms/tok':>8s} {'HR%':>5s} {'VRAM':>6s}")
    log("-" * 44)

    for ctx in [10, 50, 100, 500, 1000, 2000, 4000, 6000]:
        if kv and ctx > kv.max_seq_len - 30:
            break
        try:
            inp = tok(BASE, return_tensors="pt", truncation=True, max_length=ctx).to("cuda")
            plen = inp["input_ids"].shape[1]
            if kv:
                kv.reset()
            for p in model._offload_pipelines:
                if p.cache:
                    p.cache.reset_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=20, do_sample=False, past_key_values=kv)
            torch.cuda.synchronize()
            el = time.perf_counter() - t0
            n = out.shape[1] - plen
            hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
            misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
            hr = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
            free = torch.cuda.mem_get_info(0)[0] / 1e6
            log(f"{plen:>7d} {n:>4d} {n/el:>6.1f} {el/n*1000:>7.1f} {hr:>4.0f}% {free:>5.0f}M")
        except Exception:
            log(f"{ctx:>7d}  ERR: {traceback.format_exc().splitlines()[-1]}")
            break

    log("Done.")


if __name__ == "__main__":
    main()
