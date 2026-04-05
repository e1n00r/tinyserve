"""Benchmark decode throughput at various context lengths.

Usage:
    python -m scripts.bench_context
    python -m scripts.bench_context --model openai/gpt-oss-120b
    python -m scripts.bench_context --streaming
"""

import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from tinyserve.engine import OffloadConfig, load_and_offload
from tinyserve.static_kv_cache import StaticKVCache
from tinyserve.chunked import chunked_prefill


def build_long_ids(tok, target_len, device="cuda"):
    """Build token IDs of approximately target_len by repeating text."""
    base = "The quick brown fox jumps over the lazy dog. " * 100
    ids = tok.encode(base, return_tensors="pt").to(device)
    while ids.shape[1] < target_len:
        ids = torch.cat([ids, ids], dim=1)
    return ids[:, :target_len]


def bench_decode(model, tok, context_len, n_warmup=5, n_measure=10,
                 max_seq_len=4096, chunk_size=512, streaming=False):
    """Prefill at context_len, then measure decode throughput."""
    # Get or create KV cache
    kv = getattr(model, "_kv_cache", None)
    if kv is None:
        kv = getattr(getattr(model, "_model", None), "_kv_cache", None)
    if kv is not None:
        kv.reset()

    ids = build_long_ids(tok, context_len)

    # Prefill
    try:
        t0 = time.perf_counter()
        if chunk_size > 0 and context_len > chunk_size:
            out = chunked_prefill(model, ids, kv, chunk_size=chunk_size)
        else:
            with torch.no_grad():
                out = model(input_ids=ids, past_key_values=kv, use_cache=kv is not None)
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - t0
    except Exception as e:
        return {"context": context_len, "status": f"PREFILL_FAIL: {e}"}

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Warmup decode
    for _ in range(n_warmup):
        try:
            with torch.no_grad():
                out = model(input_ids=next_token, past_key_values=kv, use_cache=kv is not None)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        except Exception as e:
            return {"context": context_len, "status": f"WARMUP_FAIL: {e}"}

    # Measure decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv, use_cache=kv is not None)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t0

    tps = n_measure / decode_time
    ms_per_tok = decode_time * 1000 / n_measure
    kv_len = kv.get_seq_length(0) if kv else "N/A"

    return {
        "context": context_len,
        "status": "OK",
        "prefill_ms": round(prefill_time * 1000, 0),
        "decode_tps": round(tps, 2),
        "decode_ms": round(ms_per_tok, 1),
        "kv_seq_len": kv_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--contexts", default="0,256,512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--streaming-window", type=int, default=2048)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-measure", type=int, default=20)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print(f"=== Context Length Benchmark ===")
    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Streaming: {args.streaming}")
    if args.streaming:
        print(f"Window: {args.streaming_window}")
    print()

    contexts = [int(x) for x in args.contexts.split(",")]
    decode_budget = args.n_warmup + args.n_measure + 16  # headroom for decode steps after prefill
    if args.streaming:
        effective_max_seq = args.streaming_window + args.streaming_sink_size + decode_budget
    else:
        effective_max_seq = max(max(contexts) + decode_budget, args.max_seq_len)

    cfg = OffloadConfig(
        sliding_window_kv=args.streaming,
        kv_window_tokens=args.streaming_window,
        kv_sink_tokens=4,
        max_context_tokens=effective_max_seq,
        compress_weights=True,
        temporal_prefetch=True,
    )

    model = load_and_offload(
        args.model,
        device="cuda",
        offload_config=cfg,
    )
    tok = AutoTokenizer.from_pretrained(args.model)

    results = []

    for ctx in contexts:
        label = f"ctx={ctx}" if ctx > 0 else "no-ctx (decode-only)"
        if ctx == 0:
            # Pure decode benchmark (no prefill)
            r = bench_decode(model, tok, 1, n_warmup=args.n_warmup,
                           n_measure=args.n_measure, max_seq_len=args.max_seq_len,
                           chunk_size=args.chunk_size, streaming=args.streaming)
            r["context"] = 0
        else:
            r = bench_decode(model, tok, ctx, n_warmup=args.n_warmup,
                           n_measure=args.n_measure, max_seq_len=args.max_seq_len,
                           chunk_size=args.chunk_size, streaming=args.streaming)

        results.append(r)
        if r["status"] == "OK":
            print(f"  {ctx:>6} tokens: {r['decode_tps']:>6.1f} tok/s  ({r['decode_ms']:>6.0f} ms/tok)  "
                  f"prefill={r['prefill_ms']:.0f}ms  kv={r['kv_seq_len']}")
        else:
            print(f"  {ctx:>6} tokens: {r['status']}")

    # Expert cache stats
    pipelines = getattr(getattr(model, "_model", model), "_offload_pipelines", [])
    if pipelines and pipelines[0].cache:
        cache = pipelines[0].cache
        total = cache.hits + cache.misses
        hr = cache.hits / total if total > 0 else 0
        print(f"\nExpert cache: {cache.hits} hits, {cache.misses} misses, HR={hr:.1%}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json_out}")


if __name__ == "__main__":
    main()
