"""Local end-to-end tests: GPT-OSS-20B at various context lengths + StreamingLLM.

Usage:
    python -m scripts.test_contexts
    python -m scripts.test_contexts --streaming
    python -m scripts.test_contexts --model Qwen/Qwen3.5-122B-A10B
"""

import argparse
import time

import torch
from transformers import AutoTokenizer

from tinyserve.engine import OffloadConfig, load_and_offload
from tinyserve.static_kv_cache import StaticKVCache
from tinyserve.chunked import chunked_prefill, generate_chunked


def _get_kv_cache(model) -> StaticKVCache | None:
    return getattr(model, "_kv_cache", None) or getattr(getattr(model, "_model", None), "_kv_cache", None)


def _collect_cache_stats(model) -> dict:
    seen = set()
    hits = misses = 0
    for p in getattr(model, "_offload_pipelines", []) or getattr(getattr(model, "_model", None), "_offload_pipelines", []):
        if p.cache is not None and id(p.cache) not in seen:
            seen.add(id(p.cache))
            hits += p.cache.hits
            misses += p.cache.misses
    total = hits + misses
    return {"hits": hits, "misses": misses, "hit_rate": hits / total if total > 0 else 0}


def test_decode_baseline(model, tok, n_warmup=20, n_measure=40):
    """Short-context decode throughput."""
    prompt = "Explain the theory of relativity in simple terms."
    ids = tok.encode(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(input_ids=next_token, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        with torch.no_grad():
            out = model(input_ids=next_token, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return n_measure / elapsed, elapsed * 1000 / n_measure


def test_context_length(model, tok, context_tokens, gen_tokens=20, chunk_size=512):
    """Prefill at given context length, then decode gen_tokens."""
    kv = _get_kv_cache(model)
    if kv is not None:
        kv.reset()

    # Build a long prompt by repeating text
    base = "The quick brown fox jumps over the lazy dog. " * 50
    ids = tok.encode(base, return_tensors="pt").to("cuda")
    # Tile to desired length
    while ids.shape[1] < context_tokens:
        ids = torch.cat([ids, ids], dim=1)
    ids = ids[:, :context_tokens]

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if chunk_size > 0 and context_tokens > chunk_size:
            out = chunked_prefill(model, ids, kv, chunk_size=chunk_size)
        else:
            with torch.no_grad():
                out = model(input_ids=ids, past_key_values=kv)

        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - t0

        # Decode
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(gen_tokens):
            with torch.no_grad():
                out = model(input_ids=next_token, past_key_values=kv)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        decode_time = time.perf_counter() - t0

        kv_seq = kv.get_seq_length(0) if kv else "N/A"
        return {
            "context": context_tokens,
            "prefill_ms": round(prefill_time * 1000, 1),
            "decode_tps": round(gen_tokens / decode_time, 1),
            "decode_ms_per_tok": round(decode_time * 1000 / gen_tokens, 1),
            "kv_seq_len": kv_seq,
            "status": "OK",
        }
    except Exception as e:
        return {"context": context_tokens, "status": f"FAIL: {e}"}


def test_streaming(model, tok, total_tokens=2048, gen_per_round=20, chunk_size=512):
    """StreamingLLM: repeated generation to verify KV stays bounded."""
    kv = _get_kv_cache(model)
    if kv is None:
        return {"status": "SKIP: no KV cache"}

    is_streaming = getattr(kv, "_streaming", False)
    if not is_streaming:
        return {"status": "SKIP: streaming not enabled"}

    max_kept = kv._sink_size + kv._window_size
    kv.reset()

    base = "Write a detailed essay about the history of computing. " * 20
    ids = tok.encode(base, return_tensors="pt").to("cuda")
    ids = ids[:, :min(512, ids.shape[1])]

    rounds = total_tokens // gen_per_round
    results = []

    for r in range(rounds):
        try:
            if r == 0:
                if chunk_size > 0:
                    out = chunked_prefill(model, ids, kv, chunk_size=chunk_size)
                else:
                    with torch.no_grad():
                        out = model(input_ids=ids, past_key_values=kv)
            else:
                with torch.no_grad():
                    out = model(input_ids=next_token, past_key_values=kv)

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(gen_per_round - 1):
                with torch.no_grad():
                    out = model(input_ids=next_token, past_key_values=kv)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            seq_len = kv.get_seq_length(0)
            tps = (gen_per_round - 1) / elapsed if elapsed > 0 else 0
            results.append({"round": r, "seq_len": seq_len, "tps": round(tps, 1)})

            if seq_len > kv.max_seq_len:
                return {"status": f"FAIL: seq_len {seq_len} > max {kv.max_seq_len}", "rounds": results}

        except Exception as e:
            return {"status": f"FAIL at round {r}: {e}", "rounds": results}

    final_seq = kv.get_seq_length(0)
    return {
        "status": "OK",
        "streaming": True,
        "max_kept": max_kept,
        "final_seq_len": final_seq,
        "seq_bounded": final_seq <= kv.max_seq_len,
        "rounds": len(results),
        "avg_tps": round(sum(r["tps"] for r in results) / len(results), 1) if results else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--streaming-sink-size", type=int, default=4)
    parser.add_argument("--streaming-window-size", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--contexts", default="256,512,1024,2048,4096", help="Context lengths to test")
    args = parser.parse_args()

    print(f"=== tinyserve Context Length Tests ===")
    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Streaming: {args.streaming} (sink={args.streaming_sink_size}, window={args.streaming_window_size})")
    print()

    cfg = OffloadConfig(
        sliding_window_kv=args.streaming,
        kv_sink_tokens=args.streaming_sink_size,
        kv_window_tokens=args.streaming_window_size,
    )
    model = load_and_offload(
        args.model,
        device="cuda",
        fp8=True,
        adaptive_fate=True,
        max_seq_len=args.max_seq_len,
        offload_config=cfg,
    )
    tok = AutoTokenizer.from_pretrained(args.model)

    # Test 1: Decode baseline
    print("--- Test 1: Decode baseline (warm cache) ---")
    tps, ms = test_decode_baseline(model, tok)
    print(f"  {tps:.1f} tok/s ({ms:.1f} ms/tok)")

    # Test 2: Context lengths
    contexts = [int(x) for x in args.contexts.split(",")]
    print(f"\n--- Test 2: Context lengths (chunk_size={args.chunk_size}) ---")
    for ctx in contexts:
        if ctx > args.max_seq_len and not args.streaming:
            print(f"  {ctx:>5} tokens: SKIP (> max_seq_len={args.max_seq_len})")
            continue
        result = test_context_length(model, tok, ctx, chunk_size=args.chunk_size)
        if result["status"] == "OK":
            print(f"  {ctx:>5} tokens: prefill={result['prefill_ms']:.0f}ms, "
                  f"decode={result['decode_tps']:.1f} tok/s ({result['decode_ms_per_tok']:.0f}ms/tok), "
                  f"kv_seq={result['kv_seq_len']}")
        else:
            print(f"  {ctx:>5} tokens: {result['status']}")

    # Test 3: Streaming (if enabled)
    if args.streaming:
        print(f"\n--- Test 3: StreamingLLM sustained generation ---")
        result = test_streaming(model, tok, total_tokens=4096, chunk_size=args.chunk_size)
        if result["status"] == "OK":
            print(f"  {result['rounds']} rounds, avg {result['avg_tps']:.1f} tok/s")
            print(f"  Final seq_len={result['final_seq_len']} (max_kept={result['max_kept']})")
            print(f"  Bounded: {result['seq_bounded']}")
        else:
            print(f"  {result['status']}")

    # Cache stats
    stats = _collect_cache_stats(model)
    print(f"\n--- Expert Cache ---")
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}, HR: {stats['hit_rate']:.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
