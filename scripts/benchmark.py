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

# Domain-shift workload: maximises delta between good and bad cache policies.
# Phase 1 (warm-up): English technical code — builds expert residency for EN-tech vocab.
# Phase 2 (shift): Russian classical literature — forces expert cache churn.
# Russian uses a completely different tokeniser distribution and activates different
# experts than English; Latin-script languages are too similar to EN for a clean shift.
# Least-Stale evicts phase-1 experts immediately; LRU keeps them, crowding RU experts.
_WARMUP_PROMPT_TECH = (
    "Write a Python implementation of a binary search tree with insertion, "
    "deletion, and in-order traversal. Include type hints and docstrings. "
    "class BSTNode: def __init__(self, val: int): self.val = val; "
    "self.left = None; self.right = None. The time complexity of insertion "
    "is O(log n) average case. Memory management uses garbage collection."
)

_SHIFT_PROMPT_LITERATURE = (
    "Опишите главные темы романа Льва Толстого «Война и мир». "
    "Андрей Болконский и Пьер Безухов ищут смысл жизни на фоне наполеоновских войн. "
    "Толстой исследует природу истории, свободу воли и роль личности в великих событиях. "
    "Наташа Ростова воплощает жизненную силу и нравственное возрождение. "
    "Через семьи Болконских, Ростовых и Курагиных автор показывает судьбу России "
    "в эпоху великих потрясений, войны и мира, любви и утраты."
)

_POLICIES = ("lru", "slru", "lfu", "lfru", "fifo", "ls", "dali")

_FAMILY_MODELS = {
    "gpt-oss": "openai/gpt-oss-20b",
    "qwen35": "Qwen/Qwen3.5-35B-A3B",
}


def _collect_cache_stats(model) -> tuple[int, int]:
    # All pipelines share the same cache object — only count it once.
    seen: set[int] = set()
    hits = 0
    misses = 0
    for p in getattr(model, "_offload_pipelines", []):
        if p.cache is not None and id(p.cache) not in seen:
            seen.add(id(p.cache))
            hits += p.cache.hits
            misses += p.cache.misses
    return hits, misses


def _reset_cache_stats(model):
    seen: set[int] = set()
    for p in getattr(model, "_offload_pipelines", []):
        if p.cache is not None and id(p.cache) not in seen:
            seen.add(id(p.cache))
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
    cache_policy: str = "ls",
    cache_bias: float = 0.0,
    fp8: bool = True,
    adaptive_fate: bool = True,
) -> dict:
    from transformers import AutoTokenizer

    from tinyserve.offload import load_and_offload
    from tinyserve.offloaded_model import reset_temporal_routing

    cap = 0 if no_cache else cache_capacity
    model = load_and_offload(
        model_id,
        device="cuda",
        cache_capacity=cap if cap is not None else 0,
        cache_policy=cache_policy,
        cache_bias=cache_bias,
        fp8=fp8,
        adaptive_fate=adaptive_fate,
    )
    reset_temporal_routing()
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
    reset_temporal_routing()

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
        "cache_bias": cache_bias,
        "adaptive_fate": adaptive_fate,
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


def run_context_scaling(
    model_id: str = "openai/gpt-oss-20b",
    contexts: list[int] | None = None,
    gen_tokens: int = 20,
    cache_policy: str = "lfru",
    fp8: bool = True,
    adaptive_fate: bool = True,
    attn_implementation: str = "sdpa",
    static_kv: int = 0,
) -> list[dict]:
    """Benchmark with separated prefill and decode timing across context lengths.

    Uses HF KV cache (DynamicCache by default) so decode tokens attend to
    prior context.  With ``static_kv > 0`` a pre-allocated StaticKVCache is
    used instead, eliminating per-step allocation overhead.
    """
    from transformers import AutoTokenizer

    from tinyserve.offload import load_and_offload
    from tinyserve.offloaded_model import reset_temporal_routing

    if contexts is None:
        contexts = [10, 50, 100, 500, 1000, 2000, 3000]

    model = load_and_offload(
        model_id,
        device="cuda",
        cache_policy=cache_policy,
        fp8=fp8,
        adaptive_fate=adaptive_fate,
        attn_implementation=attn_implementation,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    base_text = "The history of artificial intelligence. " * 500

    # Warmup
    inp = tok("Hello", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=3, do_sample=False)

    results = []
    for ctx in contexts:
        reset_temporal_routing()
        inp = tok(base_text, return_tensors="pt", truncation=True, max_length=ctx).to("cuda")
        actual_ctx = inp["input_ids"].shape[1]

        _reset_cache_stats(model)

        # Build KV cache for this context length
        if static_kv > 0:
            from tinyserve.static_kv_cache import StaticKVCache
            past_kv = StaticKVCache.from_model_config(
                model.config, max_seq_len=static_kv, device="cuda",
            )
        else:
            past_kv = None  # HF will create DynamicCache automatically

        # Time prefill
        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        with torch.inference_mode():
            if past_kv is not None:
                out = model(**inp, past_key_values=past_kv)
            else:
                out = model(**inp)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t_prefill_start) * 1000
        past_kv = out.past_key_values

        p_hits, p_misses = _collect_cache_stats(model)
        _reset_cache_stats(model)

        # Time decode
        next_token = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        t_decode_start = time.perf_counter()
        for _ in range(gen_tokens):
            with torch.inference_mode():
                out = model(input_ids=next_token, past_key_values=past_kv)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_decode_start) * 1000

        d_hits, d_misses = _collect_cache_stats(model)
        d_total = d_hits + d_misses
        decode_tps = gen_tokens / (decode_ms / 1000)
        total_tps = gen_tokens / ((prefill_ms + decode_ms) / 1000)

        results.append({
            "ctx": actual_ctx,
            "prefill_ms": round(prefill_ms, 1),
            "decode_ms": round(decode_ms, 1),
            "decode_tps": round(decode_tps, 1),
            "total_tps": round(total_tps, 1),
            "prefill_experts_loaded": p_hits + p_misses,
            "decode_hit_rate": round(d_hits / max(1, d_total), 4),
        })

        # Free KV cache between context lengths to avoid OOM
        del past_kv

    return results


def _swap_cache_policy(model, policy: str, device: str | torch.device = "cuda"):
    """Replace the cache policy on all pipelines without reloading the model.

    Reinitialises ExpertCache with the same capacity but a new policy,
    preserving all non-expert GPU state. Expert weights on CPU are unaffected.
    """
    from tinyserve.expert_store import ExpertCache

    pipelines = getattr(model, "_offload_pipelines", [])
    if not pipelines or pipelines[0].cache is None:
        return

    old_cache = pipelines[0].cache
    capacity = old_cache.capacity
    expert_bytes = old_cache.expert_bytes
    cache_device = old_cache.device
    # Free old cache before allocating new one to avoid double-allocating ~3 GB.
    for p in pipelines:
        p.cache = None
    del old_cache
    torch.cuda.empty_cache()
    new_cache = ExpertCache(capacity, expert_bytes, cache_device, policy=policy)
    for p in pipelines:
        p.cache = new_cache


def run_domain_shift_benchmark(
    model,
    tok,
    model_id: str = "openai/gpt-oss-20b",
    n_phase1: int = 80,
    n_cold: int = 30,
    n_warm: int = 60,
    cache_policy: str = "ls",
    cache_bias: float = 0.0,
    same_language: bool = False,
) -> dict:
    """Domain-shift benchmark: warm up on EN technical, shift to Russian literature.

    When same_language=True all three phases use the EN-tech prompt (no shift),
    isolating steady-state frequency effects from domain-shift effects.

    Accepts an already-loaded model to avoid the 3-minute reload per policy.
    Swaps the cache policy in-place, then runs all 3 phases.
    """
    _swap_cache_policy(model, cache_policy)
    for p in getattr(model, "_offload_pipelines", []):
        p.cache_bias = cache_bias

    device = "cuda"

    def _next_token(ids):
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        return out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # ── Phase 1: EN-tech warm-up ─────────────────────────────────────────────
    ids = tok.encode(_WARMUP_PROMPT_TECH, return_tensors="pt").to(device)
    _next_token(ids)  # prefill
    next_tok = _next_token(ids[:, -1:])
    _reset_cache_stats(model)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_phase1):
        next_tok = _next_token(next_tok)
    torch.cuda.synchronize()
    phase1_tps = n_phase1 / (time.perf_counter() - t0)
    p1_hits, p1_misses = _collect_cache_stats(model)

    # ── Phase 2: cold phase (same prompt or Russian after domain shift) ─────
    phase2_prompt = _WARMUP_PROMPT_TECH if same_language else _SHIFT_PROMPT_LITERATURE
    ids2 = tok.encode(phase2_prompt, return_tensors="pt").to(device)
    _next_token(ids2)  # prefill (different domain — forces new experts)
    next_tok = _next_token(ids2[:, -1:])
    _reset_cache_stats(model)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(n_cold):
        next_tok = _next_token(next_tok)
    torch.cuda.synchronize()
    cold_tps = n_cold / (time.perf_counter() - t1)
    cold_hits, cold_misses = _collect_cache_stats(model)

    # ── Phase 3: Russian literature — warm (cache re-adapted) ────────────────
    _reset_cache_stats(model)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    for _ in range(n_warm):
        next_tok = _next_token(next_tok)
    torch.cuda.synchronize()
    warm_tps = n_warm / (time.perf_counter() - t2)
    warm_hits, warm_misses = _collect_cache_stats(model)

    cold_total = cold_hits + cold_misses
    warm_total = warm_hits + warm_misses
    return {
        "model": model_id,
        "policy": cache_policy,
        "cache_bias": cache_bias,
        "phase1_tps": round(phase1_tps, 2),
        "cold_tps": round(cold_tps, 2),
        "warm_tps": round(warm_tps, 2),
        "phase1_hit_rate": round(p1_hits / max(1, p1_hits + p1_misses), 4),
        "cold_hit_rate": round(cold_hits / max(1, cold_total), 4),
        "warm_hit_rate": round(warm_hits / max(1, warm_total), 4),
        "n_phase1": n_phase1,
        "n_cold": n_cold,
        "n_warm": n_warm,
    }


def _print_domain_shift_result(result: dict):
    sep = "─" * 44
    print(f"\nDomain-Shift Benchmark — {result['model']}")
    print(f"Policy: {result['policy']}  cache_bias: {result['cache_bias']}")
    print(f"Phases: {result['n_phase1']} EN-tech warm-up → {result['n_cold']} RU cold → {result['n_warm']} RU warm")
    print(sep)
    print(f"  {'Phase':<22} {'tok/s':>8} {'hit%':>8}")
    print(sep)
    print(f"  {'EN-tech (warm-up)':<22} {result['phase1_tps']:>8.2f} {result['phase1_hit_rate'] * 100:>7.1f}%")
    print(f"  {'RU (cold, post-shift)':<22} {result['cold_tps']:>8.2f} {result['cold_hit_rate'] * 100:>7.1f}%")
    print(f"  {'RU (warm, re-adapted)':<22} {result['warm_tps']:>8.2f} {result['warm_hit_rate'] * 100:>7.1f}%")
    print(sep)
    delta = result["warm_tps"] - result["cold_tps"]
    print(f"  cold→warm recovery: {delta:+.2f} tok/s")
    print(sep)


def _print_result(result: dict, cache_capacity: int | None = None):
    sep = "\u2500" * 38
    cap_str = f"{cache_capacity} slots" if cache_capacity else "auto"
    fate_mode = "adaptive FATE+temporal" if result.get("adaptive_fate") else "FATE only"
    print(f"\nModel: {result['model']} | Policy: {result['policy']} | Cache: {cap_str} | FATE: {fate_mode}")
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


def _run_trace(args) -> None:
    """Load model, run 5 warmup + 10 torch.profiler tokens, export Chrome trace."""
    import subprocess

    from transformers import AutoTokenizer

    from tinyserve.offload import load_and_offload
    from tinyserve.offloaded_model import reset_temporal_routing

    n_warmup = 5
    n_profile = 10
    trace_path = "/tmp/gpt_oss_trace.json"
    model_id = args.model
    prompt = args.prompt
    cap = None if args.no_cache else args.cache_capacity
    policy = args.cache_policy
    fp8 = not args.no_fp8
    adaptive = not args.no_adaptive_fate

    print(f"[trace] Loading {model_id}…")
    model = load_and_offload(
        model_id,
        device="cuda",
        cache_capacity=cap if cap is not None else 0,
        cache_policy=policy,
        fp8=fp8,
        adaptive_fate=adaptive,
    )
    reset_temporal_routing()
    tok = AutoTokenizer.from_pretrained(model_id)
    input_ids = tok.encode(prompt, return_tensors="pt").to("cuda")

    # Detect attention implementation from the loaded HF model layers.
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is not None and len(layers) > 0:
        first_layer = layers[0]
        attn = getattr(first_layer, "self_attn", None) or getattr(first_layer, "attention", None)
        if attn is not None:
            print(f"[trace] Attention class: {type(attn).__name__}")
            print(f"[trace] Attention module: {type(attn).__module__}")
        else:
            print("[trace] Attention: not found on first layer")
    else:
        print("[trace] Attention: model has no .layers attribute")

    # Check subprocess fallback for _attn_implementation config attribute.
    result = subprocess.run(
        [
            "python3",
            "-c",
            f"from transformers import AutoConfig; "
            f"cfg = AutoConfig.from_pretrained('{model_id}'); "
            f"inner = getattr(cfg, 'text_config', cfg); "
            f"print('_attn_implementation:', getattr(inner, '_attn_implementation', 'not set'))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        print(f"[trace] {result.stdout.strip()}")
    else:
        print(f"[trace] _attn_implementation probe failed: {result.stderr.strip()[:200]}")

    # Prefill
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # 5 warmup tokens (outside profiler)
    print(f"[trace] Warming up ({n_warmup} tokens, outside profiler)…")
    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(input_ids=next_token, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    # Profile 10 tokens
    print(f"[trace] Profiling {n_profile} tokens…")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(n_profile):
            with torch.no_grad():
                out = model(input_ids=next_token, use_cache=False)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"[trace] Chrome trace exported → {trace_path}")

    print("\n[trace] Top-20 CUDA kernels by cuda_time_total:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def _run_profile(args) -> None:
    """Load model, run 20 warmup + N profiled tokens, print phase report."""
    from transformers import AutoTokenizer

    from tinyserve.offload import load_and_offload
    from tinyserve.offloaded_model import reset_temporal_routing
    from tinyserve.profiler import OffloadProfiler

    n_warmup = 20
    n_measure = getattr(args, "measure", 40)
    model_id = args.model
    prompt = args.prompt
    cap = None if args.no_cache else args.cache_capacity
    policy = args.cache_policy
    fp8 = not args.no_fp8
    adaptive = not args.no_adaptive_fate

    print(f"[profile] Loading {model_id}…")
    model = load_and_offload(
        model_id,
        device="cuda",
        cache_capacity=cap if cap is not None else 0,
        cache_policy=policy,
        fp8=fp8,
        adaptive_fate=adaptive,
    )
    reset_temporal_routing()
    tok = AutoTokenizer.from_pretrained(model_id)
    input_ids = tok.encode(prompt, return_tensors="pt").to("cuda")

    # Prefill
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Warmup (20 tokens, not measured)
    print(f"[profile] Warming up ({n_warmup} tokens)…")
    for _ in range(n_warmup):
        with torch.no_grad():
            out = model(input_ids=next_token, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    _reset_cache_stats(model)
    reset_temporal_routing()

    # Inject profiler into all pipelines
    device = torch.device("cuda")
    profiler = OffloadProfiler(device, enabled=True, mode="cpu")
    for p in getattr(model, "_offload_pipelines", []):
        p.profiler = profiler

    # Profiled run
    print(f"[profile] Profiling {n_measure} tokens…")
    torch.cuda.synchronize()
    for _ in range(n_measure):
        profiler.begin_token()
        with torch.no_grad():
            out = model(input_ids=next_token, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        profiler.end_token()

    # Collect cache hit/miss totals from the cache objects (authoritative counters).
    total_hits, total_misses = _collect_cache_stats(model)
    profiler.total_hits = total_hits
    profiler.total_misses = total_misses

    print(profiler.report())


def main():
    parser = argparse.ArgumentParser(description="Benchmark offloaded MoE model decode")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="HuggingFace model id or local path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--measure", type=int, default=60)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8 expert compression (default: FP8 on)")
    parser.add_argument("--cache-capacity", type=int, default=None)
    parser.add_argument("--cache-policy", default="lfru", choices=list(_POLICIES))
    parser.add_argument(
        "--cache-bias", type=float, default=0.0, help="Logit bias for GPU-resident experts (0=off, try 0.1-0.3)"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run LRU then SLRU back-to-back and print comparison table"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--context-scaling", action="store_true",
        help="Benchmark prefill vs decode timing across context lengths"
    )
    parser.add_argument(
        "--both-families", action="store_true", help="Run GPT-OSS-20B then Qwen3.5-35B-A3B and print side-by-side"
    )
    parser.add_argument(
        "--domain-shift", action="store_true", help="Run domain-shift benchmark (EN-tech warm-up → RU-literature shift)"
    )
    parser.add_argument(
        "--compare-policies", action="store_true", help="Run domain-shift benchmark across all policies and compare"
    )
    parser.add_argument(
        "--same-language",
        action="store_true",
        help="Use same prompt for all phases (no shift) — isolates frequency effects",
    )
    parser.add_argument(
        "--sweep-bias", action="store_true", help="Sweep cache_bias values with lfru to find optimal routing bias"
    )
    parser.add_argument(
        "--fate-diagnostic",
        action="store_true",
        help="Print per-layer FATE prediction accuracy table (40-token run, lfru)",
    )
    parser.add_argument(
        "--no-adaptive-fate",
        action="store_true",
        help="Disable adaptive FATE+temporal (default: on). When disabled, "
        "FATE structural prediction is used for all layers.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiling mode: 20 warmup tokens, 40 profiled tokens, print per-phase breakdown",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Run torch.profiler trace: 5 warmup + 10 profiled tokens, "
        "export Chrome trace to /tmp/gpt_oss_trace.json and print "
        "top-20 CUDA kernel summary",
    )
    parser.add_argument(
        "--capacity", type=int, default=None, help="Alias for --cache-capacity (for profiling convenience)"
    )
    parser.add_argument(
        "--static-kv", type=int, default=0,
        help="Pre-allocate StaticKVCache with this max_seq_len (0=use DynamicCache). Only for --context-scaling."
    )
    args = parser.parse_args()

    # --capacity is a shorthand for --cache-capacity
    if args.capacity is not None and args.cache_capacity is None:
        args.cache_capacity = args.capacity

    if args.context_scaling:
        results = run_context_scaling(
            model_id=args.model,
            gen_tokens=args.measure if args.measure != 60 else 20,
            cache_policy=args.cache_policy,
            fp8=not args.no_fp8,
            adaptive_fate=not args.no_adaptive_fate,
            static_kv=args.static_kv,
        )
        sep = "─" * 72
        print(f"\nContext Scaling — {args.model}")
        print(sep)
        print(f"  {'ctx':>6}  {'prefill':>10}  {'decode':>10}  {'decode':>10}  {'total':>10}  {'decode':>8}")
        print(f"  {'':>6}  {'(ms)':>10}  {'(ms)':>10}  {'(tok/s)':>10}  {'(tok/s)':>10}  {'HR%':>8}")
        print(sep)
        for r in results:
            print(
                f"  {r['ctx']:>6}  {r['prefill_ms']:>10.1f}  {r['decode_ms']:>10.1f}"
                f"  {r['decode_tps']:>10.1f}  {r['total_tps']:>10.1f}"
                f"  {r['decode_hit_rate'] * 100:>7.1f}%"
            )
        print(sep)
        if args.json:
            print(json.dumps(results, indent=2))
        return

    if args.both_families:
        results = []
        for name, mid in _FAMILY_MODELS.items():
            r = run_benchmark(
                model_id=mid,
                prompt=args.prompt,
                n_warmup=args.warmup,
                n_measure=args.measure,
                no_cache=args.no_cache,
                fp8=not args.no_fp8,
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
            hr = f"{r['cache_hit_rate'] * 100:.1f}%" if "cache_hit_rate" in r else "—"
            print(f"  {name:<20}{r['tok_s']:>8}{r['ms_per_tok']:>8}{hr:>8}")
        print(sep)
        return

    if args.domain_shift or args.compare_policies or args.sweep_bias:
        # Load model ONCE — policy is swapped in-place between runs (no reload).
        from transformers import AutoTokenizer

        from tinyserve.offload import load_and_offload

        print(f"Loading {args.model} (once for all policy runs)…")
        model = load_and_offload(
            args.model,
            device="cuda",
            cache_policy=args.cache_policy,
            cache_bias=args.cache_bias,
            fp8=not args.no_fp8,
        )
        tok = AutoTokenizer.from_pretrained(args.model)

    if args.domain_shift:
        result = run_domain_shift_benchmark(
            model,
            tok,
            model_id=args.model,
            cache_policy=args.cache_policy,
            cache_bias=args.cache_bias,
            same_language=args.same_language,
        )
        if args.json:
            print(json.dumps(result))
        else:
            _print_domain_shift_result(result)
        return

    if args.compare_policies:
        # Run across lru / lfru / ls — model already loaded above.
        sep = "─" * 62
        shift_label = (
            "EN-tech (no shift, all phases same)"
            if args.same_language
            else "EN-tech warm-up → Russian-literature shift"
        )
        print(f"\nPolicy comparison — {shift_label} ({args.model})")
        print(sep)
        print(
            f"  {'Policy':<8} {'P1 tok/s':>10} {'Cold tok/s':>10} {'Warm tok/s':>10} {'Cold hit%':>10} {'Warm hit%':>10}"
        )
        print(sep)
        for policy in _POLICIES:
            r = run_domain_shift_benchmark(
                model,
                tok,
                model_id=args.model,
                cache_policy=policy,
                cache_bias=args.cache_bias,
                same_language=args.same_language,
            )
            if args.json:
                print(json.dumps(r))
            else:
                print(
                    f"  {policy:<8} {r['phase1_tps']:>10.2f} {r['cold_tps']:>10.2f} "
                    f"{r['warm_tps']:>10.2f} {r['cold_hit_rate'] * 100:>9.1f}% "
                    f"{r['warm_hit_rate'] * 100:>9.1f}%"
                )
        print(sep)
        return

    if args.sweep_bias:
        # Sweep cache_bias with lfru — model already loaded above.
        sep = "─" * 62
        biases = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
        shift_label = "same-language" if args.same_language else "domain-shift"
        print(f"\nCache-bias sweep — lfru, {shift_label} ({args.model})")
        print(sep)
        print(
            f"  {'bias':<8} {'P1 tok/s':>10} {'Cold tok/s':>10} {'Warm tok/s':>10} {'Cold hit%':>10} {'Warm hit%':>10}"
        )
        print(sep)
        for bias in biases:
            r = run_domain_shift_benchmark(
                model,
                tok,
                model_id=args.model,
                cache_policy="lfru",
                cache_bias=bias,
                same_language=args.same_language,
            )
            if args.json:
                print(json.dumps(r))
            else:
                print(
                    f"  {bias:<8.2f} {r['phase1_tps']:>10.2f} {r['cold_tps']:>10.2f} "
                    f"{r['warm_tps']:>10.2f} {r['cold_hit_rate'] * 100:>9.1f}% "
                    f"{r['warm_hit_rate'] * 100:>9.1f}%"
                )
        print(sep)
        return

    if args.fate_diagnostic:
        from transformers import AutoTokenizer

        from tinyserve.offload import load_and_offload
        from tinyserve.offloaded_model import get_fate_accuracy_by_layer, reset_fate_stats, reset_temporal_routing

        n_diag = 40
        policy = args.cache_policy if args.cache_policy else "lfru"
        adaptive = not args.no_adaptive_fate
        mode_label = "adaptive FATE+temporal" if adaptive else "FATE only"
        print(f"Loading {args.model} for FATE diagnostic (policy={policy}, {n_diag} tokens, mode={mode_label})…")
        model = load_and_offload(
            args.model,
            device="cuda",
            cache_capacity=args.cache_capacity if args.cache_capacity is not None else 0,
            cache_policy=policy,
            fp8=not args.no_fp8,
            adaptive_fate=adaptive,
        )
        tok = AutoTokenizer.from_pretrained(args.model)
        input_ids = tok.encode(args.prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        reset_fate_stats()
        reset_temporal_routing()
        for _ in range(n_diag):
            with torch.no_grad():
                out = model(input_ids=next_token, use_cache=False)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        stats = get_fate_accuracy_by_layer()
        if not stats:
            print("No FATE stats collected. FATE may not be active for this model/policy.")
            return

        layers = sorted(stats.keys())
        sep = "─" * 46
        print(f"\nFATE Per-Layer Accuracy — {args.model} ({n_diag} tokens, policy={policy}, mode={mode_label})")
        print(sep)
        print(f"  {'Layer':>6}  {'Predictions':>12}  {'Hits':>6}  {'Accuracy':>9}")
        print(sep)
        for li in layers:
            s = stats[li]
            print(f"  {li:>6}  {s['predictions']:>12}  {s['hits']:>6}  {s['accuracy'] * 100:>8.1f}%")
        print(sep)

        sorted_by_acc = sorted(stats.items(), key=lambda kv: kv[1]["accuracy"])
        worst5 = sorted_by_acc[:5]
        best5 = sorted_by_acc[-5:][::-1]

        print("\nWorst 5 layers (lowest FATE accuracy):")
        print(f"  {'Layer':>6}  {'Predictions':>12}  {'Hits':>6}  {'Accuracy':>9}")
        for li, s in worst5:
            print(f"  {li:>6}  {s['predictions']:>12}  {s['hits']:>6}  {s['accuracy'] * 100:>8.1f}%")

        print("\nBest 5 layers (highest FATE accuracy):")
        print(f"  {'Layer':>6}  {'Predictions':>12}  {'Hits':>6}  {'Accuracy':>9}")
        for li, s in best5:
            print(f"  {li:>6}  {s['predictions']:>12}  {s['hits']:>6}  {s['accuracy'] * 100:>8.1f}%")
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
                no_cache=args.no_cache,
                fp8=not args.no_fp8,
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
                lru_hr = f"{lru['cache_hit_rate'] * 100:.1f}%"
                slru_hr = f"{slru['cache_hit_rate'] * 100:.1f}%"
                print(f"  {'hit rate':<16}{lru_hr:>10}{slru_hr:>10}")
            print(sep)
        return

    if args.profile:
        _run_profile(args)
        return

    if args.trace:
        _run_trace(args)
        return

    result = run_benchmark(
        model_id=args.model,
        prompt=args.prompt,
        n_warmup=args.warmup,
        n_measure=args.measure,
        no_cache=args.no_cache,
        fp8=not args.no_fp8,
        cache_capacity=args.cache_capacity,
        cache_policy=args.cache_policy,
        cache_bias=args.cache_bias,
        adaptive_fate=not args.no_adaptive_fate,
    )

    if args.json:
        print(json.dumps(result))
    else:
        _print_result(result, args.cache_capacity)


if __name__ == "__main__":
    main()
