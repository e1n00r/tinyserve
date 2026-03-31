"""E2E test: demand-driven VRAM rebalancing — KV overflow triggers expert eviction."""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG = "benchmarks/dynamic_rebalance_e2e_20260331.txt"
def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

def main():
    with open(LOG, "w"): pass
    log("=== Dynamic VRAM Rebalancing E2E Test ===\n")

    from transformers import AutoTokenizer
    from tinyserve.offload import load_and_offload

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    # max_seq_len=64: very small, forces overflow on any prompt >44 tokens
    # (64 - 20 gen tokens = 44 tokens of prompt before overflow)
    model = load_and_offload("openai/gpt-oss-20b", attn_implementation="sdpa", max_seq_len=64)

    cache = next(p.cache for p in model._offload_pipelines if p.cache)
    kv = model._kv_cache
    budget = model._vram_budget

    initial_expert = cache.capacity
    initial_kv = kv.max_seq_len

    log(f"Initial: {initial_expert} expert slots, {initial_kv} KV tokens")
    log(f"Tokens per expert slot: {budget.tokens_per_expert_slot}")
    log(f"Budget: min_expert={budget.min_expert_capacity}, max_expert={budget.max_expert_capacity}")

    # Phase 1: Short prompt — fits in KV, no rebalance
    log(f"\n--- Phase 1: Short prompt (fits in KV) ---")
    for li in range(len(kv._seq_lens)):
        kv._seq_lens[li] = 0

    inp = tok("Hello world", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=10, do_sample=False, past_key_values=kv)
    log(f"  KV: {max(kv._seq_lens)}/{kv.max_seq_len}, Experts: {cache.capacity}, Rebalances: {budget._rebalance_count}")

    # Phase 2: Medium prompt — triggers overflow, rebalancing kicks in
    log(f"\n--- Phase 2: Medium prompt (overflow → rebalance) ---")
    for li in range(len(kv._seq_lens)):
        kv._seq_lens[li] = 0

    medium = "Explain the theory of relativity and its implications for modern physics and cosmology in detail."
    inp = tok(medium, return_tensors="pt").to("cuda")
    prompt_len = inp["input_ids"].shape[1]
    log(f"  Prompt tokens: {prompt_len}, KV max: {kv.max_seq_len}")

    try:
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=20, do_sample=False, past_key_values=kv)
        log(f"  KV: {max(kv._seq_lens)}/{kv.max_seq_len}, Experts: {cache.capacity}, Rebalances: {budget._rebalance_count}")
    except Exception as e:
        log(f"  ERROR: {str(e)[:100]}")

    # Phase 3: Even longer prompt — more rebalancing
    log(f"\n--- Phase 3: Long prompt (more rebalancing) ---")
    for li in range(len(kv._seq_lens)):
        kv._seq_lens[li] = 0

    long_text = "The history of artificial intelligence began in antiquity. " * 20
    inp = tok(long_text, return_tensors="pt", truncation=True, max_length=200).to("cuda")
    prompt_len = inp["input_ids"].shape[1]
    log(f"  Prompt tokens: {prompt_len}, KV max: {kv.max_seq_len}")

    try:
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=20, do_sample=False, past_key_values=kv)
        log(f"  KV: {max(kv._seq_lens)}/{kv.max_seq_len}, Experts: {cache.capacity}, Rebalances: {budget._rebalance_count}")
    except Exception as e:
        log(f"  ERROR: {str(e)[:100]}")

    post_expert = cache.capacity
    post_kv = kv.max_seq_len

    # Phase 4: Release KV, restore experts
    log(f"\n--- Phase 4: Release KV, restore experts ---")
    budget.release_kv()
    log(f"  Experts: {cache.capacity} (was {post_expert})")

    # Phase 5: Fresh generation with restored cache
    log(f"\n--- Phase 5: Fresh request with restored cache ---")
    for li in range(len(kv._seq_lens)):
        kv._seq_lens[li] = 0
    cache.reset_stats()

    inp = tok("What is 2+2?", return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=10, do_sample=False, past_key_values=kv)
    elapsed = time.perf_counter() - t0
    n = out.shape[1] - inp["input_ids"].shape[1]
    hr = cache.hits / (cache.hits + cache.misses) if (cache.hits + cache.misses) > 0 else 0
    log(f"  {n} tokens, {n/elapsed:.1f} tok/s, HR={hr:.1%}")

    # Summary
    log(f"\n{'='*55}")
    log(f"SUMMARY")
    log(f"  Initial:       {initial_expert} experts, {initial_kv} KV")
    log(f"  After growth:  {post_expert} experts, {post_kv} KV")
    log(f"  After release: {cache.capacity} experts")
    log(f"  Total rebalances: {budget._rebalance_count}")
    if budget._rebalance_count > 0:
        log(f"  PASS: Dynamic rebalancing worked!")
    else:
        log(f"  FAIL: No rebalancing triggered")
    log("Done.")

if __name__ == "__main__":
    main()
