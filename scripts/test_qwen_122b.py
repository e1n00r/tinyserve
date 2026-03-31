"""Qwen 3.5-122B GGUF Q4_K — real inference test."""
import sys, os, time, torch, traceback, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG = "benchmarks/qwen_122b_test_20260331.txt"
def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f: f.write(m + "\n")

def main():
    with open(LOG, "w"): pass
    log("=== Qwen 3.5-122B GGUF Q4_K — Real Inference Test ===\n")

    import glob
    shards = sorted(glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.5-122B-A10B-GGUF/snapshots/*/Q4_K_S/*.gguf")
    ))
    log(f"Shards: {len(shards)}, total {sum(os.path.getsize(s) for s in shards)/1e9:.1f} GB")
    
    import psutil
    log(f"RAM: {psutil.virtual_memory().available/1e9:.1f} GB available")
    log(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB free")

    from tinyserve.gguf_loader import load_from_gguf

    log("\nLoading model (disk_offload=True, this will take a while)...")
    t0 = time.time()
    try:
        model = load_from_gguf(
            shards[0],
            device="cuda",
            model_id="Qwen/Qwen3.5-122B-A10B",
            disk_offload=True,
            ram_cache_gb=32,
        )
        load_time = time.time() - t0
        log(f"Loaded in {load_time:.1f}s")

        if hasattr(model, '_offload_pipelines') and model._offload_pipelines:
            p = model._offload_pipelines[0]
            cache = p.cache
            log(f"Expert cache: {cache.capacity if cache else 'None'} slots")
            log(f"CPU-on-miss: {p.cpu_on_miss}")
            log(f"Store: {p.store.num_layers} layers × {p.store.num_experts} experts")
        
        log(f"RAM after load: {psutil.virtual_memory().available/1e9:.1f} GB")
        log(f"GPU after load: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB free")

        # Generation test
        log("\n--- Generation Test ---")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B")

        prompts = [
            ("Short", "Hello, how are you?"),
            ("English", "What is the capital of France?"),
            ("Code", "Write a Python hello world."),
        ]

        # Warmup
        inp = tok("Hi", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=3, do_sample=False)
        log("Warmup done.")

        if cache:
            cache.reset_stats()

        for name, prompt in prompts:
            if cache:
                cache.reset_stats()
            inp = tok(prompt, return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=20, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n = out.shape[1] - inp["input_ids"].shape[1]
            text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            hr = 0
            if cache and (cache.hits + cache.misses) > 0:
                hr = cache.hits / (cache.hits + cache.misses)
            log(f"  {name}: {n} tok in {elapsed:.1f}s = {n/elapsed:.2f} tok/s  HR={hr:.1%}")
            log(f"    Output: {text[:100]}")

        log("\nDone.")

    except Exception as e:
        elapsed = time.time() - t0
        log(f"\nFailed after {elapsed:.1f}s")
        log(f"{type(e).__name__}: {str(e)[:300]}")
        log(traceback.format_exc()[-800:])

if __name__ == "__main__":
    main()
