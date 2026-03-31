"""Comprehensive cache benchmark — subprocess per test to avoid GPU memory leaks.

Each test runs in a fresh subprocess with clean GPU state.
"""
import json
import os
import subprocess
import sys
import time

LOG = "benchmarks/comprehensive_20260326.txt"
JSON_OUT = "benchmarks/comprehensive_20260326.json"

def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f:
        f.write(m + "\n")

def run_subprocess(code, timeout=300):
    """Run Python code in a subprocess, return stdout."""
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=timeout,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
    )
    if result.returncode != 0:
        return {"error": result.stderr[-300:]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"bad JSON: {result.stdout[-200:]}", "stderr": result.stderr[-200:]}

# ═══════════════════════════════════════════════════════════════
# PART 1: Policy comparison (one subprocess per policy)
# ═══════════════════════════════════════════════════════════════
POLICY_CODE = '''
import json, time, torch, sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from tinyserve.offload import load_and_offload
from scripts.prompts import COLD_START, CODE_PROMPTS, CREATIVE_PROMPTS, MULTILINGUAL_PROMPTS

POLICY = "{policy}"
tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = load_and_offload("openai/gpt-oss-20b", cache_policy=POLICY, attn_implementation="sdpa")
cache = next(p.cache for p in model._offload_pipelines if p.cache is not None)

prompts = COLD_START[:4] + CODE_PROMPTS[:2] + CREATIVE_PROMPTS[:2] + MULTILINGUAL_PROMPTS[:2]
inp = tok("Hello", return_tensors="pt").to("cuda")
with torch.inference_mode():
    model.generate(**inp, max_new_tokens=3, do_sample=False)

cache.reset_stats()
total_tok, t0 = 0, time.perf_counter()
for prompt in prompts:
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=25, do_sample=False)
    total_tok += out.shape[1] - inp["input_ids"].shape[1]

elapsed = time.perf_counter() - t0
hr = cache.hits / (cache.hits + cache.misses) if (cache.hits + cache.misses) > 0 else 0
tps = total_tok / elapsed
layer_stats = cache.get_layer_stats()
layer_hrs = {{li: s["hit_rate"] for li, s in layer_stats.items()}}
deep_hr = sum(layer_hrs.get(li, 0) for li in range(18, 24)) / 6
worst = min(layer_hrs.items(), key=lambda x: x[1]) if layer_hrs else (0, 0)

print(json.dumps({{
    "hr": round(hr, 4), "tps": round(tps, 1),
    "deep_hr": round(deep_hr, 4), "worst": [worst[0], round(worst[1], 4)],
    "per_layer": {{str(k): round(v, 4) for k, v in layer_hrs.items()}}
}}))
'''

# ═══════════════════════════════════════════════════════════════
# PART 2: FATE accuracy
# ═══════════════════════════════════════════════════════════════
FATE_CODE = '''
import json, torch, sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from tinyserve.offload import load_and_offload
from tinyserve.offloaded_model import get_fate_accuracy_by_layer, reset_fate_stats, reset_temporal_routing
from scripts.prompts import COLD_START, CODE_PROMPTS, MATH_PROMPTS

ADAPTIVE = {adaptive}
tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = load_and_offload("openai/gpt-oss-20b", adaptive_fate=ADAPTIVE, attn_implementation="sdpa")

inp = tok("Hello", return_tensors="pt").to("cuda")
with torch.inference_mode():
    model.generate(**inp, max_new_tokens=3, do_sample=False)

reset_fate_stats()
reset_temporal_routing()

for prompt in COLD_START[:4] + CODE_PROMPTS[:2] + MATH_PROMPTS[:2]:
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=128).to("cuda")
    with torch.inference_mode():
        model.generate(**inp, max_new_tokens=20, do_sample=False)

stats = get_fate_accuracy_by_layer()
if stats:
    per_layer = {{str(li): round(s["accuracy"], 4) for li, s in stats.items()}}
    accs = [s["accuracy"] for s in stats.values()]
    avg = sum(accs) / len(accs)
    worst = min(stats.items(), key=lambda x: x[1]["accuracy"])
    print(json.dumps({{"per_layer": per_layer, "avg": round(avg, 4),
                        "worst_layer": worst[0], "worst_acc": round(worst[1]["accuracy"], 4)}}))
else:
    print(json.dumps({{"error": "no stats"}}))
'''

# ═══════════════════════════════════════════════════════════════
# PART 3: Qwen GGUF
# ═══════════════════════════════════════════════════════════════
QWEN_CODE = '''
import json, time, torch, sys, glob, os
sys.path.insert(0, ".")

gguf_dir = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.5-122B-A10B-GGUF")
shards = sorted(glob.glob(os.path.join(gguf_dir, "snapshots/*/Q4_K_S/*.gguf")))
if not shards:
    print(json.dumps({"error": "not found"}))
    sys.exit(0)

from tinyserve.gguf_reader import open_gguf
reader = open_gguf(shards[0])
meta = reader.metadata if hasattr(reader, "metadata") else {}
n_layers = meta.get("llama.block_count", meta.get("qwen2moe.block_count", "?"))
n_experts = meta.get("llama.expert_count", meta.get("qwen2moe.expert_count", "?"))
tensors = reader.tensor_infos if hasattr(reader, "tensor_infos") else {}
print(json.dumps({
    "status": "parsed",
    "n_layers": n_layers,
    "n_experts": n_experts,
    "n_tensors": len(tensors),
    "shards": len(shards),
    "shard_0": os.path.basename(shards[0]),
}))
'''

if __name__ == "__main__":
    with open(LOG, "w") as f:
        f.write(f"=== Comprehensive MoE Benchmark === {time.strftime('%Y-%m-%d %H:%M')}\n\n")

    results = {}

    # Part 1: Policy comparison
    log("PART 1: Policy Comparison (10 diverse prompts, 25 gen tokens each)")
    log(f"{'Policy':<8} {'HR%':>6} {'tok/s':>7} {'Deep(18-23)':>12} {'Worst':>10}")
    log("-" * 50)

    policies = ["lru", "lfru", "slru", "lfu", "fifo", "ls", "dali"]
    results["policies"] = {}
    for policy in policies:
        code = POLICY_CODE.format(policy=policy)
        log(f"  Running {policy}...", )
        r = run_subprocess(code, timeout=300)
        results["policies"][policy] = r
        if "error" in r:
            log(f"{policy:<8} {'ERR':>6}  {r['error'][:60]}")
        else:
            log(f"{policy:<8} {r['hr']:>5.1%} {r['tps']:>6.1f} {r['deep_hr']:>10.0%}  L{r['worst'][0]}={r['worst'][1]:.0%}")

    # Part 2: FATE accuracy
    log("\nPART 2: FATE Prediction Accuracy (diverse prompts)")
    results["fate"] = {}
    for name, adaptive in [("temporal+FATE", "True"), ("FATE_only", "False")]:
        code = FATE_CODE.format(adaptive=adaptive)
        log(f"  Running {name}...")
        r = run_subprocess(code, timeout=300)
        results["fate"][name] = r
        if "error" not in r:
            log(f"  {name}: avg={r['avg']:.1%}  worst=L{r['worst_layer']}={r['worst_acc']:.1%}")
        else:
            log(f"  {name}: FAILED — {r['error'][:100]}")

    # Part 3: Qwen GGUF parse test
    log("\nPART 3: Qwen 3.5-122B GGUF Q4_K Parse Test")
    r = run_subprocess(QWEN_CODE, timeout=120)
    results["qwen_gguf"] = r
    if "error" not in r:
        log(f"  Parsed: {r['n_layers']} layers, {r['n_experts']} experts, {r['n_tensors']} tensors, {r['shards']} shards")
    else:
        log(f"  FAILED: {r['error'][:200]}")

    with open(JSON_OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\nResults saved to {JSON_OUT}")
    log(f"Done: {time.strftime('%H:%M:%S')}")
