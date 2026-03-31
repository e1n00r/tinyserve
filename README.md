# tinyserve — MoE Expert Offloading for Consumer GPUs

Run Mixture-of-Experts models that don't fit in VRAM on a single NVIDIA GPU.

**9–13 tok/s** on an 8 GB laptop GPU running a 20B MoE model. 95% expert cache hit rate with LFRU eviction. ~7K lines of Python.

## Performance

All numbers from **RTX PRO 2000 8 GB laptop GPU**, GPT-OSS-20B (MXFP4, 24 layers × 32 experts, top_k=4). One model, one GPU — no other hardware benchmarked. Raw logs in [`benchmarks/`](benchmarks/).

| Workload | Hit rate | tok/s |
|---|---|---|
| Sustained code | 93% | 10–12 |
| Sustained math | 93% | 10–11 |
| Sustained creative | 95% | 11–12 |
| Sustained multilingual | 95% | 11–12 |
| Sustained conversation | 95% | 8–12 |
| Domain shift (creative→math) | 94% | — |

Per-layer cache hit rate with LFRU: 79–97% across all 24 layers. LRU starves deep layers to 0–8%; LFRU fixes this structurally.

<details>
<summary>Benchmark methodology and caveats</summary>

- Diverse prompts across 5 domains (code, math, creative, multilingual, conversation)
- 4 prompts × 30 tokens per domain — point estimates, not statistically tight bounds
- Per-prompt tok/s variance is ~15–20% CV
- Cache counters reset between prompts but cache slots retained (warm, not cold)
- Source: `benchmarks/cpu_slotmap_bench_20260331.json`
- HuggingFace `device_map="auto"` baseline: 0.19 tok/s (measured once, no backing file)
- Expert frequency is flat (top 10% = 44% of accesses) — not Zipf-like
- Expert cosine similarity: 0.0006 (fully independent, no delta compression viable)
</details>

## Quick start

```bash
git clone https://github.com/e1n00r/tinyserve.git && cd tinyserve
pip install -e "."
```

```python
from tinyserve import load_and_offload

model = load_and_offload("openai/gpt-oss-20b")
output = model.generate(input_ids, max_new_tokens=100)
```

### CLI

```bash
tinyserve serve --model openai/gpt-oss-20b --port 8000   # OpenAI-compatible API
tinyserve run --model openai/gpt-oss-20b                  # Interactive REPL
tinyserve info --model openai/gpt-oss-20b                 # Model architecture profile
```

## When to use tinyserve

A new MoE model drops on HuggingFace. No GGUF yet, Ollama can't load it, you have 8 GB of VRAM. tinyserve loads directly from safetensors and runs it today.

**If your model already works in Ollama or llama.cpp, use those.** Their C++ inference loop is faster. tinyserve is for models they don't support yet, or when you want readable, hackable Python.

## How it works

1. **Expert store** — Weights in pinned CPU memory. MXFP4 loaded as raw uint8 (no dequantization).
2. **GPU LFRU cache** — Frequency-recency eviction prevents deep-layer starvation. Hit: zero-copy MXFP4 forward via Triton. Miss: CPU expert compute (~2ms via OneDNN).
3. **CPU slot map** — Cache tracking on numpy (zero CUDA overhead). GPU tensor synced lazily.
4. **FATE prefetch** — Current layer predicts next layer's experts. Overlaps with attention.
5. **Batched prefill** — Groups tokens by expert, loads each once. O(num_experts) not O(seq_len × top_k).
6. **Buddy substitution** — On miss, substitute co-activation-similar cached expert (zero stall).

## Supported models

| Model | Params | RAM needed | Status |
|---|---|---|---|
| GPT-OSS-20B | 20B (MXFP4) | ~10 GB | **Benchmarked** |
| Qwen 3.5 MoE 35B | 35B | ~18 GB | Unit tested |
| Mixtral 8x7B | 47B | ~24 GB | Unit tested |
| GPT-OSS-120B | 120B | ~60 GB | Profile only |
| DeepSeek-V3/R1 | 671B | ~350 GB | Profile only |
| + 6 more families | varies | varies | Profile only |

**Status:** "Benchmarked" = real weights, real tokens, throughput measured. "Unit tested" = mock weights. "Profile only" = metadata only.

**Formats:** HuggingFace safetensors (BF16, FP8, MXFP4). GGUF (Q4_K/Q5_K/Q6_K) parsing verified.

## Configuration

```python
model = load_and_offload(
    "openai/gpt-oss-20b",
    cache_capacity=0,              # 0 = auto-size from VRAM
    cache_policy="lfru",           # lru, lfru, slru, lfu, fifo, ls, dali
    max_seq_len=4096,              # static KV cache (0 = dynamic)
    gpu_memory_utilization=0.90,
    buddy_table_path="benchmarks/buddy_tables_gptoss20b.json",
)
```

## Limitations

- NVIDIA only (CUDA streams, Triton PTX)
- Single GPU, batch size 1 decode only
- One model benchmarked — all performance claims are GPT-OSS-20B on one GPU
- ~36% of theoretical ceiling — Python overhead dominates; C++ rewrite needed for >20 tok/s
- GGUF parsing works but end-to-end generation not tested

<details>
<summary>What we have NOT measured</summary>

- No Ollama/llama.cpp comparison (they don't support GPT-OSS-20B)
- No multi-user or batch inference benchmarks
- No H100/A100 numbers
- No confidence intervals (4 prompts per workload)
- HF baseline (0.19 tok/s) has no backing benchmark file
</details>

<details>
<summary>What we tried and ruled out</summary>

| Technique | Result |
|---|---|
| D2-MoE delta compression | Expert cosine similarity = 0.0006 — not viable |
| Cache bias routing (0.0–3.0) | No effect on GPT-OSS-20B |
| Cython hot path | 3.4x microbench, 0% end-to-end |
| GPU INT4 on 8GB | OOMs (conversion cache exceeds VRAM) |
| Expert deferral | Produces garbage output |
| FlexAttention default | pytorch #155065, 3–67x VRAM overhead |
</details>

<details>
<summary>Performance ceiling analysis</summary>

At 10–13 tok/s, we're at ~36% of the realistic ceiling (~32 tok/s):
- ~60% Python interpreter overhead (layer loop, dict ops, torch dispatch)
- ~15% CUDA synchronization and kernel launch
- ~15% PCIe miss transfers + CPU fallback compute
- ~10% HF generate framework tax

The algorithmic optimization space is exhausted for this model/hardware. Remaining gains require a C++ forward loop.
</details>

## Benchmarking

```bash
python -m scripts.cache_benchmark                    # Diverse workload benchmark
python scripts/comprehensive_bench.py                # 7-policy comparison
python scripts/benchmark.py --context-scaling        # Prefill vs decode
python scripts/calibrate_buddies.py                  # Buddy co-activation profiling
python scripts/expert_similarity.py                  # D2-MoE feasibility check
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ --ignore=tests/test_hf_models.py -x -q   # ~340 tests
```

## License

MIT
