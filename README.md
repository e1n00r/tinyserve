# tinyserve — MoE Expert Offloading for Consumer GPUs

Run Mixture-of-Experts models that don't fit in VRAM on a single NVIDIA GPU. Pure Python, no C++ compilation.

tinyserve offloads MoE expert weights to CPU RAM and caches hot experts on the GPU with predictive prefetch. A 20B MoE model (which only activates ~2B parameters per token) needs ~10 GB of CPU RAM and runs on 8 GB of VRAM.

**Realistic expectations:** On an 8 GB laptop GPU, expect **2–7 tok/s for normal chat-length context** (100–500 tokens) and **15 tok/s on short prompts** with warm cache. This is 80× faster than HuggingFace `device_map="auto"` (0.19 tok/s) but slower than llama.cpp for models it supports.

## Who this is for

A new MoE model drops on HuggingFace. There's no GGUF quantization yet. Ollama can't load it. You have a laptop with an 8 GB GPU and you want to try it *today*, not next week when someone posts a GGUF.

tinyserve loads directly from HuggingFace safetensors. If the model is on the Hub and uses a standard MoE architecture, you can probably run it.

**If your model already works in Ollama or llama.cpp, use those.** Their C++ inference loop is faster. tinyserve is for models they don't support yet, or for when you want to read and modify the inference code yourself (~3K lines of Python, no compiled extensions).

## Performance (measured)

All numbers from an RTX PRO 2000 Blackwell 8 GB **laptop** GPU running GPT-OSS-20B (MXFP4, 238 expert cache slots). Raw benchmark logs in [`benchmarks/`](benchmarks/).

| Context length | tok/s | Cache hit rate | Source file |
|---|---|---|---|
| 10 tokens | 15.2 | 90% | `bench_micro_opt_20260326.txt` |
| 50 tokens | 8.9 | 92% | same |
| 100 tokens | 6.7 | 95% | same |
| 500 tokens | 2.4 | 99% | same |
| 1,000 tokens | 1.3 | 99% | same |
| 2,000 tokens | 0.7 | 100% | same |
| 3,000 tokens | 0.5 | 100% | same |

**Baseline:** HuggingFace `device_map="auto"` on the same hardware: 0.19 tok/s.

**Why throughput drops with context:** Attention is O(n) per decoded token (scanning all KV entries). At 1K+ context, attention compute dominates over expert loading. This affects all transformer models on consumer hardware. Paged attention and fused kernels are on the roadmap.

**RTX PRO 2000 vs RTX 4060 8 GB:** Both are 8 GB cards. The 4060 desktop has higher memory bandwidth (272 vs ~256 GB/s). Expect similar or slightly better numbers. If you benchmark, please open an issue with results.

**What we have NOT measured:**
- No Ollama or llama.cpp comparison on the same GPU (those tools don't support GPT-OSS-20B natively)
- Only GPT-OSS-20B has been benchmarked end-to-end
- GGUF Q4_K loader passes unit tests but has not been tested on real model files

## Quick start

```bash
git clone https://github.com/e1n00r/tinyserve.git
cd tinyserve
pip install -e "."            # or: pip install -e ".[server]" for HTTP serving
```

> **Note:** On Ubuntu 24.04+, use a venv (`python3 -m venv .venv && source .venv/bin/activate`) or add `--break-system-packages`.

```python
from tinyserve import load_and_offload

model = load_and_offload("openai/gpt-oss-20b")
# That's it — generates like any HF model
output = model.generate(input_ids, max_new_tokens=100)
```

For models already loaded via HuggingFace:

```python
from tinyserve import offload_model

model = offload_model(hf_model, device="cuda")
```

### CLI

```bash
# Start OpenAI-compatible HTTP server
tinyserve serve --model openai/gpt-oss-20b --port 8000

# Interactive generation REPL
tinyserve run --model openai/gpt-oss-20b

# Print model profile (expert layout, routing, sizes)
tinyserve info --model openai/gpt-oss-20b
```

All commands are also available via `python -m tinyserve <command>`.

### Serving (OpenAI-compatible API)

```bash
pip install -e ".[server]"
tinyserve serve --model openai/gpt-oss-20b --port 8000
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50,"stream":true}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`. Supports streaming (SSE), request timeouts, concurrent request interleaving via GPU lock, and paged KV cache for multi-request memory sharing.

### Docker

```bash
docker build -t tinyserve .
docker run --gpus all -p 8000:8000 -v hf-cache:/cache/huggingface tinyserve
```

Or with docker compose (set `HF_TOKEN` for gated models):

```bash
HF_TOKEN=hf_... docker compose up
```

## How it works

1. **Expert store** — Expert weights are packed into flat byte buffers on pinned CPU memory. For MXFP4 models (GPT-OSS), weights are loaded directly from safetensors as raw uint8 blocks + scales, bypassing HF dequantization. For other models, weights are optionally compressed to FP8.

2. **GPU LRU cache** — A pre-allocated VRAM tensor holds `capacity` expert slots. On a cache hit, the template module's parameters are set to views of the cache slot (zero-copy). On a miss, the double-buffered pipeline loads from CPU.

3. **Double-buffered H2D pipeline** — Two GPU buffers alternate: while expert _i_ runs its forward pass on the compute stream, expert _i+1_ is being DMA'd from CPU on the transfer stream.

4. **FATE cross-layer prefetch** — Adjacent MoE layers have high gate input cosine similarity. After layer _N_ finishes, its hidden states predict layer _N+1_'s expert selection. Those experts are prefetched on a dedicated stream, overlapping with attention compute.

5. **Adaptive temporal fallback** — After the first token, the system reuses the previous token's routing decisions. Temporal locality matches or exceeds FATE accuracy and eliminates gate computation cost.

6. **Paged KV cache** — Multiple concurrent requests share a pool of fixed-size KV pages (256 tokens/page), enabling memory-efficient multi-request serving without per-request pre-allocation.

## Supported models

| Model family | Size | RAM needed (est.) | Fits 64 GB RAM? | Status |
|---|---|---|---|---|
| GPT-OSS-20B | 20B (MXFP4) | ~10 GB | Yes | **Benchmarked** |
| Qwen 3.5 MoE 35B | 35B | ~18 GB | Yes | Unit tested |
| Mixtral 8x7B | 47B | ~24 GB | Yes | Unit tested |
| GPT-OSS-120B | 120B | ~60 GB | Tight | Profile exists |
| Mixtral 8x22B | 141B | ~70 GB | No (128 GB+) | Unit tested |
| Qwen 3.5 MoE 397B | 397B | ~200 GB | No (workstation) | Unit tested |
| DeepSeek-V3/R1 | 671B | ~350 GB | No (server) | Unit tested |
| Llama 4, Kimi K2, OLMoE, DBRX, Phi-MoE | various | varies | varies | Profile exists |

**Status levels:**
- **Benchmarked:** Real weights loaded, tokens generated, throughput measured.
- **Unit tested:** Offloading pipeline runs on mocked weights shaped like that architecture. Real weights NOT loaded.
- **Profile exists:** Model registry has architecture metadata. No code run for that model.

**Weight formats:** HuggingFace safetensors (BF16, FP8, MXFP4). GGUF (Q4_K/Q5_K/Q6_K) loader implemented and unit-tested but not tested on real GGUF files.

**Test suite:** 318 tests. CI runs on every push.

## Configuration

```python
from tinyserve import load_and_offload

model = load_and_offload(
    "openai/gpt-oss-20b",
    cache_capacity=0,              # 0 = auto-size from available VRAM
    cache_policy="lfru",           # lru, lfru, slru, lfu, fifo, ls, dali
    max_seq_len=4096,              # pre-allocate static KV cache (0 = use HF DynamicCache)
    kv_dtype=torch.bfloat16,       # torch.float8_e4m3fn for 2x context
    gpu_memory_utilization=0.90,   # fraction of GPU VRAM to use (like vLLM)
    attn_implementation="eager",   # or "flex" for FlexAttention (opt-in)
)
```

**Expert cache vs KV cache trade-off** (GPT-OSS-20B, 8 GB, BF16 KV):

| Context | KV Cache | Expert Slots | Trade-off |
|---------|----------|-------------|-----------|
| 0 (no KV) | 0 MB | 321 | Max throughput |
| 4K | 201 MB | 306 | Balanced |
| 16K | 805 MB | 260 | Long context |
| 32K | 1.6 GB | 200 | Very long |
| 64K | 3.2 GB | 78 | Max context |

Use `--kv-fp8` for FP8 KV cache (halves KV memory, doubles max context).

## Hardware requirements

**GPU:**
- NVIDIA GPUs only (hardcoded CUDA streams, Triton PTX)
- AMD ROCm / Intel XPU / Apple MPS: not supported
- Minimum: SM 7.0 (V100, RTX 2060) for BF16 path
- Recommended: SM 8.0+ (A100, RTX 3090) for MXFP4 Triton kernels
- Best: SM 8.9+ (RTX 4090, RTX PRO) for FP8 KV cache + compute
- VRAM: 8 GB minimum, more VRAM = more expert cache slots + longer context

**System:**
- RAM: must fit all expert weights (GPT-OSS-20B: ~10 GB pinned, DeepSeek-V3: ~350 GB)
- Python 3.11+, PyTorch 2.6+, Transformers 4.50+

## Key techniques

- **Native MXFP4 expert store** — loads quantized blocks + scales directly from safetensors. 4x smaller than BF16. Supports Triton `dot_scaled` kernels.
- **GPU-side FP8 to BF16 dequant** — H2D transfers raw FP8 bytes (half BF16 size), GPU kernel dequantizes into cache slot.
- **Event-based stream sync** — CUDA events coordinate transfer, compute, and prefetch streams without CPU blocking.
- **FATE cross-layer prefetch** ([arxiv 2502.12224](https://arxiv.org/abs/2502.12224)) — structural prediction using next layer's gate on current hidden states.
- **DALI workload-aware cache** ([arxiv 2602.03495](https://arxiv.org/abs/2602.03495)) — sliding-window frequency tracking with hot/cold tiers.
- **Zero-copy cache hits** — template `.data` set to cache slot views. No buffer copy on hot path.
- **GPU slot map** — tensor-indexed cache lookup avoids CUDA sync on the cache-hit path.

## Limitations

- **NVIDIA only** — hardcoded CUDA streams and Triton PTX. No AMD/Intel/Apple support.
- **Single GPU** — no multi-GPU or tensor parallelism.
- **Context scaling** — decode throughput drops linearly with context length due to attention cost (not expert offload). FlexAttention with static KV shapes is available (`attn_implementation="flex"` + `max_seq_len=N`) — eliminates torch.compile recompilation by using constant-shape tensors with masking.
- **Prefill speed** — prefill is attention-dominated, not improved by expert caching.
- **FlashAttention** — GPT-OSS uses custom attention sinks incompatible with standard FA2/SDPA. FlexAttention with sinks is available but requires torch.compile warmup.
- **Correctness** — expert weight swapping uses a single template module. This is correct for batch_size=1 decode but not for true batched inference where different tokens need different experts simultaneously.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ --ignore=tests/test_hf_models.py -x -q  # 194+ tests
```

## Benchmarking

```bash
# Standard decode benchmark
python scripts/benchmark.py --model openai/gpt-oss-20b --measure 40

# Compare cache policies
python scripts/benchmark.py --compare-policies

# FATE prediction accuracy per layer
python scripts/benchmark.py --fate-diagnostic

# Analytical Pareto frontier (instant, no GPU)
python scripts/autotune.py --model openai/gpt-oss-20b

# CPU profiler
python scripts/benchmark.py --profile

# CUDA trace
python scripts/benchmark.py --trace
```

## License

MIT
