# tinyserve — MoE Expert Offloading for Consumer GPUs

Run GPT-OSS-20B at 20+ tok/s on an 8 GB laptop GPU.

tinyserve offloads Mixture-of-Experts (MoE) expert weights to CPU and serves them through a GPU LRU cache with adaptive prefetch. Models that need 40+ GB of VRAM run interactively on a single consumer GPU.

## Performance

**Peak throughput (warm cache, 40-token decode):**

| Hardware | tok/s | Hit Rate |
|----------|-------|----------|
| RTX PRO 2000 8 GB (Blackwell laptop) | 21-27 | 100% |

**Realistic workloads (cold start, mixed prompts):**

| Scenario | tok/s | Hit Rate |
|----------|-------|----------|
| Short decode (10 tok) | 10-11 | 82% |
| Medium decode (40 tok) | 11-15 | 84% |
| Long decode (100 tok) | 11-13 | 79% |
| Code generation (60 tok) | 8-10 | 74% |
| Long prompt + decode | 7-9 | 88% |

**vs baselines (GPT-OSS-20B on 8 GB VRAM):**

| System | tok/s | Notes |
|--------|-------|-------|
| HF `device_map=auto` | 0.19 | Naive CPU offload, 138x slower |
| Ollama (8 GB, partial offload) | ~4-5 | Estimated, layer-level offload |
| llama.cpp (12 GB, all MoE on CPU) | ~20 | C++, sends activations to CPU |
| **tinyserve** | **7-27** | Depends on cache warmth + context |

Numbers measured on RTX PRO 2000 Blackwell 8 GB laptop GPU with GPT-OSS-20B MXFP4. Your results will vary with GPU, model, and workload.

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

| Model family | model_type | Expert layout | Routing | Status |
|-------------|-----------|--------------|---------|--------|
| GPT-OSS-20B/120B | `gpt_oss` | Fused gate_up + down | Sigmoid top-k | Tested (native MXFP4) |
| Mixtral 8x7B/8x22B | `mixtral` | Fused gate_up + down | Softmax then top-k | Tested |
| DeepSeek-V3/R1 | `deepseek_v3` | gate + up + down | Top-k then softmax | Tested |
| Qwen 3.5 MoE | `qwen3_5_moe` | Fused gate_up + down | Router native | Tested |
| Qwen 3 MoE (235B) | `qwen3_moe` | gate + up + down | Softmax then top-k | Profile exists |
| Qwen 2 MoE | `qwen2_moe` | gate + up + down | Softmax then top-k | Profile exists |
| OLMoE | `olmoe` | gate + up + down | Softmax then top-k | Profile exists |

Any HuggingFace MoE model with a `layers[i].mlp.experts` architecture should work. The model registry auto-detects routing strategy, expert layout, and shared expert handling from the config.

**Not yet supported:** Llama 4 Scout/Maverick, Kimi K2, DBRX, Phi-MoE, Jamba. PRs welcome.

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
python -m pytest tests/ --ignore=tests/test_hf_models.py -x -q  # 157 tests
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
