# tinyserve — MoE Expert Offloading for Consumer GPUs

Run GPT-OSS-20B at 20+ tok/s on an 8 GB laptop GPU.

tinyserve offloads Mixture-of-Experts (MoE) expert weights to CPU and serves them through a GPU LRU cache with adaptive prefetch. Models that need 40+ GB of VRAM run interactively on a single consumer GPU.

## What it does

- **Expert CPU offloading with GPU LRU cache** — expert weights live in pinned CPU memory; a VRAM cache holds hot experts (289 slots / 3.8 GB on 8 GB GPU)
- **Native MXFP4 expert store** — loads quantized weights directly from safetensors, skipping HuggingFace dequantization. 4x smaller than BF16
- **Adaptive FATE temporal prefetch** — predicts next-layer experts using cross-layer gate similarity with temporal fallback. 95-100% cache hit rate
- **Zero-copy cache hits** — template parameter `.data` points directly into cache slot views. No memcpy on hit
- **Double-buffered async PCIe pipeline** — overlaps H2D transfer with GPU compute for cache misses
- **7 pluggable cache eviction policies** — LRU, LFRU, SLRU, LFU, FIFO, Least-Stale, DALI

## Performance

**Peak throughput (warm cache, 40-token decode):**

| Hardware | tok/s | Hit Rate |
|----------|-------|----------|
| RTX PRO 2000 8 GB (Blackwell laptop) | 21-26 | 100% |

**Realistic workloads:**

| Scenario | tok/s | Hit Rate |
|----------|-------|----------|
| Short decode (10 tok) | 10.3 | 82% |
| Medium decode (40 tok) | 11.3 | 84% |
| Long decode (100 tok) | 11.3 | 79% |
| Code generation (60 tok) | 8.0 | 74% |
| Russian text (40 tok) | 9.6 | 78% |
| Long prompt + decode | 6.8 | 88% |

**vs baselines:**

| System | tok/s | vs tinyserve |
|--------|-------|-------------|
| HF `device_map=auto` | 0.19 | 58x slower |
| Ollama (8 GB, partial offload, est.) | ~4-5 | ~2x slower |
| llama.cpp (all MoE on CPU, 12 GB, DDR4) | ~20 | comparable |
| tinyserve (8 GB, native MXFP4) | 6.8-26 | -- |

## Quick start

```bash
pip install -e .
```

```python
from tinyserve.offload import load_and_offload

model = load_and_offload("openai/gpt-oss-20b")
# That's it — generate as normal
output = model.generate(input_ids, max_new_tokens=100)
```

`load_and_offload` downloads the model, moves non-expert weights to GPU, extracts expert weights into a pinned CPU store, and auto-sizes the VRAM cache from remaining free memory. All expert dispatch is handled transparently inside the model's forward pass.

For models already loaded via HuggingFace:

```python
from transformers import AutoModelForCausalLM
from tinyserve.offload import offload_model

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1", device_map="cpu", torch_dtype=torch.bfloat16
)
model = offload_model(model, device="cuda")
```

## How it works

1. **Expert store** — Expert weights are extracted from the HuggingFace model and packed into flat byte buffers on pinned CPU memory. For MXFP4 models (GPT-OSS), weights are loaded directly from safetensors as raw uint8 blocks + scales, bypassing the HF dequantization pipeline entirely.

2. **GPU LRU cache** — A pre-allocated VRAM tensor holds `capacity` expert slots. On a cache hit, the template module's parameters are set to views of the cache slot (zero-copy). On a miss, the double-buffered pipeline kicks in.

3. **Double-buffered H2D pipeline** — Two GPU buffers alternate: while expert _i_ runs its forward pass on the compute stream, expert _i+1_ is being DMA'd from CPU on the transfer stream. For FP8-compressed stores, the raw bytes are transferred at half size and dequantized on-GPU.

4. **FATE cross-layer prefetch** — Adjacent MoE layers have >83% gate input cosine similarity. After layer _N_ finishes, its hidden states are run through layer _N+1_'s gate to predict which experts will be needed. Those experts are prefetched into the cache on a dedicated stream, overlapping with attention compute.

5. **Adaptive temporal fallback** — After the first token, the system reuses the previous token's routing decisions instead of FATE gate predictions. Temporal locality matches or exceeds FATE accuracy (~99%+ on most layers) and eliminates the gate computation cost.

6. **Cache-aware routing** — Optional logit bias nudges expert selection toward GPU-resident experts, reducing cache misses with minimal quality impact.

## Supported models

| Model | Weights | Status |
|-------|---------|--------|
| GPT-OSS-20B | Native MXFP4 | Tested |
| Mixtral 8x7B | BF16 / FP8 | Tested |
| DeepSeek-V3 | BF16 / FP8 | Tested |
| Qwen 3.5 MoE | BF16 / FP8 | Tested |

Any HuggingFace MoE model with a standard `layers[i].mlp.experts` architecture should work. The model registry auto-detects routing strategy, expert layout, and shared expert handling from the model config.

## Key techniques

- **GPU-side FP8 to BF16 dequant** — H2D transfers raw FP8 bytes (half the BF16 size), then a GPU kernel dequantizes into the cache slot. Cuts effective PCIe bandwidth cost in half.
- **Event-based stream sync** — CUDA events coordinate transfer, compute, and prefetch streams without blocking the CPU.
- **FATE cross-layer prefetch** ([arxiv 2502.12224](https://arxiv.org/abs/2502.12224)) — Structural prediction using the next layer's gate on the current hidden states. Uses top-k+1 candidates to cover border-case experts.
- **DALI workload-aware cache** ([arxiv 2602.03495](https://arxiv.org/abs/2602.03495)) — Sliding-window frequency tracking. Hot experts are protected from eviction; cold experts managed by LRU.
- **Least-Stale eviction** ([SpecMD, arxiv 2602.03921](https://arxiv.org/abs/2602.03921)) — Experts from previous forward passes are marked stale and evicted first, since MoE access within one token is sequential.
- **Native MXFP4 expert store** — Loads quantized blocks + scales directly from safetensors. No HF dequantization overhead. Supports Triton fused dequant-vecmat kernels when available.
- **Zero-copy cache hits** — Template parameter `.data` set to cache slot views. No buffer copy on the hot path.
- **Inlined expert forward** — Bakes tensor offsets at init time, eliminating per-call dict lookups and getattr chains.

## Limitations

- **Batch inference** — Designed for interactive single-sequence decode (batch_size=1). Not optimized for batch inference.
- **FlashAttention** — Not available for GPT-OSS (custom attention sinks). Falls back to eager attention.
- **Multi-GPU** — Single GPU only.
- **Prefill speed** — Prefill is dominated by attention compute, not expert offload. Offloading helps decode, not prefill.
- **Context scaling** — Decode throughput drops linearly with context length due to attention cost, independent of expert offloading.

## Requirements

- NVIDIA GPU with 8 GB+ VRAM
- System RAM > model expert weights
- Python 3.11+, PyTorch 2.6+, Transformers 4.50+

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -x -q
```

## Benchmarking

```bash
# Standard decode benchmark (auto-sizes cache)
python -m scripts.benchmark --model openai/gpt-oss-20b --measure 40

# Compare all 7 cache policies on domain-shift workload
python -m scripts.benchmark --compare-policies

# Domain-shift benchmark (EN-tech warmup -> Russian literature)
python -m scripts.benchmark --domain-shift

# FATE prediction accuracy per layer
python -m scripts.benchmark --fate-diagnostic

# Torch profiler trace (exports Chrome trace to /tmp)
python -m scripts.benchmark --trace
```

## License

MIT
