# gpt-oss-offload

Run **GPT-OSS-120B** (117B params, 5.1B active MoE) on a single 8GB GPU.

**14 tok/s** steady-state decode on an RTX PRO 2000 Blackwell laptop.

## How it works

Expert weights (57 GB MXFP4) live in CPU RAM as memory-mapped files. A double-buffered PCIe pipeline streams cache-miss experts to GPU on demand. An LRU cache in VRAM (373 slots, 4.6 GB) keeps hot experts resident — after ~160 tokens the cache hits 99%+ and decode is GPU-compute bound.

Key optimizations:
- **FP4 Tensor Core matmul** via Triton `tl.dot_scaled` on Blackwell (SM 12.0) — 5x over software dequant
- **FP8 attention** (`torch._scaled_mm`) + **INT8 embeddings** — frees 2 GB VRAM for more cache
- **GPU-resident KV cache** — pre-allocated, zero CPU transfers
- **Contiguous packed binary** — one 12.6 MB DMA per expert instead of 6 scattered reads

## Performance

Measured on RTX PRO 2000 Blackwell (8 GB VRAM), 64 GB DDR5, PCIe 5.0:

| Tokens generated | tok/s | Cache hit rate | Character |
|---|---|---|---|
| 0–80 | 1.9–2.5 | 48–56% | Cold — filling LRU cache |
| 80–160 | 2.6–5.1 | 53–78% | Warming — working set stabilizing |
| 160–400 | 12–14 | 98–100% | Hot — GPU-compute bound |

Older GPUs (Ampere, Ada) fall back to software MXFP4 dequant — expect ~3–5 tok/s steady-state.

## Requirements

- NVIDIA GPU with **8 GB+ VRAM** (Blackwell recommended for FP4 Tensor Cores)
- **64 GB+ system RAM**
- **~57 GB disk** for model weights
- Python 3.11+, PyTorch 2.6+, CUDA 12.8+
- Linux (mmap + madvise used for expert storage)

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/gpt-oss-offload.git
cd gpt-oss-offload
pip install -e .

# Download and split weights (~30 min, downloads ~60 GB)
python -m scripts.split_weights --output-dir ./weights

# Convert to packed binary format (faster I/O, ~10 min)
python -m scripts.repack_experts --weights-dir ./weights

# Generate text
python -m src.generate --weights-dir ./weights --prompt "The meaning of life is" --max-tokens 100

# Benchmark decode throughput
python -m scripts.benchmark --warmup 60 --measure 120
```

If you already have the model downloaded via `huggingface-cli`:

```bash
python -m scripts.split_weights \
    --model-dir ~/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/HASH \
    --output-dir ./weights
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│ GPU (8 GB VRAM)                                 │
│                                                 │
│  Non-expert weights (FP8/INT8)    ~2 GB         │
│  KV cache (FP8, pre-allocated)    72 MB         │
│  Expert LRU cache (373 slots)     4.6 GB        │
│  Double buffers (A/B)             25 MB         │
│                                                 │
│  ┌───────────┐    ┌───────────┐                 │
│  │ Transfer  │    │ Compute   │                 │
│  │ Stream    │    │ Stream    │                 │
│  │           │    │           │                 │
│  │ Load E[i] │───>│ FP4 TC    │                 │
│  │ Load E[i+1]│   │ SwiGLU    │                 │
│  └───────────┘    └───────────┘                 │
│        ▲                                        │
└────────│────────────────────────────────────────┘
         │ PCIe 5.0
┌────────│────────────────────────────────────────┐
│ CPU (64 GB RAM)                                 │
│                                                 │
│  mmap'd expert files   57 GB                    │
│  36 layers × 128 experts × 12.6 MB each         │
│  madvise(WILLNEED) for SSD readahead            │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Model details

GPT-OSS-120B is a 36-layer MoE with 128 experts per layer, top-4 routing, and a custom SwiGLU activation with learned attention sinks. Weights are MXFP4 (E2M1 values, E8M0 block scales, block size 32).

| Parameter | Value |
|---|---|
| Total params | 117B |
| Active params | 5.1B |
| Layers | 36 (alternating sliding/full attention) |
| Experts | 128 per layer, top-4 routed |
| Hidden dim | 2880 |
| Attention | 64 Q heads, 8 KV heads, GQA |
| Context | 131K (YaRN RoPE) |
| Sliding window | 128 tokens (odd layers) |

## Limitations

- **Cold start**: First ~160 tokens run at 2–5 tok/s while the LRU cache fills
- **Blackwell-only FP4 TC**: Older GPUs use software dequant (3–5x slower compute)
- **Single-batch decode only**: No batched inference
- **Linux only**: Relies on `mmap` + `madvise` syscalls
- **RAM hungry**: 64 GB minimum for the mmap'd expert files

## License

MIT
