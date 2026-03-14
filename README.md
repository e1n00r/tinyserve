# moe-offload

Run large MoE models on a single consumer GPU by offloading expert weights to CPU RAM.

**Supports Mixtral, Qwen3-MoE, DeepSeek-V3, GPT-OSS** — any HuggingFace MoE model.

```python
from transformers import AutoModelForCausalLM
from src import offload_model

model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16)
model = offload_model(model, device="cuda", cache_capacity=100)
output = model.generate(input_ids, max_new_tokens=200)
```

## How it works

Expert weights live in CPU RAM. A double-buffered PCIe pipeline streams cache-miss experts to GPU on demand. An LRU cache in VRAM keeps hot experts resident — after the cache warms, decode becomes GPU-compute bound.

- **Double-buffered PCIe pipeline** — overlaps expert transfer with compute
- **LRU expert cache** — frequently-used experts stay in VRAM
- **Auto-detection** — identifies model family from HF config, applies correct routing
- **Shared expert support** — DeepSeek-V3's shared experts stay on GPU permanently
- **FP4 Tensor Core matmul** on Blackwell via Triton `tl.dot_scaled` (GPT-OSS MXFP4)

## Supported models

| Model | Experts | Top-K | Tested |
|---|---|---|---|
| **Mixtral 8x7B / 8x22B** | 8 | 2 | Exact logit match |
| **Qwen3-MoE / Qwen3.5-MoE** | 128 | 8 | Exact logit match |
| **DeepSeek-V3 (685B)** | 256 + shared | 8 | Token-level match |
| **GPT-OSS-120B** | 128 | 4 | 14 tok/s (FP4 TC) |

Any HuggingFace MoE model with `nn.ModuleList` experts should work via `offload_model()`.

## Performance (GPT-OSS-120B on RTX PRO 2000 Blackwell 8GB)

| Tokens generated | tok/s | Cache hit rate |
|---|---|---|
| 0–80 | 1.9–2.5 | 48–56% |
| 80–160 | 2.6–5.1 | 53–78% |
| 160–400 | 12–14 | 98–100% |

## Requirements

- NVIDIA GPU with **8 GB+ VRAM**
- System RAM > model expert weights (e.g., 64 GB for GPT-OSS-120B)
- Python 3.11+, PyTorch 2.6+
- Linux (mmap used for expert storage)

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import offload_model

model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16)
model = offload_model(model, device="cuda", cache_capacity=100)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
inputs = tokenizer("The meaning of life is", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

For GPT-OSS-120B with optimized MXFP4 pipeline:

```bash
git clone https://github.com/YOUR_USERNAME/moe-offload.git
cd moe-offload
pip install -e .

python -m scripts.split_weights --output-dir ./weights
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
