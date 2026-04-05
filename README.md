# tinyserve — MoE Expert Offloading for Consumer GPUs

Run Mixture-of-Experts models that don't fit in VRAM on a single NVIDIA GPU. **30 tok/s** decode for a 20B MoE model on an 8 GB laptop GPU. **Flat throughput to 32K context** via StreamingLLM. Zero dequantization — native MXFP4 and GGUF quant formats via ggml CUDA kernels.

## Performance

All numbers from **RTX PRO 2000 8 GB laptop GPU**, GPT-OSS-20B (MXFP4, 24 layers × 32 experts, top_k=4). Raw logs in [`benchmarks/`](benchmarks/).

### Decode throughput vs context length

| Context | tok/s | ms/tok | Prefill |
|---|---|---|---|
| 0 (decode-only) | 31.2 | 32 ms | — |
| 256 | 30.0 | 33 ms | 1.4 s |
| 512 | 30.2 | 33 ms | 1.7 s |
| 1 024 | 30.4 | 33 ms | 3.4 s |
| 2 048 | 29.9 | 33 ms | 7.0 s |
| 4 096 | 31.9 | 31 ms | 13.9 s |
| 8 192 *(StreamingLLM)* | 29.7 | 34 ms | 28 s |
| 16 384 *(StreamingLLM)* | 29.8 | 34 ms | 57 s |
| 32 768 *(StreamingLLM)* | 29.4 | 34 ms | 114 s |

Throughput is **flat** — adding context costs only prefill time, not decode speed.

**HuggingFace baseline (device_map="auto"):** 0.19 tok/s — 155× slower.

<details>
<summary>Benchmark methodology</summary>

- Diverse prompts across 5 domains (code, math, creative, multilingual, conversation)
- n_warmup=5, n_measure=20 per context length
- StaticKVCache (BF16), streaming window = 2048 for contexts > max_seq_len
- Source: `benchmarks/gptoss20b_context.json`, `benchmarks/gptoss20b_streaming.json`
</details>

## Quick start

```bash
git clone --recurse-submodules https://github.com/e1n00r/tinyserve.git && cd tinyserve
pip install -e "."
python build_ggml.py   # compile ggml CUDA kernels (optional but recommended)
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

## How it works

1. **Zero-copy expert store** — GGUF files are mmap'd directly. Expert weights are raw quantized bytes in pinned CPU memory — no dequantization at load time, no conversion to BF16.

2. **ggml CUDA MMVQ kernels** — At inference time, fused dequant+matmul runs entirely on-GPU in native quant format (Q4_K, Q5_K, Q6_K, Q8_0). Each expert forward is 3 kernel launches: gate, up, down. No intermediate BF16 materialization.

3. **GPU LFRU cache** — Frequency-recency eviction prevents deep-layer starvation. Cache tracks experts by (layer, expert_id); GPU slot map synced lazily.

4. **FATE prefetch** — Current-layer expert activations predict next-layer needs. Overlaps H2D transfer with attention compute.

5. **StreamingLLM** — Sink tokens (4) + sliding window. Infinite context at constant decode speed and constant VRAM.

6. **Batched prefill** — Groups tokens by expert, loads each once. O(num\_unique\_experts) not O(seq\_len × top\_k). 288K → 32 loads per layer at 3K context.

7. **StaticKVCache** — Pre-allocated BF16 KV buffers. No dynamic allocation during generation.

## Supported models

| Model | Format | Status |
|---|---|---|
| GPT-OSS-20B | MXFP4 safetensors | **Benchmarked** |
| GPT-OSS-120B | MXFP4 safetensors | Benchmarked |
| GPT-OSS-20B | GGUF Q4\_K\_M | End-to-end verified |
| Qwen 3.5 MoE 30B-A3B | GGUF | End-to-end verified |
| Qwen 122B | GGUF Q4\_K\_M / Q5\_K\_M | Unit tested |
| Mixtral 8×7B | BF16 safetensors | Unit tested |
| DeepSeek-V3/R1 | BF16 safetensors | Profile only |

**Formats:** HuggingFace safetensors (BF16, FP8, MXFP4). GGUF (Q4\_K, Q5\_K, Q6\_K, Q8\_0) with native kernel compute.

## Configuration

```python
model = load_and_offload(
    "openai/gpt-oss-20b",
    cache_capacity=0,              # 0 = auto-size from VRAM
    cache_policy="lfru",           # lru, lfru, slru, lfu, fifo
    max_seq_len=4096,              # static KV cache size
    gpu_memory_utilization=0.90,
    streaming=True,                # StreamingLLM for infinite context
    streaming_window_size=2048,
    fp8=True,                      # FP8 attention (saves ~0.5 GB VRAM)
    adaptive_fate=True,            # FATE temporal prefetch
)
```

## Limitations

- NVIDIA only (CUDA, ggml CUDA kernels, Triton PTX)
- Single GPU, batch size 1 decode
- GPT-OSS-120B benchmarks pending (download in progress)
- Qwen 3.x generation quality under investigation (weight mapping)

<details>
<summary>What we tried and ruled out</summary>

| Technique | Result |
|---|---|
| D2-MoE delta compression | Expert cosine similarity = 0.0006 — not viable |
| Cache bias routing (0.0–3.0) | No effect on GPT-OSS-20B |
| Cython hot path | 3.4× microbench, 0% end-to-end |
| Expert deferral | Produces garbage output |
| FlexAttention default | pytorch #155065, 3–67× VRAM overhead |
| Triton MMVQ kernel (custom) | ggml CUDA kernels already exist and are faster |
| Full dequant → BF16 at load | 1–3 GB wasted VRAM per expert tier; replaced by zero-copy mmap |
</details>

## Testing

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ --ignore=tests/test_hf_models.py -x -q   # 481 tests
```

## Benchmarking

```bash
python -m scripts.bench_context                        # Decode throughput vs context length
python -m scripts.bench_context --streaming            # StreamingLLM long context
python -m scripts.cache_benchmark                      # Expert cache policy comparison
python scripts/comprehensive_bench.py                  # 7-policy sweep
python scripts/calibrate_buddies.py                    # Buddy co-activation profiling
```

## License

MIT
