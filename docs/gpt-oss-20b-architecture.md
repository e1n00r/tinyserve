# GPT-OSS-20B Architecture Reference

Last updated: 2026-03-31.

## Model Identity

- **HuggingFace ID:** `openai/gpt-oss-20b`
- **Family:** GPT-OSS (also 120B variant)
- **Architecture class:** `GptOssForCausalLM` (custom HF implementation)
- **Parameters:** ~20B total, ~2B active per token
- **Quantization:** MXFP4 (E2M1 values + E8M0 block scales, 32-element groups)

## Structure

| Component | Value |
|---|---|
| Layers | 24 |
| Hidden size | 2880 |
| Intermediate size | 2880 |
| Attention heads | 64 (query), 8 (KV) — GQA ratio 8:1 |
| Head dim | 64 |
| Vocab size | 201,088 |
| Max position | 229,376 |
| RoPE | yarn (factor=56, original_max=4096) |

## MoE Structure

| Component | Value |
|---|---|
| Experts per layer | 32 |
| Active experts (top_k) | 4 |
| Total expert params | 768 (24 layers × 32) |
| Shared expert | Yes (runs in parallel on shared_stream) |
| Router | `GptOssTopKRouter` (softmax then topk) |
| Returns router logits | No (router_native=False) |

## Expert Weights (per expert)

All stored as fused tensors `[num_experts, ...]` in the `GptOssExperts` module:

| Tensor | Shape | Dtype | Size/expert |
|---|---|---|---|
| `gate_up_proj` | [32, 2880, 5760] | BF16 (MXFP4 native) | ~13.2 MB |
| `gate_up_proj_bias` | [32, 5760] | BF16 | ~11 KB |
| `down_proj` | [32, 2880, 2880] | BF16 (MXFP4 native) | ~6.6 MB |
| `down_proj_bias` | [32, 2880] | BF16 | ~5.6 KB |

**Expert module path:** `model.layers[i].mlp.experts`
**Router path:** `model.layers[i].mlp.router`

## Attention Structure

Alternating attention patterns per layer:

| Layer type | Config | Layers |
|---|---|---|
| `sliding_attention` | window=128 | 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 |
| `full_attention` | no window | 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 |

**Attention sinks:** GPT-OSS uses virtual sink tokens in attention. The custom SDPA hook in tinyserve handles this by skipping sinks for decode (is_causal=False) and using is_causal=True for prefill.

## MXFP4 Format Details

Expert weights are stored natively in MXFP4 in safetensors:
- **Value encoding:** E2M1 (4-bit: 1 sign + 2 exponent + 1 mantissa)
- **Block scale:** E8M0 (8-bit exponent, no mantissa) per 32-element group
- **Storage:** uint8 blocks (2 values per byte) + uint8 scales
- **LUT values:** [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] × {+, -}

tinyserve loads these as raw uint8 and uses Triton `dot_scaled` for native GPU compute (no dequantization). CPU path converts to INT4 packed format via `mxfp4_to_int4pack()`.

## Activation Function

GPT-OSS uses a **custom SwiGLU variant** (not standard SiLU):
```python
gate = gate_up[..., ::2].clamp(max=7.0)
up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
gated = (up + 1) * gate * torch.sigmoid(gate * 1.702)
```

The gate_up_proj output is interleaved (not chunked): even indices are gate, odd indices are up. This differs from Mixtral/Qwen which use `.chunk(2, dim=-1)`.

## Cache Performance (measured, 2026-03-31)

With 238 GPU cache slots (31% of 768 total experts) on RTX PRO 2000 8 GB:

| Policy | Overall HR | Deep layers (18-23) | tok/s |
|---|---|---|---|
| LRU | 87.8% | 8% | 10.7 |
| **LFRU** | **89.7%** | **52%** | **9.4** |
| SLRU | 89.4% | 55% | 8.9 |
| LFU | 76.4% | 80% | 5.5 |

LFRU is the default. CPU-on-miss reduces miss penalty from ~20ms to ~2ms.

## Expert Weight Properties

**Inter-expert cosine similarity:** 0.0006 (essentially zero)
**SVD rank for 95% energy:** 29.3/32 (no low-rank delta structure)

Implication: experts are fully independent. D2-MoE delta compression (shared base + low-rank deltas) is NOT viable for this model. Each expert must be stored and loaded independently.

**Expert frequency distribution:** Flat (top 10% of experts handle only 19% of accesses). Frequency-based caching offers limited advantage over recency-based policies.

## Related Models

| Model | Experts | top_k | Hidden | Intermediate | Notes |
|---|---|---|---|---|---|
| GPT-OSS-20B | 32 | 4 | 2880 | 2880 | This doc |
| GPT-OSS-120B | 128 | 4 | 2880 | 2880 | 5× more experts |
| GPT-OSS-Puzzle-88B | 64-128 per layer | 4 | 2880 | 2880 | Heterogeneous expert counts (NVIDIA Puzzle NAS) |
