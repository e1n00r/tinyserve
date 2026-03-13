"""Offloaded GPT-OSS-120B with hybrid KV cache and pipelined expert loading."""

import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .attention import attention_forward, attention_forward_fp8, rms_norm
from .config import (
    EXPERT_BYTES,
    HIDDEN_SIZE,
    LAYER_TYPES,
    NUM_EXPERTS_PER_TOK,
    NUM_LAYERS,
    SLIDING_WINDOW,
)
from .expert_store import ExpertBuffer, ExpertStore
from .experts import expert_forward
from .kv_cache import HybridKVCache
from .lru_cache import ExpertLRUCache
from .pipeline import ExpertPipeline
from .rope import build_rope_cache


class OffloadedGptOss:
    """GPT-OSS-120B with expert weights in CPU pinned memory.

    Non-expert weights (attention, embeddings, router, norms) live on GPU.
    Expert weights are loaded to GPU on demand via double-buffered pipeline.
    KV cache: sliding-window layers on GPU, full-attention layers on CPU.
    """

    def __init__(
        self,
        weights_dir: str,
        device: str = "cuda",
        pipeline: bool = True,
        cache_capacity: int | None = None,
    ):
        """
        Args:
            weights_dir: directory with non_expert.safetensors + experts_L*.safetensors
            device: GPU device
            pipeline: use double-buffered pipeline (True) or blocking (False)
            cache_capacity: number of experts to cache in VRAM. None = auto-size
                from free VRAM. 0 = disable cache.
        """
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.use_pipeline = pipeline

        print("Loading non-expert weights to GPU...")
        ne_path = f"{weights_dir}/non_expert.safetensors"
        self.weights = load_file(ne_path, device=device)
        ne_bytes = sum(t.nbytes for t in self.weights.values())
        print(f"  Non-expert on GPU: {ne_bytes / 1024**3:.2f} GB")

        # Quantize to free VRAM for expert cache
        saved_attn = self._quantize_attention_fp8()
        saved_embed = self._quantize_embeddings_int8()
        print(f"  Quantized: FP8 attn saved {saved_attn / 1024**3:.2f} GB, "
              f"INT8 embed saved {saved_embed / 1024**3:.2f} GB")

        self.expert_store = ExpertStore(weights_dir)
        self.expert_store.load()

        # Auto-size cache from remaining VRAM
        if cache_capacity is None and pipeline:
            reserved = 512 * 1024 * 1024  # 512 MB headroom for dequant temps, KV, activations
            free_mem = torch.cuda.mem_get_info(self.device)[0] - reserved
            cache_capacity = ExpertLRUCache.estimate_capacity(max(0, free_mem))
            print(f"  Auto-sized expert LRU cache: {cache_capacity} experts "
                  f"({cache_capacity * EXPERT_BYTES / 1024**3:.2f} GB)")

        if pipeline:
            self.expert_pipeline = ExpertPipeline(
                self.expert_store, self.device,
                cache_capacity=cache_capacity or 0,
            )
        else:
            self.expert_buf = ExpertBuffer(self.device)

        self.cos_cache, self.sin_cache = build_rope_cache(self.device, self.dtype)
        self.kv_cache = HybridKVCache(self.device)

        mode = "pipelined" + (f"+cache({cache_capacity})" if cache_capacity else "") if pipeline else "blocking"
        print(f"Model ready ({mode}).")

    def _quantize_attention_fp8(self) -> int:
        """Post-training quantize Q/K/V/O weights to FP8 E4M3 with per-tensor scale.

        Returns bytes saved.
        """
        saved = 0
        self._fp8_scales: dict[str, torch.Tensor] = {}

        for li in range(NUM_LAYERS):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                key = f"model.layers.{li}.self_attn.{proj}.weight"
                w = self.weights[key]
                old_bytes = w.nbytes

                # Per-tensor scale: max(abs(w)) / max_fp8
                amax = w.abs().amax()
                scale = amax / torch.finfo(torch.float8_e4m3fn).max
                scale = scale.clamp(min=1e-12)

                w_fp8 = (w / scale).to(torch.float8_e4m3fn)
                self.weights[key] = w_fp8
                self._fp8_scales[key] = scale.float()

                saved += old_bytes - w_fp8.nbytes

        torch.cuda.empty_cache()
        return saved

    def _quantize_embeddings_int8(self) -> int:
        """INT8 quantization for embed_tokens, FP8 for lm_head.

        embed_tokens uses INT8 (per-row scale) for embedding lookups.
        lm_head uses FP8 with _scaled_mm (avoids materializing full bf16 weight).
        Processes in chunks to avoid OOM from temporaries.
        Returns bytes saved.
        """
        saved = 0
        self._int8_scales: dict[str, torch.Tensor] = {}

        # embed_tokens → INT8 (only selected rows dequantized during forward)
        key = "model.embed_tokens.weight"
        w = self.weights[key]
        old_bytes = w.nbytes
        rows = w.shape[0]

        amax = w.abs().amax(dim=1)
        scale = (amax / 127.0).clamp(min=1e-12)

        chunk = 8192
        w_int8 = torch.empty_like(w, dtype=torch.int8)
        for start in range(0, rows, chunk):
            end = min(start + chunk, rows)
            w_int8[start:end] = (w[start:end] / scale[start:end, None]).round().clamp(-128, 127).to(torch.int8)

        self.weights[key] = w_int8
        self._int8_scales[key] = scale.float()
        del w
        saved += old_bytes - w_int8.nbytes

        # lm_head → FP8 (use _scaled_mm, no dequant needed)
        key = "lm_head.weight"
        w = self.weights[key]
        old_bytes = w.nbytes

        amax = w.abs().amax()
        lm_scale = (amax / torch.finfo(torch.float8_e4m3fn).max).clamp(min=1e-12)

        w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
        for start in range(0, w.shape[0], chunk):
            end = min(start + chunk, w.shape[0])
            w_fp8[start:end] = (w[start:end] / lm_scale).to(torch.float8_e4m3fn)

        self.weights[key] = w_fp8
        self._fp8_scales[key] = lm_scale.float()
        del w
        saved += old_bytes - w_fp8.nbytes

        torch.cuda.empty_cache()
        return saved

    def _w(self, key: str) -> torch.Tensor:
        return self.weights[key]

    def reset(self):
        self.kv_cache.reset()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch, seq_len = input_ids.shape
        timings = {"attn": 0.0, "router": 0.0, "experts": 0.0}

        # INT8 embedding: dequant on the fly (only selected rows, cheap)
        embed_w = self._w("model.embed_tokens.weight")
        embed_scale = self._int8_scales["model.embed_tokens.weight"]
        h = (embed_w[input_ids.view(-1)].to(self.dtype) * embed_scale[input_ids.view(-1)].unsqueeze(-1).to(self.dtype))
        h = h.view(batch, seq_len, -1)

        cos = self.cos_cache[position_ids]
        sin = self.sin_cache[position_ids]

        for li in range(NUM_LAYERS):
            p = f"model.layers.{li}"
            residual = h

            h = rms_norm(h, self._w(f"{p}.input_layernorm.weight"))

            t0 = time.perf_counter()
            sliding = SLIDING_WINDOW if LAYER_TYPES[li] == "sliding_attention" else None
            past_kv = self.kv_cache.get_kv(li)

            fp8_weights = {}
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                wkey = f"{p}.self_attn.{proj}.weight"
                fp8_weights[proj] = (self._w(wkey), self._fp8_scales[wkey])

            h_attn, new_k, new_v = attention_forward_fp8(
                h,
                fp8_weights["q_proj"], self._w(f"{p}.self_attn.q_proj.bias"),
                fp8_weights["k_proj"], self._w(f"{p}.self_attn.k_proj.bias"),
                fp8_weights["v_proj"], self._w(f"{p}.self_attn.v_proj.bias"),
                fp8_weights["o_proj"], self._w(f"{p}.self_attn.o_proj.bias"),
                self._w(f"{p}.self_attn.sinks"),
                cos, sin,
                past_kv,
                sliding,
            )

            self.kv_cache.update(li, new_k, new_v)
            h = residual + h_attn
            timings["attn"] += time.perf_counter() - t0

            residual = h
            h = rms_norm(h, self._w(f"{p}.post_attention_layernorm.weight"))

            t0 = time.perf_counter()
            router_logits = F.linear(
                h.reshape(-1, HIDDEN_SIZE),
                self._w(f"{p}.mlp.router.weight"),
                self._w(f"{p}.mlp.router.bias"),
            )
            top_vals, top_idx = torch.topk(router_logits, NUM_EXPERTS_PER_TOK, dim=-1)
            routing_weights = F.softmax(top_vals, dim=-1, dtype=top_vals.dtype)
            timings["router"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            h_flat = h.reshape(-1, HIDDEN_SIZE)

            if self.use_pipeline:
                expert_out = self.expert_pipeline.execute_layer_experts(
                    h_flat, li, top_idx, routing_weights,
                )
            else:
                expert_out = self._blocking_experts(
                    h_flat, li, top_idx, routing_weights,
                )

            h = residual + expert_out.view_as(h)
            timings["experts"] += time.perf_counter() - t0

        h = rms_norm(h, self._w("model.norm.weight"))
        # FP8 lm_head via _scaled_mm
        from .attention import _fp8_linear
        logits = _fp8_linear(
            h, self._w("lm_head.weight"),
            self._fp8_scales["lm_head.weight"], bias=None,
        )
        return logits, timings

    def _blocking_experts(
        self,
        h_flat: torch.Tensor,
        layer_idx: int,
        top_idx: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: blocking expert loading (Phase 2 behavior)."""
        expert_out = torch.zeros_like(h_flat)
        for tok in range(h_flat.shape[0]):
            for k in range(NUM_EXPERTS_PER_TOK):
                eidx = top_idx[tok, k].item()
                w = routing_weights[tok, k]
                self.expert_store.copy_to_buffer(
                    self.expert_buf, layer_idx, eidx, non_blocking=False
                )
                out = expert_forward(
                    h_flat[tok:tok + 1],
                    self.expert_buf.gate_up_blocks,
                    self.expert_buf.gate_up_scales,
                    self.expert_buf.gate_up_bias,
                    self.expert_buf.down_blocks,
                    self.expert_buf.down_scales,
                    self.expert_buf.down_bias,
                    dtype=self.dtype,
                )
                expert_out[tok] += w * out.squeeze(0)
        return expert_out
