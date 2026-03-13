"""Offloaded GPT-OSS-120B with hybrid KV cache and pipelined expert loading."""

import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .attention import attention_forward, rms_norm
from .config import (
    HIDDEN_SIZE,
    LAYER_TYPES,
    NUM_EXPERTS_PER_TOK,
    NUM_LAYERS,
    SLIDING_WINDOW,
)
from .expert_store import ExpertBuffer, ExpertStore
from .experts import expert_forward
from .kv_cache import HybridKVCache
from .pipeline import ExpertPipeline
from .rope import build_rope_cache


class OffloadedGptOss:
    """GPT-OSS-120B with expert weights in CPU pinned memory.

    Non-expert weights (attention, embeddings, router, norms) live on GPU.
    Expert weights are loaded to GPU on demand via double-buffered pipeline.
    KV cache: sliding-window layers on GPU, full-attention layers on CPU.
    """

    def __init__(self, weights_dir: str, device: str = "cuda", pipeline: bool = True):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.use_pipeline = pipeline

        print("Loading non-expert weights to GPU...")
        ne_path = f"{weights_dir}/non_expert.safetensors"
        self.weights = load_file(ne_path, device=device)
        ne_bytes = sum(t.nbytes for t in self.weights.values())
        print(f"  Non-expert on GPU: {ne_bytes / 1024**3:.2f} GB")

        self.expert_store = ExpertStore(weights_dir)
        self.expert_store.load()

        if pipeline:
            self.expert_pipeline = ExpertPipeline(self.expert_store, self.device)
        else:
            self.expert_buf = ExpertBuffer(self.device)

        self.cos_cache, self.sin_cache = build_rope_cache(self.device, self.dtype)
        self.kv_cache = HybridKVCache(self.device)

        mode = "pipelined" if pipeline else "blocking"
        print(f"Model ready ({mode} expert loading).")

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

        h = F.embedding(input_ids, self._w("model.embed_tokens.weight"))

        cos = self.cos_cache[position_ids]
        sin = self.sin_cache[position_ids]

        for li in range(NUM_LAYERS):
            p = f"model.layers.{li}"
            residual = h

            h = rms_norm(h, self._w(f"{p}.input_layernorm.weight"))

            t0 = time.perf_counter()
            sliding = SLIDING_WINDOW if LAYER_TYPES[li] == "sliding_attention" else None
            past_kv = self.kv_cache.get_kv(li)

            h_attn, new_k, new_v = attention_forward(
                h,
                self._w(f"{p}.self_attn.q_proj.weight"),
                self._w(f"{p}.self_attn.q_proj.bias"),
                self._w(f"{p}.self_attn.k_proj.weight"),
                self._w(f"{p}.self_attn.k_proj.bias"),
                self._w(f"{p}.self_attn.v_proj.weight"),
                self._w(f"{p}.self_attn.v_proj.bias"),
                self._w(f"{p}.self_attn.o_proj.weight"),
                self._w(f"{p}.self_attn.o_proj.bias"),
                self._w(f"{p}.self_attn.sinks"),
                cos, sin,
                past_kv,
                sliding,
            )

            self.kv_cache.update(li, new_k, new_v)
            h = residual + h_attn
            timings["attn"] += time.perf_counter() - t0

            # Post-attn norm + MoE
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

            # Expert computation
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
        logits = F.linear(h, self._w("lm_head.weight"))
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
