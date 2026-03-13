"""Phase 2: Offloaded GPT-OSS-120B with blocking expert loading."""

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
from .rope import build_rope_cache


class OffloadedGptOss:
    """GPT-OSS-120B with expert weights in CPU pinned memory.

    Non-expert weights (attention, embeddings, router, norms) live on GPU.
    Expert weights are loaded to GPU on demand for each forward pass.
    """

    def __init__(self, weights_dir: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        # Load non-expert weights to GPU
        print("Loading non-expert weights to GPU...")
        ne_path = f"{weights_dir}/non_expert.safetensors"
        self.weights = load_file(ne_path, device=device)
        ne_bytes = sum(t.nbytes for t in self.weights.values())
        print(f"  Non-expert on GPU: {ne_bytes / 1024**3:.2f} GB")

        # Load expert weights to pinned CPU RAM
        self.expert_store = ExpertStore(weights_dir)
        self.expert_store.load()

        # Pre-allocate GPU expert buffer (single expert at a time for Phase 2)
        self.expert_buf = ExpertBuffer(self.device)

        # Build RoPE cache
        self.cos_cache, self.sin_cache = build_rope_cache(self.device, self.dtype)

        # KV caches: list of (K, V) per layer, initially None
        self.kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in range(NUM_LAYERS)
        ]

        print("Model ready.")

    def _get_weight(self, key: str) -> torch.Tensor:
        return self.weights[key]

    def reset_kv_cache(self):
        self.kv_caches = [None for _ in range(NUM_LAYERS)]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for a single step.

        Args:
            input_ids: [batch, seq_len] on device
            position_ids: [batch, seq_len] on device

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape

        # 1. Embedding
        embed_w = self._get_weight("model.embed_tokens.weight")
        h = F.embedding(input_ids, embed_w)  # [batch, seq_len, hidden_size]

        # 2. RoPE
        cos = self.cos_cache[position_ids]  # [batch, seq_len, head_dim//2]
        sin = self.sin_cache[position_ids]

        # 3. Layer loop
        timings = {"attn": 0.0, "router": 0.0, "transfer": 0.0, "expert_compute": 0.0}

        for layer_idx in range(NUM_LAYERS):
            prefix = f"model.layers.{layer_idx}"
            residual = h

            # 3a. Pre-attention norm
            ln_w = self._get_weight(f"{prefix}.input_layernorm.weight")
            h = rms_norm(h, ln_w)

            # 3b. Self-attention
            t0 = time.perf_counter()
            sliding_window = SLIDING_WINDOW if LAYER_TYPES[layer_idx] == "sliding_attention" else None
            h_attn, new_kv = attention_forward(
                h,
                self._get_weight(f"{prefix}.self_attn.q_proj.weight"),
                self._get_weight(f"{prefix}.self_attn.q_proj.bias"),
                self._get_weight(f"{prefix}.self_attn.k_proj.weight"),
                self._get_weight(f"{prefix}.self_attn.k_proj.bias"),
                self._get_weight(f"{prefix}.self_attn.v_proj.weight"),
                self._get_weight(f"{prefix}.self_attn.v_proj.bias"),
                self._get_weight(f"{prefix}.self_attn.o_proj.weight"),
                self._get_weight(f"{prefix}.self_attn.o_proj.bias"),
                self._get_weight(f"{prefix}.self_attn.sinks"),
                cos, sin,
                self.kv_caches[layer_idx],
                sliding_window,
            )
            self.kv_caches[layer_idx] = new_kv
            h = residual + h_attn
            timings["attn"] += time.perf_counter() - t0

            # 3c. Post-attention norm
            residual = h
            ln_w = self._get_weight(f"{prefix}.post_attention_layernorm.weight")
            h = rms_norm(h, ln_w)

            # 3d. Router
            t0 = time.perf_counter()
            router_w = self._get_weight(f"{prefix}.mlp.router.weight")
            router_b = self._get_weight(f"{prefix}.mlp.router.bias")
            router_logits = F.linear(h.reshape(-1, HIDDEN_SIZE), router_w, router_b)
            top_values, top_indices = torch.topk(router_logits, NUM_EXPERTS_PER_TOK, dim=-1)
            routing_weights = F.softmax(top_values, dim=-1, dtype=top_values.dtype)
            timings["router"] += time.perf_counter() - t0

            # 3e. Expert computation
            h_flat = h.reshape(-1, HIDDEN_SIZE)
            expert_output = torch.zeros_like(h_flat)

            for token_idx in range(h_flat.shape[0]):
                for k in range(NUM_EXPERTS_PER_TOK):
                    expert_idx = top_indices[token_idx, k].item()
                    weight = routing_weights[token_idx, k]

                    # Transfer expert to GPU (blocking)
                    t0 = time.perf_counter()
                    self.expert_store.copy_to_buffer(
                        self.expert_buf, layer_idx, expert_idx, non_blocking=False
                    )
                    timings["transfer"] += time.perf_counter() - t0

                    # Compute expert
                    t0 = time.perf_counter()
                    out = expert_forward(
                        h_flat[token_idx : token_idx + 1],
                        self.expert_buf.gate_up_blocks,
                        self.expert_buf.gate_up_scales,
                        self.expert_buf.gate_up_bias,
                        self.expert_buf.down_blocks,
                        self.expert_buf.down_scales,
                        self.expert_buf.down_bias,
                        dtype=self.dtype,
                    )
                    expert_output[token_idx] += weight * out.squeeze(0)
                    timings["expert_compute"] += time.perf_counter() - t0

            h = residual + expert_output.view_as(h)

        # 4. Final norm
        final_ln_w = self._get_weight("model.norm.weight")
        h = rms_norm(h, final_ln_w)

        # 5. LM head
        lm_head_w = self._get_weight("lm_head.weight")
        logits = F.linear(h, lm_head_w)

        return logits, timings
