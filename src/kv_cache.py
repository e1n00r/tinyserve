"""Hybrid KV cache: sliding-window layers on GPU, full-attention layers on CPU."""

import torch

from .config import HEAD_DIM, LAYER_TYPES, NUM_KV_HEADS, NUM_LAYERS, SLIDING_WINDOW


class HybridKVCache:
    """KV cache that keeps sliding-window layers on GPU and full-attention on CPU.

    Sliding-window layers only need the last SLIDING_WINDOW tokens, so their
    KV is tiny and always resident on GPU. Full-attention layers need all past
    tokens, so their KV lives in pinned CPU memory and is transferred to GPU
    on demand per layer.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float8_e4m3fn):
        self.device = device
        self.dtype = dtype
        self.seq_len = 0

        # Per-layer storage. None until first update.
        # Sliding-window layers: (K, V) on GPU, circular buffer of SLIDING_WINDOW tokens
        # Full-attention layers: (K, V) on CPU pinned memory
        self._k: list[torch.Tensor | None] = [None] * NUM_LAYERS
        self._v: list[torch.Tensor | None] = [None] * NUM_LAYERS

    def _is_sliding(self, layer_idx: int) -> bool:
        return LAYER_TYPES[layer_idx] == "sliding_attention"

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ):
        """Append new K, V for this layer. Handles GPU vs CPU placement.

        Args:
            new_k: [batch, num_kv_heads, new_len, head_dim] in compute dtype
            new_v: same shape
        """
        # Quantize to cache dtype
        k_store = new_k.to(self.dtype)
        v_store = new_v.to(self.dtype)

        if self._is_sliding(layer_idx):
            # Keep on GPU, truncate to sliding window
            if self._k[layer_idx] is None:
                self._k[layer_idx] = k_store
                self._v[layer_idx] = v_store
            else:
                self._k[layer_idx] = torch.cat([self._k[layer_idx], k_store], dim=2)
                self._v[layer_idx] = torch.cat([self._v[layer_idx], v_store], dim=2)
                # Trim to sliding window
                if self._k[layer_idx].shape[2] > SLIDING_WINDOW:
                    self._k[layer_idx] = self._k[layer_idx][:, :, -SLIDING_WINDOW:]
                    self._v[layer_idx] = self._v[layer_idx][:, :, -SLIDING_WINDOW:]
        else:
            # Move to CPU pinned memory
            k_cpu = k_store.cpu().pin_memory()
            v_cpu = v_store.cpu().pin_memory()
            if self._k[layer_idx] is None:
                self._k[layer_idx] = k_cpu
                self._v[layer_idx] = v_cpu
            else:
                self._k[layer_idx] = torch.cat([self._k[layer_idx], k_cpu], dim=2)
                self._v[layer_idx] = torch.cat([self._v[layer_idx], v_cpu], dim=2)

    def get_kv(
        self,
        layer_idx: int,
        non_blocking: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get KV for attention computation, on GPU.

        For sliding-window layers, already on GPU. For full-attention layers,
        transfers from CPU pinned memory.

        Returns None if no cache exists yet (first token).
        """
        if self._k[layer_idx] is None:
            return None

        k = self._k[layer_idx]
        v = self._v[layer_idx]

        if self._is_sliding(layer_idx):
            # Already on GPU, just upcast
            return k.to(self.device), v.to(self.device)
        else:
            # Transfer from CPU pinned → GPU
            return (
                k.to(self.device, non_blocking=non_blocking),
                v.to(self.device, non_blocking=non_blocking),
            )

    def reset(self):
        self._k = [None] * NUM_LAYERS
        self._v = [None] * NUM_LAYERS
        self.seq_len = 0
