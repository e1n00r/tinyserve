"""GPU-resident KV cache with pre-allocated buffers."""

import torch

from .config import HEAD_DIM, LAYER_TYPES, NUM_KV_HEADS, NUM_LAYERS, SLIDING_WINDOW

_DEFAULT_MAX_SEQ = 2048


class KVCache:
    """All-GPU KV cache with pre-allocated buffers.

    Sliding-window layers use a circular buffer of SLIDING_WINDOW tokens.
    Full-attention layers use a linear buffer up to max_seq_len tokens.
    No CPU↔GPU transfers, no torch.cat per step.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float8_e4m3fn,
        max_seq_len: int = _DEFAULT_MAX_SEQ,
    ):
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # Pre-allocate all buffers on GPU
        self._k = torch.zeros(
            NUM_LAYERS, 1, NUM_KV_HEADS, max_seq_len, HEAD_DIM,
            dtype=dtype, device=device,
        )
        self._v = torch.zeros(
            NUM_LAYERS, 1, NUM_KV_HEADS, max_seq_len, HEAD_DIM,
            dtype=dtype, device=device,
        )
        # Per-layer sequence length (how many tokens stored)
        self._seq_lens = [0] * NUM_LAYERS

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ):
        """Append new K, V tokens for this layer (in-place, no allocation).

        Args:
            new_k: [batch, num_kv_heads, new_len, head_dim]
            new_v: same shape
        """
        new_len = new_k.shape[2]
        k_store = new_k.to(self.dtype)
        v_store = new_v.to(self.dtype)

        if LAYER_TYPES[layer_idx] == "sliding_attention":
            cur = self._seq_lens[layer_idx]
            if cur + new_len <= SLIDING_WINDOW:
                self._k[layer_idx, :, :, cur:cur + new_len] = k_store
                self._v[layer_idx, :, :, cur:cur + new_len] = v_store
                self._seq_lens[layer_idx] = cur + new_len
            else:
                # Shift old tokens left, append new at end
                keep = SLIDING_WINDOW - new_len
                if keep > 0 and cur > 0:
                    src_start = min(cur, self.max_seq_len) - keep
                    if src_start >= 0:
                        self._k[layer_idx, :, :, :keep] = self._k[layer_idx, :, :, src_start:src_start + keep].clone()
                        self._v[layer_idx, :, :, :keep] = self._v[layer_idx, :, :, src_start:src_start + keep].clone()
                self._k[layer_idx, :, :, keep:keep + new_len] = k_store
                self._v[layer_idx, :, :, keep:keep + new_len] = v_store
                self._seq_lens[layer_idx] = SLIDING_WINDOW
        else:
            cur = self._seq_lens[layer_idx]
            end = cur + new_len
            if end > self.max_seq_len:
                raise RuntimeError(
                    f"KV cache overflow: {end} > {self.max_seq_len}. "
                    f"Increase max_seq_len."
                )
            self._k[layer_idx, :, :, cur:end] = k_store
            self._v[layer_idx, :, :, cur:end] = v_store
            self._seq_lens[layer_idx] = end

    def get_kv(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get KV for attention. Returns views (no copy, no transfer)."""
        seq_len = self._seq_lens[layer_idx]
        if seq_len == 0:
            return None
        return (
            self._k[layer_idx, :, :, :seq_len],
            self._v[layer_idx, :, :, :seq_len],
        )

    def reset(self):
        self._seq_lens = [0] * NUM_LAYERS

    def vram_bytes(self) -> int:
        return self._k.nbytes + self._v.nbytes
