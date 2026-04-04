"""Static pre-allocated KV cache for zero-allocation decode.

Replaces HuggingFace's DynamicCache (which does torch.cat per step)
with a fixed-size tensor. Supports BF16 and FP8 storage with
on-the-fly quantization/dequantization.

Usage:
    cache = StaticKVCache.from_model_config(config, max_seq_len=4096, device="cuda")
    output = model.generate(input_ids, past_key_values=cache)
"""

import torch


class KVCacheOverflow(RuntimeError):
    """Raised when KV cache update exceeds max_seq_len.

    Carries `tokens_needed` — how many additional tokens are required.
    VRAMBudget catches this and frees expert slots to extend the KV cache.
    """

    def __init__(self, tokens_needed: int):
        self.tokens_needed = tokens_needed
        super().__init__(f"KV cache overflow: need {tokens_needed} more tokens")


class StaticKVCache:
    """Pre-allocated KV cache — zero allocation during decode.

    Implements the HuggingFace Cache interface duck-type (update,
    get_seq_length, get_max_cache_shape, reset, etc.) without inheriting
    from Cache (which has complex init requirements).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        static_shapes: bool = False,
        storage_device: torch.device | None = None,
    ):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._dtype = dtype
        self._compute_dtype = torch.bfloat16
        self._device = device
        self._storage_device = storage_device or device
        self._compute_device = device
        self.static_shapes = static_shapes
        self._k = torch.zeros(
            num_layers,
            1,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=self._storage_device,
        )
        self._v = torch.zeros(
            num_layers,
            1,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=self._storage_device,
        )
        if self._storage_device != self._compute_device and self._storage_device == torch.device("cpu"):
            self._k = self._k.pin_memory()
            self._v = self._v.pin_memory()
        self._seq_lens = [0] * num_layers
        self.is_sliding = [False] * num_layers
        self._sliding_window: int | None = None
        self._vram_budget = None  # Set by offload.py when VRAMBudget is created

    @classmethod
    def from_model_config(cls, config, max_seq_len=4096, device="cuda", dtype=torch.bfloat16, storage_device=None):
        effective = getattr(config, "text_config", config)
        head_dim = getattr(effective, "head_dim", None)
        if head_dim is None:
            head_dim = effective.hidden_size // effective.num_attention_heads
        cache = cls(
            max_seq_len=max_seq_len,
            num_layers=effective.num_hidden_layers,
            num_kv_heads=effective.num_key_value_heads,
            head_dim=head_dim,
            device=torch.device(device),
            dtype=dtype,
            storage_device=torch.device(storage_device) if storage_device is not None else None,
        )
        sliding_window = getattr(effective, "sliding_window", None)
        layer_types = getattr(effective, "layer_types", None)
        if sliding_window is not None and layer_types is not None:
            cache._sliding_window = sliding_window
            cache.is_sliding = [lt == "sliding_attention" for lt in layer_types[: effective.num_hidden_layers]]
        elif sliding_window is not None:
            cache._sliding_window = sliding_window
            cache.is_sliding = [True] * effective.num_hidden_layers
        return cache

    @staticmethod
    def bytes_per_token(num_layers, num_kv_heads, head_dim, dtype=torch.bfloat16):
        elem_size = torch.tensor([], dtype=dtype).element_size()
        return num_layers * 2 * num_kv_heads * head_dim * elem_size

    @property
    def vram_bytes(self) -> int:
        return self._k.nelement() * self._k.element_size() + self._v.nelement() * self._v.element_size()

    def extend(self, additional_tokens: int) -> None:
        """Grow KV cache capacity by additional_tokens."""
        new_max = self.max_seq_len + additional_tokens
        new_k = torch.zeros(
            self.num_layers,
            1,
            self.num_kv_heads,
            new_max,
            self.head_dim,
            dtype=self._dtype,
            device=self._storage_device,
        )
        new_v = torch.zeros(
            self.num_layers,
            1,
            self.num_kv_heads,
            new_max,
            self.head_dim,
            dtype=self._dtype,
            device=self._storage_device,
        )
        new_k[:, :, :, : self.max_seq_len, :] = self._k
        new_v[:, :, :, : self.max_seq_len, :] = self._v
        if self._storage_device != self._compute_device and self._storage_device == torch.device("cpu"):
            new_k = new_k.pin_memory()
            new_v = new_v.pin_memory()
        self._k = new_k
        self._v = new_v
        self.max_seq_len = new_max

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None) -> tuple[torch.Tensor, torch.Tensor]:
        new_tokens = key_states.shape[2]
        start = self._seq_lens[layer_idx]
        end = start + new_tokens
        if end > self.max_seq_len:
            # Try self-healing before raising
            healed = False
            if getattr(self, "_streaming", False):
                self._evict_streaming(layer_idx)
                start = self._seq_lens[layer_idx]
                end = start + new_tokens
                healed = end <= self.max_seq_len
            if not healed and self._vram_budget is not None:
                if self._vram_budget.handle_overflow(end - self.max_seq_len):
                    start = self._seq_lens[layer_idx]
                    end = start + new_tokens
                    healed = end <= self.max_seq_len
            if not healed:
                raise KVCacheOverflow(end - self.max_seq_len)
        store_val_k = key_states
        store_val_v = value_states
        if self._dtype != self._compute_dtype:
            store_val_k = store_val_k.to(self._dtype)
            store_val_v = store_val_v.to(self._dtype)
        if self._storage_device != self._compute_device:
            self._k[layer_idx, :, :, start:end] = store_val_k.to(self._storage_device)
            self._v[layer_idx, :, :, start:end] = store_val_v.to(self._storage_device)
        else:
            self._k[layer_idx, :, :, start:end] = store_val_k
            self._v[layer_idx, :, :, start:end] = store_val_v
        self._seq_lens[layer_idx] = end
        if self.static_shapes:
            k_out = self._k[layer_idx, :, :, : self.max_seq_len]
            v_out = self._v[layer_idx, :, :, : self.max_seq_len]
        else:
            seq_len = self._seq_lens[layer_idx]
            if (
                self._storage_device != self._compute_device
                and self.is_sliding[layer_idx]
                and self._sliding_window is not None
                and seq_len > self._sliding_window
            ):
                window_start = seq_len - self._sliding_window
                k_out = self._k[layer_idx, :, :, window_start:seq_len]
                v_out = self._v[layer_idx, :, :, window_start:seq_len]
            else:
                k_out = self._k[layer_idx, :, :, :seq_len]
                v_out = self._v[layer_idx, :, :, :seq_len]
        needs_device_transfer = self._storage_device != self._compute_device
        needs_dtype_cast = self._dtype != self._compute_dtype
        if needs_device_transfer or needs_dtype_cast:
            target_dtype = self._compute_dtype if needs_dtype_cast else None
            non_blocking = self._storage_device == torch.device("cpu") and needs_device_transfer
            k_out = k_out.to(device=self._compute_device, dtype=target_dtype, non_blocking=non_blocking)
            v_out = v_out.to(device=self._compute_device, dtype=target_dtype, non_blocking=non_blocking)
        return k_out, v_out

    def get_seq_length(self, layer_idx=0) -> int:
        return self._seq_lens[layer_idx]

    def get_max_cache_shape(self):
        return None

    def get_mask_sizes(self, q_length=None, layer_idx=0):
        cur = self._seq_lens[layer_idx]
        if q_length is None:
            return cur, 0
        if isinstance(q_length, int):
            return cur + q_length, 0
        # Tensor (older HF versions)
        return cur + q_length.shape[0], 0

    def is_initialized(self, layer_idx=0):
        return self._seq_lens[layer_idx] > 0

    @staticmethod
    def is_compileable():
        return False

    @property
    def max_batch_size(self):
        return 1

    @property
    def max_cache_len(self):
        return self.max_seq_len

    def early_initialization(self, *args, **kwargs):
        pass

    def crop(self, max_length):
        for i in range(self.num_layers):
            if self._seq_lens[i] > max_length:
                self._seq_lens[i] = max_length

    def batch_repeat_interleave(self, repeats):
        return self

    def batch_select_indices(self, indices):
        return self

    def reorder_cache(self, beam_idx):
        pass

    def offload(self):
        pass

    def prefetch(self, layer_idx):
        pass

    def reset(self) -> None:
        for i in range(self.num_layers):
            self._seq_lens[i] = 0

    def __len__(self):
        return self.num_layers

    def __iter__(self):
        for i in range(self.num_layers):
            end = self._seq_lens[i]
            yield (self._k[i, :, :, :end], self._v[i, :, :, :end])

    def __getitem__(self, idx):
        end = self._seq_lens[idx]
        return (self._k[idx, :, :, :end], self._v[idx, :, :, :end])

    def __contains__(self, item):
        return False

    def __bool__(self):
        return self._seq_lens[0] > 0

    def enable_streaming(self, sink_size: int = 4, window_size: int = 1024) -> None:
        """Enable StreamingLLM-style KV eviction.

        Keeps the first ``sink_size`` tokens (attention sinks) and the last
        ``window_size`` tokens. Evicts everything in between. This caps
        KV memory at ``(sink_size + window_size) × per_token_bytes``
        regardless of total context length.

        Based on StreamingLLM (arxiv 2309.17453).
        """
        self._streaming = True
        self._sink_size = sink_size
        self._window_size = window_size

    def _evict_streaming(self, layer_idx: int) -> None:
        """Compact KV to [sinks | recent_window] if streaming is enabled."""
        if not getattr(self, "_streaming", False):
            return
        seq_len = self._seq_lens[layer_idx]
        max_kept = self._sink_size + self._window_size
        if seq_len <= max_kept:
            return  # fits, no eviction needed

        # Copy sink tokens (first N) and window tokens (last M)
        # into positions 0..max_kept
        sink_end = self._sink_size
        window_start = max(self._sink_size, seq_len - self._window_size)

        # Sinks are already at positions 0..sink_end — no copy needed
        # Window needs to shift from window_start to sink_end
        self._k[layer_idx, :, :, sink_end:max_kept] = self._k[layer_idx, :, :, window_start:seq_len].clone()
        self._v[layer_idx, :, :, sink_end:max_kept] = self._v[layer_idx, :, :, window_start:seq_len].clone()

        # Also shift scales if INT8 quantization is active
        if getattr(self, "_k_scales", None) is not None:
            self._k_scales[layer_idx, :, :, sink_end:max_kept] = self._k_scales[
                layer_idx, :, :, window_start:seq_len
            ].clone()
            self._v_scales[layer_idx, :, :, sink_end:max_kept] = self._v_scales[
                layer_idx, :, :, window_start:seq_len
            ].clone()

        self._seq_lens[layer_idx] = max_kept

    def enable_h2o(self, budget: int = 1024, sink_size: int = 4) -> None:
        """Enable H2O (Heavy Hitter Oracle) KV eviction.

        Keeps the ``sink_size`` initial tokens + the ``budget - sink_size``
        tokens with highest cumulative attention scores. Evicts the rest.
        Unlike StreamingLLM (which keeps only recent tokens), H2O preserves
        tokens that are consistently important across all decode steps.

        Based on H2O (NeurIPS 2023, arxiv 2306.14048).

        Must call ``update_h2o_scores(attn_weights, layer_idx)`` after each
        attention step to accumulate scores.
        """
        self._h2o = True
        self._h2o_budget = budget
        self._h2o_sink = sink_size
        # Cumulative attention scores per position per layer
        self._h2o_scores = torch.zeros(
            self.num_layers,
            self.max_seq_len,
            dtype=torch.float32,
            device=self._storage_device,
        )

    def update_h2o_scores(self, attn_weights: torch.Tensor, layer_idx: int) -> None:
        """Accumulate attention scores for H2O eviction.

        Args:
            attn_weights: [batch, heads, q_len, kv_len] attention weights
            layer_idx: which layer
        """
        if not getattr(self, "_h2o", False):
            return
        # Sum across batch, heads, and query positions
        scores = attn_weights.sum(dim=(0, 1, 2))  # [kv_len]
        seq_len = self._seq_lens[layer_idx]
        kv_len = min(scores.shape[0], seq_len)
        self._h2o_scores[layer_idx, :kv_len] += scores[:kv_len].to(device=self._storage_device, dtype=torch.float32)

    def _evict_h2o(self, layer_idx: int) -> None:
        """Evict low-scoring KV entries based on H2O scores."""
        if not getattr(self, "_h2o", False):
            return
        seq_len = self._seq_lens[layer_idx]
        if seq_len <= self._h2o_budget:
            return

        scores = self._h2o_scores[layer_idx, :seq_len].clone()
        # Protect sink tokens (infinite score)
        scores[: self._h2o_sink] = float("inf")

        # Select top-budget positions by score
        _, keep_indices = scores.topk(self._h2o_budget)
        keep_indices = keep_indices.sort().values  # maintain temporal order

        # Compact KV to keep only selected positions
        self._k[layer_idx, :, :, : self._h2o_budget] = self._k[layer_idx, :, :, keep_indices].clone()
        self._v[layer_idx, :, :, : self._h2o_budget] = self._v[layer_idx, :, :, keep_indices].clone()

        # Compact scores
        self._h2o_scores[layer_idx, : self._h2o_budget] = self._h2o_scores[layer_idx, keep_indices].clone()
        self._h2o_scores[layer_idx, self._h2o_budget :] = 0

        self._seq_lens[layer_idx] = self._h2o_budget
