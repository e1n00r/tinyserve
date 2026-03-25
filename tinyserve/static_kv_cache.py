"""Static pre-allocated KV cache for zero-allocation decode.

Replaces HuggingFace's DynamicCache (which does torch.cat per step)
with a fixed-size tensor. Supports BF16 and FP8 storage with
on-the-fly quantization/dequantization.

Usage:
    cache = StaticKVCache.from_model_config(config, max_seq_len=4096, device="cuda")
    output = model.generate(input_ids, past_key_values=cache)
"""

import torch


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
    ):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._dtype = dtype
        self._compute_dtype = torch.bfloat16
        self._device = device
        self.static_shapes = static_shapes
        self._k = torch.zeros(
            num_layers,
            1,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._v = torch.zeros(
            num_layers,
            1,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._seq_lens = [0] * num_layers
        self.is_sliding = [False] * num_layers

    @classmethod
    def from_model_config(cls, config, max_seq_len=4096, device="cuda", dtype=torch.bfloat16):
        effective = getattr(config, "text_config", config)
        head_dim = getattr(effective, "head_dim", None)
        if head_dim is None:
            head_dim = effective.hidden_size // effective.num_attention_heads
        return cls(
            max_seq_len=max_seq_len,
            num_layers=effective.num_hidden_layers,
            num_kv_heads=effective.num_key_value_heads,
            head_dim=head_dim,
            device=torch.device(device),
            dtype=dtype,
        )

    @staticmethod
    def bytes_per_token(num_layers, num_kv_heads, head_dim, dtype=torch.bfloat16):
        elem_size = torch.tensor([], dtype=dtype).element_size()
        return num_layers * 2 * num_kv_heads * head_dim * elem_size

    @property
    def vram_bytes(self):
        return self._k.nelement() * self._k.element_size() * 2

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        new_tokens = key_states.shape[2]
        # Use cache_position if provided (HF passes it for correct placement).
        if cache_kwargs and "cache_position" in cache_kwargs:
            pos = cache_kwargs["cache_position"]
            start = pos[0].item()
            end = start + new_tokens
        else:
            start = self._seq_lens[layer_idx]
            end = start + new_tokens
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: {end} > {self.max_seq_len}. Increase max_seq_len or reduce context."
            )
        if self._dtype != self._compute_dtype:
            self._k[layer_idx, :, :, start:end] = key_states.to(self._dtype)
            self._v[layer_idx, :, :, start:end] = value_states.to(self._dtype)
        else:
            self._k[layer_idx, :, :, start:end] = key_states
            self._v[layer_idx, :, :, start:end] = value_states
        self._seq_lens[layer_idx] = max(self._seq_lens[layer_idx], end)
        if self.static_shapes:
            k_out = self._k[layer_idx, :, :, : self.max_seq_len]
            v_out = self._v[layer_idx, :, :, : self.max_seq_len]
        else:
            k_out = self._k[layer_idx, :, :, : self._seq_lens[layer_idx]]
            v_out = self._v[layer_idx, :, :, : self._seq_lens[layer_idx]]
        if self._dtype != self._compute_dtype:
            return k_out.to(self._compute_dtype), v_out.to(self._compute_dtype)
        return k_out, v_out

    def get_seq_length(self, layer_idx=0):
        return self._seq_lens[layer_idx]

    def get_max_cache_shape(self):
        return None

    def get_mask_sizes(self, cache_position=None, layer_idx=0):
        cur = self._seq_lens[layer_idx]
        if cur == 0 and cache_position is not None:
            return cache_position.shape[0], 0
        if cache_position is not None:
            return cur + cache_position.shape[0], 0
        return cur, 0

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

    def reset(self):
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
