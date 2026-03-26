"""Tests for chunked prefill generation."""

from types import SimpleNamespace

import torch
import pytest

from tinyserve.chunked import chunked_prefill, generate_chunked
from tinyserve.static_kv_cache import StaticKVCache


def _make_dummy_model(vocab_size=32, hidden=16, num_layers=2, num_kv_heads=2, head_dim=8):
    """Return a callable that behaves like an HF CausalLM for chunked prefill tests.

    Tracks how many tokens were passed per call (for chunk-size assertions)
    and accumulates KV cache entries so that chunked vs. full prefill can be
    compared.
    """
    linear = torch.nn.Linear(hidden, vocab_size, bias=False)
    linear.eval()
    call_log = []

    def forward(input_ids, past_key_values=None):
        batch, seq_len = input_ids.shape
        call_log.append(seq_len)

        # Write into KV cache if provided
        if past_key_values is not None:
            start = past_key_values.get_seq_length(0)
            k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
            v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
            pos = torch.arange(start, start + seq_len)
            for layer_idx in range(num_layers):
                past_key_values.update(k, v, layer_idx, {"cache_position": pos})

        # Produce deterministic logits from token ids
        # Map each token id to a hidden vector, then project to vocab
        emb = torch.randn(batch, seq_len, hidden)
        logits = linear(emb)
        return SimpleNamespace(logits=logits)

    model = SimpleNamespace(__call__=forward, call_log=call_log)
    model.__call__ = forward
    # Make it callable
    model_fn = lambda **kw: forward(**kw)
    model_fn.call_log = call_log
    return model_fn


class _DummyModel:
    """Callable dummy model for chunked prefill tests."""

    def __init__(self, vocab_size=32, hidden=16, num_layers=2, num_kv_heads=2, head_dim=8):
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._linear = torch.nn.Linear(hidden, vocab_size, bias=False)
        self._linear.eval()
        self.call_log = []

    def __call__(self, input_ids, past_key_values=None):
        batch, seq_len = input_ids.shape
        self.call_log.append(seq_len)

        if past_key_values is not None:
            start = past_key_values.get_seq_length(0)
            k = torch.randn(batch, self.num_kv_heads, seq_len, self.head_dim, dtype=torch.bfloat16)
            v = torch.randn(batch, self.num_kv_heads, seq_len, self.head_dim, dtype=torch.bfloat16)
            pos = torch.arange(start, start + seq_len)
            for layer_idx in range(self.num_layers):
                past_key_values.update(k, v, layer_idx, {"cache_position": pos})

        emb = torch.randn(batch, seq_len, self.hidden)
        logits = self._linear(emb)
        return SimpleNamespace(logits=logits)


def _make_cache(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=256):
    return StaticKVCache(
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )


def test_chunked_prefill_produces_output():
    model = _DummyModel()
    cache = _make_cache()
    input_ids = torch.randint(0, 32, (1, 20))

    out = chunked_prefill(model, input_ids, cache, chunk_size=8)

    assert out is not None
    assert out.logits.shape[0] == 1
    assert out.logits.shape[2] == model.vocab_size
    # KV cache should have accumulated all 20 tokens
    assert cache.get_seq_length(0) == 20
    # Model should have been called 3 times: 8 + 8 + 4
    assert model.call_log == [8, 8, 4]


def test_chunked_matches_full_prefill():
    """Chunked and full prefill should populate KV cache to the same length."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 32, (1, 15))

    # Full prefill (chunk_size >= seq_len)
    model_full = _DummyModel()
    cache_full = _make_cache()
    out_full = chunked_prefill(model_full, input_ids, cache_full, chunk_size=1024)

    # Chunked prefill
    model_chunked = _DummyModel()
    cache_chunked = _make_cache()
    out_chunked = chunked_prefill(model_chunked, input_ids, cache_chunked, chunk_size=4)

    # Both caches should have same sequence length
    assert cache_full.get_seq_length(0) == cache_chunked.get_seq_length(0) == 15

    # Full should be 1 call, chunked should be 4 calls (4+4+4+3)
    assert model_full.call_log == [15]
    assert model_chunked.call_log == [4, 4, 4, 3]

    # Output logit shapes must match (last chunk only has 3 tokens vs full 15,
    # but both have same vocab dim)
    assert out_full.logits.shape[2] == out_chunked.logits.shape[2]


def test_chunk_size_larger_than_input():
    model = _DummyModel()
    cache = _make_cache()
    input_ids = torch.randint(0, 32, (1, 5))

    out = chunked_prefill(model, input_ids, cache, chunk_size=512)

    assert out is not None
    assert cache.get_seq_length(0) == 5
    # Single call — chunk_size > seq_len means no splitting
    assert model.call_log == [5]


def test_chunk_size_one():
    model = _DummyModel()
    cache = _make_cache()
    input_ids = torch.randint(0, 32, (1, 7))

    out = chunked_prefill(model, input_ids, cache, chunk_size=1)

    assert out is not None
    assert cache.get_seq_length(0) == 7
    # 7 calls of 1 token each
    assert model.call_log == [1, 1, 1, 1, 1, 1, 1]


def test_generate_chunked_produces_tokens():
    model = _DummyModel()
    cache = _make_cache()
    input_ids = torch.randint(0, 32, (1, 10))

    output = generate_chunked(
        model, input_ids, max_new_tokens=5, kv_cache=cache, chunk_size=4,
    )

    # Output should be input + generated tokens
    assert output.shape[0] == 1
    assert output.shape[1] == 10 + 5
    # First 10 tokens should match input
    assert torch.equal(output[:, :10], input_ids)


def test_generate_chunked_eos_stops_early():
    """Generation stops when eos_token_id is produced."""

    class _EosModel(_DummyModel):
        """Produces eos_token_id=0 on the 3rd generated token."""
        def __init__(self):
            super().__init__()
            self._gen_count = 0

        def __call__(self, input_ids, past_key_values=None):
            out = super().__call__(input_ids, past_key_values)
            # During decode (seq_len=1), count generated tokens
            if input_ids.shape[1] == 1:
                self._gen_count += 1
                if self._gen_count >= 3:
                    # Force logits to predict token 0 (eos)
                    out.logits[:, -1, :] = -1e9
                    out.logits[:, -1, 0] = 1e9
            return out

    model = _EosModel()
    cache = _make_cache()
    input_ids = torch.randint(1, 32, (1, 5))

    output = generate_chunked(
        model, input_ids, max_new_tokens=20, kv_cache=cache,
        chunk_size=3, eos_token_id=0,
    )

    # 1st generated token comes from prefill logits (no decode call).
    # Decode calls: 1st (gen_count=1), 2nd (gen_count=2), 3rd (gen_count=3, eos).
    # Total generated: 1 (prefill) + 3 (decode) = 4, but eos on 3rd decode
    # means we get 1 + 3 = 4 generated tokens.
    assert output.shape[1] == 5 + 4
