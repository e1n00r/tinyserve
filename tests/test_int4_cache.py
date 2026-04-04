"""Tests for INT4 expert store disk cache."""

import json

import torch

from tinyserve.int4_cache import (
    _deserialize_layout_specs,
    _model_hash,
    _serialize_layout_specs,
    int4_cache_path,
    load_int4_cache,
    save_int4_cache,
)


def _make_fake_data(num_layers: int = 2, num_experts: int = 4, expert_bytes: int = 128):
    """Create synthetic expert store data and layout specs."""
    data = torch.randint(0, 255, (num_layers, num_experts, expert_bytes), dtype=torch.uint8)
    layout_specs = {
        "gate_up_proj": ([512, 8, 16], "torch.uint8"),
        "gate_up_proj_scales": ([512, 8], "torch.uint8"),
    }
    return data, layout_specs


def test_cache_path_format():
    path = int4_cache_path("openai/gpt-oss-20b")
    assert path.endswith(".safetensors")
    assert "openai--gpt-oss-20b" in path
    assert ".cache/tinyserve/int4/" in path


def test_cache_path_no_slash():
    path = int4_cache_path("local-model")
    assert "local-model.safetensors" in path


def test_save_and_load_roundtrip(tmp_path):
    data, layout_specs = _make_fake_data()
    cache_file = str(tmp_path / "cache.safetensors")
    model_hash = "abc123"

    save_int4_cache(cache_file, data, layout_specs, 2, 4, model_hash)

    result = load_int4_cache(cache_file, expected_hash="abc123")
    assert result is not None
    assert result["num_layers"] == 2
    assert result["num_experts"] == 4
    assert result["model_hash"] == "abc123"
    torch.testing.assert_close(result["data"], data)
    assert set(result["layout_specs"].keys()) == set(layout_specs.keys())


def test_stale_cache_returns_none(tmp_path):
    data, layout_specs = _make_fake_data()
    cache_file = str(tmp_path / "cache.safetensors")

    save_int4_cache(cache_file, data, layout_specs, 2, 4, "hash_v1")

    result = load_int4_cache(cache_file, expected_hash="hash_v2")
    assert result is None


def test_missing_cache_returns_none(tmp_path):
    result = load_int4_cache(str(tmp_path / "nonexistent.safetensors"))
    assert result is None


def test_load_without_hash_check(tmp_path):
    data, layout_specs = _make_fake_data()
    cache_file = str(tmp_path / "cache.safetensors")

    save_int4_cache(cache_file, data, layout_specs, 2, 4, "any_hash")

    result = load_int4_cache(cache_file, expected_hash=None)
    assert result is not None
    assert result["num_layers"] == 2


def test_model_hash_from_index(tmp_path):
    index = {"weight_map": {"layer.0": "shard-00001.safetensors"}}
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))

    h1 = _model_hash(str(tmp_path))

    index["weight_map"]["layer.1"] = "shard-00002.safetensors"
    index_path.write_text(json.dumps(index))
    h2 = _model_hash(str(tmp_path))

    assert h1 != h2
    assert len(h1) == 64  # SHA-256 hex


def test_model_hash_from_file_list(tmp_path):
    (tmp_path / "shard-00001.safetensors").write_bytes(b"x" * 100)
    h1 = _model_hash(str(tmp_path))

    (tmp_path / "shard-00002.safetensors").write_bytes(b"y" * 200)
    h2 = _model_hash(str(tmp_path))

    assert h1 != h2


def test_serialize_deserialize_layout_specs():
    specs = {
        "gate_up_proj": ((512, 8, 16), torch.uint8),
        "gate_up_proj_scales": ((512, 8), torch.uint8),
        "down_proj": ((256, 4, 16), torch.bfloat16),
    }
    serialized = _serialize_layout_specs(specs)
    deserialized = _deserialize_layout_specs(serialized)

    for name in specs:
        orig_shape, orig_dtype = specs[name]
        rt_shape, rt_dtype = deserialized[name]
        assert tuple(rt_shape) == tuple(orig_shape)
        assert rt_dtype == orig_dtype
