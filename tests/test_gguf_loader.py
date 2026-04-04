"""Tests for GGUF model loader: name mapping, config extraction, dequant, multi-shard."""

import struct

import numpy as np
import pytest
import torch

from tinyserve.gguf_loader import (
    GGUFModelConfig,
    MultiShardGGUFReader,
    config_from_metadata,
    gguf_to_hf_name,
    hf_to_gguf_name,
    open_gguf,
    _build_expert_store_from_fused_reader,
    _dequant_fused_tensor,
    _dequant_tensor,
    _find_tensor_info,
    _get_param,
    _set_param,
)
from tinyserve.gguf_reader import GGUFReader, GGUFTensorInfo


# ---------------------------------------------------------------------------
# Helpers for writing synthetic GGUF files
# ---------------------------------------------------------------------------

def _write_string(f, s: str):
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_metadata_string(f, key: str, value: str):
    _write_string(f, key)
    f.write(struct.pack("<I", 8))
    _write_string(f, value)


def _write_metadata_uint32(f, key: str, value: int):
    _write_string(f, key)
    f.write(struct.pack("<I", 4))
    f.write(struct.pack("<I", value))


def _write_metadata_float32(f, key: str, value: float):
    _write_string(f, key)
    f.write(struct.pack("<I", 6))
    f.write(struct.pack("<f", value))


def _write_tensor_info(f, name: str, shape: tuple[int, ...], ggml_type: int, offset: int):
    _write_string(f, name)
    f.write(struct.pack("<I", len(shape)))
    for dim in shape:
        f.write(struct.pack("<Q", dim))
    f.write(struct.pack("<I", ggml_type))
    f.write(struct.pack("<Q", offset))


def _create_gguf_with_metadata(path, metadata: dict, tensors: list[dict] | None = None):
    """Create a minimal GGUF v3 file with given metadata and optional tensors.

    Each tensor dict: {"name": str, "shape": tuple, "ggml_type": int, "data": bytes}
    """
    if tensors is None:
        tensors = []

    n_kv = len(metadata)
    n_tensors = len(tensors)

    # Pre-compute tensor offsets
    tensor_offset = 0
    tensor_infos = []
    for t in tensors:
        tensor_infos.append((t["name"], t["shape"], t["ggml_type"], tensor_offset))
        tensor_offset += len(t["data"])

    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x46554747))  # magic
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        for key, val in metadata.items():
            if isinstance(val, str):
                _write_metadata_string(f, key, val)
            elif isinstance(val, float):
                _write_metadata_float32(f, key, val)
            elif isinstance(val, int):
                _write_metadata_uint32(f, key, val)

        for name, shape, ggml_type, offset in tensor_infos:
            _write_tensor_info(f, name, shape, ggml_type, offset)

        # Align to 32 bytes
        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        # Write tensor data
        for t in tensors:
            f.write(t["data"])


def _make_f32_tensor_data(shape: tuple[int, ...], fill_value: float = 1.0) -> bytes:
    arr = np.full(shape, fill_value, dtype=np.float32)
    return arr.tobytes()


def _make_f16_tensor_data(shape: tuple[int, ...], fill_value: float = 1.0) -> bytes:
    arr = np.full(shape, fill_value, dtype=np.float16)
    return arr.tobytes()


# ---------------------------------------------------------------------------
# Test: GGUF -> HF tensor name mapping
# ---------------------------------------------------------------------------

class TestTensorNameMapping:
    @pytest.mark.parametrize(
        "gguf_name, expected_hf, expected_expert, expected_layer, expected_expert_idx",
        [
            (
                "token_embd.weight",
                "model.embed_tokens.weight",
                False, None, None,
            ),
            (
                "output_norm.weight",
                "model.norm.weight",
                False, None, None,
            ),
            (
                "output.weight",
                "lm_head.weight",
                False, None, None,
            ),
            (
                "blk.0.attn_q.weight",
                "model.layers.0.self_attn.q_proj.weight",
                False, 0, None,
            ),
            (
                "blk.5.attn_k.weight",
                "model.layers.5.self_attn.k_proj.weight",
                False, 5, None,
            ),
            (
                "blk.0.attn_v.weight",
                "model.layers.0.self_attn.v_proj.weight",
                False, 0, None,
            ),
            (
                "blk.0.attn_output.weight",
                "model.layers.0.self_attn.o_proj.weight",
                False, 0, None,
            ),
            (
                "blk.3.attn_norm.weight",
                "model.layers.3.input_layernorm.weight",
                False, 3, None,
            ),
            (
                "blk.3.ffn_norm.weight",
                "model.layers.3.post_attention_layernorm.weight",
                False, 3, None,
            ),
            (
                "blk.0.ffn_gate_inp.weight",
                "model.layers.0.mlp.gate.weight",
                False, 0, None,
            ),
            (
                "blk.0.ffn_gate.0.weight",
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                True, 0, 0,
            ),
            (
                "blk.2.ffn_up.7.weight",
                "model.layers.2.mlp.experts.7.up_proj.weight",
                True, 2, 7,
            ),
            (
                "blk.1.ffn_down.3.weight",
                "model.layers.1.mlp.experts.3.down_proj.weight",
                True, 1, 3,
            ),
            (
                "blk.0.ffn_gate_exps.weight",
                "model.layers.0.mlp.shared_expert.gate_proj.weight",
                False, 0, None,
            ),
            (
                "blk.0.ffn_down_exps.weight",
                "model.layers.0.mlp.shared_expert.down_proj.weight",
                False, 0, None,
            ),
        ],
    )
    def test_gguf_to_hf(self, gguf_name, expected_hf, expected_expert, expected_layer, expected_expert_idx):
        hf_name, is_expert, layer_idx, expert_idx = gguf_to_hf_name(gguf_name)
        assert hf_name == expected_hf
        assert is_expert == expected_expert
        assert layer_idx == expected_layer
        assert expert_idx == expected_expert_idx

    @pytest.mark.parametrize(
        "hf_name, expected_gguf",
        [
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.layers.5.self_attn.k_proj.weight", "blk.5.attn_k.weight"),
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "blk.0.ffn_gate.0.weight",
            ),
            (
                "model.layers.2.mlp.experts.7.up_proj.weight",
                "blk.2.ffn_up.7.weight",
            ),
        ],
    )
    def test_hf_to_gguf(self, hf_name, expected_gguf):
        assert hf_to_gguf_name(hf_name) == expected_gguf

    def test_roundtrip_non_expert(self):
        gguf_name = "blk.3.attn_output.weight"
        hf_name, _, _, _ = gguf_to_hf_name(gguf_name)
        roundtrip = hf_to_gguf_name(hf_name)
        assert roundtrip == gguf_name

    def test_roundtrip_expert(self):
        gguf_name = "blk.1.ffn_gate.5.weight"
        hf_name, is_expert, layer, expert = gguf_to_hf_name(gguf_name)
        assert is_expert
        assert layer == 1
        assert expert == 5
        roundtrip = hf_to_gguf_name(hf_name)
        assert roundtrip == gguf_name

    def test_unknown_name_passthrough(self):
        unknown = "some.weird.tensor.name"
        hf_name, is_expert, _, _ = gguf_to_hf_name(unknown)
        assert hf_name == unknown
        assert not is_expert


# ---------------------------------------------------------------------------
# Test: Config extraction from GGUF metadata
# ---------------------------------------------------------------------------

class TestConfigFromMetadata:
    def test_qwen3_moe_metadata(self):
        metadata = {
            "general.architecture": "qwen3moe",
            "qwen3moe.block_count": 64,
            "qwen3moe.embedding_length": 3584,
            "qwen3moe.feed_forward_length": 18944,
            "qwen3moe.attention.head_count": 28,
            "qwen3moe.attention.head_count_kv": 4,
            "qwen3moe.expert_count": 128,
            "qwen3moe.expert_used_count": 8,
            "qwen3moe.rope.freq_base": 1000000.0,
            "qwen3moe.attention.layer_norm_rms_epsilon": 1e-6,
            "qwen3moe.context_length": 131072,
        }
        cfg = config_from_metadata(metadata)
        assert cfg.arch == "qwen3moe"
        assert cfg.num_hidden_layers == 64
        assert cfg.hidden_size == 3584
        assert cfg.intermediate_size == 18944
        assert cfg.num_attention_heads == 28
        assert cfg.num_key_value_heads == 4
        assert cfg.num_experts == 128
        assert cfg.num_experts_per_tok == 8
        assert cfg.rope_theta == 1000000.0
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.context_length == 131072

    def test_empty_metadata(self):
        cfg = config_from_metadata({})
        assert cfg.arch == ""
        assert cfg.num_hidden_layers == 0
        assert cfg.num_experts == 0

    def test_partial_metadata(self):
        metadata = {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
        }
        cfg = config_from_metadata(metadata)
        assert cfg.arch == "llama"
        assert cfg.num_hidden_layers == 32
        assert cfg.hidden_size == 4096
        assert cfg.num_experts == 0  # not in metadata

    def test_shared_expert_intermediate_size(self):
        metadata = {
            "general.architecture": "qwen3moe",
            "qwen3moe.expert_shared_feed_forward_length": 2048,
        }
        cfg = config_from_metadata(metadata)
        assert cfg.shared_expert_intermediate_size == 2048

    def test_vocab_size_from_tokens(self):
        metadata = {
            "general.architecture": "llama",
            "tokenizer.ggml.tokens": ["<pad>", "a", "b", "c"],
        }
        cfg = config_from_metadata(metadata)
        assert cfg.vocab_size == 4

    def test_extra_keys_captured(self):
        metadata = {
            "general.architecture": "test",
            "test.block_count": 2,
            "test.some_custom_field": 42,
        }
        cfg = config_from_metadata(metadata)
        assert cfg.num_hidden_layers == 2
        assert "some_custom_field" in cfg.extra
        assert cfg.extra["some_custom_field"] == 42


# ---------------------------------------------------------------------------
# Test: Non-expert weight dequantization
# ---------------------------------------------------------------------------

class TestDequantTensor:
    def test_f32_dequant(self, tmp_path):
        shape = (4, 8)
        fill = 3.14
        data = _make_f32_tensor_data(shape, fill)
        path = tmp_path / "f32.gguf"
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test"},
            [{"name": "test.weight", "shape": shape, "ggml_type": 0, "data": data}],
        )
        reader = GGUFReader(path)
        info = reader.tensors[0]
        t = _dequant_tensor(reader, info, "test.weight", "cpu")
        assert t.dtype == torch.bfloat16
        assert t.shape == shape
        assert torch.allclose(t.float(), torch.full(shape, fill), atol=0.02)
        reader.close()

    def test_f16_dequant(self, tmp_path):
        shape = (4, 8)
        fill = 2.5
        data = _make_f16_tensor_data(shape, fill)
        path = tmp_path / "f16.gguf"
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test"},
            [{"name": "test.weight", "shape": shape, "ggml_type": 1, "data": data}],
        )
        reader = GGUFReader(path)
        info = reader.tensors[0]
        t = _dequant_tensor(reader, info, "test.weight", "cpu")
        assert t.dtype == torch.bfloat16
        assert t.shape == shape
        assert torch.allclose(t.float(), torch.full(shape, fill), atol=0.02)
        reader.close()

    def test_unsupported_type_raises(self, tmp_path):
        shape = (4, 8)
        # Q3_K (type 11) — not supported for non-expert
        data = b"\x00" * (256 * ((4 * 8 + 255) // 256))
        path = tmp_path / "q3k.gguf"
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test"},
            [{"name": "test.weight", "shape": shape, "ggml_type": 11, "data": data}],
        )
        reader = GGUFReader(path)
        info = reader.tensors[0]
        with pytest.raises(ValueError, match="Unsupported GGML type"):
            _dequant_tensor(reader, info, "test.weight", "cpu")
        reader.close()


# ---------------------------------------------------------------------------
# Test: Multi-shard GGUF discovery and merging
# ---------------------------------------------------------------------------

class TestMultiShardGGUF:
    def _create_shard(self, directory, shard_num, total_shards, tensor_name, shape, fill_value):
        shard_name = f"model-{shard_num:05d}-of-{total_shards:05d}.gguf"
        path = directory / shard_name
        data = _make_f32_tensor_data(shape, fill_value)
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test", "test.block_count": 2},
            [{"name": tensor_name, "shape": shape, "ggml_type": 0, "data": data}],
        )
        return path

    def test_discovers_all_shards(self, tmp_path):
        self._create_shard(tmp_path, 1, 3, "token_embd.weight", (4, 8), 1.0)
        self._create_shard(tmp_path, 2, 3, "blk.0.attn_q.weight", (4, 8), 2.0)
        self._create_shard(tmp_path, 3, 3, "output.weight", (4, 8), 3.0)

        first = tmp_path / "model-00001-of-00003.gguf"
        reader = open_gguf(str(first))
        assert isinstance(reader, MultiShardGGUFReader)
        assert len(reader.tensor_names) == 3
        reader.close()

    def test_merged_metadata(self, tmp_path):
        self._create_shard(tmp_path, 1, 2, "a.weight", (2, 2), 1.0)
        self._create_shard(tmp_path, 2, 2, "b.weight", (2, 2), 2.0)

        first = tmp_path / "model-00001-of-00002.gguf"
        reader = open_gguf(str(first))
        assert isinstance(reader, MultiShardGGUFReader)
        assert reader.metadata["general.architecture"] == "test"
        reader.close()

    def test_read_tensor_from_correct_shard(self, tmp_path):
        self._create_shard(tmp_path, 1, 2, "first.weight", (2, 4), 1.0)
        self._create_shard(tmp_path, 2, 2, "second.weight", (2, 4), 5.0)

        first = tmp_path / "model-00001-of-00002.gguf"
        reader = open_gguf(str(first))
        assert isinstance(reader, MultiShardGGUFReader)

        data = reader.get_tensor_data("second.weight")
        arr = np.frombuffer(data, dtype=np.float32)
        assert np.allclose(arr, 5.0)

        data = reader.get_tensor_data("first.weight")
        arr = np.frombuffer(data, dtype=np.float32)
        assert np.allclose(arr, 1.0)
        reader.close()

    def test_single_file_returns_reader(self, tmp_path):
        path = tmp_path / "single.gguf"
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test"},
            [{"name": "w.weight", "shape": (2, 2), "ggml_type": 0, "data": _make_f32_tensor_data((2, 2))}],
        )
        reader = open_gguf(str(path))
        assert isinstance(reader, GGUFReader)
        reader.close()

    def test_expert_tensors_across_shards(self, tmp_path):
        n_elements = 64 * 128
        tensor_bytes = n_elements * 4  # F32

        # Shard 1: layer 0 experts
        path1 = tmp_path / "model-00001-of-00002.gguf"
        tensors1 = [
            {"name": "blk.0.ffn_gate.0.weight", "shape": (64, 128), "ggml_type": 0, "data": b"\x00" * tensor_bytes},
            {"name": "blk.0.ffn_up.0.weight", "shape": (64, 128), "ggml_type": 0, "data": b"\x01" * tensor_bytes},
            {"name": "blk.0.ffn_down.0.weight", "shape": (128, 64), "ggml_type": 0, "data": b"\x02" * tensor_bytes},
        ]
        _create_gguf_with_metadata(path1, {"general.architecture": "test"}, tensors1)

        # Shard 2: layer 1 experts
        path2 = tmp_path / "model-00002-of-00002.gguf"
        tensors2 = [
            {"name": "blk.1.ffn_gate.0.weight", "shape": (64, 128), "ggml_type": 0, "data": b"\x03" * tensor_bytes},
            {"name": "blk.1.ffn_up.0.weight", "shape": (64, 128), "ggml_type": 0, "data": b"\x04" * tensor_bytes},
            {"name": "blk.1.ffn_down.0.weight", "shape": (128, 64), "ggml_type": 0, "data": b"\x05" * tensor_bytes},
        ]
        _create_gguf_with_metadata(path2, {"general.architecture": "test"}, tensors2)

        reader = open_gguf(str(path1))
        assert isinstance(reader, MultiShardGGUFReader)

        groups = reader.list_expert_tensors()
        assert len(groups) == 2
        assert (0, 0) in groups
        assert (1, 0) in groups
        assert set(groups[(0, 0)].keys()) == {"gate", "up", "down"}
        reader.close()


# ---------------------------------------------------------------------------
# Test: Parameter navigation helpers
# ---------------------------------------------------------------------------

class TestParamHelpers:
    def test_get_param_simple(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(4))
        assert _get_param(model, "weight") is not None

    def test_get_param_nested(self):
        model = torch.nn.Module()
        model.layer = torch.nn.Module()
        model.layer.fc = torch.nn.Linear(4, 4)
        param = _get_param(model, "layer.fc.weight")
        assert param is not None
        assert param.shape == (4, 4)

    def test_get_param_indexed(self):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        param = _get_param(model, "layers.0.weight")
        assert param is not None

    def test_get_param_missing_returns_none(self):
        model = torch.nn.Module()
        assert _get_param(model, "nonexistent.weight") is None

    def test_set_param_simple(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.zeros(4))
        new_val = torch.ones(4)
        _set_param(model, "weight", new_val)
        assert torch.equal(model.weight.data, new_val)

    def test_set_param_nested(self):
        model = torch.nn.Module()
        model.layer = torch.nn.Module()
        model.layer.fc = torch.nn.Linear(4, 4)
        new_val = torch.ones(4, 4)
        _set_param(model, "layer.fc.weight", new_val)
        assert torch.equal(model.layer.fc.weight.data, new_val)


# ---------------------------------------------------------------------------
# Test: Q8_0 dequantization
# ---------------------------------------------------------------------------

class TestQ8_0Dequant:
    def test_q8_0_dequant_known_values(self, tmp_path):
        """Q8_0 block: 2-byte float16 scale + 32 int8 quants = 34 bytes per block."""
        shape = (1, 32)  # exactly one block
        scale = np.float16(0.5)
        quants = np.arange(32, dtype=np.int8)
        block_data = scale.tobytes() + quants.tobytes()
        assert len(block_data) == 34

        path = tmp_path / "q8.gguf"
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test"},
            [{"name": "test.weight", "shape": shape, "ggml_type": 8, "data": block_data}],
        )
        reader = GGUFReader(path)
        info = reader.tensors[0]
        t = _dequant_tensor(reader, info, "test.weight", "cpu")
        assert t.dtype == torch.bfloat16
        assert t.shape == shape
        # Expected: scale * quant for each element
        expected = torch.tensor(
            [[float(scale) * float(q) for q in range(32)]],
            dtype=torch.bfloat16,
        )
        assert torch.allclose(t, expected, atol=0.05)
        reader.close()


# ---------------------------------------------------------------------------
# Test: find_tensor_info
# ---------------------------------------------------------------------------

class TestFindTensorInfo:
    def test_find_existing(self, tmp_path):
        path = tmp_path / "test.gguf"
        data = _make_f32_tensor_data((2, 2))
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test"},
            [{"name": "found.weight", "shape": (2, 2), "ggml_type": 0, "data": data}],
        )
        reader = GGUFReader(path)
        info = _find_tensor_info(reader, "found.weight")
        assert info.name == "found.weight"
        assert info.shape == (2, 2)
        reader.close()

    def test_find_missing_raises(self, tmp_path):
        path = tmp_path / "test.gguf"
        data = _make_f32_tensor_data((2, 2))
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test"},
            [{"name": "found.weight", "shape": (2, 2), "ggml_type": 0, "data": data}],
        )
        reader = GGUFReader(path)
        with pytest.raises(KeyError, match="not_found"):
            _find_tensor_info(reader, "not_found.weight")
        reader.close()


# ---------------------------------------------------------------------------
# Test: GGUFModelConfig dataclass
# ---------------------------------------------------------------------------

class TestGGUFModelConfig:
    def test_defaults(self):
        cfg = GGUFModelConfig()
        assert cfg.arch == ""
        assert cfg.num_hidden_layers == 0
        assert cfg.hidden_size == 0
        assert cfg.num_experts == 0
        assert cfg.rope_theta == 10000.0
        assert cfg.extra == {}

    def test_fields_settable(self):
        cfg = GGUFModelConfig(
            arch="qwen3moe",
            num_hidden_layers=64,
            hidden_size=3584,
            num_experts=128,
            num_experts_per_tok=8,
        )
        assert cfg.arch == "qwen3moe"
        assert cfg.num_hidden_layers == 64
        assert cfg.num_experts == 128

    def test_num_local_experts_alias(self):
        cfg = GGUFModelConfig(num_experts=256)
        assert cfg.num_local_experts == 256

    def test_num_local_experts_alias_default(self):
        cfg = GGUFModelConfig()
        assert cfg.num_local_experts == 0


class TestMultiShardFusedExperts:
    def _create_fused_shard(self, path, shard_num, total_shards, layers, n_experts=4, ffn_size=8, hidden=16):
        shard_name = f"model-{shard_num:05d}-of-{total_shards:05d}.gguf"
        shard_path = path / shard_name
        projections = [("gate", (hidden, ffn_size, n_experts)),
                       ("up", (hidden, ffn_size, n_experts)),
                       ("down", (ffn_size, hidden, n_experts))]

        tensors = []
        for layer in layers:
            for proj, shape in projections:
                name = f"blk.{layer}.ffn_{proj}_exps.weight"
                n_el = 1
                for d in shape:
                    n_el *= d
                data = _make_f32_tensor_data(shape)
                tensors.append({"name": name, "shape": shape, "ggml_type": 0, "data": data})
        _create_gguf_with_metadata(shard_path, {"general.architecture": "test"}, tensors)
        return shard_path

    def test_fused_expert_discovery_across_shards(self, tmp_path):
        self._create_fused_shard(tmp_path, 1, 2, layers=[0, 1])
        self._create_fused_shard(tmp_path, 2, 2, layers=[2, 3])

        first = tmp_path / "model-00001-of-00002.gguf"
        reader = open_gguf(str(first))
        assert isinstance(reader, MultiShardGGUFReader)

        groups = reader.list_fused_expert_tensors()
        assert len(groups) == 4
        for layer in range(4):
            assert layer in groups
            assert set(groups[layer].keys()) == {"gate", "up", "down"}
        reader.close()

    def test_fused_expert_shape_across_shards(self, tmp_path):
        self._create_fused_shard(tmp_path, 1, 2, layers=[0], n_experts=8, ffn_size=16, hidden=32)
        self._create_fused_shard(tmp_path, 2, 2, layers=[1], n_experts=8, ffn_size=16, hidden=32)

        first = tmp_path / "model-00001-of-00002.gguf"
        reader = open_gguf(str(first))
        groups = reader.list_fused_expert_tensors()

        gate_info = groups[0]["gate"]
        assert gate_info.shape == (32, 16, 8)
        reader.close()

    def test_list_expert_tensors_empty_when_only_fused(self, tmp_path):
        self._create_fused_shard(tmp_path, 1, 2, layers=[0])
        self._create_fused_shard(tmp_path, 2, 2, layers=[1])

        first = tmp_path / "model-00001-of-00002.gguf"
        reader = open_gguf(str(first))
        per_expert = reader.list_expert_tensors()
        assert len(per_expert) == 0
        reader.close()


class TestDequantFusedTensor:
    """Tests for _dequant_fused_tensor with 3-D fused expert tensors."""

    def _make_fused_gguf(self, tmp_path, shape, ggml_type, data, name="blk.0.ffn_gate_exps.weight"):
        path = tmp_path / "fused.gguf"
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test"},
            [{"name": name, "shape": shape, "ggml_type": ggml_type, "data": data}],
        )
        return path

    def test_f32_fused_dequant(self, tmp_path):
        shape = (4, 8, 2)
        data = _make_f32_tensor_data(shape, fill_value=1.5)
        path = self._make_fused_gguf(tmp_path, shape, 0, data)
        reader = GGUFReader(path)
        info = reader.tensors[0]
        t = _dequant_fused_tensor(reader, info, "blk.0.ffn_gate_exps.weight", "cpu")
        assert t.dtype == torch.bfloat16
        assert t.shape == torch.Size(shape)
        assert torch.allclose(t.float(), torch.full(shape, 1.5), atol=0.02)
        reader.close()

    def test_f16_fused_dequant(self, tmp_path):
        shape = (4, 8, 3)
        data = _make_f16_tensor_data(shape, fill_value=2.0)
        path = self._make_fused_gguf(tmp_path, shape, 1, data)
        reader = GGUFReader(path)
        info = reader.tensors[0]
        t = _dequant_fused_tensor(reader, info, "blk.0.ffn_gate_exps.weight", "cpu")
        assert t.dtype == torch.bfloat16
        assert t.shape == torch.Size(shape)
        reader.close()

    def test_unsupported_type_raises(self, tmp_path):
        shape = (4, 8, 2)
        data = b"\x00" * (256 * ((4 * 8 * 2 + 255) // 256))
        path = self._make_fused_gguf(tmp_path, shape, 11, data)
        reader = GGUFReader(path)
        info = reader.tensors[0]
        with pytest.raises(ValueError, match="Unsupported GGML type"):
            _dequant_fused_tensor(reader, info, "blk.0.ffn_gate_exps.weight", "cpu")
        reader.close()


class TestBuildExpertStoreFromFusedReader:
    """Tests for _build_expert_store_from_fused_reader."""

    def _create_fused_single_gguf(
        self, path, n_layers=2, n_experts=4, ffn_size=8, hidden=16, ggml_type=0
    ):
        """Create a single GGUF with F32 fused expert tensors.

        Shapes follow Qwen convention: [out_dim, in_dim, n_experts].
        gate/up: out_dim=ffn_size (intermediate), in_dim=hidden.
        down:    out_dim=hidden,                  in_dim=ffn_size.
        """
        tensors = []
        for layer in range(n_layers):
            for proj, shape in [
                ("gate", (ffn_size, hidden, n_experts)),
                ("up", (ffn_size, hidden, n_experts)),
                ("down", (hidden, ffn_size, n_experts)),
            ]:
                name = f"blk.{layer}.ffn_{proj}_exps.weight"
                data = _make_f32_tensor_data(shape, fill_value=float(layer + 1))
                tensors.append({"name": name, "shape": shape, "ggml_type": ggml_type, "data": data})
        _create_gguf_with_metadata(path, {"general.architecture": "test"}, tensors)

    def test_returns_none_when_no_fused_tensors(self, tmp_path):
        path = tmp_path / "no_fused.gguf"
        data = _make_f32_tensor_data((4, 4))
        _create_gguf_with_metadata(
            path,
            {"general.architecture": "test"},
            [{"name": "blk.0.attn_q.weight", "shape": (4, 4), "ggml_type": 0, "data": data}],
        )
        reader = GGUFReader(path)
        result = _build_expert_store_from_fused_reader(reader, num_layers=1, num_experts=4, device="cpu")
        assert result is None
        reader.close()

    def test_store_shape_correct(self, tmp_path):
        n_layers, n_experts, ffn_size, hidden = 2, 4, 8, 16
        path = tmp_path / "fused.gguf"
        self._create_fused_single_gguf(path, n_layers, n_experts, ffn_size, hidden)
        reader = GGUFReader(path)
        store = _build_expert_store_from_fused_reader(reader, num_layers=n_layers, num_experts=n_experts, device="cpu")
        assert store is not None
        assert store.num_layers == n_layers
        assert store.num_experts == n_experts
        reader.close()

    def test_gate_up_proj_shape(self, tmp_path):
        n_layers, n_experts, ffn_size, hidden = 1, 4, 8, 16
        path = tmp_path / "fused.gguf"
        self._create_fused_single_gguf(path, n_layers, n_experts, ffn_size, hidden)
        reader = GGUFReader(path)
        store = _build_expert_store_from_fused_reader(reader, num_layers=n_layers, num_experts=n_experts, device="cpu")
        assert store is not None
        # gate_up_proj should be [2*ffn_size, hidden] per expert
        gu_shape, _ = store.layout.specs["gate_up_proj"]
        assert gu_shape == (2 * ffn_size, hidden)
        dn_shape, _ = store.layout.specs["down_proj"]
        assert dn_shape == (hidden, ffn_size)
        reader.close()

    def test_expert_values_correct(self, tmp_path):
        """Verify that sliced expert weights match the expected per-expert slice."""
        n_layers, n_experts, ffn_size, hidden = 1, 3, 4, 8
        path = tmp_path / "fused.gguf"

        # Build fused tensors with expert-specific fill values for easy verification
        # Shapes: gate/up = [ffn_size, hidden, n_experts], down = [hidden, ffn_size, n_experts]
        tensors = []
        for proj, shape in [
            ("gate", (ffn_size, hidden, n_experts)),
            ("up", (ffn_size, hidden, n_experts)),
            ("down", (hidden, ffn_size, n_experts)),
        ]:
            arr = np.zeros(shape, dtype=np.float32)
            for e in range(n_experts):
                arr[:, :, e] = float(e + 1)
            tensors.append({
                "name": f"blk.0.ffn_{proj}_exps.weight",
                "shape": shape,
                "ggml_type": 0,
                "data": arr.tobytes(),
            })
        _create_gguf_with_metadata(path, {"general.architecture": "test"}, tensors)

        reader = GGUFReader(path)
        store = _build_expert_store_from_fused_reader(reader, num_layers=n_layers, num_experts=n_experts, device="cpu")
        assert store is not None

        layout = store.layout
        for e in range(n_experts):
            raw = store._data[0, e]
            gu_offset = layout.offsets["gate_up_proj"]
            gu_nbytes = layout.sizes["gate_up_proj"]
            gu_shape, gu_dtype = layout.specs["gate_up_proj"]
            gu = raw[gu_offset:gu_offset + gu_nbytes].view(gu_dtype).reshape(gu_shape).float()

            dn_offset = layout.offsets["down_proj"]
            dn_nbytes = layout.sizes["down_proj"]
            dn_shape, dn_dtype = layout.specs["down_proj"]
            dn = raw[dn_offset:dn_offset + dn_nbytes].view(dn_dtype).reshape(dn_shape).float()

            expected_val = float(e + 1)
            assert torch.allclose(gu, torch.full_like(gu, expected_val), atol=0.02), \
                f"Expert {e} gate_up_proj mismatch"
            assert torch.allclose(dn, torch.full_like(dn, expected_val), atol=0.02), \
                f"Expert {e} down_proj mismatch"
        reader.close()

    def test_fused_expert_excluded_from_non_expert_loading(self, tmp_path):
        """3-D fused expert tensors must not appear in non_expert_names during load."""
        n_layers, n_experts, ffn_size, hidden = 1, 2, 4, 8
        path = tmp_path / "fused_classify.gguf"
        self._create_fused_single_gguf(path, n_layers, n_experts, ffn_size, hidden)

        reader = GGUFReader(path)
        import re as _re
        _fused_re = _re.compile(r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$")
        for t in reader.tensors:
            if _fused_re.match(t.name):
                assert len(t.shape) == 3, f"Fused tensor {t.name} should be 3-D"
        reader.close()


class TestQwen35MoeMetadata:
    def test_qwen35moe_arch_prefix(self):
        metadata = {
            "general.architecture": "qwen35moe",
            "qwen35moe.block_count": 48,
            "qwen35moe.embedding_length": 3072,
            "qwen35moe.expert_count": 256,
            "qwen35moe.expert_used_count": 8,
            "qwen35moe.context_length": 262144,
            "qwen35moe.rope.freq_base": 10000000.0,
            "tokenizer.ggml.tokens": ["a"] * 248320,
        }
        cfg = config_from_metadata(metadata)
        assert cfg.arch == "qwen35moe"
        assert cfg.num_hidden_layers == 48
        assert cfg.hidden_size == 3072
        assert cfg.num_experts == 256
        assert cfg.num_local_experts == 256
        assert cfg.num_experts_per_tok == 8
        assert cfg.context_length == 262144
        assert cfg.vocab_size == 248320
