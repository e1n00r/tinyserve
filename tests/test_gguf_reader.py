"""Tests for GGUF file format reader."""

import struct

import pytest

from tinyserve.gguf_reader import GGML_TYPES, GGUFReader, GGUFTensorInfo


def _write_string(f, s: str):
    """Write a GGUF string: uint64 length + raw bytes."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_metadata_string(f, key: str, value: str):
    """Write a GGUF metadata KV pair with string value (type=8)."""
    _write_string(f, key)
    f.write(struct.pack("<I", 8))  # type = string
    _write_string(f, value)


def _write_metadata_uint32(f, key: str, value: int):
    """Write a GGUF metadata KV pair with uint32 value (type=4)."""
    _write_string(f, key)
    f.write(struct.pack("<I", 4))
    f.write(struct.pack("<I", value))


def _write_metadata_float32(f, key: str, value: float):
    """Write a GGUF metadata KV pair with float32 value (type=6)."""
    _write_string(f, key)
    f.write(struct.pack("<I", 6))
    f.write(struct.pack("<f", value))


def _write_tensor_info(f, name: str, shape: tuple[int, ...], ggml_type: int, offset: int):
    """Write a single GGUF tensor info entry."""
    _write_string(f, name)
    f.write(struct.pack("<I", len(shape)))
    for dim in shape:
        f.write(struct.pack("<Q", dim))
    f.write(struct.pack("<I", ggml_type))
    f.write(struct.pack("<Q", offset))


def _create_synthetic_gguf(path, n_layers=2, n_experts=4, ggml_type=0):
    """Create a minimal GGUF v3 file with fake expert tensors.

    Uses the given ggml_type for all tensors. Default is F32 (type=0).
    Each tensor is shape (64, 128) — small enough for tests.
    Tensor data is filled with a known byte pattern based on (layer, expert, proj).
    """
    type_name, bytes_per_block, block_size = GGML_TYPES.get(ggml_type, ("F32", 4, 1))

    tensor_shape = (64, 128)
    n_elements = tensor_shape[0] * tensor_shape[1]
    n_blocks = (n_elements + block_size - 1) // block_size
    tensor_nbytes = n_blocks * bytes_per_block

    projections = ["gate", "up", "down"]
    n_tensors = n_layers * n_experts * len(projections)
    n_kv = 2  # model name + num experts

    tensor_infos = []
    data_offset = 0
    for layer in range(n_layers):
        for expert in range(n_experts):
            for proj in projections:
                name = f"blk.{layer}.ffn_{proj}.{expert}.weight"
                tensor_infos.append((name, tensor_shape, ggml_type, data_offset))
                data_offset += tensor_nbytes

    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x46554747))  # magic "GGUF"
        f.write(struct.pack("<I", 3))  # version 3
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        # Metadata
        _write_metadata_string(f, "general.name", "test-model")
        _write_metadata_uint32(f, "test.num_experts", n_experts)

        # Tensor infos
        for name, shape, gtype, offset in tensor_infos:
            _write_tensor_info(f, name, shape, gtype, offset)

        # Align to 32 bytes for data section
        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        # Tensor data: fill with known pattern
        for layer in range(n_layers):
            for expert in range(n_experts):
                for proj_idx, proj in enumerate(projections):
                    pattern_byte = (layer * 100 + expert * 10 + proj_idx) & 0xFF
                    f.write(bytes([pattern_byte]) * tensor_nbytes)

    return tensor_infos


class TestGGUFReaderParsing:
    def test_parse_magic_and_version(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path)

        reader = GGUFReader(path)
        assert reader.version == 3
        reader.close()

    def test_invalid_magic_raises(self, tmp_path):
        path = tmp_path / "bad.gguf"
        with open(path, "wb") as f:
            f.write(struct.pack("<I", 0xDEADBEEF))
            f.write(b"\x00" * 20)

        with pytest.raises(ValueError, match="Not a GGUF file"):
            GGUFReader(path)

    def test_metadata_extraction(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path)

        reader = GGUFReader(path)
        assert reader.metadata["general.name"] == "test-model"
        assert reader.metadata["test.num_experts"] == 4
        reader.close()

    def test_tensor_count(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=2, n_experts=4)

        reader = GGUFReader(path)
        # 2 layers * 4 experts * 3 projections = 24 tensors
        assert len(reader.tensors) == 24
        reader.close()

    def test_tensor_info_shapes(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path)

        reader = GGUFReader(path)
        for t in reader.tensors:
            assert t.shape == (64, 128)
            assert t.ggml_type == 0
            assert t.ggml_type_name == "F32"
        reader.close()

    def test_tensor_info_nbytes_f32(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, ggml_type=0)

        reader = GGUFReader(path)
        for t in reader.tensors:
            # F32: 4 bytes per element, 64*128 elements
            assert t.nbytes == 64 * 128 * 4
        reader.close()


class TestGGUFReaderExpertGrouping:
    def test_expert_tensor_grouping(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=2, n_experts=4)

        reader = GGUFReader(path)
        groups = reader.list_expert_tensors()

        assert len(groups) == 8  # 2 layers * 4 experts
        for layer in range(2):
            for expert in range(4):
                key = (layer, expert)
                assert key in groups
                assert set(groups[key].keys()) == {"gate", "up", "down"}
        reader.close()

    def test_expert_grouping_tensor_info_refs(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        reader = GGUFReader(path)
        groups = reader.list_expert_tensors()

        info = groups[(0, 1)]["gate"]
        assert isinstance(info, GGUFTensorInfo)
        assert info.name == "blk.0.ffn_gate.1.weight"
        reader.close()


class TestGGUFReaderData:
    def test_tensor_data_readback(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        reader = GGUFReader(path)
        groups = reader.list_expert_tensors()

        # blk.0.ffn_gate.0.weight => layer=0, expert=0, proj_idx=0
        # pattern_byte = (0*100 + 0*10 + 0) & 0xFF = 0
        info_00_gate = groups[(0, 0)]["gate"]
        data = reader.get_tensor_data(info_00_gate)
        assert len(data) == info_00_gate.nbytes
        assert all(b == 0 for b in data)

        # blk.0.ffn_up.1.weight => layer=0, expert=1, proj_idx=1
        # pattern_byte = (0*100 + 1*10 + 1) & 0xFF = 11
        info_01_up = groups[(0, 1)]["up"]
        data = reader.get_tensor_data(info_01_up)
        assert all(b == 11 for b in data)

        reader.close()

    def test_tensor_data_distinct_per_expert(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=2, n_experts=3)

        reader = GGUFReader(path)
        groups = reader.list_expert_tensors()

        # Each (layer, expert, proj) combination should have a unique pattern byte
        seen_patterns = set()
        for (layer, expert), projs in groups.items():
            for proj_idx, proj_name in enumerate(["gate", "up", "down"]):
                data = reader.get_tensor_data(projs[proj_name])
                pattern = data[0]
                expected = (layer * 100 + expert * 10 + proj_idx) & 0xFF
                assert pattern == expected
                seen_patterns.add(pattern)

        assert len(seen_patterns) == 2 * 3 * 3  # all unique
        reader.close()


class TestGGUFNbytesCalculation:
    @pytest.mark.parametrize(
        "ggml_type, expected_name, bytes_per_block, block_size",
        [
            (0, "F32", 4, 1),
            (1, "F16", 2, 1),
            (2, "Q4_0", 18, 32),
            (8, "Q8_0", 34, 32),
            (14, "Q4_K", 144, 256),
            (15, "Q5_K", 176, 256),
            (16, "Q6_K", 210, 256),
        ],
    )
    def test_nbytes_for_quant_type(self, ggml_type, expected_name, bytes_per_block, block_size):
        n_elements = 256 * 512
        n_blocks = (n_elements + block_size - 1) // block_size
        expected_nbytes = n_blocks * bytes_per_block

        info = GGUFTensorInfo(
            name="test",
            shape=(256, 512),
            ggml_type=ggml_type,
            ggml_type_name=expected_name,
            offset=0,
            nbytes=expected_nbytes,
            block_size=block_size,
        )
        assert info.nbytes == expected_nbytes
        assert info.ggml_type_name == expected_name

    def test_q4_k_nbytes_calculation(self):
        """Q4_K: 144 bytes per block of 256 elements."""
        n_elements = 4096 * 14336
        n_blocks = (n_elements + 256 - 1) // 256
        expected = n_blocks * 144
        assert expected == 229376 * 144


class TestGGUFReaderContextManager:
    def test_close_releases_file(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path)

        reader = GGUFReader(path)
        assert not reader._file.closed
        reader.close()
        assert reader._file.closed


class TestGGUFMetadataTypes:
    def test_float32_metadata(self, tmp_path):
        path = tmp_path / "meta.gguf"
        with open(path, "wb") as f:
            f.write(struct.pack("<I", 0x46554747))  # magic
            f.write(struct.pack("<I", 3))  # version
            f.write(struct.pack("<Q", 0))  # n_tensors
            f.write(struct.pack("<Q", 1))  # n_kv

            _write_metadata_float32(f, "test.temperature", 0.75)

        reader = GGUFReader(path)
        assert abs(reader.metadata["test.temperature"] - 0.75) < 1e-6
        reader.close()

    def test_bool_metadata(self, tmp_path):
        path = tmp_path / "meta.gguf"
        with open(path, "wb") as f:
            f.write(struct.pack("<I", 0x46554747))
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 0))
            f.write(struct.pack("<Q", 1))

            _write_string(f, "test.flag")
            f.write(struct.pack("<I", 7))  # type = bool
            f.write(struct.pack("<?", True))

        reader = GGUFReader(path)
        assert reader.metadata["test.flag"] is True
        reader.close()

    def test_uint64_metadata(self, tmp_path):
        path = tmp_path / "meta.gguf"
        with open(path, "wb") as f:
            f.write(struct.pack("<I", 0x46554747))
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 0))
            f.write(struct.pack("<Q", 1))

            _write_string(f, "test.big_number")
            f.write(struct.pack("<I", 10))  # type = uint64
            f.write(struct.pack("<Q", 2**40))

        reader = GGUFReader(path)
        assert reader.metadata["test.big_number"] == 2**40
        reader.close()


def _create_fused_expert_gguf(path, n_layers=2, n_experts=4, expert_ffn_size=16, hidden_size=32):
    """Create a minimal GGUF v3 file with fused expert tensors (Qwen3.5 style).

    Tensor naming: blk.<L>.ffn_{gate,up,down}_exps.weight
    Shape: (hidden_size, expert_ffn_size, n_experts) for gate/up
            (expert_ffn_size, hidden_size, n_experts) for down
    All F32 (type=0) for simplicity.
    """
    projections = [
        ("gate", (hidden_size, expert_ffn_size, n_experts)),
        ("up", (hidden_size, expert_ffn_size, n_experts)),
        ("down", (expert_ffn_size, hidden_size, n_experts)),
    ]

    tensors = []
    for layer in range(n_layers):
        for proj, shape in projections:
            name = f"blk.{layer}.ffn_{proj}_exps.weight"
            n_elements = 1
            for d in shape:
                n_elements *= d
            tensors.append((name, shape, 0, n_elements * 4))  # F32

    n_tensors = len(tensors)
    n_kv = 1

    data_offset = 0
    offsets = []
    for _, _, _, nbytes in tensors:
        offsets.append(data_offset)
        data_offset += nbytes

    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x46554747))  # magic
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        _write_string(f, "general.name")
        f.write(struct.pack("<I", 8))
        _write_string(f, "fused-test")

        for (name, shape, ggml_type, _), offset in zip(tensors, offsets):
            _write_string(f, name)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ggml_type))
            f.write(struct.pack("<Q", offset))

        pos = f.tell()
        aligned = (pos + 31) & ~31
        f.write(b"\x00" * (aligned - pos))

        for _, _, _, nbytes in tensors:
            f.write(b"\x00" * nbytes)


class TestGGUFReaderFusedExperts:
    def test_fused_expert_grouping(self, tmp_path):
        path = tmp_path / "fused.gguf"
        _create_fused_expert_gguf(path, n_layers=2)

        reader = GGUFReader(path)
        groups = reader.list_fused_expert_tensors()

        assert len(groups) == 2
        assert 0 in groups
        assert 1 in groups
        assert set(groups[0].keys()) == {"gate", "up", "down"}
        assert set(groups[1].keys()) == {"gate", "up", "down"}
        reader.close()

    def test_fused_expert_tensor_info_shape(self, tmp_path):
        path = tmp_path / "fused.gguf"
        _create_fused_expert_gguf(path, n_layers=1, n_experts=8, expert_ffn_size=16, hidden_size=32)

        reader = GGUFReader(path)
        groups = reader.list_fused_expert_tensors()

        gate_info = groups[0]["gate"]
        assert isinstance(gate_info, GGUFTensorInfo)
        assert gate_info.shape == (32, 16, 8)
        assert gate_info.ggml_type_name == "F32"
        reader.close()

    def test_list_expert_tensors_empty_for_fused_format(self, tmp_path):
        path = tmp_path / "fused.gguf"
        _create_fused_expert_gguf(path, n_layers=2)

        reader = GGUFReader(path)
        per_expert = reader.list_expert_tensors()
        assert len(per_expert) == 0
        reader.close()


class TestGGUFQuantizedTensors:
    def test_q4_k_tensor_info(self, tmp_path):
        """Parse a file with Q4_K tensors and verify nbytes calculation."""
        path = tmp_path / "q4k.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=1, ggml_type=12)

        reader = GGUFReader(path)
        t = reader.tensors[0]
        assert t.ggml_type == 12
        assert t.ggml_type_name == "Q4_K"
        assert t.block_size == 256

        n_elements = 64 * 128
        n_blocks = (n_elements + 256 - 1) // 256
        assert t.nbytes == n_blocks * 144
        reader.close()

    def test_q4_k_data_readback(self, tmp_path):
        """Q4_K tensor data can be read back correctly."""
        path = tmp_path / "q4k.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=1, ggml_type=12)

        reader = GGUFReader(path)
        t = reader.tensors[0]
        data = reader.get_tensor_data(t)
        assert len(data) == t.nbytes
        # Pattern byte for layer=0, expert=0, proj_idx=0 is 0
        assert all(b == 0 for b in data)
        reader.close()
