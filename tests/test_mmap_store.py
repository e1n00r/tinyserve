"""Tests for MmapExpertStore — zero-copy GGUF expert storage."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import requires_cuda
from tests.test_gguf_reader import _create_synthetic_gguf


class TestMmapStoreFromPerExpertGGUF:
    def test_num_layers_num_experts_from_synthetic_gguf(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=2, n_experts=4)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store.num_layers == 2
        assert store.num_experts == 4
        store.close()

    def test_expert_bytes_positive(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store.expert_bytes > 0
        store.close()

    def test_expert_bytes_equals_sum_of_projections(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        # F32, shape (64, 128) => 64*128*4 bytes per projection, 3 projections
        expected = 64 * 128 * 4 * 3
        assert store.expert_bytes == expected
        store.close()

    def test_buffer_expert_bytes_equals_expert_bytes(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store.buffer_expert_bytes == store.expert_bytes
        store.close()


class TestMmapStoreInterfaceAttributes:
    def test_has_layout(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.expert_store import TensorLayout
        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert isinstance(store.layout, TensorLayout)
        store.close()

    def test_layout_has_gate_up_down_specs(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert set(store.layout.specs.keys()) == {"gate", "up", "down"}
        store.close()

    def test_layout_specs_are_uint8(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        for name, (shape, dtype) in store.layout.specs.items():
            assert dtype == torch.uint8, f"{name} spec dtype should be uint8"
        store.close()

    def test_has_bf16_layout(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.expert_store import TensorLayout
        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert isinstance(store._bf16_layout, TensorLayout)
        store.close()

    def test_bf16_layout_same_as_layout(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store._bf16_layout is store.layout
        store.close()

    def test_has_ggml_types(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert isinstance(store.ggml_types, dict)
        assert set(store.ggml_types.keys()) == {"gate", "up", "down"}
        store.close()

    def test_has_proj_shapes(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert isinstance(store.proj_shapes, dict)
        assert set(store.proj_shapes.keys()) == {"gate", "up", "down"}
        for name, shape in store.proj_shapes.items():
            assert len(shape) == 2
        store.close()

    def test_fp8_property_returns_false(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        assert store._fp8 is False
        store.close()

    def test_allocate_buffer_returns_expert_buffer(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.expert_store import ExpertBuffer
        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        buf = store.allocate_buffer(torch.device("cpu"))
        assert isinstance(buf, ExpertBuffer)
        store.close()


class TestMmapStoreGetExpertData:
    def test_get_expert_data_returns_tensor(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=2, n_experts=4)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        data = store.get_expert_data(0, 0)
        assert isinstance(data, torch.Tensor)
        store.close()

    def test_get_expert_data_correct_size(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        data = store.get_expert_data(0, 0)
        assert data.numel() == store.expert_bytes
        store.close()

    def test_get_expert_data_is_uint8(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        data = store.get_expert_data(0, 0)
        assert data.dtype == torch.uint8
        store.close()

    def test_get_expert_data_different_experts_differ(self, tmp_path):
        """Different (layer, expert) pairs should produce different raw bytes."""
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=2, n_experts=4)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        d00 = store.get_expert_data(0, 0)
        d01 = store.get_expert_data(0, 1)
        assert not torch.equal(d00, d01)
        store.close()

    def test_get_expert_data_matches_gguf_pattern(self, tmp_path):
        """Layer=0, expert=0, gate projection should match byte pattern from _create_synthetic_gguf."""
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2, ggml_type=0)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        data = store.get_expert_data(0, 0)
        # F32 shape (64,128) = 32768 bytes per projection
        # gate projection (proj_idx=0): pattern_byte = (0*100 + 0*10 + 0) & 0xFF = 0
        gate_nbytes = store.layout.sizes["gate"]
        gate_bytes = data[:gate_nbytes]
        assert gate_bytes.tolist() == [0] * gate_nbytes
        store.close()


@requires_cuda
class TestMmapStoreCopyToBuffer:
    def test_copy_to_buffer_fills_gpu_buffer(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        device = torch.device("cuda")
        buf = store.allocate_buffer(device)
        store.copy_to_buffer(buf, 0, 1)  # expert 1 has non-zero pattern bytes
        assert buf.packed.device.type == "cuda"
        assert buf.packed.numel() == store.expert_bytes
        store.close()

    def test_copy_to_buffer_non_zero_data(self, tmp_path):
        """Expert 1 of layer 0 has pattern byte 10, so packed should be non-zero."""
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        device = torch.device("cuda")
        buf = store.allocate_buffer(device)
        store.copy_to_buffer(buf, 0, 1)
        assert buf.packed.sum().item() > 0
        store.close()

    def test_copy_to_buffer_non_blocking(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        device = torch.device("cuda")
        buf = store.allocate_buffer(device)
        store.copy_to_buffer(buf, 0, 1, non_blocking=True)
        torch.cuda.synchronize()
        assert buf.packed.sum().item() > 0
        store.close()

    def test_copy_to_buffer_slot(self, tmp_path):
        path = tmp_path / "test.gguf"
        _create_synthetic_gguf(path, n_layers=1, n_experts=2)

        from tinyserve.expert_cache import ExpertCache
        from tinyserve.mmap_store import MmapExpertStore

        store = MmapExpertStore(path)
        device = torch.device("cuda")
        cache = ExpertCache(
            capacity=4,
            expert_bytes=store.expert_bytes,
            device=device,
        )
        store.copy_to_buffer_slot(cache, 0, 0, 1)
        torch.cuda.synchronize()
        assert cache._packed[0].sum().item() > 0
        store.close()
