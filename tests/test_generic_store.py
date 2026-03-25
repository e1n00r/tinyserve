"""Test generic expert store that handles arbitrary weight layouts."""

import pytest
import torch

from tests.conftest import requires_cuda


class QTensor:
    """Minimal optimum-quanto QTensor stand-in for unit testing.

    _is_qtensor checks type.__name__ == 'QTensor', so the class must be named exactly that.
    """

    def __init__(self, int_data: torch.Tensor, scale: torch.Tensor):
        self.int_data = int_data
        self.scale = scale

    @property
    def data(self):
        return self.int_data.float()  # quanto returns dequantized on .data

    @property
    def shape(self):
        return self.int_data.shape

    @property
    def dtype(self):
        return self.int_data.dtype


def test_expand_param_qtensor():
    """QTensor params are expanded to blocks+scales with correct naming."""
    from tinyserve.generic_store import _expand_param, _is_qtensor

    int_data = torch.randint(0, 255, (4, 8, 16), dtype=torch.uint8)
    scale = torch.randint(0, 255, (4, 8), dtype=torch.uint8)
    qt = QTensor(int_data, scale)

    assert _is_qtensor(qt)

    result = _expand_param("gate_up_proj", qt, expert_idx=None)
    assert set(result.keys()) == {"gate_up_proj", "gate_up_proj_scales"}
    torch.testing.assert_close(result["gate_up_proj"], int_data.cpu())
    torch.testing.assert_close(result["gate_up_proj_scales"], scale.cpu())


def test_expand_param_qtensor_sliced():
    """Expert-index slicing on QTensor expansion picks the right expert row."""
    from tinyserve.generic_store import _expand_param

    int_data = torch.randint(0, 255, (8, 16, 4), dtype=torch.uint8)
    scale = torch.randint(0, 255, (8, 16), dtype=torch.uint8)
    qt = QTensor(int_data, scale)

    result = _expand_param("gate_up_proj", qt, expert_idx=3)
    torch.testing.assert_close(result["gate_up_proj"], int_data[3].cpu())
    torch.testing.assert_close(result["gate_up_proj_scales"], scale[3].cpu())


@requires_cuda
def test_store_and_retrieve_expert_weights():
    """Store BF16 expert weights on CPU, retrieve to GPU, values match."""
    from tinyserve.generic_store import GenericExpertStore

    num_layers, num_experts, hidden = 2, 4, 64
    intermediate = 128

    expert_weights = {}
    for li in range(num_layers):
        for ei in range(num_experts):
            expert_weights[(li, ei)] = {
                "gate_proj.weight": torch.randn(intermediate, hidden, dtype=torch.bfloat16),
                "up_proj.weight": torch.randn(intermediate, hidden, dtype=torch.bfloat16),
                "down_proj.weight": torch.randn(hidden, intermediate, dtype=torch.bfloat16),
            }

    store = GenericExpertStore.from_dict(expert_weights, num_layers, num_experts)

    device = torch.device("cuda")
    buf = store.allocate_buffer(device)

    store.copy_to_buffer(buf, layer_idx=0, expert_idx=2, non_blocking=False)
    torch.cuda.synchronize()

    for name, original in expert_weights[(0, 2)].items():
        retrieved = buf.get_tensor(name)
        torch.testing.assert_close(retrieved.cpu(), original, rtol=0, atol=0)


@requires_cuda
def test_buffer_reuse_across_experts():
    """Same buffer can be reused for different experts without corruption."""
    from tinyserve.generic_store import GenericExpertStore

    expert_weights = {}
    for ei in range(3):
        expert_weights[(0, ei)] = {
            "w.weight": torch.full((8, 8), float(ei), dtype=torch.bfloat16),
        }

    store = GenericExpertStore.from_dict(expert_weights, 1, 3)
    buf = store.allocate_buffer(torch.device("cuda"))

    for ei in range(3):
        store.copy_to_buffer(buf, 0, ei, non_blocking=False)
        torch.cuda.synchronize()
        retrieved = buf.get_tensor("w.weight")
        assert retrieved.float().mean().item() == pytest.approx(float(ei), abs=1e-3)


@requires_cuda
def test_expert_bytes_computed_from_weights():
    """Expert byte size is derived from actual weight shapes, not hardcoded."""
    from tinyserve.generic_store import GenericExpertStore

    expert_weights = {
        (0, 0): {
            "a.weight": torch.randn(32, 16, dtype=torch.bfloat16),
            "b.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        },
        (0, 1): {
            "a.weight": torch.randn(32, 16, dtype=torch.bfloat16),
            "b.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        },
    }
    store = GenericExpertStore.from_dict(expert_weights, 1, 2)

    expected_bytes = (32 * 16 + 16 * 32) * 2
    assert store.expert_bytes == expected_bytes


@requires_cuda
def test_lru_cache_with_generic_store():
    """LRU cache works with generic store's buffer format."""
    from tinyserve.generic_store import GenericExpertStore, GenericLRUCache

    expert_weights = {}
    for ei in range(4):
        expert_weights[(0, ei)] = {
            "w.weight": torch.full((8, 8), float(ei), dtype=torch.bfloat16),
        }

    store = GenericExpertStore.from_dict(expert_weights, 1, 4)
    device = torch.device("cuda")
    cache = GenericLRUCache(capacity=2, expert_bytes=store.expert_bytes, device=device)
    buf = store.allocate_buffer(device)

    store.copy_to_buffer(buf, 0, 1, non_blocking=False)
    torch.cuda.synchronize()
    slot = cache.allocate(0, 1)
    cache.store_from_buffer(slot, buf)

    assert cache.lookup(0, 1) == slot

    cache.load_to_buffer(slot, buf)
    val = buf.get_tensor("w.weight").float().mean().item()
    assert val == pytest.approx(1.0, abs=1e-3)


@requires_cuda
def test_async_copy_with_stream():
    """Non-blocking copy on a CUDA stream produces correct results."""
    from tinyserve.generic_store import GenericExpertStore

    expert_weights = {
        (0, 0): {"w.weight": torch.randn(64, 64, dtype=torch.bfloat16)},
    }
    store = GenericExpertStore.from_dict(expert_weights, 1, 1)
    device = torch.device("cuda")
    buf = store.allocate_buffer(device)

    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        store.copy_to_buffer(buf, 0, 0, non_blocking=True)
    stream.synchronize()

    torch.testing.assert_close(
        buf.get_tensor("w.weight").cpu(),
        expert_weights[(0, 0)]["w.weight"],
        rtol=0,
        atol=0,
    )


@requires_cuda
def test_fp8_double_buffer_no_race():
    """FP8 copy_to_buffer uses per-buffer staging — no race between buf_a and buf_b."""
    from tinyserve.generic_store import GenericExpertStore

    num_layers, num_experts = 1, 4
    expert_weights = {}
    for ei in range(num_experts):
        expert_weights[(0, ei)] = {
            "w.weight": torch.full((32, 32), float(ei + 1), dtype=torch.bfloat16),
        }

    store = GenericExpertStore.from_dict(expert_weights, num_layers, num_experts, fp8=True)
    device = torch.device("cuda")
    buf_a = store.allocate_buffer(device)
    buf_b = store.allocate_buffer(device)

    transfer_stream = torch.cuda.Stream(device)

    # Simulate double-buffered pipeline: interleave copies to buf_a and buf_b.
    with torch.cuda.stream(transfer_stream):
        store.copy_to_buffer(buf_a, 0, 0, non_blocking=True)
        store.copy_to_buffer(buf_b, 0, 1, non_blocking=True)
    transfer_stream.synchronize()

    a_val = buf_a.get_tensor("w.weight").float().mean().item()
    b_val = buf_b.get_tensor("w.weight").float().mean().item()

    assert a_val == pytest.approx(1.0, abs=0.1), f"buf_a got {a_val}, expected 1.0"
    assert b_val == pytest.approx(2.0, abs=0.1), f"buf_b got {b_val}, expected 2.0"


@requires_cuda
def test_fp8_roundtrip_accuracy():
    """FP8 compression→decompression preserves values within FP8 quantization error."""
    from tinyserve.generic_store import GenericExpertStore

    expert_weights = {
        (0, 0): {
            "gate.weight": torch.randn(64, 32, dtype=torch.bfloat16),
            "down.weight": torch.randn(32, 64, dtype=torch.bfloat16),
        },
    }

    store = GenericExpertStore.from_dict(expert_weights, 1, 1, fp8=True)
    device = torch.device("cuda")
    buf = store.allocate_buffer(device)

    store.copy_to_buffer(buf, 0, 0, non_blocking=False)
    torch.cuda.synchronize()

    for name, original in expert_weights[(0, 0)].items():
        retrieved = buf.get_tensor(name).cpu()
        # FP8 e4m3 has ~3 mantissa bits — expect ~10% relative error at most
        torch.testing.assert_close(retrieved, original, rtol=0.15, atol=0.05)
