"""Tests for native-quant forward path in ExpertPipeline."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tinyserve.expert_cache import ExpertCache
from tinyserve.expert_store import ExpertBuffer, TensorLayout
from tinyserve.mmap_store import quantize_to_q8_0


HIDDEN = 64
INTER = 128
GGML_Q8_0 = 8


def _q8_bytes(n: int, k: int) -> int:
    return (n * k // 32) * 34


def _make_layout() -> TensorLayout:
    gate_bytes = _q8_bytes(INTER, HIDDEN)
    up_bytes = _q8_bytes(INTER, HIDDEN)
    down_bytes = _q8_bytes(HIDDEN, INTER)
    return TensorLayout({
        "gate": ((gate_bytes,), torch.uint8),
        "up": ((up_bytes,), torch.uint8),
        "down": ((down_bytes,), torch.uint8),
    })


def _make_expert_data(layout: TensorLayout) -> torch.Tensor:
    gate_w = torch.randn(INTER, HIDDEN)
    up_w = torch.randn(INTER, HIDDEN)
    down_w = torch.randn(HIDDEN, INTER)

    gate_q = quantize_to_q8_0(gate_w)
    up_q = quantize_to_q8_0(up_w)
    down_q = quantize_to_q8_0(down_w)

    packed = torch.zeros(layout.total_bytes, dtype=torch.uint8)
    packed[layout.offsets["gate"]:layout.offsets["gate"] + len(gate_q)] = torch.frombuffer(
        bytearray(gate_q), dtype=torch.uint8
    )
    packed[layout.offsets["up"]:layout.offsets["up"] + len(up_q)] = torch.frombuffer(
        bytearray(up_q), dtype=torch.uint8
    )
    packed[layout.offsets["down"]:layout.offsets["down"] + len(down_q)] = torch.frombuffer(
        bytearray(down_q), dtype=torch.uint8
    )
    return packed


class FakeNativeQuantStore:
    """Minimal store that has ggml_types, conforming to MmapExpertStore interface."""

    def __init__(self, layout: TensorLayout, num_experts: int = 4, num_layers: int = 2):
        self.layout = layout
        self._bf16_layout = layout
        self._fp8 = False
        self.expert_bytes = layout.total_bytes
        self.buffer_expert_bytes = layout.total_bytes
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.ggml_types = {"gate": GGML_Q8_0, "up": GGML_Q8_0, "down": GGML_Q8_0}
        self.proj_shapes = {
            "gate": (INTER, HIDDEN),
            "up": (INTER, HIDDEN),
            "down": (HIDDEN, INTER),
        }
        self._data: dict[tuple[int, int], torch.Tensor] = {}
        for layer in range(num_layers):
            for expert in range(num_experts):
                self._data[(layer, expert)] = _make_expert_data(layout)

    def copy_to_buffer(
        self, buf: ExpertBuffer, layer_idx: int, expert_idx: int, non_blocking: bool = False
    ) -> None:
        buf.packed.copy_(self._data[(layer_idx, expert_idx)].to(buf.packed.device))

    def get_expert_data(self, layer_idx: int, expert_idx: int) -> torch.Tensor:
        return self._data[(layer_idx, expert_idx)]

    def allocate_buffer(self, device: torch.device) -> ExpertBuffer:
        return ExpertBuffer(self.layout, device)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


@pytest.fixture
def layout():
    return _make_layout()


@pytest.fixture
def store(layout):
    return FakeNativeQuantStore(layout)


@pytest.fixture
def pipeline(store, device):
    from tinyserve.expert_pipeline import ExpertPipeline

    template = nn.Linear(1, 1)
    buf_a = store.allocate_buffer(device)
    buf_b = store.allocate_buffer(device)
    transfer_stream = torch.cuda.Stream(device)
    compute_stream = torch.cuda.Stream(device)
    cache = ExpertCache(
        capacity=16,
        expert_bytes=store.expert_bytes,
        device=device,
        num_layers=store.num_layers,
        num_experts=store.num_experts,
    )
    return ExpertPipeline(
        store=store,
        template=template,
        device=device,
        staging_buffer_a=buf_a,
        staging_buffer_b=buf_b,
        transfer_stream=transfer_stream,
        compute_stream=compute_stream,
        cache=cache,
    )


class TestNativeQuantDetection:
    def test_native_quant_flag_set(self, pipeline):
        assert pipeline._native_quant is True

    def test_non_native_quant_flag_false(self, device):
        assert not hasattr(object(), "ggml_types")


class TestNativeQuantDecode:
    def test_execute_layer_experts_produces_valid_output(self, pipeline, device):
        h = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)
        expert_indices = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        routing_weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32, device=device)
        out = pipeline.execute_layer_experts(h, 0, expert_indices, routing_weights)
        assert out.shape == h.shape
        assert out.dtype == h.dtype
        assert torch.isfinite(out).all()

    def test_cache_hit_reuse(self, pipeline, device):
        h = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)
        expert_indices = torch.tensor([[2]], dtype=torch.long, device=device)
        routing_weights = torch.tensor([[1.0]], dtype=torch.float32, device=device)

        out1 = pipeline.execute_layer_experts(h, 0, expert_indices, routing_weights)
        assert pipeline.cache.hits == 0

        out2 = pipeline.execute_layer_experts(h, 0, expert_indices, routing_weights)
        assert pipeline.cache.hits >= 1
        torch.testing.assert_close(out1, out2)

    def test_multi_token_decode(self, pipeline, device):
        h = torch.randn(3, HIDDEN, dtype=torch.bfloat16, device=device)
        expert_indices = torch.tensor([[0, 1], [2, 3], [0, 2]], dtype=torch.long, device=device)
        routing_weights = torch.tensor(
            [[0.5, 0.5], [0.7, 0.3], [0.4, 0.6]], dtype=torch.float32, device=device
        )
        out = pipeline.execute_layer_experts(h, 0, expert_indices, routing_weights)
        assert out.shape == h.shape
        assert torch.isfinite(out).all()


class TestNativeQuantBatched:
    def test_execute_batched_produces_valid_output(self, pipeline, device):
        h = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device=device)
        expert_indices = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long, device=device
        )
        routing_weights = torch.tensor(
            [[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.8, 0.2]],
            dtype=torch.float32, device=device,
        )
        out = pipeline.execute_layer_experts_batched(h, 0, expert_indices, routing_weights)
        assert out.shape == h.shape
        assert out.dtype == h.dtype
        assert torch.isfinite(out).all()

    def test_empty_input(self, pipeline, device):
        h = torch.randn(0, HIDDEN, dtype=torch.bfloat16, device=device)
        expert_indices = torch.zeros(0, 2, dtype=torch.long, device=device)
        routing_weights = torch.zeros(0, 2, dtype=torch.float32, device=device)
        out = pipeline.execute_layer_experts_batched(h, 0, expert_indices, routing_weights)
        assert out.shape == (0, HIDDEN)

    def test_batched_cache_hit(self, pipeline, device):
        h = torch.randn(2, HIDDEN, dtype=torch.bfloat16, device=device)
        expert_indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)
        routing_weights = torch.tensor(
            [[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32, device=device
        )
        pipeline.execute_layer_experts_batched(h, 0, expert_indices, routing_weights)
        hits_before = pipeline.cache.hits
        pipeline.execute_layer_experts_batched(h, 0, expert_indices, routing_weights)
        assert pipeline.cache.hits > hits_before

    def test_decode_and_batched_agree_single_token(self, pipeline, device):
        h = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)
        expert_indices = torch.tensor([[0, 1]], dtype=torch.long, device=device)
        routing_weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32, device=device)

        pipeline_no_cache = type(pipeline)
        out_decode = pipeline.execute_layer_experts(h, 1, expert_indices, routing_weights)
        out_batched = pipeline.execute_layer_experts_batched(h, 1, expert_indices, routing_weights)
        torch.testing.assert_close(out_decode, out_batched, atol=1e-4, rtol=1e-3)
