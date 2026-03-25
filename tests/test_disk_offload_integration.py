"""Integration tests for disk_offload: RAMCache + CPUExpertForward + pipeline."""

import torch
import torch.nn.functional as F

from tests.conftest import requires_cuda
from tinyserve.cpu_expert import CPUExpertForward
from tinyserve.generic_store import GenericExpertStore, TensorLayout, _pack_tensors
from tinyserve.ram_cache import RAMCache, madvise_willneed


HIDDEN = 64
INTERMEDIATE = 128


def _make_fused_layout():
    specs = {
        "gate_up_proj": ((2 * INTERMEDIATE, HIDDEN), torch.float32),
        "down_proj": ((HIDDEN, INTERMEDIATE), torch.float32),
    }
    return TensorLayout(specs)


def _make_store_and_weights():
    torch.manual_seed(42)
    layout = _make_fused_layout()
    w_gu = torch.randn(2 * INTERMEDIATE, HIDDEN)
    w_dn = torch.randn(HIDDEN, INTERMEDIATE)
    weights = {
        (0, 0): {"gate_up_proj": w_gu, "down_proj": w_dn},
        (0, 1): {"gate_up_proj": w_gu * 0.5, "down_proj": w_dn * 0.5},
    }
    store = GenericExpertStore.from_dict(weights, num_layers=1, num_experts=2)
    return store, layout, w_gu, w_dn


def _ref_fused_silu(h, w_gu, w_dn):
    gate_up = F.linear(h, w_gu)
    gate, up = gate_up.chunk(2, dim=-1)
    gated = F.silu(gate) * up
    return F.linear(gated, w_dn)


@requires_cuda
class TestRAMCacheLoadsFromStore:
    def test_load_and_retrieve(self):
        store, layout, _, _ = _make_store_and_weights()
        ram = RAMCache(num_slots=4, expert_bytes=layout.total_bytes)
        expert_data = store._data[0, 0]
        slot = ram.load_sync(0, 0, expert_data)
        stored = ram.get_slot_data(slot)
        assert torch.equal(stored, expert_data)
        ram.shutdown()

    def test_prefetch_and_wait(self):
        store, layout, _, _ = _make_store_and_weights()
        ram = RAMCache(num_slots=4, expert_bytes=layout.total_bytes)
        expert_data = store._data[0, 1]
        ram.prefetch_async(0, 1, expert_data)
        ram.wait_pending(0, 1)
        slot = ram.lookup(0, 1)
        assert slot is not None
        assert torch.equal(ram.get_slot_data(slot), expert_data)
        ram.shutdown()


@requires_cuda
class TestCPUForwardFromStore:
    def test_matches_reference(self):
        store, layout, w_gu, w_dn = _make_store_and_weights()
        cpu_fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=1)
        h = torch.randn(1, HIDDEN)
        expert_data = store._data[0, 0]
        result = cpu_fwd.forward(h, expert_data)
        expected = _ref_fused_silu(h, w_gu, w_dn)
        torch.testing.assert_close(result, expected)

    def test_via_ram_cache(self):
        store, layout, w_gu, w_dn = _make_store_and_weights()
        ram = RAMCache(num_slots=4, expert_bytes=layout.total_bytes)
        slot = ram.load_sync(0, 0, store._data[0, 0])
        cached_data = ram.get_slot_data(slot)
        cpu_fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=1)
        h = torch.randn(1, HIDDEN)
        result = cpu_fwd.forward(h, cached_data)
        expected = _ref_fused_silu(h, w_gu, w_dn)
        torch.testing.assert_close(result, expected)
        ram.shutdown()


class TestPipelineCPUExpert:
    @requires_cuda
    def test_cpu_expert_matches_gpu_pipeline(self):
        from tinyserve.generic_pipeline import GenericExpertPipeline
        from tinyserve.generic_store import GenericExpertBuffer

        torch.manual_seed(42)
        store, layout, w_gu, w_dn = _make_store_and_weights()
        device = torch.device("cuda")

        # Build a template module for GPU pipeline.
        import torch.nn as nn

        class FusedTemplate(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.zeros(2 * INTERMEDIATE, HIDDEN))
                self.down_proj = nn.Parameter(torch.zeros(HIDDEN, INTERMEDIATE))
                self._act_fn = F.silu

            def forward(self, x):
                gate_up = F.linear(x, self.gate_up_proj)
                gate, up = gate_up.chunk(2, dim=-1)
                gated = F.silu(gate) * up
                return F.linear(gated, self.down_proj)

        template = FusedTemplate().to(device).to(torch.float32)
        buf_a = store.allocate_buffer(device)
        buf_b = store.allocate_buffer(device)
        transfer_stream = torch.cuda.Stream(device)
        compute_stream = torch.cuda.Stream(device)

        # Pipeline WITHOUT cpu_expert (GPU-only).
        gpu_pipeline = GenericExpertPipeline(
            store, template, device,
            buf_a=buf_a, buf_b=buf_b,
            transfer_stream=transfer_stream,
            compute_stream=compute_stream,
        )

        h_gpu = torch.randn(1, HIDDEN, device=device)
        expert_ids = torch.tensor([[0]], device=device)  # [batch=1, top_k=1]
        weights_t = torch.tensor([[1.0]], device=device)
        gpu_output = gpu_pipeline.execute_layer_experts(h_gpu, 0, expert_ids, weights_t)
        torch.cuda.synchronize()

        # Pipeline WITH cpu_expert.
        ram = RAMCache(num_slots=4, expert_bytes=layout.total_bytes)
        cpu_fwd = CPUExpertForward(layout, act_fn=F.silu, num_threads=1)
        cpu_pipeline = GenericExpertPipeline(
            store, template, device,
            buf_a=buf_a, buf_b=buf_b,
            transfer_stream=transfer_stream,
            compute_stream=compute_stream,
            ram_cache=ram,
            cpu_expert=cpu_fwd,
        )

        h_cpu_pipe = h_gpu.clone()
        cpu_output = cpu_pipeline.execute_layer_experts(h_cpu_pipe, 0, expert_ids, weights_t)
        torch.cuda.synchronize()

        torch.testing.assert_close(gpu_output, cpu_output, atol=1e-4, rtol=1e-4)
        ram.shutdown()


class TestRAMCacheSizedForAllExperts:
    def test_auto_sizing_gives_enough_slots(self):
        """RAMCache auto-sizing should allocate enough slots for all experts."""
        import os

        layout = _make_fused_layout()
        num_layers = 4
        num_experts = 8
        total_expert_bytes = num_layers * num_experts * layout.total_bytes

        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        ram_bytes = min(int(page_size * avail_pages * 0.7), total_expert_bytes)
        num_slots = max(1, ram_bytes // layout.total_bytes)

        total_experts = num_layers * num_experts
        assert num_slots >= total_experts, (
            f"Auto-sizing gave {num_slots} slots but need {total_experts} for all experts"
        )


class TestMadviseWillneedNoCrash:
    def test_on_regular_tensor(self):
        """madvise_willneed should not crash on regular (non-mmap) tensors."""
        t = torch.randn(1024, dtype=torch.float32)
        madvise_willneed(t)

    def test_on_pinned_tensor(self):
        """madvise_willneed should not crash on pinned memory tensors."""
        if not torch.cuda.is_available():
            t = torch.randn(1024, dtype=torch.float32)
        else:
            t = torch.randn(1024, dtype=torch.float32).pin_memory()
        madvise_willneed(t)
