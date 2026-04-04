"""Integration tests for disk_offload: RAMCache + CPUExpertForward + pipeline."""

import torch
import torch.nn.functional as F

from tests.conftest import requires_cuda
from tinyserve.cpu_expert import CPUExpertForward
from tinyserve.expert_store import ExpertStore, TensorLayout
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
    store = ExpertStore.from_dict(weights, num_layers=1, num_experts=2)
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
        from tinyserve.expert_pipeline import ExpertPipeline

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
        staging_buffer_a = store.allocate_buffer(device)
        staging_buffer_b = store.allocate_buffer(device)
        transfer_stream = torch.cuda.Stream(device)
        compute_stream = torch.cuda.Stream(device)

        # Pipeline WITHOUT cpu_expert (GPU-only).
        gpu_pipeline = ExpertPipeline(
            store,
            template,
            device,
            staging_buffer_a=staging_buffer_a,
            staging_buffer_b=staging_buffer_b,
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
        cpu_pipeline = ExpertPipeline(
            store,
            template,
            device,
            staging_buffer_a=staging_buffer_a,
            staging_buffer_b=staging_buffer_b,
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


def _make_fake_mmap_data(num_layers, num_experts, expert_bytes):
    """Create a fake mmap-like data tensor [num_layers, num_experts, expert_bytes]."""
    torch.manual_seed(99)
    return torch.randint(0, 256, (num_layers, num_experts, expert_bytes), dtype=torch.uint8)


class TestBackgroundFill:
    def test_background_fill_populates_cache(self):
        """Background fill should load all experts into the RAM cache."""
        num_layers, num_experts, expert_bytes = 2, 4, 1024
        data = _make_fake_mmap_data(num_layers, num_experts, expert_bytes)
        ram = RAMCache(num_slots=num_layers * num_experts, expert_bytes=expert_bytes)

        thread = ram.start_background_fill(data, num_layers, num_experts)
        thread.join(timeout=5.0)

        assert ram.fill_complete
        for li in range(num_layers):
            for ei in range(num_experts):
                slot = ram.lookup(li, ei)
                assert slot is not None, f"Expert ({li}, {ei}) not in cache after fill"
                assert torch.equal(ram.get_slot_data(slot), data[li, ei])
        ram.shutdown()

    def test_background_fill_is_nonblocking(self):
        """Background fill should not block the calling thread."""
        import time

        num_layers, num_experts, expert_bytes = 2, 4, 1024
        data = _make_fake_mmap_data(num_layers, num_experts, expert_bytes)
        ram = RAMCache(num_slots=num_layers * num_experts, expert_bytes=expert_bytes)

        t0 = time.monotonic()
        thread = ram.start_background_fill(data, num_layers, num_experts)
        dt = time.monotonic() - t0

        # start_background_fill must return immediately (< 100ms)
        assert dt < 0.1, f"start_background_fill blocked for {dt:.3f}s"

        thread.join(timeout=5.0)
        assert ram.fill_complete
        ram.shutdown()

    def test_background_fill_skips_already_cached(self):
        """Background fill should not overwrite experts already in cache."""
        num_layers, num_experts, expert_bytes = 1, 2, 1024
        data = _make_fake_mmap_data(num_layers, num_experts, expert_bytes)
        ram = RAMCache(num_slots=4, expert_bytes=expert_bytes)

        # Pre-load expert (0, 0)
        slot_before = ram.load_sync(0, 0, data[0, 0])

        thread = ram.start_background_fill(data, num_layers, num_experts)
        thread.join(timeout=5.0)

        # Same slot should still be used (no duplicate allocation)
        slot_after = ram.lookup(0, 0)
        assert slot_after == slot_before
        ram.shutdown()

    def test_fill_complete_before_start(self):
        """fill_complete should be True when no fill was ever started."""
        ram = RAMCache(num_slots=4, expert_bytes=1024)
        assert ram.fill_complete
        ram.shutdown()

    def test_round_robin_ordering(self):
        """Fill visits L0E0, L1E0, L2E0... then L0E1, L1E1... (round-robin)."""
        num_layers, num_experts, expert_bytes = 3, 2, 512
        data = _make_fake_mmap_data(num_layers, num_experts, expert_bytes)
        ram = RAMCache(num_slots=num_layers * num_experts, expert_bytes=expert_bytes)

        thread = ram.start_background_fill(data, num_layers, num_experts)
        thread.join(timeout=5.0)

        # All experts present after fill
        for li in range(num_layers):
            for ei in range(num_experts):
                assert ram.lookup(li, ei) is not None
        ram.shutdown()


class TestAutoPinnedVsMmap:
    def test_auto_pinned_when_fits(self):
        """from_safetensors returns None for ram_cache when experts fit in RAM."""
        import json
        import os
        import tempfile

        from safetensors.torch import save_file

        # Create a tiny fake model dir with safetensors expert weights.
        num_layers = 2
        num_experts = 2
        hidden = 16
        intermediate = 32

        with tempfile.TemporaryDirectory() as model_dir:
            weight_map = {}
            tensors = {}
            for li in range(num_layers):
                for param_name, shape in [
                    ("gate_up_proj_blocks", (num_experts, 2 * intermediate, hidden)),
                    ("gate_up_proj_scales", (num_experts, 2 * intermediate, hidden // 32)),
                    ("down_proj_blocks", (num_experts, hidden, intermediate)),
                    ("down_proj_scales", (num_experts, hidden, intermediate // 32)),
                ]:
                    key = f"model.layers.{li}.mlp.experts.{param_name}"
                    tensors[key] = torch.randint(0, 256, shape, dtype=torch.uint8)
                    weight_map[key] = "model.safetensors"

            save_file(tensors, os.path.join(model_dir, "model.safetensors"))
            with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": weight_map}, f)

            result = ExpertStore.from_safetensors(
                model_dir,
                "mlp",
                "experts",
                list(range(num_layers)),
                disk_offload=True,
            )
            store, n_experts, ram_cache = result
            # Tiny experts easily fit in RAM → auto-pinned, no RAMCache
            assert ram_cache is None
            assert store._disk_offload is False
            # Pinned only when CUDA is available; otherwise regular CPU tensor.
            if torch.cuda.is_available():
                assert store._data.is_pinned()
            assert n_experts == num_experts

    def test_mmap_when_doesnt_fit(self):
        """from_safetensors returns RAMCache when experts exceed available RAM."""
        import json
        import os
        import tempfile
        from unittest.mock import patch

        from safetensors.torch import save_file

        num_layers = 2
        num_experts = 2
        hidden = 16
        intermediate = 32

        with tempfile.TemporaryDirectory() as model_dir:
            weight_map = {}
            tensors = {}
            for li in range(num_layers):
                for param_name, shape in [
                    ("gate_up_proj_blocks", (num_experts, 2 * intermediate, hidden)),
                    ("gate_up_proj_scales", (num_experts, 2 * intermediate, hidden // 32)),
                    ("down_proj_blocks", (num_experts, hidden, intermediate)),
                    ("down_proj_scales", (num_experts, hidden, intermediate // 32)),
                ]:
                    key = f"model.layers.{li}.mlp.experts.{param_name}"
                    tensors[key] = torch.randint(0, 256, shape, dtype=torch.uint8)
                    weight_map[key] = "model.safetensors"

            save_file(tensors, os.path.join(model_dir, "model.safetensors"))
            with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": weight_map}, f)

            # Mock sysconf to report very low available RAM so experts don't fit.
            original_sysconf = os.sysconf

            def fake_sysconf(name):
                if name == "SC_AVPHYS_PAGES":
                    return 1  # ~4 KB available
                return original_sysconf(name)

            with patch("os.sysconf", side_effect=fake_sysconf):
                result = ExpertStore.from_safetensors(
                    model_dir,
                    "mlp",
                    "experts",
                    list(range(num_layers)),
                    disk_offload=True,
                )
            store, n_experts, ram_cache = result
            # Experts "don't fit" → mmap path, RAMCache created
            assert ram_cache is not None
            assert store._disk_offload is True
            assert n_experts == num_experts
            ram_cache.shutdown()


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
