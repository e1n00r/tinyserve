import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tinyserve.fast_io import FastExpertReader


class TestFastExpertReaderInit:
    def test_opens_file(self, tmp_path):
        fpath = tmp_path / "experts.bin"
        fpath.write_bytes(b"\x00" * 128)
        reader = FastExpertReader(str(fpath), expert_offsets={}, expert_bytes=64)
        assert reader._fd >= 0
        reader.close()

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            FastExpertReader("/tmp/nonexistent_xyz_42.bin", {}, 64)

    def test_close_releases_fd(self, tmp_path):
        fpath = tmp_path / "experts.bin"
        fpath.write_bytes(b"\x00" * 64)
        reader = FastExpertReader(str(fpath), expert_offsets={}, expert_bytes=64)
        fd = reader._fd
        reader.close()
        with pytest.raises(OSError):
            os.read(fd, 1)


class TestReadExpert:
    def test_reads_correct_data(self, tmp_path):
        expert_bytes = 256
        num_experts = 4
        fpath = tmp_path / "experts.bin"
        data = bytearray()
        for i in range(num_experts):
            data.extend(bytes([i & 0xFF] * expert_bytes))
        fpath.write_bytes(bytes(data))

        offsets = {(0, i): i * expert_bytes for i in range(num_experts)}
        reader = FastExpertReader(str(fpath), offsets, expert_bytes)
        try:
            for i in range(num_experts):
                dest = torch.empty(expert_bytes, dtype=torch.uint8)
                reader.read_expert(0, i, dest)
                expected = torch.full((expert_bytes,), i & 0xFF, dtype=torch.uint8)
                assert torch.equal(dest, expected), f"Expert {i} data mismatch"
        finally:
            reader.close()

    def test_reads_at_correct_offset(self, tmp_path):
        expert_bytes = 64
        fpath = tmp_path / "experts.bin"
        blob = bytearray(256)
        blob[128:192] = bytes([0xAB] * expert_bytes)
        fpath.write_bytes(bytes(blob))

        offsets = {(2, 3): 128}
        reader = FastExpertReader(str(fpath), offsets, expert_bytes)
        try:
            dest = torch.empty(expert_bytes, dtype=torch.uint8)
            reader.read_expert(2, 3, dest)
            assert torch.all(dest == 0xAB)
        finally:
            reader.close()

    def test_missing_key_raises(self, tmp_path):
        fpath = tmp_path / "experts.bin"
        fpath.write_bytes(b"\x00" * 64)
        reader = FastExpertReader(str(fpath), {}, 64)
        try:
            dest = torch.empty(64, dtype=torch.uint8)
            with pytest.raises(KeyError):
                reader.read_expert(0, 0, dest)
        finally:
            reader.close()


class TestReadExpertAsync:
    def test_async_returns_correct_data(self, tmp_path):
        expert_bytes = 128
        fpath = tmp_path / "experts.bin"
        fpath.write_bytes(bytes([0x42] * expert_bytes))

        offsets = {(0, 0): 0}
        reader = FastExpertReader(str(fpath), offsets, expert_bytes)
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            dest = torch.empty(expert_bytes, dtype=torch.uint8)
            future = reader.read_expert_async(0, 0, dest, executor)
            future.result(timeout=5.0)
            assert torch.all(dest == 0x42)
        finally:
            executor.shutdown(wait=True)
            reader.close()

    def test_multiple_async_reads(self, tmp_path):
        expert_bytes = 64
        num_experts = 8
        fpath = tmp_path / "experts.bin"
        data = bytearray()
        for i in range(num_experts):
            data.extend(bytes([i] * expert_bytes))
        fpath.write_bytes(bytes(data))

        offsets = {(0, i): i * expert_bytes for i in range(num_experts)}
        reader = FastExpertReader(str(fpath), offsets, expert_bytes)
        executor = ThreadPoolExecutor(max_workers=4)
        try:
            dests = [torch.empty(expert_bytes, dtype=torch.uint8) for _ in range(num_experts)]
            futures = [
                reader.read_expert_async(0, i, dests[i], executor)
                for i in range(num_experts)
            ]
            for fut in futures:
                fut.result(timeout=5.0)
            for i in range(num_experts):
                expected = torch.full((expert_bytes,), i, dtype=torch.uint8)
                assert torch.equal(dests[i], expected), f"Expert {i} mismatch"
        finally:
            executor.shutdown(wait=True)
            reader.close()


class TestPreadMatchesMmap:
    def test_pread_matches_mmap_data(self, tmp_path):
        expert_bytes = 512
        num_layers = 2
        num_experts = 4
        fpath = tmp_path / "experts.bin"

        import numpy as np
        total_size = num_layers * num_experts * expert_bytes
        rng = np.random.RandomState(42)
        raw = rng.randint(0, 256, size=total_size, dtype=np.uint8)
        fpath.write_bytes(bytes(raw))

        mmap_arr = np.memmap(str(fpath), dtype=np.uint8, mode="r",
                             shape=(num_layers, num_experts, expert_bytes))
        mmap_tensor = torch.from_numpy(mmap_arr)

        offsets = {}
        for li in range(num_layers):
            for ei in range(num_experts):
                offsets[(li, ei)] = (li * num_experts + ei) * expert_bytes

        reader = FastExpertReader(str(fpath), offsets, expert_bytes)
        try:
            for li in range(num_layers):
                for ei in range(num_experts):
                    dest = torch.empty(expert_bytes, dtype=torch.uint8)
                    reader.read_expert(li, ei, dest)
                    mmap_data = mmap_tensor[li, ei]
                    assert torch.equal(dest, mmap_data), (
                        f"Mismatch at layer={li}, expert={ei}"
                    )
        finally:
            reader.close()
            del mmap_tensor, mmap_arr


class TestContextManager:
    def test_context_manager_closes_fd(self, tmp_path):
        fpath = tmp_path / "experts.bin"
        fpath.write_bytes(b"\x00" * 64)
        with FastExpertReader(str(fpath), {}, 64) as reader:
            assert reader._fd >= 0
            fd = reader._fd
        with pytest.raises(OSError):
            os.read(fd, 1)


class TestRAMCacheIntegration:
    def test_ram_cache_with_fast_reader(self, tmp_path):
        from tinyserve.ram_cache import RAMCache

        expert_bytes = 128
        num_experts = 4
        fpath = tmp_path / "experts.bin"
        data = bytearray()
        for i in range(num_experts):
            data.extend(bytes([i * 10] * expert_bytes))
        fpath.write_bytes(bytes(data))

        offsets = {(0, i): i * expert_bytes for i in range(num_experts)}
        reader = FastExpertReader(str(fpath), offsets, expert_bytes)

        cache = RAMCache(num_slots=4, expert_bytes=expert_bytes, fast_reader=reader)
        try:
            slot = cache.load_sync(0, 0, src=None)
            stored = cache.get_slot_data(slot)
            expected = torch.full((expert_bytes,), 0, dtype=torch.uint8)
            assert torch.equal(stored, expected)

            slot = cache.load_sync(0, 2, src=None)
            stored = cache.get_slot_data(slot)
            expected = torch.full((expert_bytes,), 20, dtype=torch.uint8)
            assert torch.equal(stored, expected)
        finally:
            cache.shutdown()
            reader.close()

    def test_ram_cache_prefetch_with_fast_reader(self, tmp_path):
        from tinyserve.ram_cache import RAMCache

        expert_bytes = 128
        fpath = tmp_path / "experts.bin"
        fpath.write_bytes(bytes([0xCC] * expert_bytes))

        offsets = {(0, 0): 0}
        reader = FastExpertReader(str(fpath), offsets, expert_bytes)

        cache = RAMCache(num_slots=4, expert_bytes=expert_bytes, fast_reader=reader)
        try:
            cache.prefetch_async(0, 0, src=None)
            cache.wait_pending(0, 0)
            slot = cache.lookup(0, 0)
            assert slot is not None
            stored = cache.get_slot_data(slot)
            assert torch.all(stored == 0xCC)
        finally:
            cache.shutdown()
            reader.close()

    def test_ram_cache_background_fill_with_fast_reader(self, tmp_path):
        from tinyserve.ram_cache import RAMCache

        expert_bytes = 64
        num_layers = 2
        num_experts = 3
        fpath = tmp_path / "experts.bin"
        data = bytearray()
        for li in range(num_layers):
            for ei in range(num_experts):
                val = li * num_experts + ei
                data.extend(bytes([val] * expert_bytes))
        fpath.write_bytes(bytes(data))

        offsets = {}
        for li in range(num_layers):
            for ei in range(num_experts):
                offsets[(li, ei)] = (li * num_experts + ei) * expert_bytes

        reader = FastExpertReader(str(fpath), offsets, expert_bytes)
        cache = RAMCache(
            num_slots=num_layers * num_experts,
            expert_bytes=expert_bytes,
            fast_reader=reader,
        )
        try:
            thread = cache.start_background_fill(
                mmap_data=None,
                num_layers=num_layers,
                num_experts=num_experts,
            )
            thread.join(timeout=10.0)
            assert cache.fill_complete

            for li in range(num_layers):
                for ei in range(num_experts):
                    slot = cache.lookup(li, ei)
                    assert slot is not None, f"Expert ({li},{ei}) not in cache"
                    stored = cache.get_slot_data(slot)
                    expected_val = li * num_experts + ei
                    expected = torch.full((expert_bytes,), expected_val, dtype=torch.uint8)
                    assert torch.equal(stored, expected), f"Data mismatch at ({li},{ei})"
        finally:
            cache.shutdown()
            reader.close()

    def test_ram_cache_falls_back_to_src_without_reader(self):
        from tinyserve.ram_cache import RAMCache

        cache = RAMCache(num_slots=4, expert_bytes=32)
        src = torch.arange(32, dtype=torch.uint8)
        slot = cache.load_sync(0, 0, src)
        stored = cache.get_slot_data(slot)
        assert torch.equal(stored, src)
        cache.shutdown()
