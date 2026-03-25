"""Fast expert data reader using pread — bypasses mmap page fault overhead.

For disk_offload mode, replaces mmap-backed tensor.copy_() in RAMCache with
explicit pread() calls. A single pread() syscall reads the exact 13 MB expert
blob vs mmap doing ~3300 individual 4KB page faults.

Measured improvement: ~20ms (mmap page faults) -> ~4ms (pread) per expert on NVMe.
"""

import os
from concurrent.futures import Future, ThreadPoolExecutor

import torch


class FastExpertReader:
    """Read expert blobs from SSD via pread instead of mmap page faults.

    Args:
        file_path: path to the expert store file on disk.
        expert_offsets: mapping of (layer_idx, expert_idx) -> byte offset in file.
        expert_bytes: number of bytes per expert blob.
    """

    def __init__(
        self,
        file_path: str,
        expert_offsets: dict[tuple[int, int], int],
        expert_bytes: int,
    ):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expert store file not found: {file_path}")
        self._fd = os.open(file_path, os.O_RDONLY)
        self._offsets = expert_offsets
        self._expert_bytes = expert_bytes

    def read_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        dest: torch.Tensor,
    ) -> None:
        """Read one expert from SSD into dest tensor via pread.

        Single syscall for the full expert blob — no page fault overhead.
        """
        offset = self._offsets[(layer_idx, expert_idx)]
        data = os.pread(self._fd, self._expert_bytes, offset)
        dest.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))

    def read_expert_async(
        self,
        layer_idx: int,
        expert_idx: int,
        dest: torch.Tensor,
        executor: ThreadPoolExecutor,
    ) -> Future:
        """Submit async read via ThreadPoolExecutor."""
        return executor.submit(self.read_expert, layer_idx, expert_idx, dest)

    def close(self) -> None:
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
