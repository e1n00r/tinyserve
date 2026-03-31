"""GGUF file format reader for quantized expert weights.

Parses GGUF v2/v3 headers, metadata, and tensor info without dequantizing.
Tensor data is read as raw bytes — the caller decides how to handle
block-quantized formats (Q4_K_M, Q5_K_M, etc.).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 4, 1),
    1: ("F16", 2, 1),
    2: ("Q4_0", 18, 32),
    3: ("Q4_1", 20, 32),
    6: ("Q5_0", 22, 32),
    7: ("Q5_1", 24, 32),
    8: ("Q8_0", 34, 32),
    12: ("Q2_K", 256, 256),
    13: ("Q3_K", 256, 256),
    14: ("Q4_K", 144, 256),
    15: ("Q5_K", 176, 256),
    16: ("Q6_K", 210, 256),
}


@dataclass
class GGUFTensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    ggml_type_name: str
    offset: int
    nbytes: int
    block_size: int


# Metadata value type codes
_VTYPE_UINT8 = 0
_VTYPE_INT8 = 1
_VTYPE_UINT16 = 2
_VTYPE_INT16 = 3
_VTYPE_UINT32 = 4
_VTYPE_INT32 = 5
_VTYPE_FLOAT32 = 6
_VTYPE_BOOL = 7
_VTYPE_STRING = 8
_VTYPE_ARRAY = 9
_VTYPE_UINT64 = 10
_VTYPE_INT64 = 11
_VTYPE_FLOAT64 = 12

_SCALAR_READERS: dict[int, tuple[str, int]] = {
    _VTYPE_UINT8: ("<B", 1),
    _VTYPE_INT8: ("<b", 1),
    _VTYPE_UINT16: ("<H", 2),
    _VTYPE_INT16: ("<h", 2),
    _VTYPE_UINT32: ("<I", 4),
    _VTYPE_INT32: ("<i", 4),
    _VTYPE_FLOAT32: ("<f", 4),
    _VTYPE_BOOL: ("<?", 1),
    _VTYPE_UINT64: ("<Q", 8),
    _VTYPE_INT64: ("<q", 8),
    _VTYPE_FLOAT64: ("<d", 8),
}


class GGUFReader:
    MAGIC = 0x46554747  # b"GGUF" as little-endian uint32

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        self._tensors: list[GGUFTensorInfo] = []
        self._metadata: dict = {}
        self._data_offset: int = 0
        self.version: int = 0
        self._parse()

    def _parse(self):
        magic = struct.unpack("<I", self._file.read(4))[0]
        if magic != self.MAGIC:
            raise ValueError(f"Not a GGUF file: magic={magic:#x}")
        self.version = struct.unpack("<I", self._file.read(4))[0]
        n_tensors = struct.unpack("<Q", self._file.read(8))[0]
        n_kv = struct.unpack("<Q", self._file.read(8))[0]

        self._metadata = self._read_metadata(n_kv)
        self._tensors = self._read_tensor_infos(n_tensors)

        pos = self._file.tell()
        self._data_offset = (pos + 31) & ~31

    def _read_string(self) -> str:
        length = struct.unpack("<Q", self._file.read(8))[0]
        return self._file.read(length).decode("utf-8")

    def _read_scalar(self, vtype: int):
        fmt, size = _SCALAR_READERS[vtype]
        return struct.unpack(fmt, self._file.read(size))[0]

    def _read_metadata(self, n_kv: int) -> dict:
        meta: dict = {}
        for _ in range(n_kv):
            key = self._read_string()
            vtype = struct.unpack("<I", self._file.read(4))[0]

            if vtype == _VTYPE_STRING:
                meta[key] = self._read_string()
            elif vtype == _VTYPE_ARRAY:
                atype = struct.unpack("<I", self._file.read(4))[0]
                alen = struct.unpack("<Q", self._file.read(8))[0]
                if atype == _VTYPE_STRING:
                    meta[key] = [self._read_string() for _ in range(alen)]
                elif atype in _SCALAR_READERS:
                    fmt, size = _SCALAR_READERS[atype]
                    meta[key] = [struct.unpack(fmt, self._file.read(size))[0] for _ in range(alen)]
                else:
                    meta[key] = None
            elif vtype in _SCALAR_READERS:
                meta[key] = self._read_scalar(vtype)
            else:
                meta[key] = None
        return meta

    def _read_tensor_infos(self, n_tensors: int) -> list[GGUFTensorInfo]:
        tensors = []
        for _ in range(n_tensors):
            name = self._read_string()
            n_dims = struct.unpack("<I", self._file.read(4))[0]
            shape = tuple(struct.unpack("<Q", self._file.read(8))[0] for _ in range(n_dims))
            ggml_type = struct.unpack("<I", self._file.read(4))[0]
            offset = struct.unpack("<Q", self._file.read(8))[0]

            type_name, bytes_per_block, block_size = GGML_TYPES.get(ggml_type, ("UNKNOWN", 1, 1))
            n_elements = 1
            for d in shape:
                n_elements *= d
            n_blocks = (n_elements + block_size - 1) // block_size
            nbytes = n_blocks * bytes_per_block

            tensors.append(GGUFTensorInfo(
                name=name,
                shape=shape,
                ggml_type=ggml_type,
                ggml_type_name=type_name,
                offset=offset,
                nbytes=nbytes,
                block_size=block_size,
            ))
        return tensors

    def get_tensor_data(self, info: GGUFTensorInfo) -> bytes:
        self._file.seek(self._data_offset + info.offset)
        return self._file.read(info.nbytes)

    def list_expert_tensors(self) -> dict[tuple[int, int], dict[str, GGUFTensorInfo]]:
        """Group expert tensors by (layer, expert_idx)."""
        pattern = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)\.(\d+)\.weight")
        groups: dict[tuple[int, int], dict[str, GGUFTensorInfo]] = {}
        for t in self._tensors:
            m = pattern.match(t.name)
            if m:
                layer, proj, expert = int(m.group(1)), m.group(2), int(m.group(3))
                key = (layer, expert)
                if key not in groups:
                    groups[key] = {}
                groups[key][proj] = t
        return groups

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def tensors(self) -> list[GGUFTensorInfo]:
        return self._tensors

    def close(self):
        self._file.close()
