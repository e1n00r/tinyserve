"""Model-agnostic expert weight storage and GPU buffer management.

Stores expert weights as flat byte buffers on CPU. Each expert is a set of
named tensors (e.g., gate_proj.weight, up_proj.weight) packed contiguously.
The buffer layout is computed at init time from actual weight shapes.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .ram_cache import RAMCache

from .cache_policy import make_policy
from .expert_cache import ExpertCache as ExpertCache  # re-export for backward compat

logger = logging.getLogger(__name__)


def _is_qtensor(param) -> bool:
    """Detect optimum-quanto QTensor without importing the library."""
    return type(param).__name__ == "QTensor"


def _expand_param(name: str, param, expert_idx: int | None = None) -> dict[str, torch.Tensor]:
    """Return flat tensor dict for one param, sliced to one expert if given.

    QTensors (optimum-quanto MXFP4) are expanded into:
      ``name``         → int_data (uint8 blocks)
      ``name_scales``  → scale   (uint8 E8M0 scales)
    This matches the naming expected by ``_FusedExpertTemplate``.
    """
    if _is_qtensor(param):
        int_data = param.int_data
        scale = param.scale
        if expert_idx is not None:
            int_data = int_data[expert_idx]
            scale = scale[expert_idx]
        return {
            name: int_data.cpu().contiguous(),
            name + "_scales": scale.cpu().contiguous(),
        }
    data = param.data
    if expert_idx is not None:
        data = data[expert_idx]
    return {name: data.cpu().contiguous()}


class TensorLayout:
    """Describes how named tensors are packed into a flat byte buffer."""

    def __init__(self, tensor_specs: dict[str, tuple[tuple[int, ...], torch.dtype]]):
        self.specs = tensor_specs
        self.offsets: dict[str, int] = {}
        self.sizes: dict[str, int] = {}
        offset = 0
        for name, (shape, dtype) in tensor_specs.items():
            nbytes = 1
            for dim in shape:
                nbytes *= dim
            nbytes *= torch.tensor([], dtype=dtype).element_size()
            self.offsets[name] = offset
            self.sizes[name] = nbytes
            offset += nbytes
        self.total_bytes = offset

    @staticmethod
    def from_tensors(tensors: dict[str, torch.Tensor]) -> "TensorLayout":
        return TensorLayout({name: (tensor.shape, tensor.dtype) for name, tensor in tensors.items()})


def _pack_tensors(
    dest: torch.Tensor,
    layout: "TensorLayout",
    tensors: dict[str, torch.Tensor],
):
    """Pack a dict of tensors into a flat uint8 dest according to layout."""
    for name, tensor in tensors.items():
        offset = layout.offsets[name]
        nbytes = layout.sizes[name]
        raw = tensor.contiguous().view(-1).view(torch.uint8)
        dest[offset : offset + nbytes].copy_(raw)


class ExpertBuffer:
    """Pre-allocated GPU buffer for one expert's weights."""

    def __init__(
        self,
        layout: TensorLayout,
        device: torch.device,
        fp8_layout: "TensorLayout | None" = None,
    ):
        self.layout = layout
        self.packed = torch.empty(layout.total_bytes, dtype=torch.uint8, device=device)
        # GPU staging buffer for FP8 raw bytes (half the BF16 size).
        # Populated by H2D DMA; GPU dequant reads from here → writes to packed.
        self.fp8_stage: torch.Tensor | None = None
        self.fp8_layout: TensorLayout | None = fp8_layout
        if fp8_layout is not None:
            self.fp8_stage = torch.empty(fp8_layout.total_bytes, dtype=torch.uint8, device=device)

    def get_tensor(self, name: str) -> torch.Tensor:
        shape, dtype = self.layout.specs[name]
        offset = self.layout.offsets[name]
        nbytes = self.layout.sizes[name]
        return self.packed[offset : offset + nbytes].view(dtype).view(shape)


def _fp8_layout(bf16_layout: "TensorLayout") -> "TensorLayout":
    """Return a layout where float tensors are stored as float8_e4m3fn (1 byte/elem)."""
    fp8_specs = {}
    for name, (shape, dtype) in bf16_layout.specs.items():
        if dtype.is_floating_point:
            fp8_specs[name] = (shape, torch.float8_e4m3fn)
        else:
            fp8_specs[name] = (shape, dtype)
    return TensorLayout(fp8_specs)


def _quantize_to_fp8(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Quantize float tensors to float8_e4m3fn; leave uint8 tensors unchanged."""
    out = {}
    for name, t in tensors.items():
        if t.dtype.is_floating_point:
            out[name] = t.to(torch.float8_e4m3fn)
        else:
            out[name] = t
    return out


class ExpertStore:
    """Stores all expert weights on CPU as flat byte buffers.

    When ``fp8=True``, floating-point weights are compressed to float8_e4m3fn
    before storage (~2x smaller than BF16).  GPU buffers always hold BF16 so
    the template forward is unaffected; ``copy_to_buffer`` dequantises on the
    fly during the H2D transfer.

    Attributes:
        expert_bytes: bytes per expert *in CPU storage* (FP8 or BF16).
        buffer_expert_bytes: bytes per expert *for GPU buffer/cache* (always BF16).
    """

    def __init__(
        self,
        data: torch.Tensor,
        layout: TensorLayout,
        num_layers: int,
        num_experts: int,
        bf16_layout: "TensorLayout | None" = None,
    ):
        if not data.is_pinned():
            raise RuntimeError(
                "ExpertStore: expert data must be pinned (cudaMallocHost) for "
                "async DMA. Unpinned memory forces CUDA to stage through an internal "
                "bounce buffer, doubling effective bandwidth cost."
            )
        self._data = data
        self.layout = layout
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_bytes = layout.total_bytes
        # GPU buffers always need BF16-sized slots.
        self._bf16_layout = bf16_layout if bf16_layout is not None else layout
        self.buffer_expert_bytes = self._bf16_layout.total_bytes
        self._disk_offload = False

    @classmethod
    def from_dict(
        cls,
        expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
        num_layers: int,
        num_experts: int,
        fp8: bool = False,
    ) -> "ExpertStore":
        sample_key = next(iter(expert_weights))
        bf16_layout = TensorLayout.from_tensors(expert_weights[sample_key])
        store_layout = _fp8_layout(bf16_layout) if fp8 else bf16_layout

        data = torch.empty(num_layers, num_experts, store_layout.total_bytes, dtype=torch.uint8).pin_memory()

        for (layer_idx, expert_idx), tensors in expert_weights.items():
            packed = _quantize_to_fp8(tensors) if fp8 else tensors
            offset = 0
            for name, tensor in packed.items():
                raw = tensor.contiguous().view(-1).view(torch.uint8)
                data[layer_idx, expert_idx, offset : offset + raw.numel()] = raw
                offset += raw.numel()

        return cls(data, store_layout, num_layers, num_experts, bf16_layout=bf16_layout if fp8 else None)

    @classmethod
    def build(
        cls,
        moe_layers: list,
        moe_block_attr: str,
        expert_list_attr: str,
        fp8: bool = False,
    ) -> tuple["ExpertStore", int]:
        import torch.nn as nn

        first_container = getattr(getattr(moe_layers[0][1], moe_block_attr), expert_list_attr)
        if isinstance(first_container, nn.ModuleList):
            num_experts = len(first_container)
            sample_tensors: dict[str, torch.Tensor] = {}
            for n, p in first_container[0].named_parameters():
                sample_tensors.update(_expand_param(n, p))
        else:
            num_experts = next(iter(first_container.parameters())).shape[0]
            sample_tensors = {}
            for n, p in first_container.named_parameters():
                sample_tensors.update(_expand_param(n, p, expert_idx=0))

        bf16_layout = TensorLayout.from_tensors(sample_tensors)
        store_layout = _fp8_layout(bf16_layout) if fp8 else bf16_layout
        num_layers = len(moe_layers)

        # Write to mmap first (streaming, avoids holding full tensor in pageable RAM).
        tmp = tempfile.NamedTemporaryFile(suffix=".expert_store", delete=False)
        tmp_name = tmp.name
        tmp.close()
        mmap = np.memmap(tmp_name, dtype=np.uint8, mode="w+", shape=(num_layers, num_experts, store_layout.total_bytes))
        data = torch.from_numpy(mmap)

        for store_idx, (_, layer) in enumerate(moe_layers):
            container = getattr(getattr(layer, moe_block_attr), expert_list_attr)
            if isinstance(container, nn.ModuleList):
                for expert_idx, expert in enumerate(container):
                    tensors: dict[str, torch.Tensor] = {}
                    for n, p in expert.named_parameters():
                        tensors.update(_expand_param(n, p))
                    if fp8:
                        tensors = _quantize_to_fp8(tensors)
                    _pack_tensors(data[store_idx, expert_idx], store_layout, tensors)
                for param in container.parameters():
                    param.data = torch.empty(0, device="cpu")
            else:
                for expert_idx in range(num_experts):
                    tensors = {}
                    for n, p in container.named_parameters():
                        tensors.update(_expand_param(n, p, expert_idx=expert_idx))
                    if fp8:
                        tensors = _quantize_to_fp8(tensors)
                    _pack_tensors(data[store_idx, expert_idx], store_layout, tensors)
                for param in container.parameters():
                    param.data = torch.empty(0, device="cpu")
            gc.collect()

        # Flush mmap to ensure all writes are visible, then promote to pinned RAM.
        # This is the only point where mmap page faults occur; thereafter all
        # H2D transfers go through pinned DMA with no OS involvement.
        mmap.flush()
        pinned = torch.empty(num_layers, num_experts, store_layout.total_bytes, dtype=torch.uint8).pin_memory()
        pinned.copy_(data)

        # Release mmap and tempfile — data now lives entirely in pinned RAM.
        del data
        del mmap
        gc.collect()
        try:
            os.unlink(tmp_name)
        except OSError:
            logger.warning("Failed to delete temp file %s", tmp_name, exc_info=True)

        store = cls(pinned, store_layout, num_layers, num_experts, bf16_layout=bf16_layout if fp8 else None)
        return store, num_experts

    @classmethod
    def from_safetensors(
        cls,
        model_id: str,
        moe_block_attr: str,
        expert_list_attr: str,
        layer_indices: list[int],
        disk_offload: bool = False,
        ram_cache_gb: float = 0,
    ) -> "tuple[ExpertStore, int] | tuple[ExpertStore, int, RAMCache | None]":
        """Load expert weights directly from safetensors, bypassing HF dequantization.

        For MXFP4 models where ``*_blocks`` / ``*_scales`` tensors exist in the
        checkpoint, loads them as raw uint8 and strips the ``_blocks`` suffix so
        the layout matches ``_FusedExpertTemplate`` naming (``gate_up_proj`` for
        blocks, ``gate_up_proj_scales`` for scales).

        Args:
            model_id: HuggingFace repo id or local path to the model directory.
            moe_block_attr: attribute name of the MoE block on each layer (e.g. ``mlp``).
            expert_list_attr: attribute name of the expert container (e.g. ``experts``).
            layer_indices: list of layer indices that have MoE blocks.
        """
        import json
        import os
        import re

        from safetensors import safe_open

        from .int4_cache import (
            _deserialize_layout_specs,
            _model_hash,
            _serialize_layout_specs,
            int4_cache_path,
            load_int4_cache,
            save_int4_cache,
        )

        if os.path.isdir(model_id):
            model_dir = model_id
        else:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(model_id, ignore_patterns=["*.bin", "*.msgpack", "*.h5"])

        # --- INT4 cache: fast reload path ---
        cache_path = int4_cache_path(model_id)
        model_hash_val = _model_hash(model_dir)
        cached = load_int4_cache(cache_path, expected_hash=model_hash_val)
        if cached is not None and not disk_offload:
            raw_specs = cached["layout_specs"]
            specs = _deserialize_layout_specs(raw_specs)
            layout = TensorLayout(specs)
            num_moe_layers = cached["num_layers"]
            num_experts = cached["num_experts"]
            pinned = torch.empty(
                num_moe_layers, num_experts, layout.total_bytes,
                dtype=torch.uint8,
            ).pin_memory()
            pinned.copy_(cached["data"])
            store = cls(pinned, layout, num_moe_layers, num_experts)
            store._disk_offload = False
            return store, num_experts

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                shard_files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            shard_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))

        expert_prefix = f"{moe_block_attr}.{expert_list_attr}."
        layer_param_name_re = re.compile(r"layers\.(\d+)\." + re.escape(expert_prefix) + r"(.+)$")

        # Collect: layer_idx -> param_base_name -> tensor
        layer_tensors: dict[int, dict[str, torch.Tensor]] = {}
        try:
            from tqdm import tqdm
            shard_iter = tqdm(shard_files, desc="Loading expert shards", unit="shard")
        except ImportError:
            shard_iter = shard_files
        for shard_name in shard_iter:
            shard_path = os.path.join(model_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    m = layer_param_name_re.search(key)
                    if m is None:
                        continue
                    li = int(m.group(1))
                    raw_name = m.group(2)
                    # Strip _blocks suffix: gate_up_proj_blocks -> gate_up_proj
                    param_name = re.sub(r"_blocks$", "", raw_name)
                    if li not in layer_tensors:
                        layer_tensors[li] = {}
                    layer_tensors[li][param_name] = f.get_tensor(key)

        moe_layer_indices = [li for li in layer_indices if li in layer_tensors]
        if not moe_layer_indices:
            raise ValueError(
                f"No expert tensors found in safetensors for layers {layer_indices}. "
                f"Pattern: *.layers.<i>.{expert_prefix}*"
            )

        first_layer_tensors = layer_tensors[moe_layer_indices[0]]
        num_experts = next(iter(first_layer_tensors.values())).shape[0]
        sample_tensors = {n: t[0] for n, t in first_layer_tensors.items()}
        layout = TensorLayout.from_tensors(sample_tensors)
        num_moe_layers = len(moe_layer_indices)

        tmp = tempfile.NamedTemporaryFile(suffix=".expert_store", delete=False)
        tmp_name = tmp.name
        tmp.close()
        mmap = np.memmap(tmp_name, dtype=np.uint8, mode="w+", shape=(num_moe_layers, num_experts, layout.total_bytes))
        data = torch.from_numpy(mmap)

        for store_idx, li in enumerate(moe_layer_indices):
            tensors = layer_tensors[li]
            for expert_idx in range(num_experts):
                per_expert = {n: t[expert_idx].contiguous() for n, t in tensors.items()}
                _pack_tensors(data[store_idx, expert_idx], layout, per_expert)
            del layer_tensors[li]
            gc.collect()

        mmap.flush()

        # --- INT4 cache: save for next load ---
        try:
            serialized_specs = _serialize_layout_specs(layout.specs)
            save_int4_cache(
                cache_path, data, serialized_specs,
                num_moe_layers, num_experts, model_hash_val,
            )
        except Exception:
            logger.warning("Expert cache save failed (non-fatal)", exc_info=True)

        if disk_offload:
            # Phase 2: Auto-detect whether experts fit in available RAM.
            # If they do, promote to a single pinned buffer (no mmap, no RAMCache).
            # If they don't, keep mmap + RAMCache for hot subset.
            total_expert_bytes = num_moe_layers * num_experts * layout.total_bytes
            available_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")

            if total_expert_bytes < available_ram * 0.7:
                # Experts fit in RAM — use single pinned buffer (no mmap overhead).
                pinned = torch.empty(
                    num_moe_layers, num_experts, layout.total_bytes,
                    dtype=torch.uint8,
                )
                if torch.cuda.is_available():
                    pinned = pinned.pin_memory()
                pinned.copy_(data)
                del data, mmap
                gc.collect()
                try:
                    os.unlink(tmp_name)
                except OSError:
                    logger.warning("Failed to delete temp file %s", tmp_name, exc_info=True)
                # Bypass __init__'s is_pinned() check (no CUDA in test envs).
                store = cls.__new__(cls)
                store._data = pinned
                store.layout = layout
                store.num_layers = num_moe_layers
                store.num_experts = num_experts
                store.expert_bytes = layout.total_bytes
                store._bf16_layout = layout
                store.buffer_expert_bytes = layout.total_bytes
                store._disk_offload = False
                return store, num_experts, None

            # Experts don't fit — keep mmap + RAMCache for hot subset.
            # Use MADV_HUGEPAGE for transparent huge pages — better for large
            # sequential reads per expert blob than MADV_RANDOM readahead disable.
            MADV_HUGEPAGE = 14
            mmap_addr = ctypes.c_void_p(data.data_ptr())
            mmap_len = ctypes.c_size_t(data.numel())
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.madvise(mmap_addr, mmap_len, ctypes.c_int(MADV_HUGEPAGE))

            # Build store bypassing __init__'s is_pinned() check.
            store = cls.__new__(cls)
            store._data = data
            store.layout = layout
            store.num_layers = num_moe_layers
            store.num_experts = num_experts
            store.expert_bytes = layout.total_bytes
            store._bf16_layout = layout
            store.buffer_expert_bytes = layout.total_bytes
            store._disk_offload = True
            # Keep references to prevent GC of the mmap.
            store._mmap = mmap
            store._tmp_name = tmp_name

            # Create FastExpertReader for pread-based SSD reads (bypasses mmap
            # page fault overhead: single syscall per expert vs ~3300 4KB faults).
            from .fast_io import FastExpertReader
            from .ram_cache import RAMCache as _RAMCache

            expert_offsets: dict[tuple[int, int], int] = {}
            for li in range(num_moe_layers):
                for ei in range(num_experts):
                    expert_offsets[(li, ei)] = (li * num_experts + ei) * layout.total_bytes
            fast_reader = FastExpertReader(tmp_name, expert_offsets, layout.total_bytes)

            total_expert_slots = num_moe_layers * num_experts
            if ram_cache_gb > 0:
                num_slots = max(1, int(ram_cache_gb * 1024**3) // layout.total_bytes)
            else:
                # Target: hold ALL experts. The mmap data is on SSD, RAMCache
                # acts as a fast pinned-memory mirror. After warmup, zero SSD reads.
                num_slots = total_expert_slots
            ram_cache = _RAMCache(
                num_slots=num_slots,
                expert_bytes=layout.total_bytes,
                fast_reader=fast_reader,
            )

            return store, num_experts, ram_cache

        pinned = torch.empty(num_moe_layers, num_experts, layout.total_bytes, dtype=torch.uint8).pin_memory()
        pinned.copy_(data)

        del data
        del mmap
        gc.collect()
        try:
            os.unlink(tmp_name)
        except OSError:
            logger.warning("Failed to delete temp file %s", tmp_name, exc_info=True)

        store = cls(pinned, layout, num_moe_layers, num_experts)
        store._disk_offload = False
        return store, num_experts

    @property
    def _fp8(self) -> bool:
        return self._bf16_layout is not self.layout

    def allocate_buffer(self, device: torch.device) -> "ExpertBuffer":
        fp8_layout = self.layout if self._fp8 else None
        return ExpertBuffer(self._bf16_layout, device, fp8_layout=fp8_layout)

    def get_expert_data(self, layer_idx: int, expert_idx: int) -> torch.Tensor:
        """Return raw packed expert data from CPU store (pinned memory)."""
        return self._data[layer_idx, expert_idx]

    def copy_to_buffer_slot(
        self,
        cache: "ExpertCache",
        slot: int,
        layer_idx: int,
        expert_idx: int,
    ) -> None:
        """Copy expert data directly into a cache slot (no intermediate buffer).

        Used by imatrix seeding to pre-populate the GPU cache without
        allocating a temporary ExpertBuffer per expert.
        """
        if not self._fp8:
            cache._packed[slot].copy_(self._data[layer_idx, expert_idx])
            return
        # FP8 path: dequantize via a temporary buffer, then write to slot.
        buf = self.allocate_buffer(cache.device)
        self.copy_to_buffer(buf, layer_idx, expert_idx)
        cache._packed[slot].copy_(buf.packed)

    def copy_to_buffer(
        self,
        buf: "ExpertBuffer",
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ):
        if not self._fp8:
            # BF16 / native-quant path — single async H2D, minimal overhead.
            buf.packed.copy_(self._data[layer_idx, expert_idx], non_blocking=non_blocking)
            return

        # Fix A: GPU-side FP8→BF16 dequant.
        # Step 1: H2D raw FP8 bytes (half the BF16 size → ~1ms at PCIe 4.0).
        #         non_blocking=True: CPU returns immediately; GPU copy runs on
        #         the calling stream (transfer_stream) and completes before any
        #         subsequent ops enqueued on that same stream.
        buf.fp8_stage.copy_(self._data[layer_idx, expert_idx], non_blocking=non_blocking)

        # Step 2: GPU dequant per tensor — runs on calling stream after Step 1.
        #         ~0.1ms GPU kernel instead of 9.5ms CPU loop.
        for name, (fp8_shape, fp8_dtype) in self.layout.specs.items():
            fp8_off = self.layout.offsets[name]
            fp8_sz = self.layout.sizes[name]
            bf16_off = self._bf16_layout.offsets[name]
            bf16_sz = self._bf16_layout.sizes[name]
            bf16_shape, bf16_dtype = self._bf16_layout.specs[name]
            src = buf.fp8_stage[fp8_off : fp8_off + fp8_sz].view(fp8_dtype).view(fp8_shape)
            dst = buf.packed[bf16_off : bf16_off + bf16_sz].view(bf16_dtype).view(bf16_shape)
            if fp8_dtype.is_floating_point:
                dst.copy_(src.to(bf16_dtype))
            else:
                dst.copy_(src.view(torch.uint8).view(bf16_shape))


