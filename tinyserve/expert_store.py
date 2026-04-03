"""Model-agnostic expert weight storage and GPU buffer management.

Stores expert weights as flat byte buffers on CPU. Each expert is a set of
named tensors (e.g., gate_proj.weight, up_proj.weight) packed contiguously.
The buffer layout is computed at init time from actual weight shapes.
"""

from __future__ import annotations

import ctypes
import gc
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .ram_cache import RAMCache

from .cache_policy import make_policy


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
            pass

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
            pass  # Cache save is best-effort; don't block loading.

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
                    pass
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
            pass

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


class ExpertCache:
    """LRU cache for generic expert buffers in GPU VRAM."""

    def __init__(
        self,
        capacity: int,
        expert_bytes: int,
        device: torch.device,
        policy: str = "lru",
        num_layers: int = 1,
        num_experts: int = 1,
    ):
        self.capacity = capacity
        self.expert_bytes = expert_bytes
        self.device = device
        self._packed = torch.empty(capacity, expert_bytes, dtype=torch.uint8, device=device)
        self._policy = make_policy(policy, capacity)
        self._free_slots = list(range(capacity - 1, -1, -1))
        self.hits = 0
        self.misses = 0
        # Per-layer statistics
        self._layer_hits: dict[int, int] = {}
        self._layer_misses: dict[int, int] = {}
        self._layer_miss_latencies: dict[int, list[float]] = {}
        self._expert_access_count: dict[tuple[int, int], int] = {}
        # Per-step tracking
        self._step_experts: set[tuple[int, int]] | None = None
        self._step_lookups: int = 0
        # Slot map: CPU array is primary (written per-allocate, no CUDA kernel),
        # GPU tensor is synced lazily before lookup_slots reads it.
        import numpy as np
        self._slot_map: torch.Tensor | None = None
        self._slot_map_cpu: np.ndarray | None = None
        self._slot_map_dirty: bool = False
        self._slot_map_dims = (num_layers, num_experts)
        if num_layers > 1 or num_experts > 1:
            self._slot_map_cpu = np.full((num_layers, num_experts), -1, dtype=np.int32)
            self._slot_map = torch.from_numpy(self._slot_map_cpu).to(dtype=torch.int32, device=device)

    def lookup(self, layer_idx: int, expert_idx: int) -> int | None:
        slot = self._policy.lookup((layer_idx, expert_idx))
        key = (layer_idx, expert_idx)
        self._expert_access_count[key] = self._expert_access_count.get(key, 0) + 1
        if self._step_experts is not None:
            self._step_experts.add(key)
            self._step_lookups += 1
        if slot is not None:
            self.hits += 1
            self._layer_hits[layer_idx] = self._layer_hits.get(layer_idx, 0) + 1
        else:
            self.misses += 1
            self._layer_misses[layer_idx] = self._layer_misses.get(layer_idx, 0) + 1
        return slot

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        """Check if expert is in cache without updating policy state or stats."""
        return self._policy.contains((layer_idx, expert_idx))

    def begin_pass(self) -> None:
        """Notify the policy that a new token forward pass is starting.

        Only meaningful for LeastStalePolicy — rotates fresh→stale so experts
        loaded for the previous token become eviction candidates. No-op for
        all other policies.
        """
        if hasattr(self._policy, "begin_pass"):
            self._policy.begin_pass()

    def _ensure_slot_map(self, layer_idx: int, expert_idx: int):
        """Lazily create or grow the slot map to fit (layer_idx, expert_idx)."""
        import numpy as np
        nl = max(self._slot_map_dims[0], layer_idx + 1)
        ne = max(self._slot_map_dims[1], expert_idx + 1)
        if self._slot_map_cpu is None:
            self._slot_map_cpu = np.full((nl, ne), -1, dtype=np.int32)
            self._slot_map = torch.from_numpy(self._slot_map_cpu).to(dtype=torch.int32, device=self.device)
            self._slot_map_dims = (nl, ne)
        elif nl > self._slot_map_cpu.shape[0] or ne > self._slot_map_cpu.shape[1]:
            new_cpu = np.full((nl, ne), -1, dtype=np.int32)
            old = self._slot_map_cpu
            new_cpu[:old.shape[0], :old.shape[1]] = old
            self._slot_map_cpu = new_cpu
            self._slot_map = torch.from_numpy(new_cpu).to(dtype=torch.int32, device=self.device)
            self._slot_map_dims = (nl, ne)
            self._slot_map_dirty = False

    def allocate(self, layer_idx: int, expert_idx: int) -> int:
        key = (layer_idx, expert_idx)
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            evict_key, slot = self._policy.select_evict()
            self._policy.remove(evict_key)
            if self._slot_map is not None:
                self._slot_map_cpu[evict_key[0], evict_key[1]] = -1
                self._slot_map_dirty = True
        self._policy.insert(key, slot)
        self._ensure_slot_map(layer_idx, expert_idx)
        self._slot_map_cpu[layer_idx, expert_idx] = slot
        self._slot_map_dirty = True
        return slot

    def flush_slot_updates(self):
        """Sync CPU slot_map to GPU. Called automatically by lookup_slots."""
        if not self._slot_map_dirty or self._slot_map_cpu is None:
            return
        self._slot_map.copy_(torch.from_numpy(self._slot_map_cpu))
        self._slot_map_dirty = False

    def lookup_slots(self, layer_idx: int, expert_ids: torch.Tensor) -> torch.Tensor:
        """GPU tensor cache lookup — syncs from CPU if dirty.

        Args:
            layer_idx: which MoE layer
            expert_ids: [top_k] int tensor on GPU

        Returns:
            [top_k] int32 tensor on GPU. Values >= 0 are cache slot indices,
            -1 means cache miss.
        """
        if self._slot_map_dirty:
            self.flush_slot_updates()
        if self._slot_map is None:
            return torch.full_like(expert_ids, -1, dtype=torch.int32)
        if layer_idx >= self._slot_map.shape[0]:
            return torch.full_like(expert_ids, -1, dtype=torch.int32)
        ne = self._slot_map.shape[1]
        ids = expert_ids.long().to(self.device)
        safe = ids.clamp(max=ne - 1)
        result = self._slot_map[layer_idx, safe]
        result = torch.where(ids < ne, result,
            torch.tensor(-1, dtype=torch.int32, device=self.device))
        return result

    def store_from_buffer(self, slot: int, buf: ExpertBuffer):
        self._packed[slot].copy_(buf.packed)

    def load_to_buffer(self, slot: int, buf: ExpertBuffer):
        buf.packed.copy_(self._packed[slot])

    def get_packed(self, slot: int) -> torch.Tensor:
        return self._packed[slot]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_miss_latency(self, layer_idx: int, latency_ms: float):
        if layer_idx not in self._layer_miss_latencies:
            self._layer_miss_latencies[layer_idx] = []
        self._layer_miss_latencies[layer_idx].append(latency_ms)

    def get_layer_stats(self) -> dict[int, dict]:
        layers = set(self._layer_hits.keys()) | set(self._layer_misses.keys())
        layers |= set(range(self._slot_map_dims[0]))
        result = {}
        for li in sorted(layers):
            h = self._layer_hits.get(li, 0)
            m = self._layer_misses.get(li, 0)
            result[li] = {
                "hits": h,
                "misses": m,
                "hit_rate": h / (h + m) if (h + m) > 0 else 0.0,
                "miss_latency_ms": self._layer_miss_latencies.get(li, []),
            }
        return result

    def get_expert_frequencies(self) -> dict[tuple[int, int], int]:
        return dict(self._expert_access_count)

    def begin_step(self):
        self._step_experts = set()
        self._step_lookups = 0

    def end_step(self) -> dict:
        result = {
            "unique_experts_accessed": len(self._step_experts) if self._step_experts else 0,
            "total_lookups": self._step_lookups,
        }
        self._step_experts = None
        self._step_lookups = 0
        return result

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self._layer_hits.clear()
        self._layer_misses.clear()
        self._layer_miss_latencies.clear()
        self._expert_access_count.clear()

    def clear(self):
        """Evict all entries. Cache is empty after this call."""
        while len(self._policy) > 0:
            key, slot = self._policy.select_evict()
            self._policy.remove(key)
            self._free_slots.append(slot)
        if self._slot_map_cpu is not None:
            self._slot_map_cpu[:] = -1
            self._slot_map_dirty = True
        self.reset_stats()

    def shrink(self, n_slots: int) -> int:
        """Reduce capacity by n_slots. Evicts LRU experts as needed.

        Returns bytes freed (n_slots × expert_bytes).
        """
        if n_slots <= 0 or n_slots > self.capacity:
            return 0

        # Evict LRU experts until we have n_slots free
        while len(self._free_slots) < n_slots and len(self._policy) > 0:
            key, slot = self._policy.select_evict()
            self._policy.remove(key)
            self._free_slots.append(slot)
            if self._slot_map_cpu is not None:
                self._slot_map_cpu[key[0], key[1]] = -1
                self._slot_map_dirty = True

        # Take the n_slots highest-numbered free slots to remove from the end
        self._free_slots.sort(reverse=True)
        self._free_slots = self._free_slots[n_slots:]

        new_capacity = self.capacity - n_slots
        self._packed = self._packed[:new_capacity].clone()
        self.capacity = new_capacity

        if self._slot_map_dirty:
            self.flush_slot_updates()

        return n_slots * self.expert_bytes

    def grow(self, n_slots: int) -> None:
        """Increase capacity by n_slots."""
        if n_slots <= 0:
            return
        old_capacity = self.capacity
        new_capacity = old_capacity + n_slots
        new_packed = torch.empty(
            new_capacity, self.expert_bytes, dtype=torch.uint8, device=self.device
        )
        new_packed[:old_capacity] = self._packed
        self._packed = new_packed
        self.capacity = new_capacity
        self._free_slots.extend(range(old_capacity, new_capacity))

    @staticmethod
    def estimate_capacity(available_bytes: int, expert_bytes: int) -> int:
        return available_bytes // expert_bytes
