"""Model-agnostic expert weight storage and GPU buffer management.

Stores expert weights as flat byte buffers on CPU. Each expert is a set of
named tensors (e.g., gate_proj.weight, up_proj.weight) packed contiguously.
The buffer layout is computed at init time from actual weight shapes.
"""

import gc
import os
import tempfile
import numpy as np
import torch

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
        return TensorLayout({
            name: (tensor.shape, tensor.dtype)
            for name, tensor in tensors.items()
        })


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
        dest[offset:offset + nbytes].copy_(raw)


class GenericExpertBuffer:
    """Pre-allocated GPU buffer for one expert's weights."""

    def __init__(self, layout: TensorLayout, device: torch.device):
        self.layout = layout
        self.packed = torch.empty(layout.total_bytes, dtype=torch.uint8, device=device)

    def get_tensor(self, name: str) -> torch.Tensor:
        shape, dtype = self.layout.specs[name]
        offset = self.layout.offsets[name]
        nbytes = self.layout.sizes[name]
        return self.packed[offset:offset + nbytes].view(dtype).view(shape)


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


class GenericExpertStore:
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
        self._data = data
        self.layout = layout
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_bytes = layout.total_bytes
        # GPU buffers always need BF16-sized slots.
        self._bf16_layout = bf16_layout if bf16_layout is not None else layout
        self.buffer_expert_bytes = self._bf16_layout.total_bytes

    @classmethod
    def from_dict(
        cls,
        expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
        num_layers: int,
        num_experts: int,
        fp8: bool = False,
    ) -> "GenericExpertStore":
        sample_key = next(iter(expert_weights))
        bf16_layout = TensorLayout.from_tensors(expert_weights[sample_key])
        store_layout = _fp8_layout(bf16_layout) if fp8 else bf16_layout

        data = torch.empty(num_layers, num_experts, store_layout.total_bytes,
                           dtype=torch.uint8).pin_memory()

        for (layer_idx, expert_idx), tensors in expert_weights.items():
            packed = _quantize_to_fp8(tensors) if fp8 else tensors
            offset = 0
            for name, tensor in packed.items():
                raw = tensor.contiguous().view(-1).view(torch.uint8)
                data[layer_idx, expert_idx, offset:offset + raw.numel()] = raw
                offset += raw.numel()

        return cls(data, store_layout, num_layers, num_experts,
                   bf16_layout=bf16_layout if fp8 else None)

    @classmethod
    def build(
        cls,
        moe_layers: list,
        moe_block_attr: str,
        expert_list_attr: str,
        fp8: bool = False,
    ) -> tuple["GenericExpertStore", int]:
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
        mmap = np.memmap(tmp_name, dtype=np.uint8, mode="w+",
                         shape=(num_layers, num_experts, store_layout.total_bytes))
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
        pinned = torch.empty(num_layers, num_experts, store_layout.total_bytes,
                             dtype=torch.uint8).pin_memory()
        pinned.copy_(data)

        # Release mmap and tempfile — data now lives entirely in pinned RAM.
        del data
        del mmap
        gc.collect()
        try:
            os.unlink(tmp_name)
        except OSError:
            pass

        store = cls(pinned, store_layout, num_layers, num_experts,
                    bf16_layout=bf16_layout if fp8 else None)
        return store, num_experts

    @classmethod
    def from_safetensors(
        cls,
        model_id: str,
        moe_block_attr: str,
        expert_list_attr: str,
        layer_indices: list[int],
    ) -> tuple["GenericExpertStore", int]:
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

        if os.path.isdir(model_id):
            model_dir = model_id
        else:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(model_id, ignore_patterns=["*.bin", "*.msgpack", "*.h5"])

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                shard_files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            shard_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))

        expert_prefix = f"{moe_block_attr}.{expert_list_attr}."
        layer_param_name_re = re.compile(
            r"layers\.(\d+)\." + re.escape(expert_prefix) + r"(.+)$"
        )

        # Collect: layer_idx -> param_base_name -> tensor
        layer_tensors: dict[int, dict[str, torch.Tensor]] = {}
        for shard_name in shard_files:
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
        mmap = np.memmap(tmp_name, dtype=np.uint8, mode="w+",
                         shape=(num_moe_layers, num_experts, layout.total_bytes))
        data = torch.from_numpy(mmap)

        for store_idx, li in enumerate(moe_layer_indices):
            tensors = layer_tensors[li]
            for expert_idx in range(num_experts):
                per_expert = {n: t[expert_idx].contiguous() for n, t in tensors.items()}
                _pack_tensors(data[store_idx, expert_idx], layout, per_expert)
            del layer_tensors[li]
            gc.collect()

        mmap.flush()
        pinned = torch.empty(num_moe_layers, num_experts, layout.total_bytes,
                             dtype=torch.uint8).pin_memory()
        pinned.copy_(data)

        del data
        del mmap
        gc.collect()
        try:
            os.unlink(tmp_name)
        except OSError:
            pass

        store = cls(pinned, layout, num_moe_layers, num_experts)
        return store, num_experts

    @property
    def _fp8(self) -> bool:
        return self._bf16_layout is not self.layout

    def allocate_buffer(self, device: torch.device) -> "GenericExpertBuffer":
        return GenericExpertBuffer(self._bf16_layout, device)

    def copy_to_buffer(
        self,
        buf: "GenericExpertBuffer",
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ):
        if not self._fp8:
            buf.packed.copy_(self._data[layer_idx, expert_idx], non_blocking=non_blocking)
            return
        # FP8 path: dequant to BF16 on CPU using a per-buffer pinned staging area,
        # then single bulk H2D.  One staging buffer per GenericExpertBuffer avoids
        # contention between the two pipeline slots (buf_a / buf_b).
        if not hasattr(self, "_cpu_bf16_stages"):
            self._cpu_bf16_stages: dict[int, torch.Tensor] = {}
        buf_key = id(buf)
        if buf_key not in self._cpu_bf16_stages:
            self._cpu_bf16_stages[buf_key] = torch.empty(
                self.buffer_expert_bytes, dtype=torch.uint8
            ).pin_memory()
        cpu_stage = self._cpu_bf16_stages[buf_key]
        fp8_data = self._data[layer_idx, expert_idx]
        for name, (bf16_shape, bf16_dtype) in self._bf16_layout.specs.items():
            fp8_offset = self.layout.offsets[name]
            fp8_bytes = self.layout.sizes[name]
            fp8_shape, fp8_dtype = self.layout.specs[name]
            bf16_offset = self._bf16_layout.offsets[name]
            bf16_bytes = self._bf16_layout.sizes[name]
            src = fp8_data[fp8_offset:fp8_offset + fp8_bytes].view(fp8_dtype).view(fp8_shape)
            dst = cpu_stage[bf16_offset:bf16_offset + bf16_bytes].view(bf16_dtype).view(bf16_shape)
            dst.copy_(src)  # CPU: FP8→BF16 dtype cast
        buf.packed.copy_(cpu_stage, non_blocking=non_blocking)

    def prefetch(self, layer_idx: int, expert_idx: int):
        pass


class GenericLRUCache:
    """LRU cache for generic expert buffers in GPU VRAM."""

    def __init__(self, capacity: int, expert_bytes: int, device: torch.device, policy: str = "lru"):
        self.capacity = capacity
        self.expert_bytes = expert_bytes
        self.device = device
        self._packed = torch.empty(capacity, expert_bytes, dtype=torch.uint8, device=device)
        self._policy = make_policy(policy, capacity)
        self._free_slots = list(range(capacity - 1, -1, -1))
        self.hits = 0
        self.misses = 0

    def lookup(self, layer_idx: int, expert_idx: int) -> int | None:
        slot = self._policy.lookup((layer_idx, expert_idx))
        if slot is not None:
            self.hits += 1
        else:
            self.misses += 1
        return slot

    def allocate(self, layer_idx: int, expert_idx: int) -> int:
        key = (layer_idx, expert_idx)
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            evict_key, slot = self._policy.select_evict()
            self._policy.remove(evict_key)
        self._policy.insert(key, slot)
        return slot

    def store_from_buffer(self, slot: int, buf: GenericExpertBuffer):
        self._packed[slot].copy_(buf.packed)

    def load_to_buffer(self, slot: int, buf: GenericExpertBuffer):
        buf.packed.copy_(self._packed[slot])

    def get_packed(self, slot: int) -> torch.Tensor:
        return self._packed[slot]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_stats(self):
        self.hits = 0
        self.misses = 0

    @staticmethod
    def estimate_capacity(available_bytes: int, expert_bytes: int) -> int:
        return available_bytes // expert_bytes
