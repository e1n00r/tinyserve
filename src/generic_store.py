"""Model-agnostic expert weight storage and GPU buffer management.

Stores expert weights as flat byte buffers on CPU. Each expert is a set of
named tensors (e.g., gate_proj.weight, up_proj.weight) packed contiguously.
The buffer layout is computed at init time from actual weight shapes.
"""

from collections import OrderedDict

import torch


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


class GenericExpertStore:
    """Stores all expert weights on CPU as flat byte buffers."""

    def __init__(
        self,
        data: torch.Tensor,
        layout: TensorLayout,
        num_layers: int,
        num_experts: int,
    ):
        self._data = data
        self.layout = layout
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_bytes = layout.total_bytes

    @classmethod
    def from_dict(
        cls,
        expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
        num_layers: int,
        num_experts: int,
    ) -> "GenericExpertStore":
        sample_key = next(iter(expert_weights))
        layout = TensorLayout.from_tensors(expert_weights[sample_key])

        data = torch.empty(num_layers, num_experts, layout.total_bytes, dtype=torch.uint8)

        for (layer_idx, expert_idx), tensors in expert_weights.items():
            offset = 0
            for name, tensor in tensors.items():
                raw = tensor.contiguous().view(-1).view(torch.uint8)
                data[layer_idx, expert_idx, offset:offset + raw.numel()] = raw
                offset += raw.numel()

        return cls(data, layout, num_layers, num_experts)

    def allocate_buffer(self, device: torch.device) -> GenericExpertBuffer:
        return GenericExpertBuffer(self.layout, device)

    def copy_to_buffer(
        self,
        buf: GenericExpertBuffer,
        layer_idx: int,
        expert_idx: int,
        non_blocking: bool = False,
    ):
        buf.packed.copy_(self._data[layer_idx, expert_idx], non_blocking=non_blocking)

    def prefetch(self, layer_idx: int, expert_idx: int):
        pass


class GenericLRUCache:
    """LRU cache for generic expert buffers in GPU VRAM."""

    def __init__(self, capacity: int, expert_bytes: int, device: torch.device):
        self.capacity = capacity
        self.expert_bytes = expert_bytes
        self.device = device
        self._packed = torch.empty(capacity, expert_bytes, dtype=torch.uint8, device=device)
        self._lru: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._free_slots = list(range(capacity - 1, -1, -1))
        self.hits = 0
        self.misses = 0

    def lookup(self, layer_idx: int, expert_idx: int) -> int | None:
        key = (layer_idx, expert_idx)
        if key in self._lru:
            self._lru.move_to_end(key)
            self.hits += 1
            return self._lru[key]
        self.misses += 1
        return None

    def allocate(self, layer_idx: int, expert_idx: int) -> int:
        key = (layer_idx, expert_idx)
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            _, slot = self._lru.popitem(last=False)
        self._lru[key] = slot
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
