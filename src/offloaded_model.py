"""Unified offloaded MoE model wrapper.

Takes any nn.Module with MoE layers, extracts expert weights to CPU,
installs offloaded dispatch hooks, moves non-expert weights to GPU.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generic_store import GenericExpertStore, _is_qtensor
from .generic_pipeline import GenericExpertPipeline
from .mxfp4 import dequant_mxfp4_no_transpose

_MXFP4_BACKEND = "pytorch"
try:
    from .triton_dot_scaled import dot_scaled_vecmat as _dot_scaled_vecmat
    _MXFP4_BACKEND = "dot_scaled"
except Exception:
    try:
        from .triton_dequant import fused_dequant_vecmat as _fused_dequant_vecmat
        _MXFP4_BACKEND = "triton_sw"
    except Exception:
        pass


def _mxfp4_linear(
    x: torch.Tensor,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.shape[0] == 1 and _MXFP4_BACKEND == "dot_scaled":
        return _dot_scaled_vecmat(x, blocks, scales, bias)
    if x.shape[0] == 1 and _MXFP4_BACKEND == "triton_sw":
        return _fused_dequant_vecmat(x, blocks, scales, bias)
    w = dequant_mxfp4_no_transpose(blocks, scales, x.dtype)
    return F.linear(x, w, bias.to(x.dtype) if bias is not None else None)


class OffloadedModel(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        pipelines: list[GenericExpertPipeline],
    ):
        super().__init__()
        self.model = model
        self.pipelines = pipelines

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def cache_stats(self) -> dict:
        total_hits = 0
        total_misses = 0
        for pipeline in self.pipelines:
            if pipeline.cache is not None:
                total_hits += pipeline.cache.hits
                total_misses += pipeline.cache.misses
        return {"total_hits": total_hits, "total_misses": total_misses}

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        moe_block_attr: str,
        expert_list_attr: str,
        router_attr: str,
        top_k: int,
        device: torch.device,
        cache_capacity: int = 0,
        returns_router_logits: bool = False,
        softmax_order: str = "topk_then_softmax",
        first_moe_layer: int = 0,
        model_id: str | None = None,
        cache_policy: str = "lru",
    ) -> "OffloadedModel":
        model.eval()
        layers = model.layers

        moe_layers = [
            (li, layer) for li, layer in enumerate(layers)
            if li >= first_moe_layer and hasattr(getattr(layer, moe_block_attr, None), expert_list_attr)
        ]

        first_container = getattr(getattr(moe_layers[0][1], moe_block_attr), expert_list_attr)

        if model_id is not None:
            # Native quantized path: load expert weights directly from safetensors.
            # Non-expert weights (attention, norms, etc.) are taken from `model`.
            layer_indices = [li for li, _ in moe_layers]
            store, _ = GenericExpertStore.from_safetensors(
                model_id, moe_block_attr, expert_list_attr, layer_indices
            )
            template = _FusedExpertTemplate.from_layout(store.layout, first_container)
            template = template.to(device)
            if not template._is_mxfp4:
                template = template.to(torch.bfloat16)
            else:
                for name in template._param_names:
                    param = getattr(template, name)
                    if param.dtype.is_floating_point:
                        param.data = param.data.to(torch.bfloat16)
            # Zero expert params in the (dequantized) model — they're not needed.
            for _, layer in moe_layers:
                container = getattr(getattr(layer, moe_block_attr), expert_list_attr)
                for param in container.parameters():
                    param.data = torch.empty(0, device="cpu")
        else:
            # Standard path: extract weights from the model's current parameters.
            template = _make_template(first_container, device)
            store, _ = GenericExpertStore.build(moe_layers, moe_block_attr, expert_list_attr)

        free_vram = torch.cuda.mem_get_info(device)[0]
        reserved = 2 * store.expert_bytes + 512 * 1024 * 1024
        available_for_cache = max(0, free_vram - reserved)
        auto_capacity = store.expert_bytes > 0 and available_for_cache // store.expert_bytes or 0
        if cache_capacity > 0:
            cache_capacity = min(cache_capacity, auto_capacity)

        shared_buf_a = store.allocate_buffer(device)
        shared_buf_b = store.allocate_buffer(device)
        transfer_stream = torch.cuda.Stream(device)
        compute_stream = torch.cuda.Stream(device)
        shared_stream = torch.cuda.Stream(device)
        from .generic_store import GenericLRUCache
        cache = GenericLRUCache(cache_capacity, store.expert_bytes, device, policy=cache_policy) if cache_capacity > 0 else None

        print(f"  Cache capacity: {cache_capacity} experts ({cache_capacity * store.expert_bytes / 1e9:.2f} GB GPU)")

        pipelines = []
        for store_idx, (li, layer) in enumerate(moe_layers):
            moe_block = getattr(layer, moe_block_attr)
            pipeline = GenericExpertPipeline(
                store, template, device,
                buf_a=shared_buf_a,
                buf_b=shared_buf_b,
                transfer_stream=transfer_stream,
                compute_stream=compute_stream,
                cache=cache,
                shared_stream=shared_stream,
            )
            pipelines.append(pipeline)

            _install_offloaded_forward(
                moe_block, pipeline, store_idx, router_attr, top_k,
                returns_router_logits=returns_router_logits,
                softmax_order=softmax_order,
            )

        model = model.to(device).to(torch.bfloat16)
        return cls(model, pipelines)


def _make_template(expert_container, device):
    """Create a template expert module for weight swapping."""
    if isinstance(expert_container, nn.ModuleList):
        return copy.deepcopy(expert_container[0]).to(device).to(torch.bfloat16)
    template = _FusedExpertTemplate(expert_container).to(device)
    if template._is_mxfp4:
        # uint8 blocks/scales must stay uint8; only cast floating-point params.
        for name in template._param_names:
            param = getattr(template, name)
            if param.dtype.is_floating_point:
                param.data = param.data.to(torch.bfloat16)
    else:
        template = template.to(torch.bfloat16)
    return template


class _FusedExpertTemplate(nn.Module):
    """Template for fused-parameter experts (GPT-OSS, Qwen3.5).

    Creates nn.Parameter placeholders matching the per-expert slice shapes.
    Detects activation type and weight layout from the original container.
    Supports both bf16 and MXFP4-quantized weight layouts.
    """

    def __init__(self, fused_container: nn.Module):
        super().__init__()
        self._param_names = []
        for name, param in fused_container.named_parameters():
            if _is_qtensor(param):
                # Expand QTensor into int_data (blocks) and scale components.
                int_data = param.int_data
                scale = param.scale
                self.register_parameter(
                    name, nn.Parameter(torch.zeros(int_data.shape[1:], dtype=int_data.dtype))
                )
                self.register_parameter(
                    name + "_scales",
                    nn.Parameter(torch.zeros(scale.shape[1:], dtype=scale.dtype)),
                )
                self._param_names.extend([name, name + "_scales"])
            else:
                per_expert_shape = param.shape[1:]
                self.register_parameter(name, nn.Parameter(torch.zeros(per_expert_shape, dtype=param.dtype)))
                self._param_names.append(name)

        self._act_fn = getattr(fused_container, "act_fn", None)
        self._has_bias = "gate_up_proj_bias" in self._param_names
        self._is_mxfp4 = "gate_up_proj_scales" in self._param_names

    @classmethod
    def from_layout(cls, layout: "TensorLayout", fused_container: nn.Module) -> "_FusedExpertTemplate":
        """Create template from a TensorLayout (used with native safetensors loading)."""
        from .generic_store import TensorLayout  # noqa: F401
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj._param_names = []
        for name, (shape, dtype) in layout.specs.items():
            obj.register_parameter(name, nn.Parameter(torch.zeros(shape, dtype=dtype)))
            obj._param_names.append(name)
        obj._act_fn = getattr(fused_container, "act_fn", None)
        obj._has_bias = "gate_up_proj_bias" in obj._param_names
        obj._is_mxfp4 = "gate_up_proj_scales" in obj._param_names
        return obj

    def forward(self, hidden_states):
        params = {name: getattr(self, name) for name in self._param_names}

        if self._is_mxfp4:
            gate_up = _mxfp4_linear(
                hidden_states,
                params["gate_up_proj"],
                params["gate_up_proj_scales"],
                params.get("gate_up_proj_bias"),
            )
            gate = gate_up[..., ::2].clamp(max=7.0)
            up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
            gated = (up + 1) * gate * torch.sigmoid(gate * 1.702)
            return _mxfp4_linear(
                gated,
                params["down_proj"],
                params["down_proj_scales"],
                params.get("down_proj_bias"),
            )

        w_gu = params["gate_up_proj"]
        if w_gu.shape[0] == hidden_states.shape[-1]:
            w_gu = w_gu.t()
        gate_up = nn.functional.linear(hidden_states, w_gu, params.get("gate_up_proj_bias"))

        if self._act_fn is not None:
            gate, up = gate_up.chunk(2, dim=-1)
            gated = self._act_fn(gate) * up
        else:
            gate = gate_up[..., ::2].clamp(max=7.0)
            up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
            gated = (up + 1) * gate * torch.sigmoid(gate * 1.702)

        w_dn = params["down_proj"]
        if w_dn.shape[0] == gated.shape[-1]:
            w_dn = w_dn.t()
        return nn.functional.linear(gated, w_dn, params.get("down_proj_bias"))


def _extract_routing_fn(moe_block, router_attr, top_k, softmax_order):
    """Build a routing function matching the model's original routing logic."""
    router = getattr(moe_block, router_attr)
    renormalize = getattr(moe_block, "norm_topk_prob", True)

    if softmax_order == "router_native":
        def route(hidden_states):
            result = router(hidden_states)
            route.last_logits = None

            if isinstance(result, torch.Tensor):
                router_logits = result
                routing_weights = torch.sigmoid(router_logits)
                routing_weights, top_idx = torch.topk(routing_weights, top_k, dim=-1)
                return top_idx, routing_weights.to(hidden_states.dtype)

            if len(result) == 3:
                router_logits, routing_weights, top_idx = result
                route.last_logits = router_logits
                return top_idx, routing_weights.to(hidden_states.dtype)

            first, second = result
            if second.dtype in (torch.int32, torch.int64):
                routing_weights_full, top_idx = first, second
                if routing_weights_full.shape[-1] > top_k:
                    routing_weights = routing_weights_full.gather(-1, top_idx)
                else:
                    routing_weights = routing_weights_full
            else:
                top_idx, routing_weights = first, second
            return top_idx, routing_weights.to(hidden_states.dtype)
    elif softmax_order == "softmax_then_topk":
        def route(hidden_states):
            router_logits = router(hidden_states)
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, top_idx = torch.topk(routing_weights, top_k, dim=-1)
            if renormalize:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            route.last_logits = router_logits
            return top_idx, routing_weights.to(hidden_states.dtype)
    else:
        def route(hidden_states):
            router_logits = router(hidden_states)
            top_vals, top_idx = torch.topk(router_logits, top_k, dim=-1)
            routing_weights = F.softmax(top_vals, dim=-1).to(hidden_states.dtype)
            route.last_logits = router_logits
            return top_idx, routing_weights

    route.last_logits = None
    return route


def _install_offloaded_forward(
    moe_block, pipeline, layer_idx, router_attr, top_k,
    returns_router_logits: bool = False,
    softmax_order: str = "topk_then_softmax",
):
    route = _extract_routing_fn(moe_block, router_attr, top_k, softmax_order)

    shared_expert = getattr(moe_block, "shared_experts", None) or getattr(moe_block, "shared_expert", None)

    def offloaded_forward(hidden_states, **_kwargs):
        if hidden_states.dim() == 3:
            batch, seq_len, hidden = hidden_states.shape
            flat = hidden_states.view(-1, hidden)
        else:
            flat = hidden_states
            batch, seq_len = None, None

        top_idx, routing_weights = route(flat)

        shared_event = None
        shared_out = None
        if shared_expert is not None:
            with torch.cuda.stream(pipeline.shared_stream):
                shared_out = shared_expert(flat)
                shared_event = torch.cuda.Event()
                shared_event.record(pipeline.shared_stream)

        output = pipeline.execute_layer_experts(flat, layer_idx, top_idx, routing_weights)

        if shared_event is not None:
            torch.cuda.current_stream().wait_event(shared_event)
            output = output + shared_out

        if batch is not None:
            output = output.view(batch, seq_len, -1)

        if returns_router_logits:
            return output, route.last_logits
        return output

    moe_block.forward = offloaded_forward
