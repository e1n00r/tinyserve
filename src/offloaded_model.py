"""Unified offloaded MoE model wrapper.

Takes any nn.Module with MoE layers, extracts expert weights to CPU,
installs offloaded dispatch hooks, moves non-expert weights to GPU.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generic_store import GenericExpertStore
from .generic_pipeline import GenericExpertPipeline


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
    ) -> "OffloadedModel":
        model.eval()
        layers = model.layers

        moe_layers = [
            (li, layer) for li, layer in enumerate(layers)
            if li >= first_moe_layer and hasattr(getattr(layer, moe_block_attr, None), expert_list_attr)
        ]

        expert_weights, num_experts = _extract_expert_weights(
            moe_layers, moe_block_attr, expert_list_attr, top_k,
        )

        num_moe_layers = len(moe_layers)
        store = GenericExpertStore.from_dict(expert_weights, num_moe_layers, num_experts)

        pipelines = []
        for store_idx, (li, layer) in enumerate(moe_layers):
            moe_block = getattr(layer, moe_block_attr)
            expert_container = getattr(moe_block, expert_list_attr)
            template = _make_template(expert_container, device)
            pipeline = GenericExpertPipeline(store, template, device, cache_capacity=cache_capacity)
            pipelines.append(pipeline)

            _install_offloaded_forward(
                moe_block, pipeline, store_idx, router_attr, top_k,
                returns_router_logits=returns_router_logits,
                softmax_order=softmax_order,
            )

            for param in expert_container.parameters():
                param.data = torch.empty(0, device="cpu")

        model = model.to(device).to(torch.bfloat16)
        return cls(model, pipelines)


def _extract_expert_weights(moe_layers, moe_block_attr, expert_list_attr, top_k):
    """Extract expert weights from either nn.ModuleList or fused nn.Parameter."""
    expert_weights = {}
    first_moe = getattr(moe_layers[0][1], moe_block_attr)
    expert_container = getattr(first_moe, expert_list_attr)

    if isinstance(expert_container, nn.ModuleList):
        num_experts = len(expert_container)
        for store_idx, (_, layer) in enumerate(moe_layers):
            moe_block = getattr(layer, moe_block_attr)
            expert_list = getattr(moe_block, expert_list_attr)
            for expert_idx, expert in enumerate(expert_list):
                expert_tensors = {}
                for name, param in expert.named_parameters():
                    expert_tensors[name] = param.data.detach().cpu().clone()
                expert_weights[(store_idx, expert_idx)] = expert_tensors
    else:
        num_experts = next(iter(expert_container.parameters())).shape[0]
        for store_idx, (_, layer) in enumerate(moe_layers):
            moe_block = getattr(layer, moe_block_attr)
            expert_container = getattr(moe_block, expert_list_attr)
            for expert_idx in range(num_experts):
                expert_tensors = {}
                for name, param in expert_container.named_parameters():
                    expert_tensors[name] = param.data[expert_idx].detach().cpu().clone()
                expert_weights[(store_idx, expert_idx)] = expert_tensors

    return expert_weights, num_experts


def _make_template(expert_container, device):
    """Create a template expert module for weight swapping."""
    if isinstance(expert_container, nn.ModuleList):
        return copy.deepcopy(expert_container[0]).to(device).to(torch.bfloat16)
    else:
        return _FusedExpertTemplate(expert_container).to(device).to(torch.bfloat16)


class _FusedExpertTemplate(nn.Module):
    """Template for fused-parameter experts (GPT-OSS, Qwen3.5).

    Creates nn.Parameter placeholders matching the per-expert slice shapes.
    Detects activation type and weight layout from the original container.
    """

    def __init__(self, fused_container: nn.Module):
        super().__init__()
        self._param_names = []
        for name, param in fused_container.named_parameters():
            per_expert_shape = param.shape[1:]
            self.register_parameter(name, nn.Parameter(torch.zeros(per_expert_shape, dtype=param.dtype)))
            self._param_names.append(name)

        self._act_fn = getattr(fused_container, "act_fn", None)
        self._has_bias = "gate_up_proj_bias" in self._param_names

    def forward(self, hidden_states):
        params = {name: getattr(self, name) for name in self._param_names}

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
        output = pipeline.execute_layer_experts(flat, layer_idx, top_idx, routing_weights)

        if shared_expert is not None:
            output = output + shared_expert(flat)

        if batch is not None:
            output = output.view(batch, seq_len, -1)

        if returns_router_logits:
            return output, route.last_logits
        return output

    moe_block.forward = offloaded_forward
