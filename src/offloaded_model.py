"""Unified offloaded MoE model wrapper.

Takes any nn.Module with MoE layers, extracts expert weights to CPU,
installs offloaded dispatch hooks, moves non-expert weights to GPU.
"""

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
        expert_module_cls: type,
        expert_cls_kwargs: dict,
        moe_block_attr: str,
        expert_list_attr: str,
        router_attr: str,
        top_k: int,
        device: torch.device,
        cache_capacity: int = 0,
        returns_router_logits: bool = False,
    ) -> "OffloadedModel":
        model.eval()
        layers = model.layers

        expert_weights = _extract_all_expert_weights(
            layers, moe_block_attr, expert_list_attr,
        )

        num_layers = len(layers)
        num_experts = len(getattr(getattr(layers[0], moe_block_attr), expert_list_attr))
        store = GenericExpertStore.from_dict(expert_weights, num_layers, num_experts)

        pipelines = []
        for layer_idx, layer in enumerate(layers):
            moe_block = getattr(layer, moe_block_attr)
            template = expert_module_cls(**expert_cls_kwargs).to(device).to(torch.bfloat16)
            pipeline = GenericExpertPipeline(store, template, device, cache_capacity=cache_capacity)
            pipelines.append(pipeline)

            _install_offloaded_forward(
                moe_block, pipeline, layer_idx, router_attr, top_k,
                returns_router_logits=returns_router_logits,
            )

            expert_list = getattr(moe_block, expert_list_attr)
            for expert in expert_list:
                for param in expert.parameters():
                    param.data = torch.empty(0)

        model = model.to(device).to(torch.bfloat16)
        return cls(model, pipelines)


def _extract_all_expert_weights(layers, moe_block_attr, expert_list_attr):
    weights = {}
    for layer_idx, layer in enumerate(layers):
        moe_block = getattr(layer, moe_block_attr)
        expert_list = getattr(moe_block, expert_list_attr)
        for expert_idx, expert in enumerate(expert_list):
            expert_tensors = {}
            for name, param in expert.named_parameters():
                expert_tensors[name] = param.data.detach().cpu().clone()
            weights[(layer_idx, expert_idx)] = expert_tensors
    return weights


def _install_offloaded_forward(
    moe_block, pipeline, layer_idx, router_attr, top_k,
    returns_router_logits: bool = False,
):
    router = getattr(moe_block, router_attr)

    def offloaded_forward(hidden_states, **_kwargs):
        if hidden_states.dim() == 3:
            batch, seq_len, hidden = hidden_states.shape
            flat = hidden_states.view(-1, hidden)
        else:
            flat = hidden_states
            batch, seq_len = None, None

        router_logits = router(flat)
        top_vals, top_idx = torch.topk(router_logits, top_k, dim=-1)
        routing_weights = F.softmax(top_vals, dim=-1).to(flat.dtype)

        output = pipeline.execute_layer_experts(flat, layer_idx, top_idx, routing_weights)

        if batch is not None:
            output = output.view(batch, seq_len, -1)

        if returns_router_logits:
            return output, router_logits
        return output

    moe_block.forward = offloaded_forward
