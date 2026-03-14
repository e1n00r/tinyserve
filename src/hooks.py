"""HF model graph interception for expert offloading.

Walks the HF model, extracts expert weights to CPU storage,
and replaces MoE block forward methods with our offloaded pipeline.
"""

import torch
import torch.nn as nn

from .model_registry import ModelProfile
from .pipeline import ExpertPipeline


class OffloadedExpertDispatch:
    """Replaces the expert loop in an HF MoE block with our offloaded pipeline.

    The router stays on GPU (cheap). Expert weights are served from our
    LRU cache (hits) or streamed from CPU via double-buffered pipeline (misses).
    """

    def __init__(
        self,
        pipeline: ExpertPipeline,
        layer_idx: int,
        num_experts_per_tok: int,
    ):
        self.pipeline = pipeline
        self.layer_idx = layer_idx
        self.num_experts_per_tok = num_experts_per_tok

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run selected experts through our offloaded pipeline.

        Args:
            hidden_states: [num_tokens, hidden_size]
            expert_indices: [num_tokens, top_k] expert IDs
            routing_weights: [num_tokens, top_k] routing scores

        Returns:
            [num_tokens, hidden_size] weighted expert outputs
        """
        return self.pipeline.execute_layer_experts(
            hidden_states, self.layer_idx, expert_indices, routing_weights,
        )


def extract_expert_weights(
    model: nn.Module,
    profile: ModelProfile,
) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    """Extract all expert weights from HF model, return as CPU tensors.

    Returns:
        {(layer_idx, expert_idx): {"weight_name": tensor, ...}}
    """
    weights: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    layers = model.model.layers

    for layer_idx in range(profile.num_layers):
        if layer_idx < profile.first_moe_layer:
            continue

        layer = layers[layer_idx]
        moe_block = getattr(layer, profile.moe_block_attr)
        expert_container = getattr(moe_block, profile.expert_list_attr)

        if isinstance(expert_container, nn.ModuleList):
            for expert_idx, expert_module in enumerate(expert_container):
                expert_weights = {}
                for name in profile.expert_layout.weight_names:
                    linear = getattr(expert_module, name)
                    expert_weights[f"{name}.weight"] = linear.weight.data.cpu()
                    if linear.bias is not None:
                        expert_weights[f"{name}.bias"] = linear.bias.data.cpu()
                for name in profile.expert_layout.bias_names:
                    param = getattr(expert_module, name, None)
                    if param is not None:
                        expert_weights[name] = param.data.cpu()
                weights[(layer_idx, expert_idx)] = expert_weights
        else:
            for name in profile.expert_layout.weight_names:
                param = getattr(expert_container, name)
                for expert_idx in range(profile.num_experts):
                    key = (layer_idx, expert_idx)
                    if key not in weights:
                        weights[key] = {}
                    weights[key][f"{name}.weight"] = param.data[expert_idx].cpu()

            for name in profile.expert_layout.bias_names:
                param = getattr(expert_container, name, None)
                if param is not None:
                    for expert_idx in range(profile.num_experts):
                        weights[(layer_idx, expert_idx)][name] = param.data[expert_idx].cpu()

    return weights


def install_offloaded_dispatch(
    model: nn.Module,
    profile: ModelProfile,
    pipeline: ExpertPipeline,
):
    """Replace MoE expert forward with offloaded dispatch in-place."""
    layers = model.model.layers

    for layer_idx in range(profile.num_layers):
        if layer_idx < profile.first_moe_layer:
            continue

        layer = layers[layer_idx]
        moe_block = getattr(layer, profile.moe_block_attr)

        dispatch = OffloadedExpertDispatch(
            pipeline=pipeline,
            layer_idx=layer_idx - profile.first_moe_layer,
            num_experts_per_tok=profile.num_experts_per_tok,
        )

        _patch_moe_forward(moe_block, dispatch, profile)


def _patch_moe_forward(
    moe_block: nn.Module,
    dispatch: OffloadedExpertDispatch,
    profile: ModelProfile,
):
    """Monkey-patch the MoE block's forward to use offloaded experts.

    Keeps the router on GPU. Replaces the expert loop with our pipeline.
    """
    def offloaded_forward(hidden_states: torch.Tensor, **_kwargs):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)

        if hasattr(moe_block, "gate"):
            router = moe_block.gate
        elif hasattr(moe_block, "router"):
            router = moe_block.router
        else:
            raise AttributeError("Cannot find router in MoE block")

        if isinstance(router, nn.Linear):
            logits = router(flat)
            top_vals, top_idx = torch.topk(logits, dispatch.num_experts_per_tok, dim=-1)
            routing_weights = torch.softmax(top_vals, dim=-1)
        else:
            routing_result = router(flat)
            if isinstance(routing_result, tuple) and len(routing_result) == 2:
                first, second = routing_result
                if first.shape[-1] == dispatch.num_experts_per_tok:
                    routing_weights, top_idx = first, second
                else:
                    top_idx, routing_weights = first, second
            else:
                raise ValueError(f"Unexpected router output: {type(routing_result)}")

        expert_out = dispatch.dispatch(flat, top_idx, routing_weights.to(flat.dtype))

        if profile.shared_expert_attr is not None:
            shared = getattr(moe_block, profile.shared_expert_attr)
            expert_out = expert_out + shared(flat)

        return expert_out.view(batch_size, seq_len, hidden_dim)

    moe_block.forward = offloaded_forward
