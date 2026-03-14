"""One-call API: offload any HF MoE model's experts to CPU.

Usage:
    from transformers import AutoModelForCausalLM
    from src.offload import offload_model

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    model = offload_model(model, device="cuda")
    output = model(input_ids)
"""

import torch

from .model_registry import profile_from_config
from .offloaded_model import OffloadedModel

_ROUTING_MAP = {
    "mixtral": ("softmax_then_topk", True),
    "qwen3_moe": ("softmax_then_topk", True),
    "qwen2_moe": ("softmax_then_topk", True),
    "deepseek_v3": ("router_native", False),
    "gpt_oss": ("topk_then_softmax", False),
}


def offload_model(
    model: torch.nn.Module,
    device: str | torch.device = "cuda",
    cache_capacity: int = 0,
) -> torch.nn.Module:
    """Offload MoE experts from an HF model to CPU with GPU LRU cache.

    Auto-detects model family from config and applies the correct
    routing strategy, expert layout, and shared expert handling.

    Args:
        model: HuggingFace CausalLM model (e.g., MixtralForCausalLM)
        device: GPU device for non-expert weights and cache
        cache_capacity: number of experts to cache in VRAM (0 = no cache)

    Returns:
        The same model object with experts offloaded. Call model(input_ids)
        as normal — expert loading is handled transparently.
    """
    device = torch.device(device)
    config = model.config
    profile = profile_from_config(config)
    model_type = config.model_type

    softmax_order, returns_logits = _ROUTING_MAP.get(
        model_type, ("softmax_then_topk", True)
    )

    inner_model = model.model if hasattr(model, "model") else model

    offloaded = OffloadedModel.from_module(
        inner_model,
        moe_block_attr=profile.moe_block_attr,
        expert_list_attr=profile.expert_list_attr,
        router_attr="gate",
        top_k=profile.num_experts_per_tok,
        device=device,
        cache_capacity=cache_capacity,
        returns_router_logits=returns_logits,
        softmax_order=softmax_order,
        first_moe_layer=profile.first_moe_layer,
    )

    if hasattr(model, "model"):
        model.model = offloaded.model
    model._offload_pipelines = offloaded.pipelines
    model = model.to(device)
    return model
