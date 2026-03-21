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
    "mixtral": ("router_native", False, "gate"),
    "qwen3_moe": ("router_native", False, "gate"),
    "qwen2_moe": ("router_native", False, "gate"),
    "deepseek_v3": ("router_native", False, "gate"),
    "gpt_oss": ("router_native", True, "router"),
    "olmoe": ("softmax_then_topk", True, "gate"),
    "qwen3_5_moe": ("router_native", False, "gate"),
    "qwen3_5_moe_text": ("router_native", False, "gate"),
}


def offload_model(
    model: torch.nn.Module,
    device: str | torch.device = "cuda",
    cache_capacity: int = 0,
    model_id: str | None = None,
    cache_policy: str = "lfru",
    fp8: bool = True,
    cache_bias: float = 0.0,
    adaptive_fate: bool = True,
) -> torch.nn.Module:
    """Offload MoE experts from an HF model to CPU with GPU LRU cache.

    Auto-detects model family from config and applies the correct
    routing strategy, expert layout, and shared expert handling.

    Args:
        model: HuggingFace CausalLM model (e.g., MixtralForCausalLM)
        device: GPU device for non-expert weights and cache
        cache_capacity: number of experts to cache in VRAM (0 = no cache)
        model_id: HuggingFace repo id or local path. When provided for models
            with native quantized expert weights (e.g. MXFP4), expert tensors
            are loaded directly from safetensors, bypassing HF dequantization.
            Non-expert weights remain as loaded in ``model``.
        cache_policy: eviction policy for the expert cache ('lru', 'slru',
            'lfu', 'lfru', 'fifo', 'ls', or 'dali'). Default 'lfru'.

    Returns:
        The same model object with experts offloaded. Call model(input_ids)
        as normal — expert loading is handled transparently.
    """
    device = torch.device(device)
    config = model.config
    effective_config = getattr(config, "text_config", config)
    profile = profile_from_config(effective_config)
    model_type = effective_config.model_type

    softmax_order, returns_logits, router_attr = _ROUTING_MAP.get(
        model_type, ("softmax_then_topk", True, "gate")
    )

    inner_model = model.model if hasattr(model, "model") else model

    offloaded, store, cache_capacity, cache_policy = OffloadedModel.from_module(
        inner_model,
        moe_block_attr=profile.moe_block_attr,
        expert_list_attr=profile.expert_list_attr,
        router_attr=router_attr,
        top_k=profile.num_experts_per_tok,
        device=device,
        cache_capacity=cache_capacity,
        returns_router_logits=returns_logits,
        softmax_order=softmax_order,
        first_moe_layer=profile.first_moe_layer,
        model_id=model_id,
        cache_policy=cache_policy,
        fp8=fp8,
        adaptive_fate=adaptive_fate,
    )

    if hasattr(model, "model"):
        model.model = offloaded.model
    model._offload_pipelines = offloaded.pipelines
    model = model.to(device).to(torch.bfloat16)

    # Auto-size or cap cache now that all non-expert weights are on GPU.
    from .generic_store import GenericLRUCache
    buf_bytes = store.buffer_expert_bytes  # always BF16 size for GPU
    if buf_bytes > 0:
        free_vram = torch.cuda.mem_get_info(device)[0]
        reserved = 2 * buf_bytes + 256 * 1024 * 1024  # double-buf + 256 MB headroom
        max_capacity = max(0, free_vram - reserved) // buf_bytes
        if cache_capacity == 0:
            cache_capacity = max_capacity
        else:
            cache_capacity = min(cache_capacity, max_capacity)

    cache = GenericLRUCache(
        cache_capacity, buf_bytes, device, policy=cache_policy,
        num_layers=store.num_layers, num_experts=store.num_experts,
    ) if cache_capacity > 0 else None
    print(f"  Cache capacity: {cache_capacity} experts ({cache_capacity * buf_bytes / 1e9:.2f} GB GPU)")
    for p in offloaded.pipelines:
        p.cache = cache
        p.cache_bias = cache_bias

    return model


def load_and_offload(
    model_id: str,
    device: str | torch.device = "cuda",
    cache_capacity: int = 0,
    cache_policy: str = "lfru",
    cache_bias: float = 0.0,
    flash_attention: bool = True,
    torch_dtype=torch.bfloat16,
    fp8: bool = True,
    adaptive_fate: bool = True,
    **hf_kwargs,
) -> torch.nn.Module:
    """Load a HuggingFace MoE model and immediately offload its experts.

    Args:
        model_id: HuggingFace repo id or local path
        device: GPU device
        cache_capacity: expert slots in VRAM (0 = auto)
        cache_policy: 'lru', 'slru', 'lfu', 'lfru', 'fifo', or 'ls' (least-stale, default)
        flash_attention: use flash_attention_2 if available (default True)
        torch_dtype: weight dtype (default bfloat16)
        **hf_kwargs: passed through to AutoModelForCausalLM.from_pretrained
    """
    from transformers import AutoModelForCausalLM

    attn_impl = "eager"
    if flash_attention:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="cpu",
        **hf_kwargs,
    )
    return offload_model(model, device=device, cache_capacity=cache_capacity,
                         model_id=model_id, cache_policy=cache_policy, fp8=fp8,
                         cache_bias=cache_bias, adaptive_fate=adaptive_fate)
