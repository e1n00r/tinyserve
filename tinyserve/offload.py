"""One-call API: offload any HF MoE model's experts to CPU.

Usage:
    from transformers import AutoModelForCausalLM
    from tinyserve.offload import offload_model

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    model = offload_model(model, device="cuda")
    output = model(input_ids)
"""

import logging

import torch

from .model_registry import profile_from_config
from .offloaded_model import OffloadedModel

logger = logging.getLogger(__name__)


def _register_flex_attention() -> str:
    """Register FlexAttention with sink support for GPT-OSS models.

    Appends a virtual sink token to K/V and uses score_mod to inject
    the per-head sink logit at the virtual position. When a StaticKVCache
    with static_shapes=True is used, K/V always have max_seq_len shape
    and the score_mod masks padding positions with -inf, eliminating
    torch.compile recompilation on context length changes.

    Returns the attn_implementation string to use, or 'eager' on failure.
    """
    try:
        import transformers
        from torch.nn.attention.flex_attention import flex_attention

        _compiled_flex = torch.compile(flex_attention)

        def flex_attention_with_sinks(module, query, key, value, attention_mask, scaling, dropout=0.0, **_):
            N, H, L, E = query.shape
            _, G, S, _ = key.shape
            key_expanded = key.repeat_interleave(H // G, dim=1)
            value_expanded = value.repeat_interleave(H // G, dim=1)
            # Append virtual sink token (zero K/V).
            sink_k = torch.zeros(N, H, 1, E, device=key.device, dtype=key.dtype)
            sink_v = torch.zeros(N, H, 1, E, device=value.device, dtype=value.dtype)
            k_ext = torch.cat([key_expanded, sink_k], dim=2)
            v_ext = torch.cat([value_expanded, sink_v], dim=2)
            sinks = module.sinks  # [H]
            max_kv_len = S  # includes padding when static_shapes=True

            # Determine actual valid sequence length for masking.
            # When static_shapes=True, S == max_seq_len but only
            # current_seq_len positions contain real data.
            kv_cache = getattr(module, "_kv_cache_ref", None)
            if kv_cache is not None and kv_cache.static_shapes:
                current_seq_len = kv_cache.get_seq_length(0)

                def score_mod(score, b, h, q_idx, kv_idx):
                    is_valid = kv_idx < current_seq_len
                    is_sink = kv_idx == max_kv_len
                    return torch.where(
                        is_valid, score, torch.where(is_sink, sinks[h], float("-inf"))
                    )
            else:

                def score_mod(score, b, h, q_idx, kv_idx):
                    return torch.where(kv_idx == max_kv_len, sinks[h], score)

            out = _compiled_flex(query, k_ext, v_ext, score_mod=score_mod, scale=scaling)
            return out.transpose(1, 2).contiguous(), None

        transformers.AttentionInterface.register("flex", flex_attention_with_sinks)
        try:
            transformers.AttentionMaskInterface.register("flex", transformers.masking_utils.eager_mask)
        except Exception:
            pass
        gpt_oss_mod = getattr(transformers.models, "gpt_oss", None)
        if gpt_oss_mod:
            gpt_oss_mod.modeling_gpt_oss.GptOssPreTrainedModel._supports_flex = True
        return "flex"
    except Exception:
        return "eager"


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
    max_seq_len: int = 0,
    kv_dtype: torch.dtype = torch.bfloat16,
    gpu_memory_utilization: float = 0.90,
    attn_implementation: str | None = None,
    disk_offload: bool = False,
    ram_cache_gb: float = 0,
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
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "offload_model requires CUDA but no GPU is available. "
            "Install a CUDA-enabled PyTorch build or pass device='cpu'."
        )
    config = model.config
    effective_config = getattr(config, "text_config", config)
    profile = profile_from_config(effective_config)
    model_type = effective_config.model_type

    softmax_order, returns_logits, router_attr = _ROUTING_MAP.get(model_type, ("softmax_then_topk", True, "gate"))

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
        disk_offload=disk_offload,
        ram_cache_gb=ram_cache_gb,
    )

    if hasattr(model, "model"):
        model.model = offloaded.model
    model._offload_pipelines = offloaded.pipelines
    model = model.to(device).to(torch.bfloat16)

    # Allocate KV cache first (if requested), then give remainder to expert cache.
    from .generic_store import GenericLRUCache
    from .static_kv_cache import StaticKVCache

    buf_bytes = store.buffer_expert_bytes
    total_vram = torch.cuda.get_device_properties(device).total_memory
    used_vram = total_vram - torch.cuda.mem_get_info(device)[0]
    # Cap usable VRAM like vLLM's --gpu-memory-utilization
    usable_vram = int(total_vram * gpu_memory_utilization) - used_vram
    free_vram = max(0, usable_vram)
    reserved = 2 * buf_bytes + 128 * 1024 * 1024  # double-buf + 128 MB headroom

    kv_cache = None
    kv_vram = 0
    use_flex = attn_implementation == "flex"
    if max_seq_len > 0:
        kv_cache = StaticKVCache.from_model_config(config, max_seq_len=max_seq_len, device=device, dtype=kv_dtype)
        if use_flex:
            kv_cache.static_shapes = True
        kv_vram = kv_cache.vram_bytes
        model._kv_cache = kv_cache

    if buf_bytes > 0:
        available = max(0, free_vram - reserved - kv_vram)
        max_capacity = available // buf_bytes
        if cache_capacity == 0:
            cache_capacity = max_capacity
        else:
            cache_capacity = min(cache_capacity, max_capacity)

    cache = (
        GenericLRUCache(
            cache_capacity,
            buf_bytes,
            device,
            policy=cache_policy,
            num_layers=store.num_layers,
            num_experts=store.num_experts,
        )
        if cache_capacity > 0
        else None
    )
    if kv_cache is not None:
        logger.info("KV cache: %d tokens (%.2f GB, %s)", max_seq_len, kv_vram / 1e9, kv_dtype)
    logger.info("Expert cache: %d experts (%.2f GB GPU)", cache_capacity, cache_capacity * buf_bytes / 1e9)
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
    attn_implementation: str | None = None,
    max_seq_len: int = 0,
    kv_dtype: torch.dtype = torch.bfloat16,
    gpu_memory_utilization: float = 0.90,
    disk_offload: bool = False,
    ram_cache_gb: float = 0,
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

    if attn_implementation is not None:
        attn_impl = attn_implementation
        if attn_impl == "flex":
            _register_flex_attention()
    else:
        attn_impl = "eager"
        if flash_attention:
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                pass
                # FlexAttention requires static KV cache for efficient compilation.
                # Enable via load_and_offload(attn_implementation='flex') when
                # static KV cache is implemented.

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="cpu",
        **hf_kwargs,
    )
    return offload_model(
        model,
        device=device,
        cache_capacity=cache_capacity,
        model_id=model_id,
        cache_policy=cache_policy,
        fp8=fp8,
        cache_bias=cache_bias,
        adaptive_fate=adaptive_fate,
        max_seq_len=max_seq_len,
        kv_dtype=kv_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        attn_implementation=attn_impl,
        disk_offload=disk_offload,
        ram_cache_gb=ram_cache_gb,
    )
