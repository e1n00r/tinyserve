"""One-call API: offload any HF MoE model's experts to CPU.

Usage:
    from transformers import AutoModelForCausalLM
    from tinyserve.offload import offload_model

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    model = offload_model(model, device="cuda")
    output = model(input_ids)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

import torch

from ._model_hooks import OffloadedModel
from .model_registry import profile_from_config


class AttentionBackend(str, Enum):
    EAGER = "eager"
    SDPA = "sdpa"
    FLEX = "flex"
    FLASHINFER = "flashinfer"
    FLASH_ATTENTION_2 = "flash_attention_2"


class RoutingSpec(NamedTuple):
    softmax_order: str  # "router_native" | "softmax_then_topk"
    returns_logits: bool  # whether router returns raw logits
    router_attr: str  # attribute name on MoE block ("gate" | "router")


logger = logging.getLogger(__name__)

_RESERVED_VRAM_BYTES = 128 * 1024 * 1024  # 128 MB for CUDA kernels + defragmentation
_HEAD_ATTENTION_SEQ_THRESHOLD = 2048
_HEAD_PREFILL_KV_THRESHOLD = 1024


@dataclass
class TinyserveConfig:
    """Configuration for expert offloading.

    Pass to load_and_offload or offload_model instead of individual kwargs.
    All fields have sensible defaults — only override what you need.
    """

    cache_capacity: int = 0
    cache_policy: str = "lfru"
    cache_bias: float = 0.0
    adaptive_fate: bool = True
    max_seq_len: int = 0
    kv_dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16)
    gpu_memory_utilization: float = 0.90
    fp8: bool = True
    disk_offload: bool = False
    ram_cache_gb: float = 0
    kv_offload: bool = False
    buddy_table_path: str | None = None
    imatrix_path: str | None = None
    attn_implementation: str | AttentionBackend | None = None
    streaming: bool = False
    streaming_sink_size: int = 4
    streaming_window_size: int = 1024


class OffloadedLM:
    """Typed wrapper for an offloaded MoE model.

    Replaces monkey-patched attributes (_kv_cache, _vram_budget, _offload_pipelines)
    with typed fields. Forwards all other attribute access to the underlying model.
    """

    def __init__(self, model, pipelines, kv_cache=None, vram_budget=None, offload_config=None):
        self._model = model
        self.pipelines = pipelines
        self.kv_cache = kv_cache
        self.vram_budget = vram_budget
        self.offload_config = offload_config

    # Backward-compat aliases for code that reads the monkey-patched attributes.
    @property
    def _offload_pipelines(self):
        return self.pipelines

    @property
    def _kv_cache(self):
        return self.kv_cache

    @property
    def _vram_budget(self):
        return self.vram_budget

    def generate(self, *args, **kwargs) -> torch.Tensor:
        if self.kv_cache is not None:
            kwargs.setdefault("past_key_values", self.kv_cache)
        return self._model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str):
        if name.startswith("_") or name in ("pipelines", "kv_cache", "vram_budget", "offload_config"):
            raise AttributeError(name)
        return getattr(self._model, name)

    def to(self, *args, **kwargs) -> "OffloadedLM":
        self._model = self._model.to(*args, **kwargs)
        return self


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
        from torch.nn.attention.flex_attention import (
            flex_attention,
        )

        _compiled_flex = torch.compile(flex_attention)

        def flex_attention_with_sinks(
            module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **_
        ):
            N, H, L, E = query.shape
            _, G, S, _ = key.shape

            # Fix 1: Native GQA — pass K/V in original shape, let the kernel
            # handle the GQA expansion instead of materializing 8x larger tensors.
            sink_k = torch.zeros(N, G, 1, E, device=key.device, dtype=key.dtype)
            sink_v = torch.zeros(N, G, 1, E, device=value.device, dtype=value.dtype)
            k_ext = torch.cat([key, sink_k], dim=2)  # (N, G, S+1, E)
            v_ext = torch.cat([value, sink_v], dim=2)  # (N, G, S+1, E)

            sinks = module.sinks  # [H] — per query-head sink logits
            kv_len = S  # position of virtual sink token

            # Fix 2: Sliding window mask — GPT-OSS alternates full/sliding
            # attention per layer; each call gets its own sliding_window value.
            sw = sliding_window

            # Determine actual valid sequence length for masking.
            # When static_shapes=True, S == max_seq_len but only
            # current_seq_len positions contain real data.
            kv_cache = getattr(module, "_kv_cache_ref", None)
            if kv_cache is not None and kv_cache.static_shapes:
                current_seq_len = kv_cache.get_seq_length(0)

                def score_mod(score, b, h, q_idx, kv_idx):
                    is_valid = kv_idx < current_seq_len
                    is_sink = kv_idx == kv_len
                    if sw is not None:
                        in_window = kv_idx >= (q_idx - sw)
                        is_valid = is_valid & in_window
                    return torch.where(is_valid, score, torch.where(is_sink, sinks[h], float("-inf")))

                # Fix 3: Block mask for hardware-level block skipping.
                def mask_mod(b, h, q_idx, kv_idx):
                    is_valid = kv_idx < current_seq_len
                    is_sink = kv_idx == kv_len
                    if sw is not None:
                        in_window = kv_idx >= (q_idx - sw)
                        is_valid = is_valid & in_window
                    return is_valid | is_sink

                # block_mask disabled: create_block_mask triggers torch.compile
                # recompilation per shape, causing VRAM OOM on small GPUs.
                # TODO: cache block_mask per (seq_len, sliding_window) pair.
                block_mask = None
            else:

                def score_mod(score, b, h, q_idx, kv_idx):
                    is_sink = kv_idx == kv_len
                    if sw is not None:
                        in_window = kv_idx >= (q_idx - sw)
                        return torch.where(is_sink, sinks[h], torch.where(in_window, score, float("-inf")))
                    return torch.where(is_sink, sinks[h], score)

                block_mask = None

            out = _compiled_flex(
                query,
                k_ext,
                v_ext,
                score_mod=score_mod,
                block_mask=block_mask,
                scale=scaling,
                enable_gqa=True,
            )
            return out.transpose(1, 2).contiguous(), None

        transformers.AttentionInterface.register("flex", flex_attention_with_sinks)
        try:
            transformers.AttentionMaskInterface.register("flex", transformers.masking_utils.eager_mask)
        except (AttributeError, TypeError):
            logger.warning("FlexAttention mask interface registration failed", exc_info=True)
        gpt_oss_mod = getattr(transformers.models, "gpt_oss", None)
        if gpt_oss_mod:
            gpt_oss_mod.modeling_gpt_oss.GptOssPreTrainedModel._supports_flex = True
        return AttentionBackend.FLEX
    except (ImportError, AttributeError, RuntimeError):
        logger.warning("FlexAttention registration failed, falling back to eager", exc_info=True)
        return AttentionBackend.EAGER


def _register_sdpa_attention() -> str:
    """Register SDPA with sink support. No torch.compile — zero VRAM overhead."""
    try:
        import transformers

        def sdpa_attention_with_sinks(
            module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **_
        ):
            N, H, L, E = query.shape
            _, G, S, _ = key.shape

            if L == 1:
                # Decode: single query position
                k_decode = key
                v_decode = value
                if sliding_window is not None and sliding_window < S:
                    k_decode = key[:, :, -sliding_window:]
                    v_decode = value[:, :, -sliding_window:]
                out = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    k_decode,
                    v_decode,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scaling,
                    enable_gqa=True,
                )
            elif S > _HEAD_ATTENTION_SEQ_THRESHOLD or (L > 256 and S > _HEAD_PREFILL_KV_THRESHOLD):
                # Long KV context: head-wise attention to avoid VRAM OOM.
                # Triggers when KV length exceeds 2048, or when both chunk and
                # KV are moderate. Processes one GQA group at a time — peak VRAM
                # is O(chunk × head_dim) instead of O(chunk × num_heads × head_dim).
                from .head_attention import head_wise_sdpa

                return head_wise_sdpa(
                    query,
                    key,
                    value,
                    scaling,
                    sliding_window=sliding_window,
                    is_causal=True,
                )
            else:
                # Short prefill: standard SDPA is faster
                out = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=scaling,
                    enable_gqa=True,
                )
            return out.transpose(1, 2).contiguous(), None

        transformers.AttentionInterface.register("sdpa", sdpa_attention_with_sinks)
        try:
            transformers.AttentionMaskInterface.register("sdpa", transformers.masking_utils.eager_mask)
        except (AttributeError, TypeError):
            logger.warning("SDPA mask interface registration failed", exc_info=True)
        gpt_oss_mod = getattr(transformers.models, "gpt_oss", None)
        if gpt_oss_mod:
            gpt_oss_mod.modeling_gpt_oss.GptOssPreTrainedModel._supports_sdpa = True
        return AttentionBackend.SDPA
    except (ImportError, AttributeError, RuntimeError):
        logger.warning("SDPA registration failed, falling back to eager", exc_info=True)
        return AttentionBackend.EAGER


def _register_flashinfer_attention() -> str:
    """Register FlashInfer attention backend. Near-optimal GQA decode kernels."""
    try:
        import flashinfer
        import transformers

        def flashinfer_attention_with_sinks(
            module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **_
        ):
            N, H, L, E = query.shape
            _, G, S, _ = key.shape

            if L == 1:
                q_2d = query[0, :, 0, :]
                k_nhd = key[0].permute(1, 0, 2).contiguous()
                v_nhd = value[0].permute(1, 0, 2).contiguous()

                if sliding_window is not None and sliding_window < S:
                    k_nhd = k_nhd[-sliding_window:]
                    v_nhd = v_nhd[-sliding_window:]

                out_2d = flashinfer.decode.single_decode_with_kv_cache(
                    q_2d,
                    k_nhd,
                    v_nhd,
                    kv_layout="NHD",
                    sm_scale=scaling,
                )
                out = out_2d.unsqueeze(0).unsqueeze(0)
            else:
                q_nhd = query[0].permute(1, 0, 2).contiguous()
                k_nhd = key[0].permute(1, 0, 2).contiguous()
                v_nhd = value[0].permute(1, 0, 2).contiguous()

                out_nhd = flashinfer.prefill.single_prefill_with_kv_cache(
                    q_nhd,
                    k_nhd,
                    v_nhd,
                    kv_layout="NHD",
                    causal=True,
                    sm_scale=scaling,
                )
                out = out_nhd.unsqueeze(0)

            return out.contiguous(), None

        transformers.AttentionInterface.register("flashinfer", flashinfer_attention_with_sinks)
        try:
            transformers.AttentionMaskInterface.register("flashinfer", transformers.masking_utils.eager_mask)
        except (AttributeError, TypeError):
            logger.warning("FlashInfer mask interface registration failed", exc_info=True)
        gpt_oss_mod = getattr(transformers.models, "gpt_oss", None)
        if gpt_oss_mod:
            gpt_oss_mod.modeling_gpt_oss.GptOssPreTrainedModel._supports_flashinfer = True
        return AttentionBackend.FLASHINFER
    except (ImportError, AttributeError, RuntimeError):
        logger.warning("FlashInfer registration failed, falling back to eager", exc_info=True)
        return AttentionBackend.EAGER


def _apply_offload_config(offload_config: TinyserveConfig, locals_dict: dict) -> dict:
    """Merge TinyserveConfig fields into a kwargs dict.

    Config fields set all corresponding local variables; attn_implementation
    is only overridden when config provides a non-None value.
    """
    fields = [
        "cache_capacity",
        "cache_policy",
        "cache_bias",
        "adaptive_fate",
        "max_seq_len",
        "kv_dtype",
        "gpu_memory_utilization",
        "fp8",
        "disk_offload",
        "ram_cache_gb",
        "kv_offload",
        "buddy_table_path",
        "imatrix_path",
        "streaming",
        "streaming_sink_size",
        "streaming_window_size",
    ]
    for f in fields:
        locals_dict[f] = getattr(offload_config, f)
    if offload_config.attn_implementation is not None:
        locals_dict["attn_implementation"] = offload_config.attn_implementation
    return locals_dict


_ROUTING_MAP: dict[str, RoutingSpec] = {
    "mixtral": RoutingSpec("router_native", False, "gate"),
    "qwen3_moe": RoutingSpec("router_native", False, "gate"),
    "qwen2_moe": RoutingSpec("router_native", False, "gate"),
    "deepseek_v3": RoutingSpec("router_native", False, "gate"),
    "gpt_oss": RoutingSpec("router_native", True, "router"),
    "olmoe": RoutingSpec("softmax_then_topk", True, "gate"),
    "qwen3_5_moe": RoutingSpec("router_native", False, "gate"),
    "qwen3_5_moe_text": RoutingSpec("router_native", False, "gate"),
    "llama4": RoutingSpec("router_native", False, "gate"),
    "kimi_k2": RoutingSpec("router_native", False, "gate"),
    "dbrx": RoutingSpec("router_native", False, "gate"),
    "phimoe": RoutingSpec("softmax_then_topk", False, "gate"),
}


def offload_model(
    model: torch.nn.Module,
    device: str | torch.device = "cuda",
    offload_config: TinyserveConfig | None = None,
    model_id: str | None = None,
    cache_capacity: int = 0,
    cache_policy: str = "lfru",
    fp8: bool = True,
    cache_bias: float = 0.0,
    adaptive_fate: bool = True,
    max_seq_len: int = 0,
    kv_dtype: torch.dtype = torch.bfloat16,
    gpu_memory_utilization: float = 0.90,
    attn_implementation: str | AttentionBackend | None = None,
    disk_offload: bool = False,
    ram_cache_gb: float = 0,
    kv_offload: bool = False,
    buddy_table_path: str | None = None,
    imatrix_path: str | None = None,
) -> "OffloadedLM":
    """Offload MoE experts from an HF model to CPU with GPU LRU cache.

    Auto-detects model family from config and applies the correct
    routing strategy, expert layout, and shared expert handling.

    Args:
        model: HuggingFace CausalLM model (e.g., MixtralForCausalLM)
        device: GPU device for non-expert weights and cache
        offload_config: TinyserveConfig object. When provided, its fields override
            the individual kwargs. Lets callers pass a single config object instead
            of 15 keyword arguments.
        cache_capacity: number of experts to cache in VRAM (0 = no cache)
        model_id: HuggingFace repo id or local path. When provided for models
            with native quantized expert weights (e.g. MXFP4), expert tensors
            are loaded directly from safetensors, bypassing HF dequantization.
            Non-expert weights remain as loaded in ``model``.
        cache_policy: eviction policy for the expert cache ('lru', 'slru',
            'lfu', 'lfru', 'fifo', 'ls', or 'dali'). Default 'lfru'.
        kv_offload: store KV cache on CPU pinned memory (zero VRAM).
        buddy_table_path: path to JSON buddy table (from calibrate_buddies.py).
            When provided, cache misses try buddy substitution before CPU compute.
            All VRAM goes to expert cache. PCIe transfer (~0.25ms at 1K ctx)
            is hidden behind expert compute (~5ms/layer).

    Returns:
        The same model object with experts offloaded. Call model(input_ids)
        as normal — expert loading is handled transparently.
    """
    if offload_config is not None:
        lv = _apply_offload_config(offload_config, locals())
        cache_capacity = lv["cache_capacity"]
        cache_policy = lv["cache_policy"]
        cache_bias = lv["cache_bias"]
        adaptive_fate = lv["adaptive_fate"]
        max_seq_len = lv["max_seq_len"]
        kv_dtype = lv["kv_dtype"]
        gpu_memory_utilization = lv["gpu_memory_utilization"]
        fp8 = lv["fp8"]
        disk_offload = lv["disk_offload"]
        ram_cache_gb = lv["ram_cache_gb"]
        kv_offload = lv["kv_offload"]
        buddy_table_path = lv["buddy_table_path"]
        imatrix_path = lv["imatrix_path"]
        attn_implementation = lv.get("attn_implementation", attn_implementation)

    if cache_capacity < 0:
        raise ValueError("cache_capacity must be >= 0")
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

    _default_spec = RoutingSpec("softmax_then_topk", True, "gate")
    spec = _ROUTING_MAP.get(model_type, _default_spec)
    if spec is _default_spec:
        logger.warning(
            "Unknown model type %r — using default routing spec %s. Add it to _ROUTING_MAP for optimal performance.",
            model_type,
            _default_spec,
        )
    softmax_order, returns_logits, router_attr = spec

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
    model = model.to(device).to(torch.bfloat16)

    # Allocate KV cache first (if requested), then give remainder to expert cache.
    from .expert_cache import ExpertCache
    from .static_kv_cache import StaticKVCache

    model_config = model.config
    buf_bytes = store.buffer_expert_bytes
    total_vram = torch.cuda.get_device_properties(device).total_memory
    used_vram = total_vram - torch.cuda.mem_get_info(device)[0]
    # Cap usable VRAM like vLLM's --gpu-memory-utilization
    usable_vram = int(total_vram * gpu_memory_utilization) - used_vram
    free_vram = max(0, usable_vram)
    reserved = 2 * buf_bytes + _RESERVED_VRAM_BYTES  # double-buf + 128 MB headroom

    kv_cache = None
    kv_vram = 0
    use_flex = attn_implementation == AttentionBackend.FLEX
    streaming = getattr(offload_config, "streaming", False) if offload_config is not None else False
    streaming_sink_size = getattr(offload_config, "streaming_sink_size", 4) if offload_config is not None else 4
    streaming_window_size = (
        getattr(offload_config, "streaming_window_size", 1024) if offload_config is not None else 1024
    )
    if max_seq_len > 0:
        storage_device = "cpu" if kv_offload else None
        kv_cache = StaticKVCache.from_model_config(
            model_config, max_seq_len=max_seq_len, device=device, dtype=kv_dtype, storage_device=storage_device
        )
        if use_flex:
            kv_cache.static_shapes = True
        if streaming:
            kv_cache.enable_streaming(sink_size=streaming_sink_size, window_size=streaming_window_size)
        kv_vram = kv_cache.vram_bytes

    if buf_bytes > 0:
        available = max(0, free_vram - reserved - kv_vram)
        max_capacity = available // buf_bytes
        if cache_capacity == 0:
            cache_capacity = max_capacity
        else:
            cache_capacity = min(cache_capacity, max_capacity)
        if cache_capacity == 0:
            logger.warning(
                "Expert cache auto-sized to 0 slots — insufficient free VRAM "
                "(free=%.2f GB, reserved=%.2f GB, expert=%.2f MB). "
                "All expert loads will be cache misses.",
                free_vram / 1e9,
                reserved / 1e9,
                buf_bytes / 1e6,
            )

    cache = (
        ExpertCache(
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

    from .vram_budget import VRAMBudget

    vram_budget = None
    if cache is not None and kv_cache is not None:
        kv_bpt = kv_cache.vram_bytes // max(1, kv_cache.max_seq_len)
        vram_budget = VRAMBudget(
            expert_cache=cache,
            kv_cache=kv_cache,
            expert_bytes=buf_bytes,
            kv_bytes_per_token=kv_bpt,
            max_expert_capacity=cache_capacity,
        )
        kv_cache._vram_budget = vram_budget  # enables self-healing on overflow

    # Seed cache from imatrix activation data (eliminates cold-start phase).
    if imatrix_path is not None and cache is not None:
        import os

        from .imatrix import parse_imatrix_dat, rank_experts_from_imatrix, seed_cache_from_ranking

        if not os.path.isfile(imatrix_path):
            raise FileNotFoundError(f"imatrix file not found: {imatrix_path}")
        counts = parse_imatrix_dat(imatrix_path)
        ranking = rank_experts_from_imatrix(counts, store.num_layers, store.num_experts)
        n_seeded = seed_cache_from_ranking(cache, store, ranking)
        logger.info("imatrix seeding: pre-loaded %d experts into cache from %s", n_seeded, imatrix_path)

    # Load buddy tables for miss substitution
    if buddy_table_path is not None:
        import json
        import os

        from .buddy_experts import BuddyTable

        if not os.path.isfile(buddy_table_path):
            raise FileNotFoundError(f"Buddy table not found: {buddy_table_path}")
        with open(buddy_table_path) as f:
            try:
                buddy_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid buddy table JSON: {e}") from e
        for p in offloaded.pipelines:
            p._buddy_tables = {}
            for layer_str, expert_buddies in buddy_data.items():
                buddies = {int(eid): bl for eid, bl in expert_buddies.items()}
                p._buddy_tables[int(layer_str)] = BuddyTable(buddies)
        logger.info("Buddy tables loaded from %s (%d layers)", buddy_table_path, len(buddy_data))

    # Start background eager fill: load all experts from mmap into pinned RAM.
    # Runs concurrently with the first inference requests. Only when disk_offload
    # is active and a RAM cache was created.
    ram_cache = offloaded.pipelines[0].ram_cache if offloaded.pipelines else None
    if disk_offload and ram_cache is not None:
        mmap_data = None if ram_cache._fast_reader is not None else store._data
        ram_cache.start_background_fill(mmap_data, store.num_layers, store.num_experts)
        logger.info(
            "Background fill started: %d experts into RAM cache (%d slots)",
            store.num_layers * store.num_experts,
            ram_cache.num_slots,
        )

    return OffloadedLM(
        model=model,
        pipelines=offloaded.pipelines,
        kv_cache=kv_cache,
        vram_budget=vram_budget,
        offload_config=offload_config,
    )


def load_and_offload(
    model_id: str,
    device: str | torch.device = "cuda",
    offload_config: TinyserveConfig | None = None,
    cache_capacity: int = 0,
    cache_policy: str = "lfru",
    cache_bias: float = 0.0,
    flash_attention: bool = True,
    torch_dtype=torch.bfloat16,
    fp8: bool = True,
    adaptive_fate: bool = True,
    attn_implementation: str | AttentionBackend | None = None,
    max_seq_len: int = 0,
    kv_dtype: torch.dtype = torch.bfloat16,
    gpu_memory_utilization: float = 0.90,
    disk_offload: bool = False,
    ram_cache_gb: float = 0,
    kv_offload: bool = False,
    buddy_table_path: str | None = None,
    imatrix_path: str | None = None,
    **hf_kwargs,
) -> "OffloadedLM":
    """Load a HuggingFace MoE model and immediately offload its experts.

    Args:
        model_id: HuggingFace repo id or local path
        device: GPU device
        offload_config: TinyserveConfig object. When provided, its fields override
            the individual kwargs.
        cache_capacity: expert slots in VRAM (0 = auto)
        cache_policy: 'lru', 'slru', 'lfu', 'lfru', 'fifo', or 'ls'
        buddy_table_path: path to JSON buddy table for miss substitution
        **hf_kwargs: passed through to AutoModelForCausalLM.from_pretrained
    """
    if offload_config is not None:
        lv = _apply_offload_config(offload_config, locals())
        cache_capacity = lv["cache_capacity"]
        cache_policy = lv["cache_policy"]
        cache_bias = lv["cache_bias"]
        adaptive_fate = lv["adaptive_fate"]
        max_seq_len = lv["max_seq_len"]
        kv_dtype = lv["kv_dtype"]
        gpu_memory_utilization = lv["gpu_memory_utilization"]
        fp8 = lv["fp8"]
        disk_offload = lv["disk_offload"]
        ram_cache_gb = lv["ram_cache_gb"]
        kv_offload = lv["kv_offload"]
        buddy_table_path = lv["buddy_table_path"]
        imatrix_path = lv["imatrix_path"]
        attn_implementation = lv.get("attn_implementation", attn_implementation)

    if cache_capacity < 0:
        raise ValueError("cache_capacity must be >= 0")
    from transformers import AutoModelForCausalLM

    if attn_implementation is not None:
        attn_impl = attn_implementation
        if attn_impl == AttentionBackend.FLEX:
            _register_flex_attention()
        elif attn_impl == AttentionBackend.SDPA:
            _register_sdpa_attention()
        elif attn_impl == AttentionBackend.FLASHINFER:
            attn_impl = _register_flashinfer_attention()
    else:
        attn_impl = AttentionBackend.EAGER
        if flash_attention:
            try:
                import flash_attn  # noqa: F401

                attn_impl = AttentionBackend.FLASH_ATTENTION_2
            except ImportError:
                logger.debug("flash_attn not available, using SDPA")
                attn_impl = _register_sdpa_attention()

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
        kv_offload=kv_offload,
        buddy_table_path=buddy_table_path,
        imatrix_path=imatrix_path,
    )
