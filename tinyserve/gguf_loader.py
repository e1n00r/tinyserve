"""Load a complete MoE model from GGUF files.

Parses GGUF metadata to extract model config, maps GGUF tensor names to
HuggingFace parameter names, dequantizes non-expert weights to BF16 on GPU,
and routes expert weights through GGUFExpertStore for INT4 CPU compute.

Supports multi-shard GGUF files (e.g. Qwen3.5-122B-A10B split across 3 shards).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .gguf_dequant import (
    _dequant_fused_tensor,  # noqa: F401 (re-exported for tests)
    _dequant_tensor,
)
from .gguf_reader import GGUFReader, GGUFTensorInfo
from .gguf_weights import (
    _build_expert_store_from_fused_reader,  # noqa: F401 (re-exported for tests)
    _find_tensor_info,
    _get_param,
    _set_param,
)

logger = logging.getLogger(__name__)


@dataclass
class GGUFModelConfig:
    """Minimal model config extracted from GGUF metadata."""

    arch: str = ""
    num_hidden_layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    vocab_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    shared_expert_intermediate_size: int = 0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    context_length: int = 4096
    extra: dict = field(default_factory=dict)

    @property
    def num_local_experts(self) -> int:
        """Alias for num_experts (compatibility with HF config naming)."""
        return self.num_experts


# GGUF metadata key -> GGUFModelConfig field. The arch prefix (e.g.
# "qwen3moe.") is stripped before matching.
_META_KEY_MAP: dict[str, str] = {
    "block_count": "num_hidden_layers",
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "context_length": "context_length",
    "expert_count": "num_experts",
    "expert_used_count": "num_experts_per_tok",
    "rope.freq_base": "rope_theta",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
}


def config_from_metadata(metadata: dict) -> GGUFModelConfig:
    """Extract a GGUFModelConfig from GGUF key-value metadata.

    Handles the ``<arch>.<key>`` prefix convention used by llama.cpp.
    """
    cfg = GGUFModelConfig()

    arch = metadata.get("general.architecture", "")
    cfg.arch = arch

    # Vocab size from tokenizer metadata
    if "tokenizer.ggml.tokens" in metadata:
        tokens = metadata["tokenizer.ggml.tokens"]
        if isinstance(tokens, list):
            cfg.vocab_size = len(tokens)

    prefix = f"{arch}." if arch else ""

    for meta_suffix, config_field in _META_KEY_MAP.items():
        key = f"{prefix}{meta_suffix}"
        if key in metadata:
            val = metadata[key]
            current = getattr(cfg, config_field)
            if isinstance(current, float):
                setattr(cfg, config_field, float(val))
            elif isinstance(current, int):
                setattr(cfg, config_field, int(val))
            else:
                setattr(cfg, config_field, val)

    # Shared expert intermediate size (Qwen3.5 MoE specific)
    shared_key = f"{prefix}expert_shared_feed_forward_length"
    if shared_key in metadata:
        cfg.shared_expert_intermediate_size = int(metadata[shared_key])

    # Collect remaining arch-prefixed keys into extra
    for key, val in metadata.items():
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if suffix not in _META_KEY_MAP and suffix != "expert_shared_feed_forward_length":
                cfg.extra[suffix] = val

    return cfg


# ---------------------------------------------------------------------------
# GGUF <-> HuggingFace tensor name mapping
# ---------------------------------------------------------------------------

# Pattern: blk.<L>.ffn_<proj>.<E>.weight  (expert)
_EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)\.(\d+)\.weight$")

# Pattern: blk.<L>.ffn_gate_exps.weight or blk.<L>.ffn_up_exps.weight etc.
# (shared expert, Qwen3.5 style: fused gate_up_proj for shared expert)
_SHARED_EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")

# Mapping for non-expert tensors: GGUF name fragment -> HF name fragment.
# These are applied after stripping ``blk.<L>.`` prefix.
_NON_EXPERT_MAP: dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

_LAYER_MAP: dict[str, str] = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    # Shared expert projections (separate gate/up/down)
    "ffn_gate_exps.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_exps.weight": "mlp.shared_expert.down_proj.weight",
    # Qwen3.5 shared expert with fused gate_up
    "ffn_gate_up_exps.weight": "mlp.shared_expert.gate_up_proj.weight",
    # Attention biases (some architectures)
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.bias": "self_attn.o_proj.bias",
}


def gguf_to_hf_name(gguf_name: str) -> tuple[str, bool, int | None, int | None]:
    """Map a GGUF tensor name to a HuggingFace parameter name.

    Returns:
        (hf_name, is_expert, layer_idx, expert_idx)

        ``is_expert`` is True for per-expert FFN weights (gate/up/down).
        ``layer_idx`` and ``expert_idx`` are populated when applicable.
    """
    # Expert tensor
    m = _EXPERT_RE.match(gguf_name)
    if m:
        layer = int(m.group(1))
        proj = m.group(2)
        expert = int(m.group(3))
        hf_proj = f"{proj}_proj"
        hf_name = f"model.layers.{layer}.mlp.experts.{expert}.{hf_proj}.weight"
        return hf_name, True, layer, expert

    # Global (non-layer) tensors
    if gguf_name in _NON_EXPERT_MAP:
        return _NON_EXPERT_MAP[gguf_name], False, None, None

    # Layer-local tensors: blk.<L>.<suffix>
    blk_match = re.match(r"^blk\.(\d+)\.(.+)$", gguf_name)
    if blk_match:
        layer = int(blk_match.group(1))
        suffix = blk_match.group(2)
        if suffix in _LAYER_MAP:
            hf_suffix = _LAYER_MAP[suffix]
            return f"model.layers.{layer}.{hf_suffix}", False, layer, None

    return gguf_name, False, None, None


def hf_to_gguf_name(hf_name: str) -> str:
    """Map a HuggingFace parameter name to a GGUF tensor name (inverse)."""
    # Reverse global map
    for gguf_n, hf_n in _NON_EXPERT_MAP.items():
        if hf_n == hf_name:
            return gguf_n

    # Expert pattern
    m = re.match(
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$",
        hf_name,
    )
    if m:
        layer, expert, proj = m.group(1), m.group(2), m.group(3)
        return f"blk.{layer}.ffn_{proj}.{expert}.weight"

    # Layer-local
    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", hf_name)
    if m:
        layer = m.group(1)
        hf_suffix = m.group(2)
        for gguf_suffix, hf_s in _LAYER_MAP.items():
            if hf_s == hf_suffix:
                return f"blk.{layer}.{gguf_suffix}"

    return hf_name


# ---------------------------------------------------------------------------
# Multi-shard GGUF reader
# ---------------------------------------------------------------------------


@dataclass
class _ShardedTensorInfo:
    """Tensor info with a reference to which shard (reader) contains it."""

    info: GGUFTensorInfo
    shard_idx: int


class MultiShardGGUFReader:
    """Merges tensor infos and metadata across multiple GGUF shard files.

    Shard naming convention: ``model-00001-of-00003.gguf``, etc.
    """

    def __init__(self, paths: list[str | Path]):
        self._readers: list[GGUFReader] = []
        self._merged_metadata: dict = {}
        self._tensor_map: dict[str, _ShardedTensorInfo] = {}

        for idx, p in enumerate(sorted(paths)):
            reader = GGUFReader(p)
            self._readers.append(reader)
            # Merge metadata (first shard wins on conflicts)
            for k, v in reader.metadata.items():
                if k not in self._merged_metadata:
                    self._merged_metadata[k] = v
            # Register tensors
            for t in reader.tensors:
                self._tensor_map[t.name] = _ShardedTensorInfo(info=t, shard_idx=idx)

        logger.info(
            "MultiShardGGUFReader: %d shards, %d tensors, %d metadata keys",
            len(self._readers),
            len(self._tensor_map),
            len(self._merged_metadata),
        )

    @property
    def metadata(self) -> dict:
        return self._merged_metadata

    @property
    def tensor_names(self) -> list[str]:
        return list(self._tensor_map.keys())

    def get_tensor_info(self, name: str) -> GGUFTensorInfo:
        return self._tensor_map[name].info

    def get_tensor_data(self, name: str) -> bytes:
        entry = self._tensor_map[name]
        reader = self._readers[entry.shard_idx]
        return reader.get_tensor_data(entry.info)

    def get_tensor_data_by_offset(self, offset: int, nbytes: int, shard_idx: int = -1) -> bytes:
        """Read raw bytes from a specific offset (for slicing fused tensors).

        If shard_idx is -1, searches all shards for one whose data range covers the offset.
        """
        if shard_idx >= 0:
            reader = self._readers[shard_idx]
            reader._file.seek(reader._data_offset + offset)
            return reader._file.read(nbytes)
        # Search: find which shard contains this offset
        for reader in self._readers:
            try:
                reader._file.seek(reader._data_offset + offset)
                data = reader._file.read(nbytes)
                if len(data) == nbytes:
                    return data
            except (OSError, ValueError):
                continue
        raise ValueError(f"Offset {offset} not found in any shard")

    def list_expert_tensors(self) -> dict[tuple[int, int], dict[str, GGUFTensorInfo]]:
        """Group per-expert tensors by (layer, expert_idx) across all shards.

        Matches ``blk.<L>.ffn_<proj>.<E>.weight``. For fused expert tensors
        (``blk.<L>.ffn_<proj>_exps.weight``), use ``list_fused_expert_tensors()``.
        """
        groups: dict[tuple[int, int], dict[str, GGUFTensorInfo]] = {}
        for name, entry in self._tensor_map.items():
            m = _EXPERT_RE.match(name)
            if m:
                layer, proj, expert = int(m.group(1)), m.group(2), int(m.group(3))
                key = (layer, expert)
                if key not in groups:
                    groups[key] = {}
                groups[key][proj] = entry.info
        return groups

    def list_fused_expert_tensors(self) -> dict[int, dict[str, GGUFTensorInfo]]:
        """Group fused expert tensors by layer index across all shards.

        Matches ``blk.<L>.ffn_gate_exps.weight``, ``blk.<L>.ffn_up_exps.weight``,
        ``blk.<L>.ffn_down_exps.weight``. The tensors have shape
        ``(out_dim, in_dim, n_experts)`` with experts in the last dimension.

        Returns:
            Dict mapping layer index to ``{"gate": info, "up": info, "down": info}``.
        """
        fused_re = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
        groups: dict[int, dict[str, GGUFTensorInfo]] = {}
        for name, entry in self._tensor_map.items():
            m = fused_re.match(name)
            if m:
                layer, proj = int(m.group(1)), m.group(2)
                if layer not in groups:
                    groups[layer] = {}
                groups[layer][proj] = entry.info
        return groups

    def close(self):
        for r in self._readers:
            r.close()


def open_gguf(path: str | Path) -> MultiShardGGUFReader | GGUFReader:
    """Open a single GGUF file or discover and open all shards.

    If ``path`` matches the shard naming pattern (``*-00001-of-*.gguf``),
    all sibling shards are discovered and a MultiShardGGUFReader is returned.
    Otherwise a single GGUFReader is returned.
    """
    p = Path(path)

    # Handle directory input: discover GGUF files inside
    if p.is_dir():
        candidates = sorted(p.glob("*-*-of-*.gguf"))
        if len(candidates) > 1:
            return MultiShardGGUFReader(candidates)
        candidates = sorted(p.glob("*.gguf"))
        if len(candidates) == 1:
            return GGUFReader(candidates[0])
        if candidates:
            return MultiShardGGUFReader(candidates)
        raise FileNotFoundError(f"No GGUF files found in {p}")

    shard_pattern = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$")
    m = shard_pattern.match(p.name)
    if m:
        prefix = m.group(1)
        n_shards = int(m.group(3))
        shard_paths = []
        for i in range(1, n_shards + 1):
            shard_name = f"{prefix}-{i:05d}-of-{n_shards:05d}.gguf"
            shard_path = p.parent / shard_name
            if shard_path.exists():
                shard_paths.append(shard_path)
        if len(shard_paths) > 1:
            return MultiShardGGUFReader(shard_paths)

    # Also check if there are multiple shards with a glob pattern
    if p.suffix == ".gguf":
        stem = p.stem
        # Try to find sibling shards
        candidates = sorted(p.parent.glob(f"{stem.rsplit('-', 2)[0]}-*-of-*.gguf"))
        if len(candidates) > 1:
            return MultiShardGGUFReader(candidates)

    return GGUFReader(p)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_from_gguf(
    gguf_path: str,
    model_id: str | None = None,
    device: str = "cuda",
    cache_capacity: int = 0,
    **kwargs,
) -> torch.nn.Module:
    """Load a complete MoE model from GGUF file(s).

    1. Parse GGUF metadata and extract model config
    2. Create HF model skeleton with ``init_empty_weights``
    3. Map GGUF tensor names to HF parameter names
    4. Non-expert weights: dequant to BF16, load onto GPU
    5. Expert weights: create MmapExpertStore (per-expert) or FusedMmapExpertStore
       via ``from_fused_gguf`` (Qwen-style fused). Old BF16 dequant paths in
       ``gguf_weights.py`` are kept as ``--force-bf16-dequant`` fallback.
    6. Apply expert offloading via ``offload_model``

    Args:
        gguf_path: path to GGUF file (single or first shard)
        model_id: HuggingFace model ID for tokenizer and config fallback.
            If None, config is extracted purely from GGUF metadata.
        device: target device for non-expert weights
        cache_capacity: number of expert cache slots in VRAM (0 = auto)
        **kwargs: passed through to ``offload_model``
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    reader = open_gguf(gguf_path)
    is_multi = isinstance(reader, MultiShardGGUFReader)
    metadata = reader.metadata

    # Step 1: Extract config from GGUF metadata
    gguf_cfg = config_from_metadata(metadata)
    logger.info(
        "GGUF config: arch=%s, layers=%d, hidden=%d, experts=%d, top_k=%d",
        gguf_cfg.arch,
        gguf_cfg.num_hidden_layers,
        gguf_cfg.hidden_size,
        gguf_cfg.num_experts,
        gguf_cfg.num_experts_per_tok,
    )

    # Step 2: Create HF model skeleton
    if model_id is not None:
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    else:
        raise ValueError(
            "model_id is required for now. Pure GGUF config extraction is not yet "
            "implemented — the HF config is needed to create the model skeleton."
        )

    # Handle multimodal wrappers (e.g., Qwen 3.5 MoE)
    # Extract text_config if present — the text model is what we offload
    effective_config = getattr(hf_config, "text_config", hf_config)
    if effective_config is not hf_config:
        logger.info("Multimodal model detected, using text_config for model creation")
        hf_config = effective_config

    # Try specific model class for known multimodal architectures
    model_type = getattr(hf_config, "model_type", "")
    with init_empty_weights():
        if "qwen3_5_moe" in model_type:
            from transformers import Qwen3_5MoeForCausalLM

            model = Qwen3_5MoeForCausalLM(hf_config).to(dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)

    # Step 3: Classify tensors into expert vs non-expert
    if is_multi:
        all_names = reader.tensor_names
    else:
        all_names = [t.name for t in reader.tensors]

    # Collect fused expert tensor names upfront so they are excluded from the
    # non-expert loading pass (they would fail shape validation against the
    # shared_expert param they incorrectly map to via gguf_to_hf_name).
    _fused_expert_re = re.compile(r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$")

    def _is_fused_expert(n: str) -> bool:
        if not _fused_expert_re.match(n):
            return False
        # Only 3-D tensors are fused-expert tensors; 2-D ones are shared experts.
        if is_multi:
            info = reader.get_tensor_info(n)
        else:
            info = _find_tensor_info(reader, n)
        return len(info.shape) == 3

    fused_expert_names: set[str] = {n for n in all_names if _is_fused_expert(n)}

    expert_names: list[str] = []
    non_expert_names: list[str] = []

    for name in all_names:
        if name in fused_expert_names:
            continue  # handled separately via from_fused_gguf
        _, is_expert, _, _ = gguf_to_hf_name(name)
        if is_expert:
            expert_names.append(name)
        else:
            non_expert_names.append(name)

    logger.info(
        "Tensors: %d total, %d expert, %d non-expert",
        len(all_names),
        len(expert_names),
        len(non_expert_names),
    )

    # Step 4: Load non-expert weights
    #
    # Use architecture-specific mapper if available (handles fused QKV, norm offsets,
    # V-head reorders, etc.). Falls back to generic gguf_to_hf_name mapper.
    #
    # Linear layers: store raw Q4_K/Q5_K on GPU as GGMLLinear (ggml kernel).
    # Non-linear (embeddings, norms, biases): dequant to BF16 (must be float).
    from .ggml_linear import replace_linear_with_ggml

    # Select mapper based on architecture
    arch = reader.metadata.get("general.architecture", "")
    if arch == "qwen35moe":
        from .qwen35moe_mapper import map_gguf_to_hf as _map_name
        logger.info("Using Qwen3.5MoE-specific GGUF mapper")
    else:
        def _map_name(gguf_name):
            hf, is_expert, _, _ = gguf_to_hf_name(gguf_name)
            return hf, False, is_expert, None

    # Extract linear_attn head config for V-head reorder transforms (Qwen3.5MoE).
    # These come from the HF config; defaults are safe for non-Qwen3.5 models
    # (vhead_mode will always be None in that case).
    _linear_num_k_heads: int = getattr(hf_config, "linear_num_key_heads", 0)
    _linear_num_v_heads: int = getattr(hf_config, "linear_num_value_heads", 0)
    _linear_k_head_dim: int = getattr(hf_config, "linear_key_head_dim", 0)
    _linear_v_head_dim: int = getattr(hf_config, "linear_value_head_dim", 0)
    _num_attn_heads: int = getattr(hf_config, "num_attention_heads", 0)
    _hidden_size: int = getattr(hf_config, "hidden_size", 0)
    _q_head_dim: int = (
        getattr(hf_config, "head_dim", 0)
        or (_hidden_size // _num_attn_heads if _num_attn_heads else 0)
    )

    if arch == "qwen35moe" and _linear_num_k_heads:
        from .qwen35moe_mapper import apply_vhead_transform as _apply_vhead
        logger.info(
            "V-head reorder config: num_k=%d num_v=%d k_head_dim=%d v_head_dim=%d q_head_dim=%d",
            _linear_num_k_heads, _linear_num_v_heads,
            _linear_k_head_dim, _linear_v_head_dim, _q_head_dim,
        )
    else:
        _apply_vhead = None  # type: ignore[assignment]

    linear_weights: dict[str, tuple[bytes, int, tuple[int, int]]] = {}
    loaded = 0
    skipped = 0

    for gguf_name in non_expert_names:
        hf_name, needs_norm_offset, is_fused_expert, vhead_mode = _map_name(gguf_name)

        if is_fused_expert or hf_name is None:
            skipped += 1
            continue

        param = _get_param(model, hf_name)
        if param is None:
            logger.debug("Skipping unmapped: %s -> %s (no param)", gguf_name, hf_name)
            skipped += 1
            continue

        if is_multi:
            info = reader.get_tensor_info(gguf_name)
            raw = reader.get_tensor_data(gguf_name)
        else:
            info = _find_tensor_info(reader, gguf_name)
            raw = reader.get_tensor_data(info)

        # Tensors that need V-head reorder MUST be dequantized to BF16 first —
        # quantized block layouts (Q4_K etc.) cannot have rows permuted in-place.
        # They are stored as plain nn.Parameter (BF16) rather than GGMLLinear.
        needs_dequant_for_vhead = vhead_mode is not None and _apply_vhead is not None

        # Linear layers with 2D weight and quantized type: keep native quant
        is_linear_weight = (
            len(info.shape) == 2
            and info.ggml_type in (2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15)
            and hf_name.endswith(".weight")
            and not needs_norm_offset  # norms must be float (need +1 offset)
            and not needs_dequant_for_vhead  # vhead reorder requires float
            and "embed" not in hf_name
        )

        if is_linear_weight:
            module_path = hf_name.rsplit(".weight", 1)[0]
            # GGML stores ne[0]=in_features, ne[1]=out_features (column-major convention)
            linear_weights[module_path] = (raw, info.ggml_type, (info.shape[1], info.shape[0]))
            loaded += 1
        else:
            # Non-linear or vhead-reorder: dequant to BF16
            try:
                from .gguf_dequant_torch import dequant_tensor as _fast_dequant
                tensor = _fast_dequant(raw, info.ggml_type, tuple(info.shape)).to(torch.bfloat16)
            except (ImportError, ValueError):
                tensor = _dequant_tensor(reader, info, gguf_name, device).cpu().to(torch.bfloat16)

            # Apply norm +1 offset (GGUF stores w-1, HF expects w)
            if needs_norm_offset:
                tensor = tensor + 1.0

            # Apply V-head inverse reorder (tiled → grouped) or A_log inverse transform
            if needs_dequant_for_vhead:
                tensor = _apply_vhead(
                    tensor,
                    vhead_mode,
                    num_k_heads=_linear_num_k_heads,
                    num_v_heads=_linear_num_v_heads,
                    num_q_heads=_num_attn_heads,
                    k_head_dim=_linear_k_head_dim,
                    v_head_dim=_linear_v_head_dim,
                    q_head_dim=_q_head_dim,
                )

            tensor = tensor.to(device)

            if tensor.shape != param.shape:
                # GGUF often stores 2D weights transposed relative to HF
                if tensor.dim() == 2 and param.dim() == 2 and tensor.shape == param.shape[::-1]:
                    tensor = tensor.T.contiguous()
                elif tensor.numel() == param.numel():
                    tensor = tensor.reshape(param.shape)
                else:
                    logger.warning("Shape mismatch for %s: GGUF %s vs model %s", hf_name, tensor.shape, param.shape)
                    skipped += 1
                    continue

            _set_param(model, hf_name, tensor)
            loaded += 1

    n_replaced = replace_linear_with_ggml(model, linear_weights, device)
    logger.info("Loaded %d weights (%d native quant, %d dequant), skipped %d",
                loaded, n_replaced, loaded - n_replaced, skipped)

    # Step 5: Expert weights -> MmapExpertStore (zero-copy, native quant)
    from .mmap_store import MmapExpertStore

    if fused_expert_names:
        # Fused format (Qwen-style): blk.<L>.ffn_{gate,up,down}_exps.weight
        # Zero-copy: expert dim is outermost in ggml, each expert is contiguous bytes.
        reader.close()
        expert_store = MmapExpertStore.from_fused(gguf_path)
    elif expert_names:
        # Per-expert format: blk.<L>.ffn_{gate,up,down}.<E>.weight
        reader.close()
        expert_store = MmapExpertStore(gguf_path)
        logger.info(
            "Expert store (per-expert mmap): %d layers, %d experts, %.2f MB/expert",
            expert_store.num_layers,
            expert_store.num_experts,
            expert_store.expert_bytes / 1e6,
        )
    else:
        reader.close()
        raise ValueError(f"No expert tensors found in {gguf_path}")

    # Step 6: Apply offloading with the pre-built mmap store
    from .offload import offload_model

    model = offload_model(
        model,
        device=device,
        cache_capacity=cache_capacity,
        model_id=model_id,
        expert_store=expert_store,
        **kwargs,
    )

    return model
