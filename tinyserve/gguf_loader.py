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

from .gguf_reader import GGML_TYPES, GGUFReader, GGUFTensorInfo

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
            suffix = key[len(prefix):]
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
    "ffn_down_exps.weight": "mlp.shared_expert.down_proj.weight",
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
# Weight dequantization
# ---------------------------------------------------------------------------

def _dequant_tensor(
    reader: GGUFReader | MultiShardGGUFReader,
    info: GGUFTensorInfo,
    name: str,
    device: str | torch.device,
) -> torch.Tensor:
    """Dequantize a GGUF tensor to BF16 on the target device.

    Supports F32, F16, Q4_K, Q8_0, and other common GGUF types.
    """
    import numpy as np

    if isinstance(reader, MultiShardGGUFReader):
        raw = reader.get_tensor_data(name)
    else:
        raw = reader.get_tensor_data(info)

    ggml_type = info.ggml_type

    if ggml_type == 0:  # F32
        t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(info.shape)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 1:  # F16
        t = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(info.shape)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 12:  # Q4_K
        from .gguf_quant import parse_q4k_blocks

        w_float = parse_q4k_blocks(raw, (info.shape[0], info.shape[1]))
        t = torch.from_numpy(w_float)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 8:  # Q8_0
        # Q8_0: 34 bytes per block of 32 elements
        # Layout: float16 scale (2 bytes) + int8[32] (32 bytes) = 34 bytes
        n_elements = 1
        for d in info.shape:
            n_elements *= d
        n_blocks = n_elements // 32
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 34:(b + 1) * 34]
            scale = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
            quants = np.frombuffer(block[2:34], dtype=np.int8).astype(np.float32)
            values[b * 32:(b + 1) * 32] = scale * quants
        t = torch.from_numpy(values.reshape(info.shape))
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 14:  # Q6_K — 210 bytes per 256 elements
        n_elements = 1
        for d in info.shape:
            n_elements *= d
        n_blocks = n_elements // 256
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 210:(b + 1) * 210]
            # Q6_K layout: ql[128] + qh[64] + scales[16] + d(float16)
            ql = np.frombuffer(block[:128], dtype=np.uint8)
            qh = np.frombuffer(block[128:192], dtype=np.uint8)
            scales = np.frombuffer(block[192:208], dtype=np.int8).astype(np.float32)
            d = np.frombuffer(block[208:210], dtype=np.float16).astype(np.float32)[0]
            # Decode 6-bit values: lower 4 bits from ql, upper 2 bits from qh
            for j in range(256):
                q_lo = int(ql[j // 2] >> (4 * (j % 2))) & 0xF
                q_hi = int(qh[j // 4] >> (2 * (j % 4))) & 0x3
                q = q_lo | (q_hi << 4)
                sc_idx = j // 16
                values[b * 256 + j] = d * scales[sc_idx] * (q - 32)
        t = torch.from_numpy(values.reshape(info.shape))
        return t.to(torch.bfloat16).to(device)

    raise ValueError(
        f"Unsupported GGML type {ggml_type} ({GGML_TYPES.get(ggml_type, ('UNKNOWN',))[0]}) "
        f"for non-expert tensor '{name}'. Only F32, F16, Q4_K, Q6_K, Q8_0 are supported."
    )


def _dequant_fused_tensor(
    reader: GGUFReader | MultiShardGGUFReader,
    info: GGUFTensorInfo,
    name: str,
    device: str | torch.device,
) -> torch.Tensor:
    """Dequantize a fused 3-D GGUF expert tensor to BF16.

    Fused expert tensors have shape ``(out_dim, in_dim, n_experts)``.  Q4_K
    quantisation applies to the flat element buffer, so we dequant into a 2-D
    array of ``n_elements`` elements and then reshape.
    """
    import numpy as np

    if isinstance(reader, MultiShardGGUFReader):
        raw = reader.get_tensor_data(name)
    else:
        raw = reader.get_tensor_data(info)

    ggml_type = info.ggml_type
    shape_3d = tuple(info.shape)  # (out_dim, in_dim, n_experts)
    n_elements = 1
    for d in shape_3d:
        n_elements *= d

    if ggml_type == 0:  # F32
        t = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape_3d)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 1:  # F16
        t = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape_3d)
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 12:  # Q4_K — 256-element blocks, 144 bytes each
        from .gguf_quant import parse_q4k_block

        n_blocks = n_elements // 256
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 144:(b + 1) * 144]
            vals, _, _ = parse_q4k_block(block)
            values[b * 256:(b + 1) * 256] = vals
        t = torch.from_numpy(values.reshape(shape_3d))
        return t.to(torch.bfloat16).to(device)

    if ggml_type == 8:  # Q8_0 — 34 bytes per block of 32 elements
        n_blocks = n_elements // 32
        values = np.empty(n_elements, dtype=np.float32)
        for b in range(n_blocks):
            block = raw[b * 34:(b + 1) * 34]
            scale = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
            quants = np.frombuffer(block[2:34], dtype=np.int8).astype(np.float32)
            values[b * 32:(b + 1) * 32] = scale * quants
        t = torch.from_numpy(values.reshape(shape_3d))
        return t.to(torch.bfloat16).to(device)

    raise ValueError(
        f"Unsupported GGML type {ggml_type} ({GGML_TYPES.get(ggml_type, ('UNKNOWN',))[0]}) "
        f"for fused expert tensor '{name}'. Only F32, F16, Q4_K, Q8_0 are supported."
    )


def _build_expert_store_from_fused_reader(
    reader: GGUFReader | MultiShardGGUFReader,
    num_layers: int,
    num_experts: int,
    device: str | torch.device,
) -> "ExpertStore | None":
    """Build a ExpertStore from fused expert tensors (Qwen 3.5 style).

    Fused expert tensors store all experts in a single tensor per projection:
    ``blk.<L>.ffn_gate_exps.weight`` with shape ``(out_dim, in_dim, n_experts)``.

    This function dequants one layer at a time (memory-efficient) and slices
    per expert, then packs into a ``ExpertStore``.

    Returns ``None`` when no fused expert tensors are present.
    """
    from .expert_store import ExpertStore

    fused_layers = reader.list_fused_expert_tensors()
    if not fused_layers:
        return None

    layers = sorted(fused_layers.keys())
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    for layer_idx in layers:
        layer_tensors = fused_layers[layer_idx]

        gate_name = f"blk.{layer_idx}.ffn_gate_exps.weight"
        up_name = f"blk.{layer_idx}.ffn_up_exps.weight"
        down_name = f"blk.{layer_idx}.ffn_down_exps.weight"

        # Dequant full fused tensors on CPU first (shape: [out, in, n_experts])
        gate_bf16 = _dequant_fused_tensor(reader, layer_tensors["gate"], gate_name, "cpu")
        up_bf16 = _dequant_fused_tensor(reader, layer_tensors["up"], up_name, "cpu")
        down_bf16 = _dequant_fused_tensor(reader, layer_tensors["down"], down_name, "cpu")

        n_exp = gate_bf16.shape[2]

        for expert_idx in range(n_exp):
            gate_e = gate_bf16[:, :, expert_idx]  # [intermediate, hidden]
            up_e = up_bf16[:, :, expert_idx]      # [intermediate, hidden]
            down_e = down_bf16[:, :, expert_idx]  # [hidden, intermediate]

            gate_up = torch.cat([gate_e, up_e], dim=0)  # [2*intermediate, hidden]

            expert_weights[(layer_idx, expert_idx)] = {
                "gate_up_proj": gate_up.to(torch.bfloat16),
                "down_proj": down_e.to(torch.bfloat16),
            }

        # Release fused tensors immediately to bound peak RAM
        del gate_bf16, up_bf16, down_bf16

    return ExpertStore.from_dict(expert_weights, num_layers, num_experts)


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
    5. Expert weights: create GGUFExpertStore (Q4_K -> INT4 pack)
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
            continue  # handled separately via _build_expert_store_from_fused_reader
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

    # Step 4: Load non-expert weights (dequant to BF16 on device)
    loaded = 0
    skipped = 0
    for gguf_name in non_expert_names:
        hf_name, _, _, _ = gguf_to_hf_name(gguf_name)

        # Navigate to the parameter in the model
        param = _get_param(model, hf_name)
        if param is None:
            logger.debug("Skipping unmapped GGUF tensor: %s -> %s", gguf_name, hf_name)
            skipped += 1
            continue

        if is_multi:
            info = reader.get_tensor_info(gguf_name)
        else:
            info = _find_tensor_info(reader, gguf_name)

        tensor = _dequant_tensor(reader, info, gguf_name, device)

        # Handle shape mismatches (e.g. 1D norm weights)
        if tensor.shape != param.shape:
            if tensor.numel() == param.numel():
                tensor = tensor.reshape(param.shape)
            else:
                logger.warning(
                    "Shape mismatch for %s: GGUF %s vs model %s, skipping",
                    hf_name,
                    tensor.shape,
                    param.shape,
                )
                skipped += 1
                continue

        _set_param(model, hf_name, tensor)
        loaded += 1

    logger.info("Loaded %d non-expert weights, skipped %d", loaded, skipped)

    # Step 5: Expert weights -> expert store
    from .gguf_store import GGUFExpertStore

    expert_groups = reader.list_expert_tensors()

    if expert_groups:
        store = _build_expert_store_from_reader(reader, expert_groups, is_multi)
        logger.info(
            "Expert store (per-expert): %d layers, %d experts, %.2f MB/expert",
            store.num_layers,
            store.num_experts,
            store.expert_bytes / 1e6,
        )
    else:
        store = None

    if store is None and fused_expert_names:
        logger.info(
            "No per-expert tensors found; attempting fused expert tensor slicing (%d fused tensors)",
            len(fused_expert_names),
        )
        store = _build_expert_store_from_fused_reader(
            reader,
            num_layers=gguf_cfg.num_hidden_layers,
            num_experts=gguf_cfg.num_experts,
            device=device,
        )
        if store is not None:
            logger.info(
                "Expert store (fused): %d layers, %d experts, %.2f MB/expert",
                store.num_layers,
                store.num_experts,
                store.expert_bytes / 1e6,
            )

    if store is None:
        logger.warning("No expert tensors found in GGUF — model has no MoE layers?")

    reader.close()

    # Step 6: Apply offloading
    from .offload import offload_model

    model = offload_model(
        model,
        device=device,
        cache_capacity=cache_capacity,
        model_id=model_id,
        **kwargs,
    )

    return model


def _find_tensor_info(reader: GGUFReader, name: str) -> GGUFTensorInfo:
    """Find a GGUFTensorInfo by name in a single-shard reader."""
    for t in reader.tensors:
        if t.name == name:
            return t
    raise KeyError(f"Tensor '{name}' not found in GGUF file")


def _get_param(model: torch.nn.Module, hf_name: str) -> torch.nn.Parameter | None:
    """Navigate dotted path to find a parameter in the model."""
    parts = hf_name.split(".")
    obj = model
    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif part.isdigit() and hasattr(obj, "__getitem__"):
            obj = obj[int(part)]
        else:
            return None
    final = parts[-1]
    if hasattr(obj, final):
        attr = getattr(obj, final)
        if isinstance(attr, (torch.nn.Parameter, torch.Tensor)):
            return attr
    return None


def _set_param(model: torch.nn.Module, hf_name: str, tensor: torch.Tensor):
    """Set a parameter in the model by dotted path."""
    parts = hf_name.split(".")
    obj = model
    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif part.isdigit() and hasattr(obj, "__getitem__"):
            obj = obj[int(part)]
        else:
            raise AttributeError(f"Cannot navigate to {hf_name}: missing {part}")
    final = parts[-1]
    if hasattr(obj, final):
        attr = getattr(obj, final)
        if isinstance(attr, torch.nn.Parameter):
            attr.data = tensor.to(dtype=attr.data.dtype, device=attr.data.device)
        else:
            setattr(obj, final, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(obj, final, torch.nn.Parameter(tensor, requires_grad=False))


def _build_expert_store_from_reader(
    reader: GGUFReader | MultiShardGGUFReader,
    expert_groups: dict[tuple[int, int], dict[str, GGUFTensorInfo]],
    is_multi: bool,
) -> GGUFExpertStore:
    """Build a GGUFExpertStore from parsed expert tensor groups."""
    from .gguf_quant import q4k_expert_to_int4pack
    from .gguf_store import GGUFExpertStore
    from .expert_store import TensorLayout, _pack_tensors

    layers = sorted({k[0] for k in expert_groups})
    experts_per_layer = sorted({k[1] for k in expert_groups})
    num_layers = len(layers)
    num_experts = len(experts_per_layer)

    # Convert first expert to determine layout
    first_key = (layers[0], experts_per_layer[0])
    first_projs = expert_groups[first_key]

    def _read_data(info: GGUFTensorInfo, name: str) -> bytes:
        if is_multi:
            return reader.get_tensor_data(name)
        return reader.get_tensor_data(info)

    gate_info = first_projs["gate"]
    up_info = first_projs["up"]
    down_info = first_projs["down"]

    gate_name = f"blk.{layers[0]}.ffn_gate.{experts_per_layer[0]}.weight"
    up_name = f"blk.{layers[0]}.ffn_up.{experts_per_layer[0]}.weight"
    down_name = f"blk.{layers[0]}.ffn_down.{experts_per_layer[0]}.weight"

    gate_data = _read_data(gate_info, gate_name)
    up_data = _read_data(up_info, up_name)
    down_data = _read_data(down_info, down_name)

    gate_shape = (gate_info.shape[0], gate_info.shape[1])
    up_shape = (up_info.shape[0], up_info.shape[1])
    down_shape = (down_info.shape[0], down_info.shape[1])

    g_packed, g_sz, u_packed, u_sz, d_packed, d_sz = q4k_expert_to_int4pack(
        gate_data, up_data, down_data,
        gate_shape, up_shape, down_shape,
    )

    specs = {
        "gate_packed": (tuple(g_packed.shape), g_packed.dtype),
        "gate_sz": (tuple(g_sz.shape), g_sz.dtype),
        "up_packed": (tuple(u_packed.shape), u_packed.dtype),
        "up_sz": (tuple(u_sz.shape), u_sz.dtype),
        "down_packed": (tuple(d_packed.shape), d_packed.dtype),
        "down_sz": (tuple(d_sz.shape), d_sz.dtype),
    }
    layout = TensorLayout(specs)

    data = torch.empty(
        num_layers, num_experts, layout.total_bytes,
        dtype=torch.uint8,
    )
    if torch.cuda.is_available():
        data = data.pin_memory()

    for li, layer_idx in enumerate(layers):
        for ei, expert_idx in enumerate(experts_per_layer):
            key = (layer_idx, expert_idx)
            projs = expert_groups[key]

            g_name = f"blk.{layer_idx}.ffn_gate.{expert_idx}.weight"
            u_name = f"blk.{layer_idx}.ffn_up.{expert_idx}.weight"
            d_name = f"blk.{layer_idx}.ffn_down.{expert_idx}.weight"

            g_data = _read_data(projs["gate"], g_name)
            u_data = _read_data(projs["up"], u_name)
            d_data = _read_data(projs["down"], d_name)

            g_s = (projs["gate"].shape[0], projs["gate"].shape[1])
            u_s = (projs["up"].shape[0], projs["up"].shape[1])
            d_s = (projs["down"].shape[0], projs["down"].shape[1])

            gp, gsz, up, usz, dp, dsz = q4k_expert_to_int4pack(
                g_data, u_data, d_data, g_s, u_s, d_s,
            )

            tensors = {
                "gate_packed": gp,
                "gate_sz": gsz,
                "up_packed": up,
                "up_sz": usz,
                "down_packed": dp,
                "down_sz": dsz,
            }
            _pack_tensors(data[li, ei], layout, tensors)

    return GGUFExpertStore(data, layout, num_layers, num_experts)
