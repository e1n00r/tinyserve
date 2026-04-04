"""Disk cache for pre-packed expert store data.

Saves the fully-parsed expert byte buffers (from safetensors shards) to a
single safetensors file so that subsequent loads skip shard scanning and
MXFP4 layout processing.  Second load takes ~2s instead of ~8min.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
from safetensors.torch import save_file


def int4_cache_path(model_id: str) -> str:
    """Return cache file path based on model ID.

    Sanitises the model_id (replaces ``/`` with ``--``) and places the file
    under ``~/.cache/tinyserve/int4/``.

    >>> int4_cache_path("openai/gpt-oss-20b")
    '.../.cache/tinyserve/int4/openai--gpt-oss-20b.safetensors'
    """
    sanitized = model_id.replace("/", "--").replace(os.sep, "--")
    cache_dir = Path.home() / ".cache" / "tinyserve" / "int4"
    return str(cache_dir / f"{sanitized}.safetensors")


def _model_hash(model_dir: str) -> str:
    """Compute a stable hash of the model's safetensors index (or file list).

    If ``model.safetensors.index.json`` exists, hash its contents.
    Otherwise hash the sorted list of ``*.safetensors`` filenames + sizes.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    entries: list[str] = []
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".safetensors"):
            size = os.path.getsize(os.path.join(model_dir, fname))
            entries.append(f"{fname}:{size}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def save_int4_cache(
    path: str,
    data: torch.Tensor,
    layout_specs: dict[str, tuple[tuple[int, ...], str]],
    num_layers: int,
    num_experts: int,
    model_hash: str,
) -> None:
    """Save pre-packed expert store data to a safetensors cache file.

    Args:
        path: Destination file path.
        data: The flat [num_layers, num_experts, expert_bytes] uint8 tensor.
        layout_specs: Serialisable form of TensorLayout.specs
            (shape tuple, dtype name string).
        num_layers: Number of MoE layers.
        num_experts: Number of experts per layer.
        model_hash: Hash string for cache invalidation.
    """
    cache_dir = os.path.dirname(path)
    os.makedirs(cache_dir, exist_ok=True)

    metadata = {
        "num_layers": str(num_layers),
        "num_experts": str(num_experts),
        "model_hash": model_hash,
        "layout_specs": json.dumps(layout_specs),
    }

    tensors = {"expert_data": data.contiguous()}
    save_file(tensors, path, metadata=metadata)


def load_int4_cache(
    path: str,
    expected_hash: str | None = None,
) -> dict | None:
    """Load cached expert store data.

    Returns a dict with keys ``data``, ``layout_specs``, ``num_layers``,
    ``num_experts`` on success.  Returns ``None`` if:
    - The cache file does not exist.
    - The stored model_hash does not match ``expected_hash``.
    """
    if not os.path.exists(path):
        return None

    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata is None:
                return None

            if expected_hash is not None:
                stored_hash = metadata.get("model_hash", "")
                if stored_hash != expected_hash:
                    return None

            data = f.get_tensor("expert_data")

        layout_specs_raw = json.loads(metadata["layout_specs"])
        layout_specs = {name: (tuple(shape), dtype_name) for name, (shape, dtype_name) in layout_specs_raw.items()}

        return {
            "data": data,
            "layout_specs": layout_specs,
            "num_layers": int(metadata["num_layers"]),
            "num_experts": int(metadata["num_experts"]),
            "model_hash": metadata["model_hash"],
        }
    except (OSError, RuntimeError, KeyError, json.JSONDecodeError):
        logger.warning("Failed to load expert cache from disk, will rebuild", exc_info=True)
        return None


def _serialize_layout_specs(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Convert TensorLayout.specs to a JSON-serialisable form."""
    return {name: (list(shape), str(dtype)) for name, (shape, dtype) in specs.items()}


def _deserialize_layout_specs(
    raw: dict[str, tuple[tuple[int, ...], str]],
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Convert serialised layout specs back to TensorLayout.specs format."""
    dtype_map = {
        "torch.uint8": torch.uint8,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float8_e4m3fn": torch.float8_e4m3fn,
        "torch.int8": torch.int8,
        "torch.int16": torch.int16,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
    }
    return {name: (tuple(shape), dtype_map[dtype_name]) for name, (shape, dtype_name) in raw.items()}
