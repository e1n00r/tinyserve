"""Phase 1: Download GPT-OSS-120B and split into non-expert + per-layer expert files.

Usage:
    python -m scripts.split_weights --model-dir /path/to/hf/cache --output-dir ./weights

If --model-dir is not provided, downloads the model via huggingface_hub.
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_ID,
    NUM_LAYERS,
    is_expert_key,
    parse_expert_key,
)


def download_model(cache_dir: str | None = None) -> str:
    """Download model and return local path."""
    print(f"Downloading {MODEL_ID}...")
    return snapshot_download(
        MODEL_ID,
        cache_dir=cache_dir,
        ignore_patterns=["*.gguf", "*.md", "*.txt"],
    )


def load_safetensors_index(model_dir: str) -> dict[str, str]:
    """Load the weight map from the safetensors index."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    return index["weight_map"]


def split_weights(model_dir: str, output_dir: str):
    """Split model weights into non-expert and per-layer expert files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weight_map = load_safetensors_index(model_dir)

    # Classify keys
    non_expert_keys: dict[str, str] = {}  # key -> shard filename
    expert_keys_by_layer: dict[int, dict[str, str]] = {i: {} for i in range(NUM_LAYERS)}

    for key, shard in weight_map.items():
        if is_expert_key(key):
            layer_idx, _ = parse_expert_key(key)
            expert_keys_by_layer[layer_idx][key] = shard
        else:
            non_expert_keys[key] = shard

    print(f"Non-expert keys: {len(non_expert_keys)}")
    print(f"Expert keys per layer: {len(expert_keys_by_layer[0])}")

    # Collect unique shard files we need to open
    all_shards = set(weight_map.values())
    shard_handles: dict[str, any] = {}
    for shard_name in sorted(all_shards):
        shard_path = os.path.join(model_dir, shard_name)
        shard_handles[shard_name] = safe_open(shard_path, framework="pt", device="cpu")

    # 1. Write non-expert weights
    print("Writing non-expert weights...")
    non_expert_tensors = {}
    for key, shard in non_expert_keys.items():
        non_expert_tensors[key] = shard_handles[shard].get_tensor(key)
    save_file(non_expert_tensors, str(output_path / "non_expert.safetensors"))
    total_ne = sum(t.nbytes for t in non_expert_tensors.values())
    print(f"  non_expert.safetensors: {total_ne / 1024**3:.2f} GB")
    del non_expert_tensors

    # 2. Write per-layer expert files
    # Each layer's experts are stored as safetensors with all 6 tensor types
    # Keys: gate_up_proj_blocks, gate_up_proj_scales, gate_up_proj_bias,
    #        down_proj_blocks, down_proj_scales, down_proj_bias
    # Each tensor has shape [NUM_EXPERTS, ...] so we can index by expert.
    print("Writing per-layer expert weights...")
    for layer_idx in range(NUM_LAYERS):
        layer_tensors = {}
        for key, shard in expert_keys_by_layer[layer_idx].items():
            _, tensor_name = parse_expert_key(key)
            tensor = shard_handles[shard].get_tensor(key)
            layer_tensors[tensor_name] = tensor

        out_file = output_path / f"experts_L{layer_idx:02d}.safetensors"
        save_file(layer_tensors, str(out_file))
        total_bytes = sum(t.nbytes for t in layer_tensors.values())
        print(f"  experts_L{layer_idx:02d}.safetensors: {total_bytes / 1024**2:.0f} MB "
              f"({len(layer_tensors)} tensors)")
        del layer_tensors

    # Close handles
    for h in shard_handles.values():
        del h

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Split GPT-OSS-120B weights")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to already-downloaded HF model directory")
    parser.add_argument("--output-dir", type=str, default="./weights",
                        help="Output directory for split weights")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory for download")
    args = parser.parse_args()

    if args.model_dir is None:
        model_dir = download_model(args.cache_dir)
    else:
        model_dir = args.model_dir

    split_weights(model_dir, args.output_dir)


if __name__ == "__main__":
    main()
