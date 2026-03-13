"""Repack per-layer safetensors expert files into contiguous binary format.

Each expert's 6 tensors are packed contiguously so a single sequential read
loads the entire expert. Layout matches ExpertBuffer.packed (same byte offsets).

Output: weights/experts_L{i:02d}.bin — raw binary, expert j at offset j * EXPERT_BYTES.

Usage:
    python -m scripts.repack_experts --weights-dir ./weights
"""

import argparse
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    EXPERT_BYTES,
    NUM_EXPERTS,
    NUM_LAYERS,
    PACK_DN_BIAS_OFF,
    PACK_DN_BLOCKS_OFF,
    PACK_DN_SCALES_OFF,
    PACK_GU_BIAS_OFF,
    PACK_GU_BLOCKS_OFF,
    PACK_GU_SCALES_OFF,
)


def repack_layer(weights_dir: Path, layer_idx: int):
    """Repack one layer's safetensors into contiguous binary."""
    src = weights_dir / f"experts_L{layer_idx:02d}.safetensors"
    dst = weights_dir / f"experts_L{layer_idx:02d}.bin"

    f = safe_open(str(src), framework="pt", device="cpu")

    gu_blocks = f.get_tensor("gate_up_proj_blocks")   # [128, 5760, 90, 16] uint8
    gu_scales = f.get_tensor("gate_up_proj_scales")    # [128, 5760, 90] uint8
    gu_bias = f.get_tensor("gate_up_proj_bias")        # [128, 5760] f32
    dn_blocks = f.get_tensor("down_proj_blocks")       # [128, 2880, 90, 16] uint8
    dn_scales = f.get_tensor("down_proj_scales")       # [128, 2880, 90] uint8
    dn_bias = f.get_tensor("down_proj_bias")           # [128, 2880] f32

    buf = bytearray(NUM_EXPERTS * EXPERT_BYTES)

    for eidx in range(NUM_EXPERTS):
        base = eidx * EXPERT_BYTES

        # gate_up_blocks: flatten to bytes
        gb = gu_blocks[eidx].contiguous().numpy().tobytes()
        buf[base + PACK_GU_BLOCKS_OFF:base + PACK_GU_BLOCKS_OFF + len(gb)] = gb

        # gate_up_scales
        gs = gu_scales[eidx].contiguous().numpy().tobytes()
        buf[base + PACK_GU_SCALES_OFF:base + PACK_GU_SCALES_OFF + len(gs)] = gs

        # gate_up_bias → raw bytes (may be bf16 or f32, store as f32)
        gbias_t = gu_bias[eidx].contiguous().float()
        gbias = gbias_t.numpy().tobytes()
        buf[base + PACK_GU_BIAS_OFF:base + PACK_GU_BIAS_OFF + len(gbias)] = gbias

        # down_blocks
        db = dn_blocks[eidx].contiguous().numpy().tobytes()
        buf[base + PACK_DN_BLOCKS_OFF:base + PACK_DN_BLOCKS_OFF + len(db)] = db

        # down_scales
        ds = dn_scales[eidx].contiguous().numpy().tobytes()
        buf[base + PACK_DN_SCALES_OFF:base + PACK_DN_SCALES_OFF + len(ds)] = ds

        # down_bias → raw bytes (may be bf16 or f32, store as f32)
        dbias_t = dn_bias[eidx].contiguous().float()
        dbias = dbias_t.numpy().tobytes()
        buf[base + PACK_DN_BIAS_OFF:base + PACK_DN_BIAS_OFF + len(dbias)] = dbias

    with open(dst, "wb") as out:
        out.write(buf)

    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=str, default="./weights")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)

    total_bytes = 0
    for layer_idx in range(NUM_LAYERS):
        dst = repack_layer(weights_dir, layer_idx)
        size = dst.stat().st_size
        total_bytes += size
        if (layer_idx + 1) % 6 == 0 or layer_idx == NUM_LAYERS - 1:
            print(f"  Repacked {layer_idx + 1}/{NUM_LAYERS} layers "
                  f"({size / 1024**2:.0f} MB each)")

    print(f"\nTotal: {total_bytes / 1024**3:.2f} GB")
    print("Done. You can now use --packed flag with the model.")


if __name__ == "__main__":
    main()
