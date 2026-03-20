"""Repack GPT-OSS expert weights from safetensors to contiguous binary.

Each expert's data is packed contiguously:
    gate_up_blocks | gate_up_scales | gate_up_bias | down_blocks | down_scales | down_bias

Expert (layer_idx, expert_idx) is at offset:
    (layer_idx * num_experts + expert_idx) * expert_bytes

Usage:
    python -m scripts.repack_experts --model openai/gpt-oss-20b --output experts.bin
"""

import argparse
import ctypes
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _madvise_willneed(data: memoryview):
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        MADV_WILLNEED = 3
        addr = ctypes.c_char_p(bytes(data[:1]))
        libc.madvise(addr, len(data), MADV_WILLNEED)
    except Exception:
        pass


def _expert_bytes(gu_blocks_shape, gu_scales_shape, gu_bias_shape,
                  dn_blocks_shape, dn_scales_shape, dn_bias_shape) -> int:
    def nbytes(shape, dtype_bytes):
        n = 1
        for d in shape:
            n *= d
        return n * dtype_bytes

    return (
        nbytes(gu_blocks_shape, 1)
        + nbytes(gu_scales_shape, 1)
        + nbytes(gu_bias_shape, 4)
        + nbytes(dn_blocks_shape, 1)
        + nbytes(dn_scales_shape, 1)
        + nbytes(dn_bias_shape, 4)
    )


def repack(model_id: str, output_path: Path):
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(model_id))
    st_files = sorted(model_dir.glob("*.safetensors"))

    layer_tensors: dict[int, dict[str, torch.Tensor]] = {}
    for st_file in st_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".mlp.experts." not in key:
                    continue
                parts = key.split(".")
                layer_idx = int(parts[2])
                tensor_name = parts[-1]
                if layer_idx not in layer_tensors:
                    layer_tensors[layer_idx] = {}
                layer_tensors[layer_idx][tensor_name] = f.get_tensor(key)

    sample = layer_tensors[next(iter(layer_tensors))]
    num_experts = sample["gate_up_proj_blocks"].shape[0]
    num_layers = len(layer_tensors)

    gu_b = sample["gate_up_proj_blocks"]
    gu_s = sample["gate_up_proj_scales"]
    gu_bias = sample["gate_up_proj_bias"]
    dn_b = sample["down_proj_blocks"]
    dn_s = sample["down_proj_scales"]
    dn_bias = sample["down_proj_bias"]

    expert_bytes = _expert_bytes(
        gu_b.shape[1:], gu_s.shape[1:], gu_bias.shape[1:],
        dn_b.shape[1:], dn_s.shape[1:], dn_bias.shape[1:],
    )
    total_bytes = num_layers * num_experts * expert_bytes

    print(f"  {num_layers} layers, {num_experts} experts, {expert_bytes / 1024**2:.1f} MB/expert")
    print(f"  Total output: {total_bytes / 1024**3:.2f} GB -> {output_path}")

    buf = bytearray(total_bytes)
    mv = memoryview(buf)
    _madvise_willneed(mv)

    for layer_idx in sorted(layer_tensors):
        tensors = layer_tensors[layer_idx]
        gu_blocks = tensors["gate_up_proj_blocks"]
        gu_scales = tensors["gate_up_proj_scales"]
        gu_bias_t = tensors["gate_up_proj_bias"].float()
        dn_blocks = tensors["down_proj_blocks"]
        dn_scales = tensors["down_proj_scales"]
        dn_bias_t = tensors["down_proj_bias"].float()

        for eidx in range(num_experts):
            base = (layer_idx * num_experts + eidx) * expert_bytes
            offset = base

            gb = gu_blocks[eidx].contiguous().numpy().tobytes()
            buf[offset:offset + len(gb)] = gb
            offset += len(gb)

            gs = gu_scales[eidx].contiguous().numpy().tobytes()
            buf[offset:offset + len(gs)] = gs
            offset += len(gs)

            gb2 = gu_bias_t[eidx].contiguous().numpy().tobytes()
            buf[offset:offset + len(gb2)] = gb2
            offset += len(gb2)

            db = dn_blocks[eidx].contiguous().numpy().tobytes()
            buf[offset:offset + len(db)] = db
            offset += len(db)

            ds = dn_scales[eidx].contiguous().numpy().tobytes()
            buf[offset:offset + len(ds)] = ds
            offset += len(ds)

            db2 = dn_bias_t[eidx].contiguous().numpy().tobytes()
            buf[offset:offset + len(db2)] = db2

        if (layer_idx + 1) % 6 == 0 or layer_idx == num_layers - 1:
            print(f"  Repacked {layer_idx + 1}/{num_layers} layers")

    with open(output_path, "wb") as f:
        f.write(buf)

    print(f"Done. {output_path.stat().st_size / 1024**3:.2f} GB written.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--output", type=str, default="experts.bin")
    args = parser.parse_args()
    repack(args.model, Path(args.output))


if __name__ == "__main__":
    main()
