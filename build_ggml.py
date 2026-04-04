# SPDX-License-Identifier: MIT
"""Build tinyserve ggml_ops CUDA extension.

Usage: python3 build_ggml.py
"""
import subprocess
import torch
from torch.utils.cpp_extension import load

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required to build ggml_ops")

cc = torch.cuda.get_device_capability()
sm = f"{cc[0]}{cc[1]}"

nvcc_ver = subprocess.run(
    ["nvcc", "--version"], capture_output=True, text=True
).stdout
nvcc_supports_sm = f"compute_{sm}" in subprocess.run(
    ["nvcc", "--list-gpu-arch"], capture_output=True, text=True
).stdout

if nvcc_supports_sm:
    arch_flag = f"-gencode=arch=compute_{sm},code=sm_{sm}"
else:
    # Fallback: compile PTX for highest supported arch, JIT at load time
    arch_flag = "-gencode=arch=compute_90,code=compute_90"
    print(f"nvcc does not support sm_{sm}, using PTX JIT via compute_90")

ggml_ops = load(
    name="tinyserve_ggml_ops",
    sources=["tinyserve/csrc/ggml_ops.cu"],
    extra_cuda_cflags=["-O2", "--use_fast_math", arch_flag],
    verbose=True,
)
print(f"Build successful! (target: sm_{sm})")
