"""Full-weight validation on real models.

Usage:
    python -m scripts.validate_real_models --model openai/gpt-oss-20b
    python -m scripts.validate_real_models --model Qwen/Qwen3.5-35B-A3B
    python -m scripts.validate_real_models --model Qwen/Qwen3.5-122B-A10B

Loads model to CPU, offloads experts to GPU, generates text.
Requires enough RAM for the full model weights.

IMPORTANT: Do NOT run this script directly via Claude's bash tool for real models.
Run via scripts/run_validation.sh in a separate terminal. Use --dry-run first.
"""
import argparse
import gc
import os
import sys
import time

import torch


def _is_mxfp4_model(model_name: str) -> bool:
    """Peek at HF config to determine if the model is MXFP4-quantized."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        qcfg = getattr(cfg, "quantization_config", None)
        if qcfg is None:
            return False
        # Config may be a dict or an object depending on the model.
        if isinstance(qcfg, dict):
            quant_method = str(qcfg.get("quant_method", "") or qcfg.get("quant_type", "") or qcfg.get("weights", ""))
        else:
            quant_method = str(getattr(qcfg, "quant_method", "") or getattr(qcfg, "quant_type", "") or getattr(qcfg, "weights", ""))
        return "mxfp4" in quant_method.lower() or "fp4" in quant_method.lower()
    except Exception:
        return False


def _estimate_model_ram_gb(model_name: str) -> float:
    """Rough bfloat16 RAM estimate (2 bytes/param + 20% overhead).

    We always load with device_map='cpu' which dequantizes to bf16 regardless
    of the model's on-disk quantization format.
    """
    import re
    name_lower = model_name.lower()
    match = re.search(r"(\d+(?:\.\d+)?)b", name_lower)
    param_billions = float(match.group(1)) if match else 20.0
    return param_billions * 2.0 * 1.20


def _available_ram_gb() -> float:
    """Read MemAvailable from /proc/meminfo (most accurate on Linux)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / 1024 / 1024
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3


def main():
    try:
        os.setsid()
    except PermissionError:
        pass  # already a session leader

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="The theory of relativity states that")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--cache-capacity", type=int, default=100)
    parser.add_argument("--max-ram-gb", type=float, default=None,
                        help="Abort if model needs more RAM than this")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print RAM estimate and exit without loading the model")
    args = parser.parse_args()

    is_mxfp4 = _is_mxfp4_model(args.model)
    needed_gb = _estimate_model_ram_gb(args.model)
    avail_gb = _available_ram_gb()

    on_disk_fmt = "MXFP4" if is_mxfp4 else "bfloat16"
    print(f"On-disk format : {on_disk_fmt} (loaded as bfloat16 via device_map=cpu)")
    print(f"RAM estimate   : {needed_gb:.1f} GB needed (bfloat16 + 20% overhead)")
    print(f"RAM available  : {avail_gb:.1f} GB (MemAvailable)")

    if args.dry_run:
        if avail_gb >= needed_gb:
            print("DRY-RUN: RAM sufficient. Run without --dry-run to proceed.")
        else:
            print(f"DRY-RUN: INSUFFICIENT RAM — need {needed_gb:.1f} GB, have {avail_gb:.1f} GB.")
        sys.exit(0)

    if avail_gb < needed_gb:
        print(
            f"ERROR: Insufficient RAM. Need {needed_gb:.1f} GB, only {avail_gb:.1f} GB available.\n"
            "Free memory (close other processes) or use a smaller model."
        )
        sys.exit(1)

    if args.max_ram_gb and avail_gb < args.max_ram_gb:
        print(f"WARNING: Only {avail_gb:.1f} GB RAM available, --max-ram-gb requires {args.max_ram_gb} GB")

    device = torch.device("cuda")

    print(f"Loading {args.model} to CPU...")
    t0 = time.perf_counter()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    # device_map="cpu" is required: it dequantizes MXFP4 models to bf16 on CPU.
    # The GPU path requires proprietary OpenAI Triton kernels (kernels-community/
    # gpt-oss-triton-kernels) that are not publicly available. Native MXFP4
    # loading via direct safetensors is planned (pin for later).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", low_cpu_mem_usage=True,
    )
    load_time = time.perf_counter() - t0
    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded in {load_time:.0f}s ({params_b:.1f}B params)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    print(f"  Prompt: {input_ids.shape[1]} tokens")

    # Add project root to path for src imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.offload import offload_model

    print("Offloading experts...")
    t0 = time.perf_counter()
    # For MXFP4 models, pass model_id so expert weights are loaded natively
    # from safetensors (uint8 blocks + scales) rather than from the dequantized
    # bf16 params.  This uses our Triton kernels and avoids OAI's proprietary ones.
    offload_kwargs = {"model_id": args.model} if is_mxfp4 else {}
    model = offload_model(model, device=device, cache_capacity=args.cache_capacity, **offload_kwargs)
    gc.collect()
    torch.cuda.empty_cache()
    offload_time = time.perf_counter() - t0
    print(f"  Offloaded in {offload_time:.0f}s")

    free_vram = torch.cuda.mem_get_info(device)[0] / 1024**3
    print(f"  Free VRAM: {free_vram:.1f} GB")

    print(f"Generating ({args.max_tokens} tokens, greedy)...")
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(input_ids, max_new_tokens=args.max_tokens, do_sample=False)
        elapsed = time.perf_counter() - t0

    n_new = out.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  {n_new} tokens in {elapsed:.1f}s ({n_new / elapsed:.1f} tok/s)")
    print(f"  Output: {text}")

    if hasattr(model, "_offload_pipelines"):
        total_hits = sum(p.cache.hits for p in model._offload_pipelines if p.cache)
        total_misses = sum(p.cache.misses for p in model._offload_pipelines if p.cache)
        total = total_hits + total_misses
        rate = total_hits / total if total > 0 else 0
        print(f"  Cache: {total_hits} hits, {total_misses} misses ({rate:.1%} hit rate)")

    print("\nDONE")


if __name__ == "__main__":
    main()
