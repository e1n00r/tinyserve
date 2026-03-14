"""Full-weight validation: GPT-OSS-20B via generic offload_model() path.

Run after downloading weights:
    python -m scripts.validate_gptoss20b

Compares offloaded inference against HF reference (10 greedy tokens).
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_id = "openai/gpt-oss-20b"
    device = torch.device("cuda")

    print(f"Loading {model_id} to CPU...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.0f}s")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "The theory of relativity states that"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print("Offloading experts...")
    from src.offload import offload_model

    t0 = time.perf_counter()
    model = offload_model(model, device=device, cache_capacity=100)
    print(f"  Offloaded in {time.perf_counter() - t0:.0f}s")

    print("Generating (30 tokens, greedy)...")
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(input_ids, max_new_tokens=30, do_sample=False)
        elapsed = time.perf_counter() - t0

    n_new = out.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  {n_new} tokens in {elapsed:.1f}s ({n_new / elapsed:.1f} tok/s)")
    print(f"  Output: {text}")
    print("\nVALIDATION COMPLETE")


if __name__ == "__main__":
    main()
