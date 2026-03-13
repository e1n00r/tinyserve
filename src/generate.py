"""Text generation loop for offloaded GPT-OSS-120B."""

import time

import torch
from transformers import AutoTokenizer

from .config import MODEL_ID
from .model import OffloadedGptOss


def generate(
    model: OffloadedGptOss,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Autoregressive text generation."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    batch, prompt_len = input_ids.shape

    model.reset_kv_cache()

    # Prefill: process entire prompt
    position_ids = torch.arange(prompt_len, device=model.device).unsqueeze(0)
    print(f"Prefilling {prompt_len} tokens...")
    t0 = time.perf_counter()
    logits, timings = model.forward(input_ids, position_ids)
    prefill_time = time.perf_counter() - t0
    print(f"  Prefill: {prefill_time:.2f}s ({prompt_len / prefill_time:.1f} tok/s)")

    # Sample first token
    next_logits = logits[:, -1, :]
    next_token = _sample(next_logits, temperature, top_p)
    generated = [next_token.item()]

    # Decode loop
    total_timings = {k: 0.0 for k in timings}
    decode_start = time.perf_counter()

    for step in range(1, max_new_tokens):
        position_ids = torch.tensor(
            [[prompt_len + step - 1]], device=model.device
        )
        logits, timings = model.forward(
            next_token.unsqueeze(0), position_ids
        )

        for k, v in timings.items():
            total_timings[k] += v

        next_logits = logits[:, -1, :]
        next_token = _sample(next_logits, temperature, top_p)
        generated.append(next_token.item())

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        if (step + 1) % 10 == 0:
            elapsed = time.perf_counter() - decode_start
            print(f"  Generated {step + 1} tokens ({(step + 1) / elapsed:.1f} tok/s)")

    decode_time = time.perf_counter() - decode_start
    n_tokens = len(generated)
    print(f"\nDecode: {n_tokens} tokens in {decode_time:.2f}s ({n_tokens / decode_time:.1f} tok/s)")
    print(f"  Breakdown: attn={total_timings['attn']:.2f}s, "
          f"router={total_timings['router']:.2f}s, "
          f"transfer={total_timings['transfer']:.2f}s, "
          f"expert_compute={total_timings['expert_compute']:.2f}s")

    output_ids = input_ids[0].tolist() + generated
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative prob above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# Need this import for _sample
import torch.nn.functional as F


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=str, default="./weights")
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OffloadedGptOss(args.weights_dir)

    output = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature)
    print(f"\n{'='*60}")
    print(output)


if __name__ == "__main__":
    main()
