"""CLI entry point: python -m tinyserve <command>.

Commands:
    serve   Start the OpenAI-compatible HTTP server
    run     Load a model and generate interactively (REPL)
    info    Print model profile (expert layout, routing, sizes)
"""

import argparse
import sys


def cmd_serve(args: argparse.Namespace) -> None:
    sys.argv = ["tinyserve-serve", "--model", args.model, "--port", str(args.port)]
    from .server import main as server_main

    server_main()


def cmd_run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    gguf_path = getattr(args, "gguf", None)

    if gguf_path:
        from .gguf_loader import load_from_gguf

        print(f"Loading from GGUF: {gguf_path} ...")
        model = load_from_gguf(gguf_path, model_id=args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        from .offload import load_and_offload

        print(f"Loading {args.model} ...")
        model = load_and_offload(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Ready. Type a prompt (Ctrl-D to quit).\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        chunk_size = getattr(args, "chunk_size", 0)
        if chunk_size > 0:
            from .chunked import generate_chunked

            kv_cache = getattr(model, "_kv_cache", None)
            if kv_cache is not None:
                kv_cache.reset()
            output = generate_chunked(
                model,
                input_ids,
                max_new_tokens=args.max_tokens,
                kv_cache=kv_cache,
                chunk_size=chunk_size,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=args.max_tokens)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(text)
        print()


def cmd_info(args: argparse.Namespace) -> None:
    from transformers import AutoConfig

    from .model_registry import profile_from_config

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    profile = profile_from_config(config)

    print(f"Model:           {args.model}")
    print(f"Model type:      {config.model_type}")
    print(f"Num layers:      {profile.num_layers}")
    print(f"Experts/layer:   {profile.num_experts}")
    print(f"Top-k:           {profile.num_experts_per_tok}")
    print(f"MoE block attr:  {profile.moe_block_attr}")
    print(f"Expert list:     {profile.expert_list_attr}")
    weights = profile.expert_layout.weight_names
    biases = profile.expert_layout.bias_names
    print(f"Weight tensors:  {', '.join(weights)}")
    if biases:
        print(f"Bias tensors:    {', '.join(biases)}")
    if profile.shared_expert_attr:
        print(f"Shared expert:   {profile.shared_expert_attr}")
    else:
        print("Shared expert:   no")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tinyserve", description="MoE expert offloading for consumer GPUs")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible HTTP server")
    p_serve.add_argument("--model", default="openai/gpt-oss-20b", help="HuggingFace model id or local path")
    p_serve.add_argument("--gguf", default=None, help="Path to GGUF file (loads model from GGUF instead of HF)")
    p_serve.add_argument("--port", type=int, default=8000)

    p_run = sub.add_parser("run", help="Interactive generation REPL")
    p_run.add_argument("--model", required=True, help="HuggingFace model id (for tokenizer + config)")
    p_run.add_argument("--gguf", default=None, help="Path to GGUF file (loads model from GGUF instead of HF)")
    p_run.add_argument("--max-tokens", type=int, default=100)
    p_run.add_argument("--chunk-size", type=int, default=0, help="Prefill chunk size (0 = full prefill)")

    p_info = sub.add_parser("info", help="Print model profile and expert layout")
    p_info.add_argument("--model", required=True, help="HuggingFace model id or local path")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {"serve": cmd_serve, "run": cmd_run, "info": cmd_info}
    dispatch[args.command](args)
