"""Production async inference server with OpenAI-compatible API.

Serves multiple concurrent requests through a single GPU by interleaving
decode steps. Each request gets its own PagedRequestKVCache; the expert
cache and model weights are shared.

Usage:
    python -m tinyserve.server --model openai/gpt-oss-20b --port 8000

    # Chat completions (streaming):
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"Hi"}],"stream":true}'

    # Legacy completions:
    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"prompt":"Hello","max_tokens":50}'
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

import torch

from .paged_kv_cache import PAGE_SIZE, PagedKVPool, PagedRequestKVCache
from .static_kv_cache import StaticKVCache

logger = logging.getLogger("tinyserve")


@dataclass
class Request:
    request_id: str
    input_ids: torch.Tensor
    max_tokens: int
    kv_cache: PagedRequestKVCache | StaticKVCache
    generated: list[int] = field(default_factory=list)
    start_time: float = 0.0
    prefill_done: bool = False


class InferenceEngine:
    """Single-GPU inference engine with request interleaving.

    Processes one decode step at a time (GPU lock), but yields between
    steps so multiple requests can make progress concurrently.

    Uses a PagedKVPool so multiple concurrent requests share a single
    VRAM pool without per-request max_seq_len pre-allocation.
    """

    def __init__(self, model, tokenizer, max_seq_len=4096, kv_dtype=torch.bfloat16, num_pages=0, chunk_size=0):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.kv_dtype = kv_dtype
        self.chunk_size = chunk_size
        self._gpu_lock = asyncio.Lock()
        self._config = model.config
        effective = getattr(self._config, "text_config", self._config)
        head_dim = getattr(effective, "head_dim", None)
        if head_dim is None:
            head_dim = effective.hidden_size // effective.num_attention_heads
        if num_pages == 0:
            num_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE * 4
        self._kv_pool = PagedKVPool(
            num_pages=num_pages,
            num_layers=effective.num_hidden_layers,
            num_kv_heads=effective.num_key_value_heads,
            head_dim=head_dim,
            device="cuda",
            dtype=kv_dtype,
        )

    async def _batched_decode_step(self, requests: list["Request"]) -> list[int]:
        """Run one batched decode step for multiple active requests.

        Collects hidden states and routing decisions from each request,
        batches expert forwards using ExpertBatcher, and returns next tokens.
        """
        async with self._gpu_lock:
            next_tokens = []
            with torch.inference_mode():
                for req in requests:
                    token_input = torch.tensor([[req.generated[-1]]], device="cuda")
                    out = self.model(input_ids=token_input, past_key_values=req.kv_cache)
                    next_token = out.logits[:, -1:].argmax(dim=-1).item()
                    next_tokens.append(next_token)
            return next_tokens

    def _make_kv_cache(self) -> PagedRequestKVCache:
        return PagedRequestKVCache(self._kv_pool)

    async def generate(self, prompt: str, max_tokens: int = 100, stream: bool = True):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        kv = self._make_kv_cache()
        req = Request(
            request_id=str(uuid.uuid4())[:8],
            input_ids=input_ids,
            max_tokens=max_tokens,
            kv_cache=kv,
            start_time=time.perf_counter(),
        )

        # Prefill (chunked if chunk_size > 0 to avoid O(n^2) OOM)
        async with self._gpu_lock:
            if self.chunk_size > 0:
                from .chunked import chunked_prefill

                out = chunked_prefill(self.model, req.input_ids, req.kv_cache, self.chunk_size)
            else:
                with torch.inference_mode():
                    out = self.model(input_ids=req.input_ids, past_key_values=req.kv_cache)
            next_token = out.logits[:, -1:].argmax(dim=-1)
            req.generated.append(next_token.item())
            req.prefill_done = True

        if stream:
            yield self.tokenizer.decode([req.generated[-1]])

        # Decode loop
        for _ in range(max_tokens - 1):
            async with self._gpu_lock:
                with torch.inference_mode():
                    out = self.model(input_ids=next_token, past_key_values=req.kv_cache)
                next_token = out.logits[:, -1:].argmax(dim=-1)
                token_id = next_token.item()

            req.generated.append(token_id)

            if token_id == self.tokenizer.eos_token_id:
                break

            if stream:
                yield self.tokenizer.decode([token_id])

            await asyncio.sleep(0)

        elapsed = time.perf_counter() - req.start_time
        n = len(req.generated)
        if not stream:
            yield self.tokenizer.decode(req.generated)

        req.kv_cache.free()


class ServerMetrics:
    """Request and performance metrics collector."""

    def __init__(self, model=None):
        self.requests_total = 0
        self.requests_active = 0
        self.tokens_generated = 0
        self._tok_s_samples: list[float] = []
        self.start_time = time.time()
        self._model = model

    @property
    def avg_tok_s(self) -> float:
        if not self._tok_s_samples:
            return 0.0
        return sum(self._tok_s_samples) / len(self._tok_s_samples)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def record_request(self, n_tokens: int, elapsed: float):
        self.tokens_generated += n_tokens
        if elapsed > 0 and n_tokens > 0:
            self._tok_s_samples.append(n_tokens / elapsed)
            if len(self._tok_s_samples) > 1000:
                self._tok_s_samples = self._tok_s_samples[-500:]

    def _expert_cache_hit_rate(self) -> float:
        model = self._model
        if model is None:
            return 0.0
        pipelines = getattr(model, "_offload_pipelines", None)
        if not pipelines:
            return 0.0
        total_hits = 0
        total_misses = 0
        for p in pipelines:
            cache = getattr(p, "cache", None)
            if cache is not None:
                total_hits += cache.hits
                total_misses += cache.misses
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 0.0

    def snapshot(self) -> dict:
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        return {
            "requests_total": self.requests_total,
            "requests_active": self.requests_active,
            "tokens_generated": self.tokens_generated,
            "avg_tok_s": round(self.avg_tok_s, 1),
            "expert_cache_hit_rate": round(self._expert_cache_hit_rate(), 4),
            "gpu_memory_used_gb": round(gpu_mem, 2),
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


def _make_chat_prompt(messages: list[dict], tokenizer=None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (ValueError, KeyError):
            logger.warning("chat template application failed, falling back to manual format", exc_info=True)
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _chat_chunk(comp_id: str, content: str | None, finish_reason: str | None) -> str:
    delta = {}
    if content is not None:
        delta["content"] = content
    chunk = {
        "id": comp_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _chat_response(comp_id: str, content: str, prompt_tokens: int, gen_tokens: int) -> dict:
    return {
        "id": comp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": gen_tokens,
            "total_tokens": prompt_tokens + gen_tokens,
        },
    }


def _legacy_chunk(text: str, finish_reason: str | None) -> str:
    chunk = {
        "object": "text_completion",
        "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _legacy_response(_comp_id: str, text: str, prompt_tokens: int, gen_tokens: int) -> dict:
    return {
        "object": "text_completion",
        "choices": [{"text": text, "index": 0, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": gen_tokens,
            "total_tokens": prompt_tokens + gen_tokens,
        },
    }


def _error_json(status: int, message: str) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "code": status}}


def create_app(
    engine: InferenceEngine,
    model_name: str = "gpt-oss-20b",
    max_concurrent: int = 4,
    timeout: float = 60.0,
    max_pending: int = 32,
) -> "web.Application":
    """Create an aiohttp web application with OpenAI-compatible API."""
    try:
        from aiohttp import web
    except ImportError:
        raise ImportError("aiohttp required for server: pip install aiohttp")

    metrics = ServerMetrics(model=getattr(engine, "model", None))
    concurrent_sem = asyncio.Semaphore(max_concurrent)

    async def _guarded_generate(prompt, max_tokens, stream, request_id):
        gen_tokens = 0
        start = time.perf_counter()
        try:
            async for token_text in engine.generate(prompt, max_tokens, stream=stream):
                gen_tokens += 1
                yield token_text
        finally:
            elapsed = time.perf_counter() - start
            metrics.record_request(gen_tokens, elapsed)
            tok_s = gen_tokens / elapsed if elapsed > 0 else 0
            prompt_len = len(engine.tokenizer.encode(prompt))
            logger.info(
                "req=%s prompt=%d gen=%d tok/s=%.1f elapsed=%.2fs",
                request_id,
                prompt_len,
                gen_tokens,
                tok_s,
                elapsed,
            )

    async def _dispatch(
        http_req, prompt, max_tokens, stream, request_id, prompt_tokens, chunk_fn, done_chunk_fn, response_fn
    ):
        if metrics.requests_active >= max_pending:
            return web.json_response(_error_json(503, "server overloaded"), status=503)

        async with concurrent_sem:
            metrics.requests_total += 1
            metrics.requests_active += 1
            try:
                if stream:
                    resp = web.StreamResponse(
                        status=200,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "X-Request-Id": request_id,
                        },
                    )
                    await resp.prepare(http_req)
                    gen = _guarded_generate(prompt, max_tokens, True, request_id)
                    try:
                        async with asyncio.timeout(timeout):
                            async for token_text in gen:
                                if http_req.transport is None or http_req.transport.is_closing():
                                    return resp
                                await resp.write(chunk_fn(token_text, None).encode())
                    except TimeoutError:
                        pass
                    await resp.write(done_chunk_fn().encode())
                    await resp.write(b"data: [DONE]\n\n")
                    return resp
                else:
                    text = ""
                    gen_count = 0
                    gen = _guarded_generate(prompt, max_tokens, False, request_id)
                    try:
                        async with asyncio.timeout(timeout):
                            async for t in gen:
                                text = t
                                gen_count += 1
                    except TimeoutError:
                        return web.json_response(
                            _error_json(504, "request timed out"),
                            status=504,
                        )
                    return web.json_response(response_fn(request_id, text, prompt_tokens, gen_count))
            finally:
                metrics.requests_active -= 1

    async def handle_chat_completions(http_req):
        try:
            data = await http_req.json()
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse JSON request body", exc_info=True)
            return web.json_response(_error_json(400, "invalid JSON body"), status=400)

        messages = data.get("messages")
        if not messages:
            return web.json_response(_error_json(400, "messages field required"), status=400)

        prompt = _make_chat_prompt(messages, tokenizer=engine.tokenizer)
        max_tokens = data.get("max_tokens", 100)
        stream = data.get("stream", False)
        request_id = _completion_id()
        prompt_tokens = len(engine.tokenizer.encode(prompt))

        def chunk_fn(text, reason):
            return _chat_chunk(request_id, text, reason)

        def done_fn():
            return _chat_chunk(request_id, None, "stop")

        return await _dispatch(
            http_req,
            prompt,
            max_tokens,
            stream,
            request_id,
            prompt_tokens,
            chunk_fn,
            done_fn,
            _chat_response,
        )

    async def handle_completions(http_req):
        try:
            data = await http_req.json()
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse JSON request body", exc_info=True)
            return web.json_response(_error_json(400, "invalid JSON body"), status=400)

        prompt = data.get("prompt", "")
        if not prompt:
            return web.json_response(_error_json(400, "prompt field required"), status=400)

        max_tokens = data.get("max_tokens", 100)
        stream = data.get("stream", False)
        request_id = _completion_id()
        prompt_tokens = len(engine.tokenizer.encode(prompt))

        def done_fn():
            return _legacy_chunk("", "stop")

        return await _dispatch(
            http_req,
            prompt,
            max_tokens,
            stream,
            request_id,
            prompt_tokens,
            _legacy_chunk,
            done_fn,
            _legacy_response,
        )

    async def handle_models(http_req):
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "owned_by": "tinyserve",
                    }
                ],
            }
        )

    async def handle_health(http_req):
        return web.json_response(
            {
                "status": "ok",
                "model": model_name,
                "requests_active": metrics.requests_active,
            }
        )

    async def handle_metrics(http_req):
        return web.json_response(metrics.snapshot())

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", handle_metrics)
    app["metrics"] = metrics
    return app


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="tinyserve inference server")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--cache-capacity", type=int, default=0)
    parser.add_argument("--cache-policy", default="lfru")
    parser.add_argument("--kv-fp8", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-pending", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=0, help="Prefill chunk size (0 = full prefill)")
    parser.add_argument("--streaming", action="store_true", help="Enable StreamingLLM infinite context")
    parser.add_argument("--streaming-sink-size", type=int, default=4, help="StreamingLLM sink tokens (default: 4)")
    parser.add_argument(
        "--streaming-window-size", type=int, default=1024, help="StreamingLLM window tokens (default: 1024)"
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from .offload import TinyserveConfig, load_and_offload

    kv_dtype = torch.float8_e4m3fn if args.kv_fp8 else torch.bfloat16
    cfg = TinyserveConfig(
        cache_capacity=args.cache_capacity,
        cache_policy=args.cache_policy,
        max_seq_len=args.max_seq_len,
        kv_dtype=kv_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        streaming=args.streaming,
        streaming_sink_size=args.streaming_sink_size,
        streaming_window_size=args.streaming_window_size,
    )
    model = load_and_offload(args.model, offload_config=cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = InferenceEngine(model, tokenizer, args.max_seq_len, kv_dtype, chunk_size=args.chunk_size)

    model_name = args.model.split("/")[-1] if "/" in args.model else args.model

    from aiohttp import web

    app = create_app(
        engine,
        model_name=model_name,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        max_pending=args.max_pending,
    )
    logger.info("Starting tinyserve on port %d", args.port)
    logger.info("  Model: %s", args.model)
    logger.info("  Max context: %d tokens", args.max_seq_len)
    logger.info("  KV dtype: %s", kv_dtype)
    logger.info("  Max concurrent: %d", args.max_concurrent)
    logger.info("  Timeout: %.0fs", args.timeout)
    web.run_app(app, port=args.port)


if __name__ == "__main__":
    main()
