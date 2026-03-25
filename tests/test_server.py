"""Tests for the production async server (OpenAI-compatible API).

Uses a mock InferenceEngine to test HTTP layer without GPU.
All async tests use aiohttp.test_utils.AioHTTPTestCase.
"""

import asyncio
import json
import time
import unittest

import pytest

from tinyserve.server import (
    ServerMetrics,
    _chat_chunk,
    _chat_response,
    _completion_id,
    _error_json,
    _legacy_chunk,
    _legacy_response,
    _make_chat_prompt,
    create_app,
)


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))

    def decode(self, ids):
        return " ".join(f"tok{i}" for i in ids)

    @property
    def eos_token_id(self):
        return 999


class FakeEngine:
    """Mock engine that yields deterministic tokens without GPU."""

    def __init__(self, tokens_to_yield=3):
        self.tokenizer = FakeTokenizer()
        self._tokens = tokens_to_yield

    async def generate(self, prompt, max_tokens=100, stream=True):
        for i in range(min(self._tokens, max_tokens)):
            yield f"word{i}"
            await asyncio.sleep(0)


# --- Unit tests for helper functions ---

class TestMakeChatPrompt:
    def test_single_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = _make_chat_prompt(messages)
        assert "<|user|>" in result
        assert "Hello" in result
        assert "<|assistant|>" in result

    def test_multi_turn(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
        ]
        result = _make_chat_prompt(messages)
        assert result.count("<|") == 5


class TestCompletionId:
    def test_format(self):
        cid = _completion_id()
        assert cid.startswith("chatcmpl-")
        assert len(cid) == len("chatcmpl-") + 24


class TestChatChunk:
    def test_with_content(self):
        raw = _chat_chunk("id-1", "hello", None)
        assert raw.startswith("data: ")
        assert raw.endswith("\n\n")
        parsed = json.loads(raw[len("data: "):])
        assert parsed["id"] == "id-1"
        assert parsed["choices"][0]["delta"]["content"] == "hello"
        assert parsed["choices"][0]["finish_reason"] is None

    def test_finish(self):
        raw = _chat_chunk("id-1", None, "stop")
        parsed = json.loads(raw[len("data: "):])
        assert parsed["choices"][0]["delta"] == {}
        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_object_type(self):
        raw = _chat_chunk("id-1", "x", None)
        parsed = json.loads(raw[len("data: "):])
        assert parsed["object"] == "chat.completion.chunk"


class TestChatResponse:
    def test_structure(self):
        resp = _chat_response("id-1", "answer", 10, 5)
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["message"]["content"] == "answer"
        assert resp["usage"]["prompt_tokens"] == 10
        assert resp["usage"]["completion_tokens"] == 5
        assert resp["usage"]["total_tokens"] == 15


class TestLegacyChunk:
    def test_structure(self):
        raw = _legacy_chunk("word", None)
        parsed = json.loads(raw[len("data: "):])
        assert parsed["choices"][0]["text"] == "word"


class TestLegacyResponse:
    def test_structure(self):
        resp = _legacy_response("id", "full text", 5, 10)
        assert resp["choices"][0]["text"] == "full text"
        assert resp["usage"]["total_tokens"] == 15


class TestErrorJson:
    def test_structure(self):
        err = _error_json(400, "bad request")
        assert err["error"]["message"] == "bad request"
        assert err["error"]["code"] == 400


# --- ServerMetrics tests ---

class TestServerMetrics:
    def test_initial_state(self):
        m = ServerMetrics()
        assert m.requests_total == 0
        assert m.requests_active == 0
        assert m.tokens_generated == 0
        assert m.avg_tok_s == 0.0

    def test_record_request(self):
        m = ServerMetrics()
        m.record_request(100, 10.0)
        assert m.tokens_generated == 100
        assert m.avg_tok_s == pytest.approx(10.0)

    def test_avg_tok_s_multiple(self):
        m = ServerMetrics()
        m.record_request(10, 1.0)
        m.record_request(20, 1.0)
        assert m.avg_tok_s == pytest.approx(15.0)

    def test_snapshot_keys(self):
        m = ServerMetrics()
        snap = m.snapshot()
        expected_keys = {
            "requests_total", "requests_active", "tokens_generated",
            "avg_tok_s", "expert_cache_hit_rate", "gpu_memory_used_gb",
            "uptime_seconds",
        }
        assert set(snap.keys()) == expected_keys

    def test_uptime_increases(self):
        m = ServerMetrics()
        m.start_time = time.time() - 100
        assert m.uptime_seconds >= 99.0

    def test_sample_buffer_capped(self):
        m = ServerMetrics()
        for _ in range(1200):
            m.record_request(1, 0.1)
        assert len(m._tok_s_samples) <= 1000

    def test_zero_elapsed_ignored(self):
        m = ServerMetrics()
        m.record_request(10, 0.0)
        assert m.avg_tok_s == 0.0
        assert m.tokens_generated == 10

    def test_zero_tokens_ignored(self):
        m = ServerMetrics()
        m.record_request(0, 1.0)
        assert m.avg_tok_s == 0.0


# --- HTTP endpoint tests (aiohttp AioHTTPTestCase) ---

from aiohttp.test_utils import AioHTTPTestCase
from aiohttp import web


class TestHealthAPI(AioHTTPTestCase):
    async def get_application(self):
        return create_app(FakeEngine(), model_name="test-model")

    async def test_health_ok(self):
        resp = await self.client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "test-model"

    async def test_health_has_active_count(self):
        data = await (await self.client.get("/health")).json()
        assert "requests_active" in data


class TestModelsAPI(AioHTTPTestCase):
    async def get_application(self):
        return create_app(FakeEngine(), model_name="test-model")

    async def test_list_models(self):
        resp = await self.client.get("/v1/models")
        assert resp.status == 200
        data = await resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"


class TestMetricsAPI(AioHTTPTestCase):
    async def get_application(self):
        return create_app(FakeEngine(), model_name="test-model")

    async def test_metrics_initial(self):
        resp = await self.client.get("/metrics")
        assert resp.status == 200
        data = await resp.json()
        assert data["requests_total"] == 0
        assert data["tokens_generated"] == 0

    async def test_metrics_after_request(self):
        await self.client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 3},
        )
        data = await (await self.client.get("/metrics")).json()
        assert data["requests_total"] == 1
        assert data["tokens_generated"] > 0


class TestChatCompletionsAPI(AioHTTPTestCase):
    async def get_application(self):
        return create_app(FakeEngine(), model_name="test-model")

    async def test_non_streaming(self):
        resp = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "stream": False,
            },
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["object"] == "chat.completion"
        assert "content" in data["choices"][0]["message"]
        assert data["usage"]["prompt_tokens"] > 0

    async def test_streaming(self):
        resp = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "stream": True,
            },
        )
        assert resp.status == 200
        assert "text/event-stream" in resp.headers["Content-Type"]
        body = await resp.text()
        lines = [l for l in body.strip().split("\n") if l.startswith("data:")]
        assert lines[-1] == "data: [DONE]"
        for line in lines[:-1]:
            payload = line[len("data: "):]
            chunk = json.loads(payload)
            assert "choices" in chunk

    async def test_missing_messages(self):
        resp = await self.client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )
        assert resp.status == 400

    async def test_invalid_json(self):
        resp = await self.client.post(
            "/v1/chat/completions",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400


class TestLegacyCompletionsAPI(AioHTTPTestCase):
    async def get_application(self):
        return create_app(FakeEngine(), model_name="test-model")

    async def test_non_streaming(self):
        resp = await self.client.post(
            "/v1/completions",
            json={"prompt": "Hello world", "max_tokens": 3},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["object"] == "text_completion"
        assert "text" in data["choices"][0]

    async def test_streaming(self):
        resp = await self.client.post(
            "/v1/completions",
            json={"prompt": "Hello world", "max_tokens": 3, "stream": True},
        )
        assert resp.status == 200
        body = await resp.text()
        assert "data: [DONE]" in body

    async def test_missing_prompt(self):
        resp = await self.client.post(
            "/v1/completions",
            json={"max_tokens": 10},
        )
        assert resp.status == 400

    async def test_request_id_in_stream_header(self):
        resp = await self.client.post(
            "/v1/completions",
            json={"prompt": "test", "max_tokens": 1, "stream": True},
        )
        assert "X-Request-Id" in resp.headers


class TestConcurrencyAPI(AioHTTPTestCase):
    async def get_application(self):
        return create_app(FakeEngine(), model_name="test-model", max_concurrent=2)

    async def test_concurrent_requests(self):
        tasks = [
            self.client.post("/v1/completions", json={"prompt": f"req{i}", "max_tokens": 2})
            for i in range(3)
        ]
        responses = await asyncio.gather(*tasks)
        for r in responses:
            assert r.status == 200


class TestTimeoutAPI(AioHTTPTestCase):
    async def get_application(self):

        class SlowEngine:
            tokenizer = FakeTokenizer()

            async def generate(self, prompt, max_tokens=100, stream=True):
                await asyncio.sleep(10)
                yield "late"

        return create_app(SlowEngine(), model_name="slow", timeout=0.1)

    async def test_timeout_non_streaming(self):
        resp = await self.client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 1},
        )
        assert resp.status == 504
