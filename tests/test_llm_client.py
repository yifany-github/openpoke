"""OpenRouter client: output cap, retry/backoff, concurrency limit."""

import asyncio
import json

import httpx
import pytest

from server.openrouter_client import client as client_mod
from server.openrouter_client.client import OpenRouterError, request_chat_completion

OK = {"choices": [{"message": {"content": "hi"}}]}


@pytest.fixture
def transport(settings, monkeypatch):
    """Route the client through an httpx.MockTransport driven by ``transport.handler``."""
    monkeypatch.setattr(client_mod, "_RETRY_BASE_DELAY", 0)

    class Box:
        handler = None
        requests = []

    async def handle(request):
        Box.requests.append(json.loads(request.content))
        return await Box.handler(request)

    monkeypatch.setattr(client_mod, "_client_factory",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    return Box


def call():
    return request_chat_completion(model="m", messages=[{"role": "user", "content": "x"}])


def test_max_tokens_is_always_sent(transport, settings):
    async def ok(request):
        return httpx.Response(200, json=OK)

    transport.handler = ok
    assert asyncio.run(call()) == OK
    assert transport.requests[0]["max_tokens"] == settings.llm_max_output_tokens


def test_retries_rate_limits_then_succeeds(transport):
    statuses = iter([429, 503, 200])

    async def flaky(request):
        code = next(statuses)
        return httpx.Response(code, json=OK if code == 200 else {"error": "busy"})

    transport.handler = flaky
    assert asyncio.run(call()) == OK
    assert len(transport.requests) == 3


def test_does_not_retry_client_errors(transport):
    async def payment_required(request):
        return httpx.Response(402, json={"message": "insufficient credits"})

    transport.handler = payment_required
    with pytest.raises(OpenRouterError, match="402"):
        asyncio.run(call())
    assert len(transport.requests) == 1


def test_concurrency_is_capped(transport, settings, monkeypatch):
    monkeypatch.setattr(settings, "llm_max_concurrency", 2)
    in_flight = {"now": 0, "max": 0}

    async def slow(request):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1
        return httpx.Response(200, json=OK)

    transport.handler = slow

    async def run():
        await asyncio.gather(*(call() for _ in range(6)))

    asyncio.run(run())
    assert in_flight["max"] == 2


def test_honors_retry_after(transport, monkeypatch):
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(client_mod.asyncio, "sleep", fake_sleep)
    statuses = iter([429, 200])

    async def limited(request):
        code = next(statuses)
        return httpx.Response(code, json=OK, headers={"retry-after": "7"} if code == 429 else {})

    transport.handler = limited
    assert asyncio.run(call()) == OK
    assert slept == [7.0]
