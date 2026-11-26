from __future__ import annotations

import asyncio
import json
import weakref
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings

OpenRouterBaseURL = "https://openrouter.ai/api/v1"

# Retry transient failures (rate limits, overloaded/unavailable upstreams).
_RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}
_MAX_ATTEMPTS = 4
_MAX_RETRY_AFTER = 60.0  # never sleep longer than this on a single retry
_RETRY_BASE_DELAY = 1.0  # seconds; doubled each attempt (patched to 0 in tests)

# Swappable for tests (httpx.MockTransport).
_client_factory = httpx.AsyncClient


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error response."""


# One semaphore per event loop: fan-out (parallel execution agents x tool loops, plus
# background summarizers and the email classifier) used to hit the API with no limit.
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(max(get_settings().llm_max_concurrency, 1))
        _semaphores[loop] = sem
    return sem


def _headers(*, api_key: Optional[str] = None) -> Dict[str, str]:
    settings = get_settings()
    key = (api_key or settings.openrouter_api_key or "").strip()
    if not key:
        raise OpenRouterError("Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    return headers


def _build_messages(messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
    if system:
        return [{"role": "system", "content": system}, *messages]
    return messages


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        return payload.get("error") or payload.get("message") or json.dumps(payload)
    except Exception:
        return response.text


async def request_chat_completion(
    *,
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    base_url: str = OpenRouterBaseURL,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Request a chat completion and return the raw JSON payload.

    - ``max_tokens`` defaults to ``settings.llm_max_output_tokens``. Without it OpenRouter
      reserves the model's full output budget (64k tokens) per request, which both
      uncaps cost and makes requests fail outright on low-balance accounts.
    - Concurrency is capped by ``settings.llm_max_concurrency``.
    - 429/5xx and network errors are retried with exponential backoff.
    """

    payload: Dict[str, object] = {
        "model": model,
        "messages": _build_messages(messages, system),
        "stream": False,
        "max_tokens": max_tokens or get_settings().llm_max_output_tokens,
    }
    if tools:
        payload["tools"] = tools

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = _headers(api_key=api_key)

    last_error: Optional[OpenRouterError] = None
    retry_after: Optional[float] = None
    for attempt in range(_MAX_ATTEMPTS):
        if attempt:
            backoff = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            # Honor the server's Retry-After (rate limits are often per-minute windows).
            await asyncio.sleep(min(max(backoff, retry_after or 0.0), _MAX_RETRY_AFTER))
        async with _semaphore():
            try:
                async with _client_factory() as client:
                    response = await client.post(
                        url, headers=headers, json=payload, timeout=get_settings().llm_timeout_seconds
                    )
            except httpx.HTTPError as exc:
                # str(ReadTimeout) is often empty; keep the exception type in the message.
                last_error = OpenRouterError(f"OpenRouter request failed: {type(exc).__name__}: {exc}")
                continue

        if response.status_code < 400:
            return response.json()
        last_error = OpenRouterError(
            f"OpenRouter request failed ({response.status_code}): {_error_detail(response)}"
        )
        retry_after = _parse_retry_after(response)
        if response.status_code not in _RETRY_STATUSES:
            break

    raise last_error or OpenRouterError("OpenRouter request failed: unknown error")


__all__ = ["OpenRouterError", "request_chat_completion", "OpenRouterBaseURL"]
