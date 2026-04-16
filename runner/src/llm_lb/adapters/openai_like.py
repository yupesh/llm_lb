"""Shared helper for OpenAI-compatible chat completions endpoints.

Used by both the official OpenAI adapter and any self-hosted endpoint that
exposes a `/v1/chat/completions` route (vLLM, TGI in OpenAI mode, llama.cpp
server, LM Studio, OpenRouter, Together, Groq, ...).
"""
from __future__ import annotations

import time

import httpx

from ..models import LLMParams
from .base import Completion

# Transient network-layer errors that we consider worth retrying. Anything else
# (bad JSON, 4xx other than 429, programming errors) bubbles up immediately.
_RETRY_EXCEPTIONS = (
    httpx.ReadError,
    httpx.WriteError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def openai_chat(
    base_url: str,
    headers: dict[str, str],
    served_name: str,
    system: str | None,
    user: str,
    params: LLMParams,
    timeout: float = 600.0,
    max_retries: int = 4,
    backoff_base: float = 2.0,
) -> Completion:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body: dict = {
        "model": served_name,
        "messages": messages,
        "temperature": params.temperature,
        "top_p": params.top_p,
        # Most endpoints (vLLM, TGI, older OpenAI) use `max_tokens`.
        # Newer OpenAI models require `max_completion_tokens` instead.
        # Try `max_tokens` first; on 400 retry with the new name.
        "max_tokens": params.max_tokens,
    }
    if params.seed is not None:
        body["seed"] = params.seed
    url = f"{base_url.rstrip('/')}/chat/completions"

    # Retry loop: transient network errors and 5xx/429 are retried with
    # exponential backoff. 4xx (other than 429) fail fast.
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=body)
                if r.status_code == 400 and "max_tokens" in (r.text or ""):
                    # Endpoint requires new-style param — retry.
                    del body["max_tokens"]
                    body["max_completion_tokens"] = params.max_tokens
                    r = client.post(url, headers=headers, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    # Retryable server-side condition.
                    last_exc = RuntimeError(
                        f"{r.status_code} from {base_url} model={served_name}: "
                        f"{(r.text or '')[:200]}"
                    )
                    if attempt < max_retries:
                        time.sleep(backoff_base**attempt)
                        continue
                    raise last_exc
                if r.is_error:
                    detail = r.text[:500] if r.text else "(empty body)"
                    raise RuntimeError(
                        f"{r.status_code} from {base_url} model={served_name}: {detail}"
                    )
                data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            return Completion(
                text=text,
                output_tokens=usage.get("completion_tokens"),
                input_tokens=usage.get("prompt_tokens"),
            )
        except _RETRY_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_base**attempt)
                continue
            raise RuntimeError(
                f"network error after {max_retries + 1} attempts to {base_url} "
                f"model={served_name}: {type(exc).__name__}: {exc}"
            ) from exc

    # Unreachable: the loop either returns or raises.
    raise RuntimeError(f"openai_chat: exhausted retries ({last_exc})")


def openai_chat_messages(
    base_url: str,
    headers: dict[str, str],
    served_name: str,
    messages: list[dict],
    params: LLMParams,
    tools: list[dict] | None = None,
    timeout: float = 600.0,
    max_retries: int = 4,
    backoff_base: float = 2.0,
) -> dict:
    """Multi-turn variant of `openai_chat` that returns the raw assistant
    message (dict with keys `role`, `content`, optional `tool_calls`).

    Used by dialog-simulation tasks where we need the full message history
    plus optional OpenAI-style function calling.
    """
    body: dict = {
        "model": served_name,
        "messages": messages,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
    }
    if params.seed is not None:
        body["seed"] = params.seed
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    url = f"{base_url.rstrip('/')}/chat/completions"

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=body)
                if r.status_code == 400 and "max_tokens" in (r.text or ""):
                    del body["max_tokens"]
                    body["max_completion_tokens"] = params.max_tokens
                    r = client.post(url, headers=headers, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_exc = RuntimeError(
                        f"{r.status_code} from {base_url} model={served_name}: "
                        f"{(r.text or '')[:200]}"
                    )
                    if attempt < max_retries:
                        time.sleep(backoff_base**attempt)
                        continue
                    raise last_exc
                if r.is_error:
                    detail = r.text[:500] if r.text else "(empty body)"
                    raise RuntimeError(
                        f"{r.status_code} from {base_url} model={served_name}: {detail}"
                    )
                data = r.json()
            msg = data["choices"][0]["message"]
            usage = data.get("usage") or {}
            return {
                "message": msg,
                "usage": usage,
            }
        except _RETRY_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_base**attempt)
                continue
            raise RuntimeError(
                f"network error after {max_retries + 1} attempts to {base_url} "
                f"model={served_name}: {type(exc).__name__}: {exc}"
            ) from exc

    raise RuntimeError(f"openai_chat_messages: exhausted retries ({last_exc})")
