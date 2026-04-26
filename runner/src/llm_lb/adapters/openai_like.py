"""Shared helper for OpenAI-compatible chat completions endpoints.

Used by both the official OpenAI adapter and any self-hosted endpoint that
exposes a `/v1/chat/completions` route (vLLM, TGI in OpenAI mode, llama.cpp
server, LM Studio, OpenRouter, Together, Groq, ...).
"""
from __future__ import annotations

import os
import time

import httpx

from ..models import LLMParams, ModelCard
from .base import Completion


def resolve_served_name(model: ModelCard) -> str:
    """Resolve what the adapter will put in the `model` field of an OpenAI
    chat request. Precedence: env override > `served_model_name` > `hf_uri` >
    the `model_id` slug with our internal `@provider` suffix stripped."""
    if model.served_model_name_env:
        from_env = os.environ.get(model.served_model_name_env, "")
        if from_env:
            return from_env
    if model.served_model_name:
        return model.served_model_name
    if model.hf_uri:
        return model.hf_uri
    return model.model_id.split("@", 1)[0]

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


def _detect_reasoning_api(served_name: str) -> str:
    """Pick the request-body shape that the served model expects.

    - "qwen": Qwen3+ chat templates honour `chat_template_kwargs.enable_thinking`.
      litellm rejects `reasoning_effort` for Qwen with HTTP 400, so we must NOT
      send it.
    - "openai": Nemotron-3, gpt-oss, OpenAI o-series read `reasoning_effort`.
    """
    n = (served_name or "").lower()
    if n.startswith("qwen/") or "/qwen" in n or n.startswith("qwen"):
        return "qwen"
    return "openai"


def _apply_reasoning_mode(body: dict, mode: str | None, api_style: str) -> None:
    """Inject reasoning-control fields into the request body.

    Two mutually-exclusive API styles (litellm validates strictly, so we can't
    send both):
      - api_style="qwen": `chat_template_kwargs.enable_thinking` (Qwen3+).
      - api_style="openai": `reasoning_effort` (Nemotron-3, gpt-oss, OpenAI).

    Modes:
      - "off": disable thinking. For Qwen → enable_thinking=False; for the
        OpenAI path → reasoning_effort="low" (closest to off across vendors).
      - "low"/"medium"/"high": passed through as reasoning_effort on the
        OpenAI path; for Qwen, treated as enable_thinking=True (graded levels
        aren't supported by Qwen3 templates).
      - None: untouched.
    """
    if mode is None:
        return
    if api_style == "qwen":
        body["chat_template_kwargs"] = {"enable_thinking": mode != "off"}
        return
    body["reasoning_effort"] = "low" if mode == "off" else mode


def openai_chat(
    base_url: str,
    headers: dict[str, str],
    served_name: str,
    system: str | None,
    user: str,
    params: LLMParams,
    timeout: float = 300.0,
    max_retries: int = 2,
    backoff_base: float = 2.0,
    reasoning_mode: str | None = None,
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
    }
    if params.max_tokens is not None:
        # Most endpoints (vLLM, TGI, older OpenAI) use `max_tokens`.
        # Newer OpenAI models require `max_completion_tokens` instead.
        # Try `max_tokens` first; on 400 retry with the new name.
        body["max_tokens"] = params.max_tokens
    if params.seed is not None:
        body["seed"] = params.seed
    _apply_reasoning_mode(body, reasoning_mode, _detect_reasoning_api(served_name))
    url = f"{base_url.rstrip('/')}/chat/completions"

    # Retry loop: transient network errors and 5xx/429 are retried with
    # exponential backoff. 4xx (other than 429) fail fast.
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=body)
                if r.status_code == 400 and "max_tokens" in body and "max_tokens" in (r.text or ""):
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
            msg = data["choices"][0]["message"]
            # vLLM reasoning parsers (nemotron_v3, deepseek_r1, ...) split
            # output into `content` (final answer) and `reasoning_content`
            # (chain-of-thought). Carry both: callers decide whether the
            # reasoning is a usable fallback (single-shot scoring) or must
            # be discarded (multi-turn dialog history).
            usage = data.get("usage") or {}
            return Completion(
                text=msg.get("content") or "",
                reasoning_text=msg.get("reasoning_content") or "",
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
    timeout: float = 300.0,
    max_retries: int = 2,
    backoff_base: float = 2.0,
    reasoning_mode: str | None = None,
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
    }
    if params.max_tokens is not None:
        body["max_tokens"] = params.max_tokens
    if params.seed is not None:
        body["seed"] = params.seed
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    _apply_reasoning_mode(body, reasoning_mode, _detect_reasoning_api(served_name))
    url = f"{base_url.rstrip('/')}/chat/completions"

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=body)
                if r.status_code == 400 and "max_tokens" in body and "max_tokens" in (r.text or ""):
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
