"""Shared helper for OpenAI-compatible chat completions endpoints.

Used by both the official OpenAI adapter and any self-hosted endpoint that
exposes a `/v1/chat/completions` route (vLLM, TGI in OpenAI mode, llama.cpp
server, LM Studio, OpenRouter, Together, Groq, ...).
"""
from __future__ import annotations

import httpx

from ..models import LLMParams
from .base import Completion


def openai_chat(
    base_url: str,
    headers: dict[str, str],
    served_name: str,
    system: str | None,
    user: str,
    params: LLMParams,
    timeout: float = 120.0,
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
        # Newer OpenAI models require `max_completion_tokens` instead of
        # `max_tokens`. Try the new name first; on 400 retry with the old one.
        "max_completion_tokens": params.max_tokens,
    }
    if params.seed is not None:
        body["seed"] = params.seed
    url = f"{base_url.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
        if r.status_code == 400 and "max_completion_tokens" in (r.text or ""):
            # Endpoint doesn't support new-style param — fall back.
            del body["max_completion_tokens"]
            body["max_tokens"] = params.max_tokens
            r = client.post(url, headers=headers, json=body)
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
