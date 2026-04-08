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
        "max_tokens": params.max_tokens,
    }
    if params.seed is not None:
        body["seed"] = params.seed
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage") or {}
    return Completion(
        text=text,
        output_tokens=usage.get("completion_tokens"),
        input_tokens=usage.get("prompt_tokens"),
    )
