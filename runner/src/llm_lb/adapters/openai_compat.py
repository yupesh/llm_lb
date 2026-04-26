"""Adapter for any OpenAI-chat-compatible endpoint: vLLM, TGI, llama.cpp,
LM Studio, OpenRouter, Together, Groq, etc."""
from __future__ import annotations

import os

from ..models import LLMParams, ModelCard
from .base import Completion, register
from .openai_like import openai_chat, openai_chat_messages, resolve_served_name


@register("openai_compat")
class OpenAICompatAdapter:
    def __init__(self, model: ModelCard) -> None:
        self.model = model
        base_url = model.endpoint_url
        if model.endpoint_url_env:
            base_url = os.environ.get(model.endpoint_url_env, "") or base_url
        if not base_url:
            raise RuntimeError(
                f"openai_compat adapter: model {model.model_id!r} requires `endpoint_url`"
                f" or `endpoint_url_env`"
            )
        # Local endpoints often don't need a key — only set Authorization if we have one.
        env_var = model.api_key_env or "LLM_LB_API_KEY"
        api_key = os.environ.get(env_var, "")
        self.headers: dict[str, str] = (
            {"Authorization": f"Bearer {api_key}"} if api_key else {}
        )
        self.base_url = base_url
        self.timeout = float(os.environ.get("LLM_LB_HTTP_TIMEOUT", "300"))
        self.served_name = resolve_served_name(model)

    def chat(self, system: str | None, user: str, params: LLMParams) -> Completion:
        return openai_chat(
            self.base_url,
            self.headers,
            self.served_name,
            system,
            user,
            params,
            self.timeout,
            reasoning_mode=params.reasoning_mode or self.model.reasoning_mode,
        )

    def chat_messages(
        self,
        messages: list[dict],
        params: LLMParams,
        tools: list[dict] | None = None,
    ) -> dict:
        return openai_chat_messages(
            self.base_url,
            self.headers,
            self.served_name,
            messages,
            params,
            tools=tools,
            timeout=self.timeout,
            reasoning_mode=params.reasoning_mode or self.model.reasoning_mode,
        )
