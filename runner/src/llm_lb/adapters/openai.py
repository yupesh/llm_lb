from __future__ import annotations

import os

from ..models import LLMParams, ModelCard
from .base import Completion, register
from .openai_like import openai_chat, openai_chat_messages, resolve_served_name

DEFAULT_BASE_URL = "https://api.openai.com/v1"


@register("openai")
class OpenAIAdapter:
    def __init__(self, model: ModelCard) -> None:
        self.model = model
        env_var = model.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(
                f"openai adapter: environment variable {env_var!r} is not set"
            )
        self.headers = {"Authorization": f"Bearer {api_key}"}
        if model.endpoint_url_env:
            self.base_url = os.environ.get(model.endpoint_url_env, "")
        if not getattr(self, "base_url", ""):
            self.base_url = model.endpoint_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.timeout = float(os.environ.get("LLM_LB_HTTP_TIMEOUT", "300"))
        self.served_name = resolve_served_name(model)

    def chat(self, system: str | None, user: str, params: LLMParams) -> Completion:
        return openai_chat(
            self.base_url, self.headers, self.served_name, system, user, params, self.timeout
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
        )
