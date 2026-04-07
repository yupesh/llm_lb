"""Adapter for any OpenAI-chat-compatible endpoint: vLLM, TGI, llama.cpp,
LM Studio, OpenRouter, Together, Groq, etc."""
from __future__ import annotations

import os

from ..models import LLMParams, ModelCard
from .base import Completion, register
from .openai_like import openai_chat


def _served_name(model: ModelCard) -> str:
    if model.served_model_name:
        return model.served_model_name
    if model.hf_uri:
        return model.hf_uri
    return model.model_id.split("@", 1)[0]


@register("openai_compat")
class OpenAICompatAdapter:
    def __init__(self, model: ModelCard) -> None:
        self.model = model
        if not model.endpoint_url:
            raise RuntimeError(
                f"openai_compat adapter: model {model.model_id!r} requires `endpoint_url`"
            )
        # Local endpoints often don't need a key — only set Authorization if we have one.
        env_var = model.api_key_env or "LLM_LB_API_KEY"
        api_key = os.environ.get(env_var, "")
        self.headers: dict[str, str] = (
            {"Authorization": f"Bearer {api_key}"} if api_key else {}
        )
        self.base_url = model.endpoint_url
        self.timeout = float(os.environ.get("LLM_LB_HTTP_TIMEOUT", "120"))
        self.served_name = _served_name(model)

    def chat(self, system: str | None, user: str, params: LLMParams) -> Completion:
        return openai_chat(
            self.base_url, self.headers, self.served_name, system, user, params, self.timeout
        )
