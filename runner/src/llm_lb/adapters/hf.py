"""Hugging Face Inference API adapter (text-generation shape).

For dedicated HF Inference Endpoints that expose an OpenAI-compatible /v1/chat
route, use the `openai_compat` provider with `endpoint_url` set instead.
"""
from __future__ import annotations

import os

import httpx

from ..models import LLMParams, ModelCard
from .base import Completion, register


@register("hf")
class HFInferenceAdapter:
    def __init__(self, model: ModelCard) -> None:
        self.model = model
        if not model.hf_uri:
            raise RuntimeError(
                f"hf adapter: model {model.model_id!r} requires `hf_uri`"
            )
        env_var = model.api_key_env or "HF_TOKEN"
        token = os.environ.get(env_var, "")
        self.headers: dict[str, str] = (
            {"Authorization": f"Bearer {token}"} if token else {}
        )
        self.url = (
            model.endpoint_url
            or f"https://api-inference.huggingface.co/models/{model.hf_uri}"
        )
        self.timeout = float(os.environ.get("LLM_LB_HTTP_TIMEOUT", "120"))

    def chat(self, system: str | None, user: str, params: LLMParams) -> Completion:
        prompt = f"{system}\n\n{user}" if system else user
        body = {
            "inputs": prompt,
            "parameters": {
                # HF rejects temperature == 0 for sampling endpoints; clamp to a tiny value.
                "temperature": max(params.temperature, 1e-5),
                "top_p": params.top_p,
                "max_new_tokens": params.max_tokens,
                "return_full_text": False,
                **({"seed": params.seed} if params.seed is not None else {}),
            },
            "options": {"wait_for_model": True},
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(self.url, headers=self.headers, json=body)
            r.raise_for_status()
            data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            text = data[0].get("generated_text", "")
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"]
        else:
            text = str(data)
        # The text-generation Inference API does not return token counts here.
        return Completion(text=text, output_tokens=None)
