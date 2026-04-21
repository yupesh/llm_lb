"""Adapter for Meta Llama Guard (3 / 4) safety classifier, served via an
OpenAI-compatible endpoint (vLLM).

Llama Guard is not a general-purpose chat model — it's a classifier with a
baked-in taxonomy (S1..S14). Two things set it apart from a regular
openai_compat model:

1. Its chat template rejects the `system` role. Sending the task-level
   `system_prompt` yields a server-side jinja error:
       "Conversation roles must alternate user/assistant/user/assistant/..."
   We therefore drop the `system` argument — the Guard model ignores
   instructions anyway and uses its own taxonomy.
2. Its response is fixed-format: the first line is `safe` or `unsafe`; if
   unsafe, a second line lists the violated Meta categories (e.g. `S5,S9`).
   Downstream `extract_label` needs just the first token — so we trim to
   the first non-empty line.

Apart from that, the wire protocol is identical to vLLM's OpenAI server.
"""
from __future__ import annotations

import os

from ..models import LLMParams, ModelCard
from .base import Completion, register
from .openai_like import openai_chat


def _served_name(model: ModelCard) -> str:
    if model.served_model_name_env:
        from_env = os.environ.get(model.served_model_name_env, "")
        if from_env:
            return from_env
    if model.served_model_name:
        return model.served_model_name
    if model.hf_uri:
        return model.hf_uri
    return model.model_id.split("@", 1)[0]


def _first_line(text: str) -> str:
    """Return the first non-empty line of `text`, stripped.

    Guard output is `safe` on one line or `unsafe\\nS1,S5,...` on two.
    We only care about the safe/unsafe verdict for classification metrics.
    """
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return text.strip()


@register("llama_guard")
class LlamaGuardAdapter:
    def __init__(self, model: ModelCard) -> None:
        self.model = model
        base_url = model.endpoint_url
        if model.endpoint_url_env:
            base_url = os.environ.get(model.endpoint_url_env, "") or base_url
        if not base_url:
            raise RuntimeError(
                f"llama_guard adapter: model {model.model_id!r} requires `endpoint_url`"
                f" or `endpoint_url_env`"
            )
        env_var = model.api_key_env or "LLM_LB_API_KEY"
        api_key = os.environ.get(env_var, "")
        self.headers: dict[str, str] = (
            {"Authorization": f"Bearer {api_key}"} if api_key else {}
        )
        self.base_url = base_url
        self.timeout = float(os.environ.get("LLM_LB_HTTP_TIMEOUT", "300"))
        self.served_name = _served_name(model)

    def chat(self, system: str | None, user: str, params: LLMParams) -> Completion:
        # Drop `system` — Guard's chat template only accepts alternating
        # user/assistant messages and rejects a system role outright.
        completion = openai_chat(
            self.base_url, self.headers, self.served_name, None, user, params, self.timeout
        )
        return Completion(
            text=_first_line(completion.text),
            output_tokens=completion.output_tokens,
            input_tokens=completion.input_tokens,
        )
