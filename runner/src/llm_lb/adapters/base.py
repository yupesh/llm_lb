from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..models import LLMParams, ModelCard


@dataclass
class Completion:
    text: str
    output_tokens: int | None = None
    input_tokens: int | None = None
    # Captured separately from `text` so callers can decide whether to use it.
    # Reasoning models (Nemotron-3, DeepSeek-R1, ...) returned via vLLM with a
    # `--reasoning-parser` flag split output into final-answer `content` and
    # chain-of-thought `reasoning_content`. Single-shot scoring treats the
    # reasoning as a fallback prediction; multi-turn dialog must NOT echo it
    # back into conversation history (would explode context).
    reasoning_text: str = ""


class LLMClient(Protocol):
    def chat(self, system: str | None, user: str, params: LLMParams) -> Completion: ...

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        params: LLMParams,
        tools: list[dict] | None = None,
    ) -> dict: ...


_REGISTRY: dict[str, type] = {}


def register(provider: str):
    def deco(cls: type) -> type:
        _REGISTRY[provider] = cls
        return cls

    return deco


def get_adapter(model: ModelCard) -> LLMClient:
    if model.provider not in _REGISTRY:
        raise ValueError(
            f"Unknown provider: {model.provider!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model.provider](model)
