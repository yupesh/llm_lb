from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import LLMParams, ModelCard


@dataclass
class Completion:
    text: str
    output_tokens: int | None = None
    input_tokens: int | None = None


class LLMClient(Protocol):
    def chat(self, system: str | None, user: str, params: LLMParams) -> Completion: ...


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
