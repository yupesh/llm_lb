"""Deterministic heuristic adapter — used by CI smoke tests so the pipeline
can be exercised end-to-end without any API keys or network access."""
from __future__ import annotations

from ..models import LLMParams, ModelCard
from .base import Completion, register

_POSITIVE = ("love", "great", "amazing", "awesome", "good", "excellent", "happy", "best", "fantastic")
_NEGATIVE = ("hate", "bad", "terrible", "awful", "worst", "horrible", "sad", "broken", "waste")


@register("dummy")
class DummyAdapter:
    def __init__(self, model: ModelCard) -> None:
        self.model = model

    def chat(self, system: str | None, user: str, params: LLMParams) -> Completion:
        text = user.lower()
        score = sum(m in text for m in _POSITIVE) - sum(m in text for m in _NEGATIVE)
        label = "positive" if score >= 0 else "negative"
        return Completion(text=label, output_tokens=1)
