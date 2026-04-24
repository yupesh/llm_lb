"""LLM-as-Judge: use a second LLM to score open-ended answers.

Design notes
------------
- The judge is just another `LLMClient`, instantiated from a normal model card
  in `models/`. This means judges are first-class artifacts in the repo:
  pinned `revision`, visible in the leaderboard, swap-able.
- We force temperature=0 on the judge call regardless of the task's
  `llm_params`, so judging is as deterministic as the provider allows.
- The judge prompt must produce a single integer in [scale_min, scale_max].
  We extract it with a regex; if extraction fails the sample is scored as
  the minimum (so flaky judges hurt scores rather than silently passing).
- Aggregate `judge_score` is normalised to [0, 1] so it composes naturally
  with accuracy / F1 in the global compare-view.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from ..adapters import get_adapter
from ..adapters.base import LLMClient
from ..adapters.openai_like import resolve_served_name
from ..models import JudgeSpec, LLMParams, ModelCard, Sample, SamplePrediction


def hash_judge_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _find_model_card(repo_root: Path, model_id: str) -> ModelCard:
    """Resolve a `judge.model` reference (e.g. 'gpt-4o-mini@openai') to its
    YAML card. Searches `models/` for a file whose `model_id` matches."""
    models_dir = repo_root / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"judge: models/ directory not found at {models_dir}")
    for p in sorted(models_dir.glob("*.yaml")):
        card = ModelCard.model_validate(yaml.safe_load(p.read_text()))
        if card.model_id == model_id:
            return card
    raise ValueError(
        f"judge: model_id {model_id!r} not found under {models_dir}. "
        f"Add a model card with that id."
    )


def build_judge(
    repo_root: Path, spec: JudgeSpec
) -> tuple[LLMClient, ModelCard, str]:
    """Instantiate the judge adapter and return it along with the resolved
    served model name for audit. Raises on served-model-name mismatch so
    silent judge swaps (e.g. JUDGE_MODEL_NAME drifting to a different model
    between CI configurations) fail loudly at run start rather than producing
    bogus scores or 404s mid-run."""
    card = _find_model_card(repo_root, spec.model)
    served = resolve_served_name(card)
    # Fully env-driven judge cards have no hardcoded identity: task.yaml
    # MUST declare what model it expects, otherwise there is no audit trail
    # linking a result file to the model that produced its scores.
    if card.served_model_name_env and not spec.served_model_name:
        raise RuntimeError(
            f"judge: model card {card.model_id!r} resolves its served name "
            f"from env var {card.served_model_name_env!r}, so the task must "
            f"declare an expected `judge.served_model_name` to validate "
            f"against. Add it to task.yaml."
        )
    if spec.served_model_name and spec.served_model_name != served:
        raise RuntimeError(
            f"judge: served model name mismatch. task.yaml expects "
            f"{spec.served_model_name!r} but the judge adapter resolved "
            f"{served!r} (card={card.model_id!r}). Either update task.yaml "
            f"or fix the env override."
        )
    return get_adapter(card), card, served


_INT_RE = re.compile(r"-?\d+")


def score_sample(
    judge_client: LLMClient,
    spec: JudgeSpec,
    sample: Sample,
    prediction: str,
    context: str = "",
) -> float:
    """Run the judge on one sample, return the *raw* integer score
    (clamped into [scale_min, scale_max]). Returns scale_min on parse failure.

    `context` is the per-sample extra context from `task.context_file` (see
    TaskSpec). Passed to the judge prompt as `{context}` so judges can also
    see e.g. a DB schema when evaluating free-form outputs.
    """
    # Empty prediction is almost always a truncation / crash signal (model
    # burned its max_tokens budget on hidden reasoning, or errored before
    # emitting anything). Judges have been observed to answer "5" for an
    # empty candidate because the rubric gives them nothing wrong to latch
    # onto — which silently inflates scores. Short-circuit to scale_min
    # instead of calling the judge.
    if not (prediction or "").strip():
        return float(spec.scale_min)

    user = spec.prompt.format(
        prompt=sample.prompt,
        expected=sample.expected,
        prediction=prediction,
        context=context,
    )
    completion = judge_client.chat(
        system=None,
        user=user,
        params=LLMParams(temperature=0.0, max_tokens=400),
    )
    m = _INT_RE.search(completion.text or "")
    if not m:
        return float(spec.scale_min)
    raw = int(m.group(0))
    return float(max(spec.scale_min, min(spec.scale_max, raw)))


def normalize(raw: float, spec: JudgeSpec) -> float:
    span = spec.scale_max - spec.scale_min
    if span <= 0:
        return 0.0
    return (raw - spec.scale_min) / span


def aggregate(preds: list[SamplePrediction], spec: JudgeSpec) -> float:
    """Average normalised judge score across samples that have one."""
    scored = [p.judge_raw_score for p in preds if p.judge_raw_score is not None]
    if not scored:
        return 0.0
    norms = [normalize(s, spec) for s in scored]
    return sum(norms) / len(norms)
