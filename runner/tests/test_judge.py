from pathlib import Path

import pytest

from llm_lb.adapters.base import Completion
from llm_lb.eval import judge as judge_mod
from llm_lb.models import JudgeSpec, LLMParams, Sample, SamplePrediction


class FakeJudge:
    """Returns whatever score the test asks for, ignoring the prompt."""

    def __init__(self, score: str) -> None:
        self.score = score

    def chat(self, system, user, params: LLMParams) -> Completion:  # noqa: ARG002
        return Completion(text=self.score)


SPEC = JudgeSpec(model="x", prompt="{prompt}|{expected}|{prediction}", scale_min=0, scale_max=5)


def test_score_sample_clamps_high():
    s = Sample(id="1", prompt="q", expected="e")
    assert judge_mod.score_sample(FakeJudge("9"), SPEC, s, "p") == 5.0


def test_score_sample_clamps_low():
    s = Sample(id="1", prompt="q", expected="e")
    assert judge_mod.score_sample(FakeJudge("-3"), SPEC, s, "p") == 0.0


def test_score_sample_unparseable_returns_min():
    s = Sample(id="1", prompt="q", expected="e")
    assert judge_mod.score_sample(FakeJudge("blah"), SPEC, s, "p") == 0.0


class RaisingJudge:
    """Fails the test if it gets called — used to prove short-circuit."""

    def chat(self, system, user, params: LLMParams) -> Completion:  # noqa: ARG002
        raise AssertionError("judge should not be called for empty prediction")


def test_score_sample_empty_prediction_skips_judge():
    s = Sample(id="1", prompt="q", expected="e")
    # Empty / whitespace-only predictions short-circuit to scale_min without
    # calling the judge (guards against judges hallucinating scores when
    # max_tokens truncation eats the model's answer).
    assert judge_mod.score_sample(RaisingJudge(), SPEC, s, "") == 0.0
    assert judge_mod.score_sample(RaisingJudge(), SPEC, s, "   \n\t ") == 0.0


def _write_judge_card(
    tmp_path: Path,
    *,
    served_model_name: str | None = None,
    served_model_name_env: str | None = None,
    hf_uri: str | None = None,
) -> Path:
    """Write a minimal model card to tmp_path/models/judge.yaml."""
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    lines = [
        "model_id: judge@openai",
        "display_name: Judge",
        "provider: openai",
        "endpoint_kind: openai_chat",
        "endpoint_url: https://api.example/v1",
    ]
    if served_model_name:
        lines.append(f"served_model_name: {served_model_name}")
    if served_model_name_env:
        lines.append(f"served_model_name_env: {served_model_name_env}")
    if hf_uri:
        lines.append(f"hf_uri: {hf_uri}")
    (models / "judge.yaml").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_build_judge_requires_expected_when_env_driven(tmp_path, monkeypatch):
    """Fully env-driven judge cards have no hardcoded identity; task.yaml
    MUST pin the expected served model, otherwise scores become unauditable."""
    _write_judge_card(tmp_path, served_model_name_env="JUDGE_MODEL_NAME")
    monkeypatch.setenv("JUDGE_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    spec = JudgeSpec(model="judge@openai", prompt="x")
    with pytest.raises(RuntimeError, match="must declare an expected"):
        judge_mod.build_judge(tmp_path, spec)


def test_build_judge_served_name_mismatch_raises(tmp_path, monkeypatch):
    """If task.yaml pins a served name that disagrees with what the adapter
    will actually send, refuse to start — guards against silent JUDGE_MODEL_NAME
    drift across CI configurations."""
    _write_judge_card(tmp_path, served_model_name_env="JUDGE_MODEL_NAME")
    monkeypatch.setenv("JUDGE_MODEL_NAME", "google/gemma-4-31B-it")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    spec = JudgeSpec(model="judge@openai", prompt="x", served_model_name="gpt-4o-mini")
    with pytest.raises(RuntimeError, match="served model name mismatch"):
        judge_mod.build_judge(tmp_path, spec)


def test_build_judge_served_name_match_passes(tmp_path, monkeypatch):
    _write_judge_card(tmp_path, served_model_name_env="JUDGE_MODEL_NAME")
    monkeypatch.setenv("JUDGE_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    spec = JudgeSpec(model="judge@openai", prompt="x", served_model_name="gpt-4o-mini")
    _, card, served = judge_mod.build_judge(tmp_path, spec)
    assert card.model_id == "judge@openai"
    assert served == "gpt-4o-mini"


def test_build_judge_static_card_no_expected_is_fine(tmp_path, monkeypatch):
    """Static (non-env) judge cards already pin identity via the card itself;
    task.yaml is not required to repeat it."""
    _write_judge_card(tmp_path, served_model_name="gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    spec = JudgeSpec(model="judge@openai", prompt="x")
    _, _, served = judge_mod.build_judge(tmp_path, spec)
    assert served == "gpt-4o-mini"


def test_aggregate_normalises():
    preds = [
        SamplePrediction(id="1", prediction="p", expected="e", correct=False, latency_ms=0, judge_raw_score=5),
        SamplePrediction(id="2", prediction="p", expected="e", correct=False, latency_ms=0, judge_raw_score=0),
        SamplePrediction(id="3", prediction="p", expected="e", correct=False, latency_ms=0, judge_raw_score=None),
    ]
    # Average of normalised [1.0, 0.0] -> 0.5
    assert judge_mod.aggregate(preds, SPEC) == 0.5
