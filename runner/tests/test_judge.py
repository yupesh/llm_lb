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


def test_aggregate_normalises():
    preds = [
        SamplePrediction(id="1", prediction="p", expected="e", correct=False, latency_ms=0, judge_raw_score=5),
        SamplePrediction(id="2", prediction="p", expected="e", correct=False, latency_ms=0, judge_raw_score=0),
        SamplePrediction(id="3", prediction="p", expected="e", correct=False, latency_ms=0, judge_raw_score=None),
    ]
    # Average of normalised [1.0, 0.0] -> 0.5
    assert judge_mod.aggregate(preds, SPEC) == 0.5
