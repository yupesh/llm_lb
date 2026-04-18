from llm_lb.eval.metrics import accuracy, macro_f1, qwk
from llm_lb.models import SamplePrediction


def _mk(expected: str, prediction: str) -> SamplePrediction:
    return SamplePrediction(
        id=f"{expected}-{prediction}",
        prediction=prediction,
        expected=expected,
        correct=prediction == expected,
        latency_ms=0.0,
    )


LABELS_ORD4 = ["1", "2", "3", "4"]
LABELS_CEFR = ["A2", "B1", "B2", "C1"]


def test_qwk_perfect_agreement():
    preds = [_mk(lab, lab) for lab in LABELS_ORD4 for _ in range(3)]
    assert qwk(preds, LABELS_ORD4) == 1.0


def test_qwk_maximal_disagreement_is_strongly_negative():
    # Swap extremes: every "1" predicted as "4" and every "4" as "1".
    preds = [_mk("1", "4")] * 5 + [_mk("4", "1")] * 5
    score = qwk(preds, LABELS_ORD4)
    assert score < -0.5, f"expected strongly negative, got {score}"


def test_qwk_close_misses_beat_distant_misses():
    # Both runs: 20 samples with 16 correct, 4 wrong. Same marginal class
    # distribution on the gold side, same accuracy — but the errors differ in
    # ordinal distance. QWK should reward the run whose wrong answers land
    # closer to the gold.
    def _make(misses: list[tuple[str, str]]) -> list[SamplePrediction]:
        samples = []
        for lab in LABELS_ORD4:
            samples.extend([_mk(lab, lab)] * 4)
        for gold, bad in misses:
            samples.append(_mk(gold, bad))
        return samples

    close = _make([("1", "2"), ("2", "3"), ("3", "4"), ("4", "3")])  # off by 1
    far = _make([("1", "4"), ("2", "4"), ("3", "1"), ("4", "1")])     # off by 2–3

    assert accuracy(close) == accuracy(far) == 16 / 20
    assert qwk(close, LABELS_ORD4) > qwk(far, LABELS_ORD4)


def test_qwk_single_class_returns_one():
    # All samples in one category — no ordinal signal; treat as perfect.
    preds = [_mk("3", "3")] * 10
    assert qwk(preds, LABELS_ORD4) == 1.0


def test_qwk_ignores_case_and_whitespace():
    preds = [_mk("a2", " A2 "), _mk("B1", "b1")]
    assert qwk(preds, LABELS_CEFR) == 1.0


def test_qwk_unrecognised_prediction_gets_max_penalty():
    # Gold is "1" (index 0). Unknown prediction should be mapped to the
    # farthest label ("4", index 3) — worst ordinal penalty.
    garbage = [_mk("1", "banana")] * 10
    max_wrong = [_mk("1", "4")] * 10
    assert qwk(garbage, LABELS_ORD4) == qwk(max_wrong, LABELS_ORD4)


def test_qwk_degenerate_inputs():
    assert qwk([], LABELS_ORD4) == 0.0
    assert qwk([_mk("1", "1")], ["only"]) == 1.0


def test_accuracy_and_macro_f1_still_work():
    preds = [_mk("1", "1"), _mk("2", "2"), _mk("3", "4")]
    assert accuracy(preds) == 2 / 3
    assert 0.0 <= macro_f1(preds, LABELS_ORD4) <= 1.0
