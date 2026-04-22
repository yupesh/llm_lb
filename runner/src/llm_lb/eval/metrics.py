from __future__ import annotations

from collections import defaultdict

from ..models import SamplePrediction

_CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def accuracy(preds: list[SamplePrediction]) -> float:
    if not preds:
        return 0.0
    return sum(1 for p in preds if p.correct) / len(preds)


def exact_match(preds: list[SamplePrediction]) -> float:
    """Same as accuracy but conceptually distinct: meant for free-form answers
    after normalisation (see `eval.extract.normalize`). The runner sets
    `correct` based on the normalised comparison, so the aggregator just
    averages it.
    """
    return accuracy(preds)


def macro_f1(preds: list[SamplePrediction], labels: list[str]) -> float:
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for p in preds:
        if p.prediction == p.expected:
            tp[p.expected] += 1
        else:
            fp[p.prediction] += 1
            fn[p.expected] += 1
    f1s: list[float] = []
    for label in labels:
        denom_p = tp[label] + fp[label]
        denom_r = tp[label] + fn[label]
        prec = tp[label] / denom_p if denom_p else 0.0
        rec = tp[label] / denom_r if denom_r else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _label_indices(labels: list[str]) -> dict[str, int]:
    canonical = {lab.strip().upper(): i for i, lab in enumerate(_CEFR_ORDER)}
    if labels and all(lab.strip().upper() in canonical for lab in labels):
        return {lab.lower(): i for i, lab in enumerate(_CEFR_ORDER)}
    return {lab.strip().lower(): i for i, lab in enumerate(labels)}


def adjacent_accuracy(preds: list[SamplePrediction], labels: list[str]) -> float:
    """Ordinal accuracy with +/-1 tolerance.

    Used by the CEFR writing-eval reports where near-misses (e.g. B1 vs B2)
    are materially better than boundary violations several levels away.
    Unrecognised predictions count as misses.
    """
    if not preds:
        return 0.0
    label_to_idx = _label_indices(labels)
    ok = 0
    for p in preds:
        ti = label_to_idx.get(p.expected.strip().lower())
        pi = label_to_idx.get(p.prediction.strip().lower())
        if ti is None or pi is None:
            continue
        if abs(pi - ti) <= 1:
            ok += 1
    return ok / len(preds)


def qwk(preds: list[SamplePrediction], labels: list[str]) -> float:
    """Quadratic Weighted Kappa for ordinal classification.

    Position in `labels` defines the ordinal rank (e.g. ['1','2','3','4','5','6']
    or ['A2','B1','B2','C1']). Penalty grows as the squared distance between
    predicted and true rank, so "5 vs gold 6" hurts less than "1 vs gold 6".
    Returns 1.0 for perfect agreement, 0.0 for chance agreement, negative when
    worse than chance. The standard ASAP essay-scoring metric.

    Unrecognised predictions (model output not in `labels`) are treated as the
    label *farthest* from the gold rank — maximum ordinal penalty — so models
    that fail to follow the rubric cannot quietly slip past the metric.
    """
    scoring_labels = _CEFR_ORDER if labels and all(lab.strip().upper() in _CEFR_ORDER for lab in labels) else labels
    n = len(scoring_labels)
    if n < 2:
        return 1.0
    label_to_idx = {lab.strip().lower(): i for i, lab in enumerate(scoring_labels)}

    confusion = [[0] * n for _ in range(n)]
    for p in preds:
        ti = label_to_idx.get(p.expected.strip().lower())
        if ti is None:
            continue  # gold outside declared labels — skip defensively
        pi = label_to_idx.get(p.prediction.strip().lower())
        if pi is None:
            pi = 0 if ti > (n - 1) / 2 else n - 1
        confusion[ti][pi] += 1

    total = sum(sum(row) for row in confusion)
    if total == 0:
        return 0.0
    row_sum = [sum(confusion[i]) for i in range(n)]
    col_sum = [sum(confusion[i][j] for i in range(n)) for j in range(n)]

    max_w = (n - 1) ** 2
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(n):
            w = ((i - j) ** 2) / max_w
            numerator += w * confusion[i][j]
            denominator += w * row_sum[i] * col_sum[j] / total

    if denominator == 0.0:
        # All observations in a single class — no ordinal signal to disagree on.
        return 1.0
    return 1.0 - numerator / denominator


def signed_diff(preds: list[SamplePrediction], labels: list[str]) -> float:
    """Mean ordinal(prediction - gold).

    Negative values mean the model tends to undershoot the gold label, positive
    values mean it tends to overshoot. Used in the B1/B2 boundary reports to
    expose systematic `B2 -> B1` bias.
    """
    if not preds:
        return 0.0
    label_to_idx = _label_indices(labels)
    total = 0.0
    count = 0
    for p in preds:
        ti = label_to_idx.get(p.expected.strip().lower())
        pi = label_to_idx.get(p.prediction.strip().lower())
        if ti is None or pi is None:
            continue
        total += pi - ti
        count += 1
    return total / count if count else 0.0


def _clip_to_boundary(label: str, low: str, high: str, label_to_idx: dict[str, int]) -> str | None:
    key = label.strip().lower()
    low_key = low.strip().lower()
    high_key = high.strip().lower()
    idx = label_to_idx.get(key)
    low_idx = label_to_idx.get(low_key)
    high_idx = label_to_idx.get(high_key)
    if idx is None or low_idx is None or high_idx is None:
        return None
    if idx <= low_idx:
        return low_key
    if idx >= high_idx:
        return high_key
    return key


def boundary_accuracy(preds: list[SamplePrediction], labels: list[str]) -> float:
    """Binary boundary accuracy after clipping off-target labels.

    For B1/B2 holdout tasks, predictions such as A2 and C1 are clipped to the
    nearest side of the boundary (A2 -> B1, C1 -> B2) before scoring.
    """
    if not preds or len(labels) != 2:
        return 0.0
    low, high = labels
    label_to_idx = _label_indices(_CEFR_ORDER if all(l.upper() in _CEFR_ORDER for l in labels) else labels)
    ok = 0
    for p in preds:
        clipped = _clip_to_boundary(p.prediction, low, high, label_to_idx)
        if clipped is None:
            continue
        if clipped.lower() == p.expected.strip().lower():
            ok += 1
    return ok / len(preds)


def boundary_kappa(preds: list[SamplePrediction], labels: list[str]) -> float:
    """Cohen's kappa on the clipped binary boundary labels."""
    if not preds or len(labels) != 2:
        return 0.0
    low, high = labels
    label_to_idx = _label_indices(_CEFR_ORDER if all(l.upper() in _CEFR_ORDER for l in labels) else labels)
    confusion = [[0, 0], [0, 0]]
    for p in preds:
        ti = 0 if p.expected.strip().lower() == low.strip().lower() else 1
        clipped = _clip_to_boundary(p.prediction, low, high, label_to_idx)
        if clipped is None:
            continue
        pi = 0 if clipped == low.strip().lower() else 1
        confusion[ti][pi] += 1

    total = sum(sum(row) for row in confusion)
    if total == 0:
        return 0.0
    po = (confusion[0][0] + confusion[1][1]) / total
    row_sum = [sum(confusion[i]) for i in range(2)]
    col_sum = [sum(confusion[i][j] for i in range(2)) for j in range(2)]
    pe = sum(row_sum[i] * col_sum[i] for i in range(2)) / (total * total)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)
