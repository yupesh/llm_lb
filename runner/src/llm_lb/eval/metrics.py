from __future__ import annotations

from collections import defaultdict

from ..models import SamplePrediction


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
    n = len(labels)
    if n < 2:
        return 1.0
    label_to_idx = {lab.strip().lower(): i for i, lab in enumerate(labels)}

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
