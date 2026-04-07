from __future__ import annotations

from collections import defaultdict

from ..models import SamplePrediction


def accuracy(preds: list[SamplePrediction]) -> float:
    if not preds:
        return 0.0
    return sum(1 for p in preds if p.correct) / len(preds)


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
