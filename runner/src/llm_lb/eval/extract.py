from __future__ import annotations


def extract_label(raw: str, labels: list[str]) -> str:
    """Pick the first task label that appears (case-insensitive) in the model output.

    Falls back to the stripped raw text if no label matches — the eval step will
    then mark the prediction as incorrect.
    """
    raw_lower = raw.lower()
    for label in labels:
        if label.lower() in raw_lower:
            return label
    return raw.strip()
