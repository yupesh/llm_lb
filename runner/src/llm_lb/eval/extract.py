from __future__ import annotations

import re

_REASONING_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(raw: str) -> str:
    """Drop `<think>...</think>` blocks emitted by reasoning models.
    Without this, `extract_label` would match whichever label is mentioned
    first inside the reasoning trace instead of the final answer.
    """
    return _REASONING_BLOCK_RE.sub("", raw).strip()


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


def extract_regex(raw: str, pattern: str) -> str:
    """Extract the answer using a regex. Returns the first capturing group if
    present, else the whole match. If nothing matches, returns the stripped
    raw text — the eval step will then mark it as incorrect.
    """
    m = re.search(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return raw.strip()
    return (m.group(1) if m.groups() else m.group(0)).strip()


def normalize(text: str) -> str:
    """Normalisation used by `exact_match`: lowercase, collapse whitespace,
    drop surrounding punctuation. Matches the SQuAD-style normaliser closely
    enough for our small benchmark tasks."""
    text = text.lower().strip()
    text = re.sub(r"^[\s\"'.,;:!?(){}\[\]]+|[\s\"'.,;:!?(){}\[\]]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
