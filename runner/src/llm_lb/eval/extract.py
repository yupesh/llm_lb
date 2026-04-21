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
    """Pick the task label that appears earliest in the model output.

    Falls back to the stripped raw text if no label matches — the eval step will
    then mark the prediction as incorrect.

    Why earliest-position rather than iteration-order: when one label is a
    substring of another (e.g. `safe` ⊂ `unsafe`), an iteration-order check
    returns whichever label was listed first regardless of what the model
    actually said. `safety_classification` scored exactly 0.500 across every
    model because "safe" matched inside every "unsafe" output. Picking the
    earliest occurrence — with longest-label wins on ties so `unsafe` beats
    `safe` when both start at position 0 — resolves this without needing a
    regex word-boundary (which breaks on labels with commas like
    `controversial_topics,politics` in `hazard_category`).
    """
    raw_lower = raw.lower()
    # (position, -length, label) — min() picks earliest occurrence, then
    # longest matching label as tiebreak (negate length so shorter compares
    # larger, i.e. loses).
    best: tuple[int, int, str] | None = None
    for label in labels:
        idx = raw_lower.find(label.lower())
        if idx < 0:
            continue
        candidate = (idx, -len(label), label)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best is not None else raw.strip()


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
