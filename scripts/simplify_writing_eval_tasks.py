#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "tasks"
CEFR_REGEX = r"\b(A1|A2|B1|B2|C1|C2)\b"
TARGET_VERSION = "1.1"


def compact_system_prompt(labels: list[str]) -> str:
    allowed = "|".join(labels)
    return (
        "You are an expert CEFR writing assessor.\n"
        "Assess the student text using the provided task/context blocks as supporting evidence.\n"
        "Use EXPERT_ANNOTATION, CALIBRATION_EXAMPLES, RETRIEVED_ERROR_SPANS, and RANKING_ANCHORS only as supporting context; "
        "never override direct evidence from the student text.\n"
        f"Return ONLY one CEFR label from: {allowed}.\n"
        "Do not output JSON.\n"
        "Do not output explanation."
    )


def extract_cefr_label(text: str) -> str:
    match = re.search(CEFR_REGEX, text)
    if match is None:
        raise ValueError(f"Could not extract CEFR label from assistant example: {text[:160]!r}")
    return match.group(1)


def simplify_messages(messages: list[dict], labels: list[str]) -> list[dict]:
    simplified: list[dict] = []
    first_system = True
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str):
            simplified.append(msg)
            continue
        if role == "system":
            if first_system:
                simplified.append({"role": "system", "content": compact_system_prompt(labels)})
                first_system = False
            continue
        if role == "assistant":
            simplified.append({"role": "assistant", "content": extract_cefr_label(content)})
            continue
        simplified.append(msg)
    return simplified


def main() -> None:
    task_dirs = sorted(TASKS_DIR.glob("writing_eval_*"))
    if not task_dirs:
        raise SystemExit("No writing_eval_* tasks found")

    for task_dir in task_dirs:
        task_path = task_dir / "task.yaml"
        task = yaml.safe_load(task_path.read_text())
        task["version"] = TARGET_VERSION
        task["description"] = (
            (task.get("description") or "").rstrip(".")
            + " Simplified in v1.1 to require label-only CEFR outputs and compact few-shot assistant examples."
        )
        task_path.write_text(yaml.safe_dump(task, sort_keys=False))

        for sample_path in sorted((task_dir / "samples").glob("*.json")):
            sample = json.loads(sample_path.read_text())
            messages = sample.get("meta", {}).get("messages")
            if isinstance(messages, list):
                sample["meta"]["messages"] = simplify_messages(messages, task["labels"])
            sample_path.write_text(json.dumps(sample, indent=2) + "\n")

        print(f"updated {task_dir.name}")


if __name__ == "__main__":
    main()
