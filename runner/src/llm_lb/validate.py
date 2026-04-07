from __future__ import annotations

from pathlib import Path

import yaml

from .models import ModelCard, Sample, TaskSpec

MIN_SAMPLES = 5  # MVP threshold; tighten to 20 once we have content


class ValidationError(Exception):
    pass


def validate_task_dir(task_dir: Path) -> TaskSpec:
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.exists():
        raise ValidationError(f"{task_dir}: missing task.yaml")
    task = TaskSpec.model_validate(yaml.safe_load(task_yaml.read_text()))

    samples_dir = task_dir / "samples"
    if not samples_dir.is_dir():
        raise ValidationError(f"{task_dir}: missing samples/ directory")
    sample_files = sorted(samples_dir.glob("*.json"))
    if len(sample_files) < MIN_SAMPLES:
        raise ValidationError(
            f"{task_dir}: need at least {MIN_SAMPLES} samples, got {len(sample_files)}"
        )
    seen: set[str] = set()
    for p in sample_files:
        s = Sample.model_validate_json(p.read_text())
        if s.id in seen:
            raise ValidationError(f"{p}: duplicate sample id {s.id!r}")
        seen.add(s.id)
        if task.labels and s.expected not in task.labels:
            raise ValidationError(
                f"{p}: expected={s.expected!r} not in task labels {task.labels}"
            )
    return task


def validate_model_file(model_path: Path) -> ModelCard:
    return ModelCard.model_validate(yaml.safe_load(model_path.read_text()))
