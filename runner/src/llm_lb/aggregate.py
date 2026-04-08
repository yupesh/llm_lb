from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .feed import write_feed
from .models import Leaderboard, LeaderboardEntry, ModelCard, RunResult, TaskSpec

# Fields that change on every aggregate run and must be ignored when deciding
# whether the on-disk file is stale. Without this guard, every CI run would see
# a fresh timestamp and report `data/index.json` as out of date.
_VOLATILE_KEYS = ("updated_at", "generated_at")


def _write_if_changed(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON only if it differs from what's already on disk (ignoring
    `updated_at` / `generated_at`). Keeps the existing timestamp when content
    is unchanged so re-running `aggregate` is a no-op for git."""
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict):
            existing_cmp = {k: v for k, v in existing.items() if k not in _VOLATILE_KEYS}
            new_cmp = {k: v for k, v in payload.items() if k not in _VOLATILE_KEYS}
            if existing_cmp == new_cmp:
                return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_task(task_dir: Path) -> TaskSpec:
    return TaskSpec.model_validate(yaml.safe_load((task_dir / "task.yaml").read_text()))


def aggregate_task(task_dir: Path) -> Leaderboard:
    """Pick the best run per model for one task and write `leaderboard.json`."""
    task = _load_task(task_dir)
    primary = task.metric.primary
    results_dir = task_dir / "results"

    best: dict[str, tuple[RunResult, str]] = {}
    if results_dir.exists():
        for p in sorted(results_dir.glob("*.json")):
            r = RunResult.model_validate_json(p.read_text())
            score = r.metrics.get(primary, 0.0)
            cur = best.get(r.model_id)
            if cur is None or score > cur[0].metrics.get(primary, 0.0):
                best[r.model_id] = (r, p.name)

    entries = [
        LeaderboardEntry(
            model_id=mid,
            score=r.metrics.get(primary, 0.0),
            metrics=r.metrics,
            tps=r.tps,
            p95_latency_ms=r.p95_latency_ms,
            cost_usd=r.cost_usd,
            result_file=f"results/{fname}",
        )
        for mid, (r, fname) in best.items()
    ]
    entries.sort(key=lambda e: (-e.score, -e.tps))

    lb = Leaderboard(
        task_id=task.name,
        task_version=task.version,
        primary_metric=primary,
        ranking=entries,
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    _write_if_changed(task_dir / "leaderboard.json", lb.model_dump(mode="json"))
    return lb


def aggregate_all(repo_root: Path) -> dict[str, Any]:
    """Rebuild every per-task leaderboard and the global `data/index.json`."""
    tasks_dir = repo_root / "tasks"
    models_dir = repo_root / "models"

    tasks_meta: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []

    if tasks_dir.exists():
        for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
            if not (task_dir / "task.yaml").exists():
                continue
            lb = aggregate_task(task_dir)
            task = _load_task(task_dir)
            n_samples = len(list((task_dir / "samples").glob("*.json")))
            tasks_meta.append(
                {
                    "id": task.name,
                    "version": task.version,
                    "primary_metric": task.metric.primary,
                    "secondary_metrics": task.metric.secondary,
                    "n_samples": n_samples,
                }
            )
            for entry in lb.ranking:
                matrix.append(
                    {
                        "task_id": task.name,
                        "task_version": task.version,
                        "model_id": entry.model_id,
                        "score": entry.score,
                        "metrics": entry.metrics,
                        "tps": entry.tps,
                        "p95_latency_ms": entry.p95_latency_ms,
                        "cost_usd": entry.cost_usd,
                        "result_file": f"tasks/{task_dir.name}/{entry.result_file}",
                    }
                )

    models_meta: list[dict[str, Any]] = []
    if models_dir.exists():
        for p in sorted(models_dir.glob("*.yaml")):
            m = ModelCard.model_validate(yaml.safe_load(p.read_text()))
            models_meta.append(
                {
                    "id": m.model_id,
                    "display_name": m.display_name,
                    "provider": m.provider,
                    "hf_uri": m.hf_uri,
                    "hardware": m.hardware.model_dump() if m.hardware else None,
                }
            )

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": tasks_meta,
        "models": models_meta,
        "matrix": matrix,
    }
    _write_if_changed(repo_root / "data" / "index.json", index)
    write_feed(repo_root)
    return index
