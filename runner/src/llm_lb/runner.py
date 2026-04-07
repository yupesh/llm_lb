from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import __version__
from .adapters import get_adapter
from .eval.extract import extract_label
from .eval.metrics import accuracy, macro_f1
from .models import ModelCard, RunResult, Sample, SamplePrediction, TaskSpec


def _hash_template(system: str | None, template: str) -> str:
    payload = (system or "") + "\n---\n" + template
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_task(task_dir: Path) -> tuple[TaskSpec, list[Sample]]:
    task = TaskSpec.model_validate(yaml.safe_load((task_dir / "task.yaml").read_text()))
    samples = [
        Sample.model_validate_json(p.read_text())
        for p in sorted((task_dir / "samples").glob("*.json"))
    ]
    return task, samples


def load_model(model_path: Path) -> ModelCard:
    return ModelCard.model_validate(yaml.safe_load(model_path.read_text()))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return s[idx]


def run(task_dir: Path, model_path: Path, out_dir: Path | None = None) -> Path:
    task, samples = load_task(task_dir)
    model = load_model(model_path)
    adapter = get_adapter(model)
    template_hash = _hash_template(task.system_prompt, task.prompt_template)

    preds: list[SamplePrediction] = []
    total_output_tokens = 0
    total_gen_time_s = 0.0

    for s in samples:
        user = task.prompt_template.format(prompt=s.prompt)
        t0 = time.perf_counter()
        completion = adapter.chat(task.system_prompt, user, task.llm_params)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if task.labels:
            pred = extract_label(completion.text, task.labels)
        else:
            pred = completion.text.strip()
        correct = pred.strip().lower() == s.expected.strip().lower()

        preds.append(
            SamplePrediction(
                id=s.id,
                prediction=pred,
                expected=s.expected,
                correct=correct,
                latency_ms=dt_ms,
                output_tokens=completion.output_tokens,
            )
        )
        if completion.output_tokens:
            total_output_tokens += completion.output_tokens
            total_gen_time_s += dt_ms / 1000.0

    metrics: dict[str, float] = {"accuracy": accuracy(preds)}
    if task.labels:
        metrics["macro_f1"] = macro_f1(preds, task.labels)

    tps = (total_output_tokens / total_gen_time_s) if total_gen_time_s > 0 else 0.0
    p95 = _p95([p.latency_ms for p in preds])

    result = RunResult(
        model_id=model.model_id,
        model_revision=model.revision,
        task_id=task.name,
        task_version=task.version,
        runner_version=__version__,
        prompt_template_hash=template_hash,
        llm_params=task.llm_params,
        metrics=metrics,
        tps=tps,
        p95_latency_ms=p95,
        n_samples=len(samples),
        samples=preds,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        env={"provider": model.provider, "endpoint_kind": model.endpoint_kind},
    )

    out_dir = out_dir or (task_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = model.model_id.replace("/", "_")
    safe_ts = result.created_at.replace(":", "-").replace("+00-00", "Z")
    out_path = out_dir / f"{safe_id}__{safe_ts}.json"
    out_path.write_text(result.model_dump_json(indent=2))
    return out_path
