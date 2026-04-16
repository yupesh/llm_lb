from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import __version__
from .adapters import get_adapter
from .eval import judge as judge_mod
from .eval.dialog_sim import simulate_retail_dialog
from .eval.extract import extract_label, extract_regex, normalize
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


def _extract_prediction(task: TaskSpec, raw: str) -> str:
    if task.labels:
        return extract_label(raw, task.labels)
    if task.answer_regex:
        return extract_regex(raw, task.answer_regex)
    return raw.strip()


def _is_correct(task: TaskSpec, prediction: str, expected: str) -> bool:
    if task.labels:
        # Classification tasks: simple case-insensitive equality.
        return prediction.strip().lower() == expected.strip().lower()
    # Free-form: SQuAD-style normalisation, used by exact_match metric.
    return normalize(prediction) == normalize(expected)


def _compute_cost_usd(model: ModelCard, in_tokens: int, out_tokens: int) -> float | None:
    if model.prompt_cost_per_1k_usd is None or model.completion_cost_per_1k_usd is None:
        return None
    return (
        in_tokens * model.prompt_cost_per_1k_usd / 1000.0
        + out_tokens * model.completion_cost_per_1k_usd / 1000.0
    )


def run(
    task_dir: Path,
    model_path: Path,
    out_dir: Path | None = None,
    limit: int | None = None,
) -> Path:
    task, samples = load_task(task_dir)
    if limit is not None and limit > 0:
        samples = samples[:limit]
    model = load_model(model_path)
    adapter = get_adapter(model)
    template_hash = _hash_template(task.system_prompt, task.prompt_template)

    # Resolve repo root from the task path so the judge can find model cards.
    repo_root = task_dir.resolve().parent.parent
    judge_client = None
    judge_card: ModelCard | None = None
    if task.judge:
        if task.judge.forbid_self_judge and task.judge.model == model.model_id:
            raise RuntimeError(
                f"Self-judge bias guard: candidate {model.model_id!r} is also the "
                f"judge. Set `judge.forbid_self_judge: false` in task.yaml to override."
            )
        judge_client, judge_card = judge_mod.build_judge(repo_root, task.judge)

    preds: list[SamplePrediction] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_gen_time_s = 0.0

    # Preload simulator artifacts once (policy, user-agent prompt, db path).
    sim_policy: str | None = None
    sim_user_prompt: str | None = None
    sim_db_path: Path | None = None
    if task.runner_kind == "dialog_simulation":
        sim_policy = (task_dir / "policy.md").read_text()
        sim_user_prompt = (task_dir / "user_agent_prompt.md").read_text()
        sim_db_path = task_dir / "db.json"

    # Preload external per-sample context (e.g. DB schemas) once. The file is
    # optional — if it's missing, {context} substitutes to an empty string so
    # local runs without the external artifact still work.
    context_data: dict[str, str] = {}
    if task.context_file:
        ctx_path = task_dir / task.context_file
        if ctx_path.exists():
            context_data = json.loads(ctx_path.read_text())

    for s in samples:
        t0 = time.perf_counter()
        if task.runner_kind == "dialog_simulation":
            pred, _trace, in_tok, out_tok = simulate_retail_dialog(
                adapter,
                policy_prompt=sim_policy or "",
                user_agent_prompt=sim_user_prompt or "",
                db_path=sim_db_path,  # type: ignore[arg-type]
                sample_prompt_json=s.prompt,
                params=task.llm_params,
                max_turns=task.max_turns or 10,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            correct = False  # judge decides; `correct` is unused for judge_score tasks
            sample_pred = SamplePrediction(
                id=s.id,
                prediction=pred,
                expected=s.expected,
                correct=correct,
                latency_ms=dt_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            if task.judge and judge_client is not None:
                sample_pred.judge_raw_score = judge_mod.score_sample(
                    judge_client, task.judge, s, pred
                )
            preds.append(sample_pred)
            if out_tok:
                total_output_tokens += out_tok
                total_gen_time_s += dt_ms / 1000.0
            if in_tok:
                total_input_tokens += in_tok
            continue

        ctx = ""
        if task.context_meta_key and context_data:
            ctx_key = s.meta.get(task.context_meta_key)
            if ctx_key:
                ctx = context_data.get(ctx_key, "")
        user = task.prompt_template.format(prompt=s.prompt, context=ctx)
        completion = adapter.chat(task.system_prompt, user, task.llm_params)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        pred = _extract_prediction(task, completion.text)
        correct = _is_correct(task, pred, s.expected)

        sample_pred = SamplePrediction(
            id=s.id,
            prediction=pred,
            expected=s.expected,
            correct=correct,
            latency_ms=dt_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )

        if task.judge and judge_client is not None:
            sample_pred.judge_raw_score = judge_mod.score_sample(
                judge_client, task.judge, s, pred, context=ctx
            )

        preds.append(sample_pred)
        if completion.output_tokens:
            total_output_tokens += completion.output_tokens
            total_gen_time_s += dt_ms / 1000.0
        if completion.input_tokens:
            total_input_tokens += completion.input_tokens

    # Metrics dispatch: pick whatever the task asks for. We always compute the
    # primary metric, then any secondaries we know how to compute cheaply.
    metrics: dict[str, float] = {}
    primary = task.metric.primary
    wanted = {primary, *task.metric.secondary}

    if task.judge:
        metrics["judge_score"] = judge_mod.aggregate(preds, task.judge)
    if "accuracy" in wanted or (task.labels and primary == "accuracy"):
        metrics["accuracy"] = accuracy(preds)
    if "exact_match" in wanted:
        metrics["exact_match"] = accuracy(preds)  # `correct` already uses normalize()
    if "macro_f1" in wanted and task.labels:
        metrics["macro_f1"] = macro_f1(preds, task.labels)

    tps = (total_output_tokens / total_gen_time_s) if total_gen_time_s > 0 else 0.0
    p95 = _p95([p.latency_ms for p in preds])
    cost_usd = _compute_cost_usd(model, total_input_tokens, total_output_tokens)

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
        cost_usd=cost_usd,
        judge_model_id=judge_card.model_id if judge_card else None,
        judge_model_revision=judge_card.revision if judge_card else None,
        judge_prompt_hash=judge_mod.hash_judge_prompt(task.judge.prompt) if task.judge else None,
    )

    out_dir = out_dir or (task_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = model.model_id.replace("/", "_")
    safe_ts = result.created_at.replace(":", "-").replace("+00-00", "Z")
    out_path = out_dir / f"{safe_id}__{safe_ts}.json"
    out_path.write_text(result.model_dump_json(indent=2))
    return out_path
