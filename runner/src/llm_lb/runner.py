from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import __version__
from .adapters import get_adapter
from .adapters.base import Completion
from .eval import judge as judge_mod
from .eval.dialog_sim import simulate_retail_dialog
from .eval.extract import extract_label, extract_regex, normalize, strip_reasoning
from .eval.metrics import (
    accuracy,
    adjacent_accuracy,
    boundary_accuracy,
    boundary_kappa,
    macro_f1,
    qwk,
    signed_diff,
)
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
    raw = strip_reasoning(raw)
    if task.answer_regex:
        return extract_regex(raw, task.answer_regex)
    if task.labels:
        return extract_label(raw, task.labels, task.label_aliases)
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


def _empty_completion_error(task: TaskSpec, completion: Completion) -> str | None:
    """Detect backend responses that spent tokens but produced no visible text.

    We have seen OpenAI-compatible endpoints return `completion_tokens > 0`
    while `message.content` is empty, especially when a model burns through its
    generation budget on hidden reasoning or malformed structured output. Those
    samples should count as infra/format failures, not ordinary wrong answers.
    """
    if (completion.text or "").strip():
        return None
    if not (completion.output_tokens or 0):
        return None
    if task.llm_params.max_tokens is not None and completion.output_tokens >= task.llm_params.max_tokens:
        return (
            "empty completion text after consuming the full output token budget "
            f"({completion.output_tokens}/{task.llm_params.max_tokens})"
        )
    return f"empty completion text despite {completion.output_tokens} output tokens"


@dataclass
class _RunContext:
    """Everything `_execute_sample` needs that doesn't change across samples."""
    task: TaskSpec
    task_dir: Path
    adapter: object
    judge_client: object | None
    judge_card: ModelCard | None
    context_data: dict[str, str]
    sim_policy: str | None
    sim_user_prompt: str | None
    sim_db_path: Path | None


def _build_context(
    task: TaskSpec, task_dir: Path, model: ModelCard
) -> _RunContext:
    adapter = get_adapter(model)
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

    sim_policy: str | None = None
    sim_user_prompt: str | None = None
    sim_db_path: Path | None = None
    if task.runner_kind == "dialog_simulation":
        sim_policy = (task_dir / "policy.md").read_text()
        sim_user_prompt = (task_dir / "user_agent_prompt.md").read_text()
        sim_db_path = task_dir / "db.json"

    context_data: dict[str, str] = {}
    if task.context_file:
        ctx_path = task_dir / task.context_file
        if ctx_path.exists():
            context_data = json.loads(ctx_path.read_text())

    return _RunContext(
        task=task,
        task_dir=task_dir,
        adapter=adapter,
        judge_client=judge_client,
        judge_card=judge_card,
        context_data=context_data,
        sim_policy=sim_policy,
        sim_user_prompt=sim_user_prompt,
        sim_db_path=sim_db_path,
    )


def _execute_sample(
    ctx: _RunContext, s: Sample
) -> tuple[SamplePrediction, int, int, float]:
    """Run one sample through the adapter.

    Returns (prediction, input_tokens_to_add, output_tokens_to_add, gen_time_s_to_add).
    The totals are zero when the sample errored — they only accumulate on success.
    Per-sample errors are caught here and recorded on the returned SamplePrediction.
    """
    task = ctx.task
    t0 = time.perf_counter()
    try:
        if task.runner_kind == "dialog_simulation":
            pred, _trace, in_tok, out_tok = simulate_retail_dialog(
                ctx.adapter,
                policy_prompt=ctx.sim_policy or "",
                user_agent_prompt=ctx.sim_user_prompt or "",
                db_path=ctx.sim_db_path,  # type: ignore[arg-type]
                sample_prompt_json=s.prompt,
                params=task.llm_params,
                max_turns=task.max_turns or 10,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            sample_pred = SamplePrediction(
                id=s.id,
                prediction=pred,
                expected=s.expected,
                correct=False,  # judge decides; unused for judge_score tasks
                latency_ms=dt_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            if task.judge and ctx.judge_client is not None:
                sample_pred.judge_raw_score = judge_mod.score_sample(
                    ctx.judge_client, task.judge, s, pred
                )
            return (
                sample_pred,
                in_tok or 0,
                out_tok or 0,
                (dt_ms / 1000.0) if out_tok else 0.0,
            )

        ctx_str = ""
        if task.context_meta_key and ctx.context_data:
            ctx_key = s.meta.get(task.context_meta_key)
            if ctx_key:
                ctx_str = ctx.context_data.get(ctx_key, "")
        sample_messages = s.meta.get("messages")
        if sample_messages is not None:
            if not hasattr(ctx.adapter, "chat_messages"):
                raise RuntimeError(
                    f"Adapter for model does not support sample-level chat histories: {type(ctx.adapter).__name__}"
                )
            result = ctx.adapter.chat_messages(sample_messages, task.llm_params)
            msg = result["message"]
            usage = result.get("usage") or {}
            text = msg.get("content") or ""
            completion = Completion(
                text=text,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            )
        else:
            user = task.prompt_template.format(prompt=s.prompt, context=ctx_str)
            completion = ctx.adapter.chat(task.system_prompt, user, task.llm_params)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        empty_err = _empty_completion_error(task, completion)
        if empty_err is not None:
            return (
                SamplePrediction(
                    id=s.id,
                    prediction="",
                    expected=s.expected,
                    correct=False,
                    latency_ms=dt_ms,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    raw_output=completion.text,
                    error=empty_err,
                ),
                0,
                0,
                0.0,
            )

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
            raw_output=completion.text,
        )
        if task.judge and ctx.judge_client is not None:
            sample_pred.judge_raw_score = judge_mod.score_sample(
                ctx.judge_client, task.judge, s, pred, context=ctx_str
            )
        return (
            sample_pred,
            completion.input_tokens or 0,
            completion.output_tokens or 0,
            (dt_ms / 1000.0) if completion.output_tokens else 0.0,
        )
    except Exception as exc:
        # Per-sample error isolation: one flaky backend call (exhausted
        # retries, timeout, 5xx) must not throw away all the samples that
        # already succeeded. Record an empty prediction marked
        # `correct=False` with the error message, keep going. Classification
        # metrics (accuracy, macro_f1) treat it as a miss; judge_score
        # short-circuits on empty predictions to scale_min (see eval/judge).
        dt_ms = (time.perf_counter() - t0) * 1000.0
        err = f"{type(exc).__name__}: {str(exc)[:400]}"
        return (
            SamplePrediction(
                id=s.id,
                prediction="",
                expected=s.expected,
                correct=False,
                latency_ms=dt_ms,
                error=err,
            ),
            0,
            0,
            0.0,
        )


def _compute_metrics(task: TaskSpec, preds: list[SamplePrediction]) -> dict[str, float]:
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
    if "adjacent_accuracy" in wanted and task.labels:
        metrics["adjacent_accuracy"] = adjacent_accuracy(preds, task.labels)
    if "qwk" in wanted and task.labels:
        metrics["qwk"] = qwk(preds, task.labels)
    if "boundary_accuracy" in wanted and task.labels:
        metrics["boundary_accuracy"] = boundary_accuracy(preds, task.labels)
    if "boundary_kappa" in wanted and task.labels:
        metrics["boundary_kappa"] = boundary_kappa(preds, task.labels)
    if "signed_diff" in wanted and task.labels:
        metrics["signed_diff"] = signed_diff(preds, task.labels)
    return metrics


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
    ctx = _build_context(task, task_dir, model)
    template_hash = _hash_template(task.system_prompt, task.prompt_template)

    preds: list[SamplePrediction] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_gen_time_s = 0.0

    for s in samples:
        pred, in_tok, out_tok, gen_time = _execute_sample(ctx, s)
        preds.append(pred)
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        total_gen_time_s += gen_time

    metrics = _compute_metrics(task, preds)
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
        n_failed_samples=sum(1 for p in preds if p.error),
        samples=preds,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        env={"provider": model.provider, "endpoint_kind": model.endpoint_kind},
        cost_usd=cost_usd,
        judge_model_id=ctx.judge_card.model_id if ctx.judge_card else None,
        judge_model_revision=ctx.judge_card.revision if ctx.judge_card else None,
        judge_prompt_hash=judge_mod.hash_judge_prompt(task.judge.prompt) if task.judge else None,
    )

    out_dir = out_dir or (task_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = model.model_id.replace("/", "_")
    safe_ts = result.created_at.replace(":", "-").replace("+00-00", "Z")
    out_path = out_dir / f"{safe_id}__{safe_ts}.json"
    out_path.write_text(result.model_dump_json(indent=2))
    return out_path


def resume(result_path: Path, task_dir: Path, model_path: Path) -> tuple[Path, int]:
    """Re-run only the error-marked samples from a prior result file.

    Loads `result_path`, identifies samples with `error != None`, retries each
    against the model, and overwrites `result_path` in place with merged
    predictions and recomputed aggregate metrics.

    Use case: per-sample transient endpoint failures (Connection reset by peer,
    network timeouts) that `_execute_sample`'s error isolation captured. The
    whole task doesn't need to be re-scored — only the broken samples do.

    Returns (result_path, n_retried). `n_retried` is 0 when no samples had
    errors — in that case the file is left untouched.
    """
    existing = RunResult.model_validate_json(result_path.read_text())
    task, samples = load_task(task_dir)
    model = load_model(model_path)

    if existing.model_id != model.model_id:
        raise RuntimeError(
            f"Model mismatch: result file has model_id={existing.model_id!r} "
            f"but the provided card is {model.model_id!r}."
        )
    if existing.task_id != task.name:
        raise RuntimeError(
            f"Task mismatch: result file has task_id={existing.task_id!r} "
            f"but the provided task is {task.name!r}."
        )

    err_ids = {p.id for p in existing.samples if p.error}
    if not err_ids:
        return result_path, 0

    ctx = _build_context(task, task_dir, model)
    new_preds: dict[str, SamplePrediction] = {}
    for s in samples:
        if s.id not in err_ids:
            continue
        pred, _, _, _ = _execute_sample(ctx, s)
        new_preds[s.id] = pred

    # Merge: swap in the retried predictions at their original positions.
    merged = [new_preds[p.id] if p.id in new_preds else p for p in existing.samples]

    # Recompute aggregates from scratch over the merged sample list. For tps
    # we approximate per-sample gen-time as latency_ms (close enough — the
    # original tracked only successful-output samples; we do the same here).
    total_in = sum((p.input_tokens or 0) for p in merged)
    total_out = sum((p.output_tokens or 0) for p in merged)
    total_gen_time_s = sum(
        p.latency_ms / 1000.0 for p in merged if p.output_tokens
    )
    tps = (total_out / total_gen_time_s) if total_gen_time_s > 0 else 0.0
    p95 = _p95([p.latency_ms for p in merged])
    cost_usd = _compute_cost_usd(model, total_in, total_out)
    metrics = _compute_metrics(task, merged)

    new_result = existing.model_copy(update=dict(
        samples=merged,
        metrics=metrics,
        tps=tps,
        p95_latency_ms=p95,
        n_samples=len(merged),
        n_failed_samples=sum(1 for p in merged if p.error),
        cost_usd=cost_usd,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    ))
    result_path.write_text(new_result.model_dump_json(indent=2))
    return result_path, len(err_ids)
