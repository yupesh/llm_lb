#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = REPO_ROOT / "run_reports.zip"
TASKS_DIR = REPO_ROOT / "tasks"

CEFR_REGEX = r"\b(A1|A2|B1|B2|C1|C2)\b"
BLOCK_SEP = "=" * 120


@dataclass(frozen=True)
class TaskConfig:
    task_name: str
    run_id: str
    description: str
    labels: tuple[str, ...]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    source_report: str


TASKS: list[TaskConfig] = [
    TaskConfig(
        "writing_eval_n10_no_ann_no_m_serial",
        "20260225_152347",
        "Base writing evaluation, N10 holdout. No expert annotation, no methodology, single-pass prompt family.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n10_no_ann_with_m_serial",
        "20260225_152531",
        "Base writing evaluation, N10 holdout. No expert annotation, methodology-enabled single-pass prompt family.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n10_no_target_ann_loocv_error_retrieval_with_m_serial",
        "20260225_025139",
        "Base writing evaluation, N10 holdout. LOOCV error-retrieval with methodology and no target annotation.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n10_no_target_ann_loocv_error_retrieval_no_m_serial",
        "20260226_000308",
        "Base writing evaluation, N10 holdout. LOOCV error-retrieval without methodology and no target annotation.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n10_with_ann_loocv_fewshot_no_m_serial",
        "20260225_031955",
        "Base writing evaluation, N10 holdout. Expert annotation with LOOCV few-shot calibration and no methodology.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n10_with_ann_loocv_fewshot_with_m_serial",
        "20260225_034953",
        "Base writing evaluation, N10 holdout. Expert annotation with LOOCV few-shot calibration and methodology.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n10_with_ann_no_m_serial",
        "20260223_052953",
        "Base writing evaluation, N10 holdout. Expert annotation without methodology, single-pass prompt family.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n10_with_ann_with_m_serial",
        "20260223_231820",
        "Base writing evaluation, N10 holdout. Expert annotation with methodology, single-pass prompt family.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n10_no_target_ann_loocv_fewshot_no_m_serial",
        "20260225_040418",
        "Base writing evaluation, N10 holdout. No target annotation, LOOCV few-shot calibration, no methodology.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n10_no_target_ann_loocv_fewshot_with_m_serial",
        "20260225_041015",
        "Base writing evaluation, N10 holdout. No target annotation, LOOCV few-shot calibration, methodology-enabled.",
        ("A2", "B1", "B2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260225.md",
    ),
    TaskConfig(
        "writing_eval_n42_no_ann_no_m_serial",
        "20260402_171309",
        "Base writing evaluation, N42 mixed CEFR holdout. No expert annotation, no methodology, single-pass prompt family.",
        ("A1", "A2", "B1", "B2", "C1", "C2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260403.md",
    ),
    TaskConfig(
        "writing_eval_n42_no_ann_with_m_serial",
        "20260402_034226",
        "Base writing evaluation, N42 mixed CEFR holdout. No expert annotation, methodology-enabled single-pass prompt family.",
        ("A1", "A2", "B1", "B2", "C1", "C2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260403.md",
    ),
    TaskConfig(
        "writing_eval_n42_no_target_ann_loocv_fewshot_no_m_serial",
        "20260402_041024",
        "Base writing evaluation, N42 mixed CEFR holdout. No target annotation, LOOCV few-shot calibration, no methodology.",
        ("A1", "A2", "B1", "B2", "C1", "C2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260403.md",
    ),
    TaskConfig(
        "writing_eval_n42_no_target_ann_loocv_fewshot_with_m_serial",
        "20260402_161550",
        "Base writing evaluation, N42 mixed CEFR holdout. No target annotation, LOOCV few-shot calibration, methodology-enabled.",
        ("A1", "A2", "B1", "B2", "C1", "C2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260403.md",
    ),
    TaskConfig(
        "writing_eval_n42_with_ann_loocv_fewshot_no_m_serial",
        "20260402_052928",
        "Base writing evaluation, N42 mixed CEFR holdout. Expert annotation with LOOCV few-shot calibration and no methodology.",
        ("A1", "A2", "B1", "B2", "C1", "C2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260403.md",
    ),
    TaskConfig(
        "writing_eval_n42_with_ann_loocv_fewshot_with_m_serial",
        "20260402_193631",
        "Base writing evaluation, N42 mixed CEFR holdout. Expert annotation with LOOCV few-shot calibration and methodology.",
        ("A1", "A2", "B1", "B2", "C1", "C2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260403.md",
    ),
    TaskConfig(
        "writing_eval_n42_with_ann_no_m_serial",
        "20260402_195010",
        "Base writing evaluation, N42 mixed CEFR holdout. Expert annotation without methodology, single-pass prompt family.",
        ("A1", "A2", "B1", "B2", "C1", "C2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260403.md",
    ),
    TaskConfig(
        "writing_eval_n42_with_ann_with_m_serial",
        "20260402_225438",
        "Base writing evaluation, N42 mixed CEFR holdout. Expert annotation with methodology, single-pass prompt family.",
        ("A1", "A2", "B1", "B2", "C1", "C2"),
        "qwk",
        ("accuracy", "adjacent_accuracy"),
        "base_writing_eval_logs_report_20260403.md",
    ),
    TaskConfig(
        "writing_eval_b1b2_no_ann_no_m_serial",
        "20260417_034949",
        "Base writing evaluation, B1/B2 boundary holdout. No expert annotation, no methodology, single-pass prompt family.",
        ("B1", "B2"),
        "boundary_accuracy",
        ("accuracy", "qwk", "boundary_kappa", "signed_diff"),
        "base_writing_eval_b1b2_logs_report_20260417.md",
    ),
    TaskConfig(
        "writing_eval_b1b2_no_ann_with_m_score_only_serial",
        "20260417_073508",
        "Base writing evaluation, B1/B2 boundary holdout. No expert annotation, score-only methodology prompt family.",
        ("B1", "B2"),
        "boundary_accuracy",
        ("accuracy", "qwk", "boundary_kappa", "signed_diff"),
        "base_writing_eval_b1b2_logs_report_20260417.md",
    ),
    TaskConfig(
        "writing_eval_b1b2_no_ann_with_m_serial",
        "20260417_034220",
        "Base writing evaluation, B1/B2 boundary holdout. No expert annotation with full methodology prompt family.",
        ("B1", "B2"),
        "boundary_accuracy",
        ("accuracy", "qwk", "boundary_kappa", "signed_diff"),
        "base_writing_eval_b1b2_logs_report_20260417.md",
    ),
    TaskConfig(
        "writing_eval_b1b2_with_ann_no_m_serial",
        "20260417_062259",
        "Base writing evaluation, B1/B2 boundary holdout. Expert annotation without methodology, single-pass prompt family.",
        ("B1", "B2"),
        "boundary_accuracy",
        ("accuracy", "qwk", "boundary_kappa", "signed_diff"),
        "base_writing_eval_b1b2_logs_report_20260417.md",
    ),
    TaskConfig(
        "writing_eval_b1b2_with_ann_with_m_serial",
        "20260417_055035",
        "Base writing evaluation, B1/B2 boundary holdout. Expert annotation with full methodology prompt family.",
        ("B1", "B2"),
        "boundary_accuracy",
        ("accuracy", "qwk", "boundary_kappa", "signed_diff"),
        "base_writing_eval_b1b2_logs_report_20260417.md",
    ),
]


def _load_manifest(archive: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    with archive.open("run_reports/share_logs_by_n_20260422/meta/manifest.csv") as handle:
        rows = csv.DictReader(line.decode("utf-8") for line in handle)
        return {row["run_id"]: row for row in rows}


def _extract_run_params(destination_dir: str) -> dict[str, Any]:
    temp_match = re.search(r"_t([^_]+)_p([^_]+)_mt([^/_]+)(?:_|$)", destination_dir)
    if temp_match is None:
        raise ValueError(f"Could not parse llm params from {destination_dir}")
    temperature_raw, top_p_raw, max_tokens_raw = temp_match.groups()
    return {
        "temperature": float(temperature_raw),
        "top_p": float(top_p_raw),
        "max_tokens": int(max_tokens_raw),
    }


def _parse_eval_log(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for chunk in text.split(BLOCK_SEP):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        try:
            payload_idx = lines.index("payload:")
        except ValueError as exc:
            raise ValueError(f"Missing payload marker in block:\n{chunk[:400]}") from exc
        meta: dict[str, Any] = {}
        for line in lines[:payload_idx]:
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            meta[key] = value
        payload_text = "\n".join(lines[payload_idx + 1 :]).lstrip()
        payload, _end = decoder.raw_decode(payload_text)
        meta["payload"] = payload
        records.append(meta)
    return records


def _run_paths(row: dict[str, str]) -> tuple[str, str]:
    rel_root = row["destination_dir"].split("share_logs_by_n_20260422/", 1)[1]
    base = f"run_reports/share_logs_by_n_20260422/{rel_root}"
    return f"{base}/eval_io.log", f"{base}/eval_results.json"


def _parse_gold_from_eval_run(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    patterns = [
        re.compile(r"Processed fold \d+/\d+ \| ref_id=(\S+) \| (\S+) -> (\S+)"),
        re.compile(r"Processed (\S+): (\S+) -> (\S+)"),
    ]
    for line in text.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue
            sample_id, expected, pred_raw = match.groups()
            rows.append({"id": sample_id, "true": expected, "pred_raw": pred_raw})
            break
    return rows


def _normalize_gold(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, list):
        return [
            {"id": row["id"], "true": row["true"], "pred_raw": row.get("pred_raw", "")}
            for row in payload
        ]
    if isinstance(payload, dict) and isinstance(payload.get("fold_results"), list):
        rows: list[dict[str, str]] = []
        for row in payload["fold_results"]:
            if row.get("status") != "ok":
                continue
            rows.append(
                {
                    "id": row["ref_id"],
                    "true": row["true_level"],
                    "pred_raw": row.get("pred_level", ""),
                }
            )
        return rows
    raise ValueError(f"Unsupported eval_results payload type: {type(payload)!r}")


def _sample_prompt(request_payload: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    text = request_payload.get("text")
    if isinstance(text, str) and text.strip():
        return text
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def _write_task(task: TaskConfig, row: dict[str, str], requests: dict[str, dict[str, Any]], responses: dict[str, dict[str, Any]], gold: list[dict[str, Any]]) -> None:
    task_dir = TASKS_DIR / task.task_name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    samples_dir = task_dir / "samples"
    samples_dir.mkdir(parents=True)

    llm_params = _extract_run_params(row["destination_dir"])
    task_payload = {
        "name": task.task_name,
        "version": "1.0",
        "description": (
            f"{task.description} Canonical prompt/messages taken from best run `{task.run_id}` "
            f"in `{task.source_report}`."
        ),
        "metric": {
            "primary": task.primary_metric,
            "secondary": list(task.secondary_metrics),
        },
        "labels": list(task.labels),
        "llm_params": llm_params,
        "prompt_template": "{prompt}",
        "answer_regex": CEFR_REGEX,
    }
    (task_dir / "task.yaml").write_text(yaml.safe_dump(task_payload, sort_keys=False))

    for sample in gold:
        sample_id = sample["id"]
        response_payload = responses[sample_id]
        request_payload = requests[sample_id]
        messages = response_payload["debugModelIo"]["messages"]
        sample_payload = {
            "id": sample_id,
            "prompt": _sample_prompt(request_payload, messages),
            "expected": sample["true"],
            "meta": {
                "messages": messages,
                "source_run_id": task.run_id,
                "source_method": request_payload.get("methodology"),
                "source_report": task.source_report,
            },
        }
        (samples_dir / f"{sample_id}.json").write_text(json.dumps(sample_payload, indent=2) + "\n")


def main() -> None:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Archive not found: {ARCHIVE_PATH}")

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        manifest = _load_manifest(archive)
        for task in TASKS:
            row = manifest[task.run_id]
            eval_io_path, eval_results_path = _run_paths(row)
            records = _parse_eval_log(archive.read(eval_io_path).decode("utf-8"))
            requests = {
                record["ref_id"]: record["payload"]
                for record in records
                if record.get("phase") == "request"
            }
            responses = {
                record["ref_id"]: record["payload"]
                for record in records
                if record.get("phase") == "response"
            }
            if eval_results_path in archive.namelist():
                gold = _normalize_gold(json.loads(archive.read(eval_results_path).decode("utf-8")))
            else:
                eval_run_path = eval_results_path.replace("eval_results.json", "eval_run.log")
                gold = _parse_gold_from_eval_run(archive.read(eval_run_path).decode("utf-8"))

            missing_requests = [sample["id"] for sample in gold if sample["id"] not in requests]
            missing_responses = [sample["id"] for sample in gold if sample["id"] not in responses]
            if missing_requests or missing_responses:
                raise ValueError(
                    f"{task.task_name}: missing request/response blocks. "
                    f"requests={missing_requests} responses={missing_responses}"
                )

            _write_task(task, row, requests, responses, gold)
            print(f"wrote {task.task_name}")


if __name__ == "__main__":
    main()
