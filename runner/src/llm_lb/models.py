"""Pydantic models — single source of truth for all artifact schemas.

JSON Schema files in `schemas/` are generated from these via `llm-lb export-schemas`.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# `protected_namespaces=()` silences pydantic's warning about field names starting
# with `model_` (model_id, model_revision, ...), which collide with pydantic's own
# `model_*` API but are intentional in our domain.
_BASE = ConfigDict(extra="forbid", protected_namespaces=())


class GPUSpec(BaseModel):
    model_config = _BASE
    gpu_count: int = Field(ge=0)
    gpu_type: str
    quantization: Optional[str] = None


Provider = Literal["openai", "openai_compat", "hf", "vllm", "tgi", "dummy", "llama_guard"]
EndpointKind = Literal["openai_chat", "openai_completion", "hf_inference", "dummy"]


class ModelCard(BaseModel):
    model_config = _BASE
    model_id: str
    display_name: str
    provider: Provider
    endpoint_kind: EndpointKind
    hf_uri: Optional[str] = None
    revision: Optional[str] = None
    context_window: Optional[int] = None
    hardware: Optional[GPUSpec] = None
    license: Optional[str] = None
    added_at: Optional[str] = None  # ISO date
    # Endpoint config (LLMasJ_URI). Used by openai_compat / self-hosted adapters.
    endpoint_url: Optional[str] = None
    # Name of env var that holds the endpoint URL. When set, the runtime value
    # of this env var overrides `endpoint_url` (handy for judge models whose
    # endpoint varies between local / CI / prod).
    endpoint_url_env: Optional[str] = None
    # Name under which the served model is advertised by the endpoint
    # (e.g. vLLM `--served-model-name`). Falls back to `hf_uri`, then to `model_id`.
    served_model_name: Optional[str] = None
    # Name of env var that holds the served model name. When set, the runtime
    # value of this env var overrides `served_model_name`.
    served_model_name_env: Optional[str] = None
    # Name of env var that holds the API key for this endpoint. Defaults are
    # provider-specific (OPENAI_API_KEY for openai, HF_TOKEN for hf).
    api_key_env: Optional[str] = None
    # Optional pricing for cost tracking. Both fields are USD per 1k tokens.
    # When both are set, the runner records `cost_usd` in each RunResult.
    prompt_cost_per_1k_usd: Optional[float] = Field(default=None, ge=0)
    completion_cost_per_1k_usd: Optional[float] = Field(default=None, ge=0)


class LLMParams(BaseModel):
    model_config = _BASE
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    seed: Optional[int] = None


class MetricSpec(BaseModel):
    model_config = _BASE
    primary: str
    secondary: list[str] = Field(default_factory=list)


class JudgeSpec(BaseModel):
    """LLM-as-Judge configuration. The judge model scores each candidate
    answer on an integer scale; the runner normalises into [0, 1].

    The judge prompt may use placeholders {prompt}, {expected}, {prediction}.
    """
    model_config = _BASE
    model: str  # model_id of an entry under models/
    prompt: str
    scale_min: int = 0
    scale_max: int = 5
    # If true, the runner will refuse to run when the candidate model and the
    # judge model share the same `model_id` (self-judge bias guard).
    forbid_self_judge: bool = True


class TaskSpec(BaseModel):
    model_config = _BASE
    name: str
    version: str
    description: Optional[str] = None
    metric: MetricSpec
    labels: Optional[list[str]] = None  # for classification tasks
    # Synonym → canonical label map. Extends `extract_label`'s search space
    # without changing the task's reported label set. Purpose-built for safety
    # tasks where different models use different vocabulary — Llama-Guard
    # outputs `unsafe`, mainstream chat models output `jailbreak` — but both
    # mean the same thing for `jailbreak_detection`.
    label_aliases: Optional[dict[str, str]] = None
    llm_params: LLMParams
    system_prompt: Optional[str] = None
    prompt_template: str  # placeholder: {prompt}
    # Free-form regex used by `eval.extract.extract_regex` for tasks where
    # the answer is buried in surrounding text. First group is the answer.
    answer_regex: Optional[str] = None
    # Optional LLM-as-Judge configuration. When present, the runner uses the
    # judge to compute the `judge_score` metric instead of (or alongside)
    # accuracy / exact_match.
    judge: Optional[JudgeSpec] = None
    # Execution mode. `None`/`"single_shot"` → classic per-sample
    # prompt→response. `"dialog_simulation"` → multi-turn user-agent ↔
    # support-agent simulation (see eval/dialog_sim). Simulation tasks
    # require additional files in the task directory (policy.md, db.json,
    # user_agent_prompt.md) and use adapters that support `chat_messages`.
    runner_kind: Optional[str] = None
    # Optional turn cap for dialog_simulation runs. Ignored otherwise.
    max_turns: Optional[int] = None
    # Optional per-sample context, looked up from an external JSON file.
    # When set, the runner loads `task_dir / context_file` once (if it exists)
    # as {key: value_str}, then for each sample takes value_str at
    # sample.meta[context_meta_key] and makes it available as `{context}` in
    # `prompt_template` (and optionally in `judge.prompt`). Missing file or
    # missing keys resolve to empty string — so local runs without the
    # external data keep working (at a quality cost). Used e.g. to inject
    # DB schemas into text2sql prompts without committing them to the repo.
    context_file: Optional[str] = None
    context_meta_key: Optional[str] = None


class Sample(BaseModel):
    model_config = _BASE
    id: str
    prompt: str
    expected: str
    meta: dict[str, Any] = Field(default_factory=dict)


class SamplePrediction(BaseModel):
    model_config = _BASE
    id: str
    prediction: str
    expected: str
    correct: bool
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # Unprocessed model output — before `strip_reasoning` and before any
    # label/regex extraction. Stored so extractor bugs (e.g. the
    # `safe`⊂`unsafe` substring bug that pinned safety_classification at
    # 0.500) can be fixed offline by re-running the extractor over the
    # original text, without a fresh GPU run. Nullable: older result files
    # predate this field, and dialog-simulation tasks don't produce a single
    # completion string.
    raw_output: Optional[str] = None
    # Per-sample judge score in the task's raw judge scale, when the task
    # configures an LLM-as-Judge metric. Normalised aggregate goes into
    # `RunResult.metrics["judge_score"]`.
    judge_raw_score: Optional[float] = None
    # Set when the sample failed mid-run (network timeout, adapter exhausted
    # retries, etc.). The prediction is empty and `correct=False`; the run
    # itself completes so other samples still contribute to the metrics.
    error: Optional[str] = None


class RunResult(BaseModel):
    model_config = _BASE
    model_id: str
    model_revision: Optional[str] = None
    task_id: str
    task_version: str
    runner_version: str
    prompt_template_hash: str
    llm_params: LLMParams
    metrics: dict[str, float]
    tps: float
    p95_latency_ms: float
    n_samples: int
    n_failed_samples: int = 0
    samples: list[SamplePrediction]
    created_at: str
    env: dict[str, Any] = Field(default_factory=dict)
    # Cost in USD, summed over all samples. Only set when the model card has
    # both `prompt_cost_per_1k_usd` and `completion_cost_per_1k_usd`.
    cost_usd: Optional[float] = None
    # LLM-as-Judge attribution. Only set when the task uses a judge metric.
    judge_model_id: Optional[str] = None
    judge_model_revision: Optional[str] = None
    judge_prompt_hash: Optional[str] = None


class LeaderboardEntry(BaseModel):
    model_config = _BASE
    model_id: str
    score: float
    metrics: dict[str, float]
    tps: float
    p95_latency_ms: float
    cost_usd: Optional[float] = None
    result_file: str


class Leaderboard(BaseModel):
    model_config = _BASE
    task_id: str
    task_version: str
    primary_metric: str
    ranking: list[LeaderboardEntry]
    updated_at: str
