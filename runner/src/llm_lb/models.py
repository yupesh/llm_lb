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


Provider = Literal["openai", "openai_compat", "hf", "vllm", "tgi", "dummy"]
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
    # Name under which the served model is advertised by the endpoint
    # (e.g. vLLM `--served-model-name`). Falls back to `hf_uri`, then to `model_id`.
    served_model_name: Optional[str] = None
    # Name of env var that holds the API key for this endpoint. Defaults are
    # provider-specific (OPENAI_API_KEY for openai, HF_TOKEN for hf).
    api_key_env: Optional[str] = None


class LLMParams(BaseModel):
    model_config = _BASE
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: Optional[int] = None


class MetricSpec(BaseModel):
    model_config = _BASE
    primary: str
    secondary: list[str] = Field(default_factory=list)


class TaskSpec(BaseModel):
    model_config = _BASE
    name: str
    version: str
    description: Optional[str] = None
    metric: MetricSpec
    labels: Optional[list[str]] = None  # for classification tasks
    llm_params: LLMParams
    system_prompt: Optional[str] = None
    prompt_template: str  # placeholder: {prompt}


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
    output_tokens: Optional[int] = None


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
    samples: list[SamplePrediction]
    created_at: str
    env: dict[str, Any] = Field(default_factory=dict)


class LeaderboardEntry(BaseModel):
    model_config = _BASE
    model_id: str
    score: float
    metrics: dict[str, float]
    tps: float
    p95_latency_ms: float
    result_file: str


class Leaderboard(BaseModel):
    model_config = _BASE
    task_id: str
    task_version: str
    primary_metric: str
    ranking: list[LeaderboardEntry]
    updated_at: str
