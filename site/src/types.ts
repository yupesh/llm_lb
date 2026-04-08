// Mirrors the pydantic models in runner/src/llm_lb/models.py and the shape
// produced by `llm-lb aggregate` in data/index.json.

export interface GPUSpec {
  gpu_count: number
  gpu_type: string
  quantization: string | null
}

export interface TaskMeta {
  id: string
  version: string
  primary_metric: string
  secondary_metrics: string[]
  n_samples: number
}

export interface ModelMeta {
  id: string
  display_name: string
  provider: string
  hf_uri: string | null
  hardware: GPUSpec | null
}

export interface MatrixEntry {
  task_id: string
  task_version: string
  model_id: string
  score: number
  metrics: Record<string, number>
  tps: number
  p95_latency_ms: number
  cost_usd: number | null
  result_file: string
}

export interface Index {
  generated_at: string
  tasks: TaskMeta[]
  models: ModelMeta[]
  matrix: MatrixEntry[]
}
