# LLM Leaderboard

GitHub-driven leaderboard for evaluating LLMs on small, reproducible tasks.

See [CLAUDE.md](CLAUDE.md) for the full specification.

## Quickstart (dev)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd runner
uv sync

# validate task definitions
uv run llm-lb validate ../tasks/text_classification

# run a model against a task
uv run llm-lb run --task ../tasks/text_classification --model ../models/dummy@local.yaml

# rebuild per-task leaderboards + global data/index.json
uv run llm-lb aggregate --root ..

# export pydantic models as JSON Schema files
uv run llm-lb export-schemas --out-dir ../schemas
```

## Repository layout

- `tasks/<task_id>/` — task definitions, samples, results, per-task leaderboard
- `models/` — model cards (one YAML per model)
- `runner/` — Python package (`llm_lb`) with CLI, adapters, eval, aggregator
- `schemas/` — JSON Schema artifacts (generated from pydantic via `llm-lb export-schemas`)
- `data/` — global aggregated index (`index.json`) consumed by the static site
- `site/` — static UI (added in a later step)
- `.github/workflows/` — CI: validate, score (opt-in), Pages build (added in a later step)
