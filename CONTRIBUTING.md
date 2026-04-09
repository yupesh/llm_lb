# Contributing

This repo is a GitHub-driven LLM leaderboard. All changes go through pull
requests. There are three contributor roles, each with its own workflow.

> For a hands-on end-to-end demo (including a local dry run with the dummy
> adapter), see [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).
> For metrics and LLM-as-Judge configuration, see [docs/METRICS.md](docs/METRICS.md).

## Roles at a glance

| Role         | What you change                                                              | Who merges    |
|--------------|------------------------------------------------------------------------------|---------------|
| Task author  | `tasks/<task_id>/task.yaml`, `tasks/<task_id>/samples/*.json`                | admins        |
| Model author | `models/*.yaml`                                                              | admins        |
| Admin        | `tasks/*/results/*.json`, `**/leaderboard.json`, `data/index.json`, runner   | admins        |

`CODEOWNERS` enforces admin review on results, leaderboards, the global index,
the runner package, the schemas and CI workflows.

## Setup

```bash
# Python runner (uv)
cd runner
uv sync
uv run pytest -q

# Static site (bun) — only needed if you touch site/
cd ../site
bun install
bun run dev   # opens http://localhost:5173/
```

## Adding a task

1. Create a branch:
   ```bash
   git checkout -b task/<task_id>
   ```
2. Create a directory `tasks/<task_id>/` with:
   - `task.yaml` — see `tasks/text_classification/task.yaml` for a working example.
     For choosing a metric, see [docs/METRICS.md](docs/METRICS.md).
   - `samples/sample_001.json`, `sample_002.json`, ... (≥ 5 samples for MVP, target 20).
3. Validate locally:
   ```bash
   cd runner
   uv run llm-lb validate ../tasks/<task_id>
   ```
4. Rebuild the global index (CI checks that it's up to date):
   ```bash
   uv run llm-lb aggregate --root ..
   ```
5. Commit `tasks/<task_id>/` and `data/index.json`, push, open a PR.
   The `task` label is auto-applied. CI runs schema validation, lint, tests
   and re-aggregation checks.
6. An admin reviews and merges. Scoring runs happen in **separate** PRs after
   merge — task PRs do not contain results.

### Task design rules

- Use `temperature: 0` and a fixed `seed` for reproducibility.
- The `prompt_template` field must contain the placeholder `{prompt}`.
- For classification tasks, list every valid label in `labels:`. The validator
  checks that every sample's `expected` is one of those labels.
- Don't change a task's content silently — if you alter prompts, samples or
  scoring rules, bump `version` so old results stay attributable to the old
  version.

## Adding a model

1. Create a branch:
   ```bash
   git checkout -b model/<model_id>
   ```
2. Create `models/<model_id>.yaml`. The recommended `model_id` format is
   `<name>@<provider>` (e.g. `gpt-4o-mini@openai`, `llama-3.1-8b@vllm`).
3. Required fields: `model_id`, `display_name`, `provider`, `endpoint_kind`.
4. Pin `revision` whenever the provider exposes one (HF commit hash, API
   revision tag, vLLM image tag).
5. **Never** put API keys, tokens or private endpoint URLs in the file.
   The model card is plain text in the repo — anything you write there
   becomes public the moment you push. Instead, write only the *name* of
   an environment variable that holds the secret, and set the variable
   either locally (`export ...`) or via GitHub repo secrets for CI.

   Example: a private vLLM endpoint protected by a bearer token.

   `models/qwen3-30b@vllm.yaml` (committed to the repo, no secrets):
   ```yaml
   model_id: qwen3-30b@vllm
   display_name: Qwen3-30B (vLLM)
   provider: openai_compat
   endpoint_kind: openai_chat
   hf_uri: Qwen/Qwen3-30B-A3B
   revision: "2025-12-01"
   endpoint_url: https://lnsigo.mipt.ru:4000/v1
   api_key_env: VLLM_PROXY_KEY     # <-- name of env var, not the key
   hardware:
     gpu_count: 1
     gpu_type: H200
   ```

   When an admin later scores this model, they set the env var and run
   `llm-lb run` (see [Scoring](#scoring-a-model-admin) below).

   In GitHub Actions the same secret goes under
   *Settings → Secrets and variables → Actions* with the name `VLLM_PROXY_KEY`.

   Defaults if you omit `api_key_env`:
   - `provider: openai`        → reads `OPENAI_API_KEY`
   - `provider: hf`            → reads `HF_TOKEN`
   - `provider: openai_compat` → no key sent unless `api_key_env` is set
     (most local vLLM/TGI deployments don't need one)
6. Validate, rebuild index, commit and open a PR:
   ```bash
   cd runner
   uv run llm-lb validate ../models/<model_id>.yaml
   uv run llm-lb aggregate --root ..
   ```

### Provider matrix

| `provider`     | Use for                                              | Required fields                       |
|----------------|------------------------------------------------------|---------------------------------------|
| `openai`       | OpenAI API or any provider exposing `OPENAI_BASE_URL`| `OPENAI_API_KEY` env var              |
| `openai_compat`| vLLM, TGI in OpenAI mode, LM Studio, OpenRouter, ... | `endpoint_url` (+ optional `api_key_env`) |
| `hf`           | HF Inference API (text-generation shape)             | `hf_uri`, `HF_TOKEN` env var          |
| `dummy`        | CI smoke tests                                       | —                                     |

## Scoring a model (admin)

Scoring is **never** triggered automatically by a contributor PR — it requires
secrets and may cost money. Two paths:

### Local
```bash
cd runner
export OPENAI_API_KEY=sk-...
uv run llm-lb run --task ../tasks/<task_id> --model ../models/<model>.yaml
uv run llm-lb aggregate --root ..
git checkout -b score/<task>-<model>
git add ../tasks/<task_id>/results ../tasks/<task_id>/leaderboard.json ../data/index.json
git commit -m "score: <model> on <task>"
git push -u origin HEAD
```
Open the PR — the `score` label is auto-applied, CI re-runs validation +
schema-up-to-date checks. An admin merges.

### Via GitHub Actions
Run the `score` workflow manually (Actions → score → Run workflow). Inputs:
- `task` — e.g. `tasks/text_classification`
- `model` — e.g. `models/gpt-4o-mini@openai.yaml`
- `runner` — `github` (for hosted-API providers) or `self-hosted` (for local
  inference on a runner labelled `score-runner`)

The workflow runs the model, rebuilds aggregates and opens a PR for review.
Configure the relevant secrets (`OPENAI_API_KEY`, `HF_TOKEN`, ...) under
*Settings → Secrets and variables → Actions* before the first run.

## What CI checks on every PR

`validate.yml` runs on every PR and:
1. Lints the Python runner with ruff
2. Runs `pytest` (includes a dummy end-to-end run)
3. Validates every `tasks/*/` directory
4. Validates every `models/*.yaml` file
5. Re-exports JSON Schemas and fails if `schemas/` is stale
6. Re-runs `llm-lb aggregate` and fails if `tasks/*/leaderboard.json` or
   `data/index.json` is stale

If the last two fail, run the suggested command and commit the result.

## Repo layout

See [README.md](README.md#repository-layout).
