# End-to-end walkthrough

This document walks through the full lifecycle of the leaderboard so you can
verify locally that everything works before opening a PR. It is the same flow
admins use day to day, just with the dummy adapter instead of paid APIs.

## 0. Prerequisites

- [uv](https://docs.astral.sh/uv/) for the Python runner
- [bun](https://bun.sh/) (preferred) or Node.js + npm for the static site
- A clone of this repository

```bash
git clone https://github.com/yupesh/llm_lb.git
cd llm_lb
```

## 1. One-shot demo cycle

The fastest way to see the entire pipeline is the demo script. It validates
every task and model card, runs the dummy model on the classification tasks,
rebuilds aggregates and builds the static site — all without network access
and without API keys.

```bash
bash scripts/demo-cycle.sh
```

When it finishes you will have:

- `tasks/*/leaderboard.json` — per-task leaderboards
- `data/index.json` — global aggregated index consumed by the SPA
- `data/feed.xml` — Atom feed of recent runs
- `site/dist/` — production build of the static site

Open `site/dist/index.html` in a browser, or run a live dev server:

```bash
cd site && bun run dev   # or: npm run dev
```

## 2. Contributing a new task (task-author flow)

1. Create a branch:
   ```bash
   git checkout -b task/my_new_task
   ```
2. Add `tasks/my_new_task/task.yaml` and at least 5 sample files in
   `tasks/my_new_task/samples/`. Use `tasks/text_classification/` as a
   reference.
3. Validate locally:
   ```bash
   cd runner
   uv run llm-lb validate ../tasks/my_new_task
   ```
4. **Rebuild the global index** (CI checks that it's up to date):
   ```bash
   cd runner
   uv run llm-lb aggregate --root ..
   ```
   This adds the new task to `data/index.json` (no `leaderboard.json` is
   created until the first scoring run).
5. Commit everything, push, open a PR. CI (`validate.yml`) will:
   - lint with ruff
   - run pytest (includes a dummy end-to-end smoke test)
   - validate every task and model card
   - check that JSON Schemas and aggregates are up to date
6. An admin reviews and merges. **No results are produced in this PR** —
   scoring happens separately.

## 3. Contributing a new model card (model-author flow)

1. Branch: `git checkout -b model/<model_id>`
2. Add `models/<model_id>.yaml`. Required fields: `model_id`, `display_name`,
   `provider`, `endpoint_kind`. Pin `revision` whenever the provider exposes one.
3. **Never put secrets in the YAML.** Use `api_key_env: NAME_OF_ENV_VAR` and
   set the secret locally or in GitHub Actions secrets. See
   [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-model) for the full example.
4. Validate, commit, push, open PR.

## 4. Scoring a model (admin flow)

Scoring requires real credentials and may cost money, so it is gated.

### 4a. Locally

```bash
cd runner
export OPENAI_API_KEY=sk-...
uv run llm-lb run --task ../tasks/text_classification --model ../models/gpt-4o-mini@openai.yaml
uv run llm-lb aggregate --root ..

git checkout -b score/text_classification-gpt-4o-mini
git add ../tasks/text_classification/results \
        ../tasks/text_classification/leaderboard.json \
        ../data/index.json \
        ../data/feed.xml
git commit -m "score: gpt-4o-mini on text_classification"
git push -u origin HEAD
```

Open the PR. The `score` label is auto-applied; CI re-runs validation and
freshness checks. An admin merges, Pages rebuilds automatically.

### 4b. Via GitHub Actions (semi-automated)

1. Go to **Actions → score → Run workflow**.
2. Inputs:
   - `task` — e.g. `tasks/text_classification`
   - `model` — e.g. `models/gpt-4o-mini@openai.yaml`
   - `runner` — `github` for hosted-API providers, `self-hosted` for local
     inference on a runner labelled `score-runner`
3. The workflow runs the model, rebuilds aggregates and opens a PR for review.
   Configure the relevant secrets (`OPENAI_API_KEY`, `HF_TOKEN`, ...) under
   *Settings → Secrets and variables → Actions* before the first run.

## 5. Inspecting the results

- **Live site:** <https://yupesh.github.io/llm_lb/>
- **Global JSON index:** <https://yupesh.github.io/llm_lb/data/index.json>
- **Atom feed (RSS readers):** <https://yupesh.github.io/llm_lb/data/feed.xml>
- **Per-task leaderboards:** `tasks/<task_id>/leaderboard.json` in the repo
- **Raw run results:** `tasks/<task_id>/results/*.json` (one file per scoring run)

## 6. Cleaning up demo data

The dummy results live alongside real ones. To wipe them before a real run:

```bash
rm tasks/*/results/dummy@local__*.json
cd runner && uv run llm-lb aggregate --root ..
```

Commit the resulting changes to `leaderboard.json`, `data/index.json` and
`data/feed.xml`.
