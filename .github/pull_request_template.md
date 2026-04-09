<!--
  Pick ONE section that matches your PR and DELETE the other two.
  task  — new or updated task (no results needed)
  model — new or updated model card
  score — admin scoring run (includes results + leaderboard update)
-->

## Type

- [ ] **task** — new or updated task (`tasks/<task_id>/`)
- [ ] **model** — new or updated model card (`models/*.yaml`)
- [ ] **score** — scoring run results (`tasks/*/results/*.json`, `**/leaderboard.json`, `data/index.json`) — admin only

---

### task PR

- Task id: `...`
- Version: `...`
- Number of samples: `...`
- Primary metric: `...`

Checklist:
- [ ] `task.yaml` and all `samples/*.json` validate (`uv run llm-lb validate tasks/<id>`)
- [ ] Bumped `version` if I changed an existing task in a way that affects scoring
- [ ] Prompt template uses deterministic params (`temperature: 0`, fixed `seed`)

---

### model PR

- Model id: `...`
- Provider: `...`
- HF URI / endpoint: `...`

Checklist:
- [ ] `uv run llm-lb validate models/<file>.yaml` passes
- [ ] `revision` (commit hash / tag / API revision) is pinned where possible
- [ ] No secrets, API keys or private endpoints in the file

---

### score PR (admin)

- Task: `tasks/<task_id>` (version `...`)
- Model: `models/<model>.yaml` (revision `...`)
- Where it ran: `local` / `self-hosted` / `github-hosted`

Checklist:
- [ ] Result file committed under `tasks/<task_id>/results/`
- [ ] `uv run llm-lb aggregate --root .` ran and committed updates to `leaderboard.json` + `data/index.json`
- [ ] `runner_version`, `prompt_template_hash`, `task_version`, `model_revision`, `seed`, `llm_params` are all present in the result JSON
