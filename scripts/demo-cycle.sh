#!/usr/bin/env bash
# End-to-end local demo of the contributor flow.
#
# Walks through the same steps an admin runs by hand:
#   1. validate every task and model card
#   2. run the dummy model on every classification task
#   3. rebuild per-task leaderboards and the global data/index.json
#   4. typecheck the static site and produce a build
#
# No network and no API keys required — uses the dummy adapter.
# Run from the repo root: `bash scripts/demo-cycle.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Step 1/4: validate tasks and models"
for d in tasks/*/; do
  (cd runner && uv run llm-lb validate "../$d")
done
for f in models/*.yaml; do
  (cd runner && uv run llm-lb validate "../$f")
done

echo "==> Step 2/4: run dummy on every classification task"
# The dummy adapter only knows positive/negative, so it scores 100% on
# text_classification, ~0% on topic_classification, and ~0% on extractive_qa.
# That's intentional — it gives the leaderboard non-trivial demo data.
for d in tasks/text_classification tasks/topic_classification tasks/extractive_qa; do
  (cd runner && uv run llm-lb run --task "../$d" --model "../models/dummy@local.yaml")
done

echo "==> Step 3/4: rebuild leaderboards and global index"
(cd runner && uv run llm-lb aggregate --root ..)

echo "==> Step 4/4: typecheck and build the static site"
if command -v bun >/dev/null 2>&1; then
  (cd site && bun install --frozen-lockfile && bun run build)
elif command -v npm >/dev/null 2>&1; then
  (cd site && npm install --silent && node scripts/sync.mjs && npx tsc --noEmit && npx vite build)
else
  echo "  (skipped: install bun or npm to build the site)"
fi

echo
echo "Demo cycle complete."
echo "  - Per-task leaderboards: tasks/*/leaderboard.json"
echo "  - Global index:          data/index.json"
echo "  - Built site:            site/dist/  (open site/dist/index.html)"
echo
echo "To preview the site locally with live-reload:"
echo "  cd site && (bun run dev || npm run dev)"
