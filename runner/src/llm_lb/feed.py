"""Atom feed generation for the global leaderboard.

The feed lists the most recent scoring runs across every task. It is fully
content-derived (no wall-clock time) so the freshness check stays stable: the
feed-level `<updated>` is the max of all entry timestamps, and entries come
from the `created_at` field of each `RunResult` JSON file.

The feed is consumed by RSS readers, search engines and anyone who wants to
follow new model results without polling the site.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .models import RunResult

# Default canonical site URL. Override at call time if the repo is forked.
DEFAULT_SITE_URL = "https://yupesh.github.io/llm_lb/"
FEED_TITLE = "LLM Leaderboard — recent runs"
FEED_ID = "tag:yupesh.github.io,2026:llm_lb/feed"
MAX_ENTRIES = 50


def _entry_id(model_id: str, task_id: str, task_version: str, created_at: str) -> str:
    safe = f"{task_id}/{task_version}/{model_id}/{created_at}".replace(" ", "_")
    return f"tag:yupesh.github.io,2026:llm_lb/run/{safe}"


def _collect_runs(repo_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    tasks_dir = repo_root / "tasks"
    if not tasks_dir.exists():
        return runs
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        results_dir = task_dir / "results"
        if not results_dir.exists():
            continue
        for p in sorted(results_dir.glob("*.json")):
            try:
                r = RunResult.model_validate_json(p.read_text())
            except Exception:
                continue
            primary = next(iter(r.metrics), None)
            score = r.metrics.get(primary, 0.0) if primary else 0.0
            runs.append(
                {
                    "model_id": r.model_id,
                    "task_id": r.task_id,
                    "task_version": r.task_version,
                    "primary_metric": primary or "score",
                    "score": score,
                    "metrics": r.metrics,
                    "created_at": r.created_at,
                    "result_path": str(p.relative_to(repo_root)),
                }
            )
    runs.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return runs[:MAX_ENTRIES]


def build_atom_feed(repo_root: Path, site_url: str = DEFAULT_SITE_URL) -> str:
    """Render an Atom 1.0 feed of recent runs as a deterministic string."""
    runs = _collect_runs(repo_root)
    feed_updated = runs[0]["created_at"] if runs else "1970-01-01T00:00:00+00:00"

    self_href = site_url.rstrip("/") + "/data/feed.xml"
    lines: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<feed xmlns="http://www.w3.org/2005/Atom">',
        f"  <title>{escape(FEED_TITLE)}</title>",
        f"  <id>{FEED_ID}</id>",
        f'  <link href="{escape(site_url)}" rel="alternate" type="text/html"/>',
        f'  <link href="{escape(self_href)}" rel="self" type="application/atom+xml"/>',
        f"  <updated>{feed_updated}</updated>",
        '  <author><name>LLM Leaderboard</name></author>',
    ]

    for r in runs:
        title = (
            f"{r['model_id']} scored {r['score']:.3f} {r['primary_metric']}"
            f" on {r['task_id']} v{r['task_version']}"
        )
        summary_parts = [f"{k}={v:.3f}" for k, v in r["metrics"].items()]
        summary = ", ".join(summary_parts)
        link = f"{site_url.rstrip('/')}/#/tasks/{r['task_id']}"
        entry_id = _entry_id(
            r["model_id"], r["task_id"], r["task_version"], r["created_at"] or ""
        )
        lines.extend(
            [
                "  <entry>",
                f"    <id>{escape(entry_id)}</id>",
                f"    <title>{escape(title)}</title>",
                f"    <updated>{r['created_at']}</updated>",
                f'    <link href="{escape(link)}" rel="alternate" type="text/html"/>',
                f"    <summary>{escape(summary)}</summary>",
                "  </entry>",
            ]
        )

    lines.append("</feed>")
    return "\n".join(lines) + "\n"


def write_feed(repo_root: Path, site_url: str = DEFAULT_SITE_URL) -> Path:
    """Write `data/feed.xml` only if its content changed."""
    out = repo_root / "data" / "feed.xml"
    new_content = build_atom_feed(repo_root, site_url)
    if out.exists() and out.read_text() == new_content:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(new_content)
    return out
