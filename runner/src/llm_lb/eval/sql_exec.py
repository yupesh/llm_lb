"""Execution-based SQL scoring for text2sql tasks.

Used by `runner_kind: sql_exec` tasks. Instead of asking an LLM judge whether
two SQL queries are semantically equivalent, we execute both against the actual
SQLite database and compare their result sets. This is closer to how BIRD and
Spider score officially.

The DBs aren't shipped with the repo (~600 MB for BIRD dev). The task expects
sqlite files under `<task_dir>/data/<db_id>/<db_id>.sqlite`; populate via
`scripts/fetch_bird_dev.py`. Missing DB files surface as a per-sample error so
the rest of the run still completes.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

# Hard cap so a runaway candidate query (cartesian join, missing WHERE) doesn't
# stall the whole run. SQLite has no built-in query timeout — we abort via a
# progress handler that fires every N VM ops.
_PROGRESS_OPS = 1_000_000
_MAX_PROGRESS_TICKS = 60  # ~60M VM ops total before we kill a query


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model response.

    Handles: bare SQL, ```sql fenced blocks, ``` fenced blocks, prefixes like
    "Answer:" or "SQL:", and trailing prose after a semicolon. Returns the SQL
    with surrounding whitespace stripped; empty string if nothing looks like
    SQL.
    """
    s = text.strip()
    # Prefer fenced code blocks when present — most reliable signal.
    fence = re.search(r"```(?:sql)?\s*(.+?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    # Drop leading "SQL:" / "Answer:" / "Query:" labels.
    s = re.sub(r"^\s*(sql|answer|query)\s*:\s*", "", s, flags=re.IGNORECASE)
    # If the model wrote prose followed by a SELECT/WITH, snip from the first
    # top-level statement keyword.
    m = re.search(r"\b(WITH|SELECT)\b", s, flags=re.IGNORECASE)
    if m:
        s = s[m.start():]
    # Trim trailing semicolons and any prose after the first one.
    s = s.split(";")[0].strip()
    return s


def _has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE) is not None


def _normalize_rows(rows: list[tuple]) -> list[tuple]:
    out: list[tuple] = []
    for row in rows:
        norm = []
        for cell in row:
            if isinstance(cell, float):
                # 6-digit rounding handles minor float drift between
                # equivalent queries (e.g. AVG vs SUM/COUNT).
                norm.append(round(cell, 6))
            elif isinstance(cell, bytes):
                norm.append(cell.decode("utf-8", errors="replace"))
            else:
                norm.append(cell)
        out.append(tuple(norm))
    return out


def _execute(db_path: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        ticks = [0]
        def _progress():
            ticks[0] += 1
            return 1 if ticks[0] > _MAX_PROGRESS_TICKS else 0
        conn.set_progress_handler(_progress, _PROGRESS_OPS)
        cur = conn.execute(sql)
        return cur.fetchall()
    finally:
        conn.close()


def score_sample(
    data_dir: Path, db_id: str, gold_sql: str, candidate_sql: str
) -> tuple[bool, str | None]:
    """Run both queries against the DB; return (correct, error).

    `error` is set when the candidate query fails to execute (syntax, missing
    table, timeout, ...). A failed candidate always counts as incorrect; we
    surface the error message so it ends up in `SamplePrediction.error`.

    A failure of the *gold* query is fatal — it means the dataset itself is
    broken — and is raised, not swallowed.
    """
    db_path = data_dir / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        # Infra failure, not a candidate failure — raise so the caller's
        # error-handling can mark the sample errored (and the all-samples-failed
        # guard fires if every sample hits this). Returning (False, ...) here
        # would silently produce a clean 0/N result, which masks misconfiguration.
        raise RuntimeError(f"DB file not found: {db_path}")

    candidate_sql = (candidate_sql or "").strip()
    if not candidate_sql:
        return False, "empty SQL"

    try:
        gold_rows = _execute(db_path, gold_sql)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Gold SQL failed for db={db_id!r}: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        cand_rows = _execute(db_path, candidate_sql)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"

    gold_norm = _normalize_rows(gold_rows)
    cand_norm = _normalize_rows(cand_rows)
    if _has_order_by(gold_sql):
        return gold_norm == cand_norm, None
    return sorted(gold_norm, key=repr) == sorted(cand_norm, key=repr), None
