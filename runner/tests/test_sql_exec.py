from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from llm_lb.eval.sql_exec import extract_sql, score_sample


@pytest.mark.parametrize("raw,expected", [
    ("SELECT 1", "SELECT 1"),
    ("```sql\nSELECT 1;\n```", "SELECT 1"),
    ("Here is the query:\nSELECT * FROM t WHERE x=1;", "SELECT * FROM t WHERE x=1"),
    ("SQL: SELECT id FROM users", "SELECT id FROM users"),
    ("```\nWITH cte AS (SELECT 1) SELECT * FROM cte;\n```", "WITH cte AS (SELECT 1) SELECT * FROM cte"),
    ("SELECT 1; -- trailing comment", "SELECT 1"),
])
def test_extract_sql(raw: str, expected: str) -> None:
    assert extract_sql(raw) == expected


def _make_db(tmp_path: Path, db_id: str) -> Path:
    db_dir = tmp_path / db_id
    db_dir.mkdir()
    db_path = db_dir / f"{db_id}.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE t (id INTEGER, name TEXT, val REAL);"
        "INSERT INTO t VALUES (1, 'a', 1.5), (2, 'b', 2.5), (3, 'c', 3.5);"
    )
    conn.commit()
    conn.close()
    return tmp_path


def test_score_sample_match(tmp_path: Path) -> None:
    data_dir = _make_db(tmp_path, "demo")
    correct, err = score_sample(
        data_dir, "demo",
        "SELECT id, name FROM t WHERE val > 1.5",
        "SELECT name, id FROM t WHERE val > 1.5",  # different col order
    )
    # Order of *columns* differs — that's a wrong result set.
    assert correct is False
    assert err is None


def test_score_sample_equivalent_unordered(tmp_path: Path) -> None:
    data_dir = _make_db(tmp_path, "demo")
    correct, err = score_sample(
        data_dir, "demo",
        "SELECT id FROM t WHERE val > 1.5",
        "SELECT id FROM t WHERE val > 1.5 ORDER BY id DESC",  # gold has no ORDER BY
    )
    assert correct is True
    assert err is None


def test_score_sample_order_sensitive(tmp_path: Path) -> None:
    data_dir = _make_db(tmp_path, "demo")
    correct, _ = score_sample(
        data_dir, "demo",
        "SELECT id FROM t ORDER BY id ASC",
        "SELECT id FROM t ORDER BY id DESC",
    )
    assert correct is False


def test_score_sample_syntax_error(tmp_path: Path) -> None:
    data_dir = _make_db(tmp_path, "demo")
    correct, err = score_sample(
        data_dir, "demo",
        "SELECT id FROM t",
        "SELEKT id FROM t",
    )
    assert correct is False
    assert err is not None
    assert "OperationalError" in err or "syntax" in err.lower()


def test_score_sample_missing_db(tmp_path: Path) -> None:
    # Infra error must raise — silent (False, err) would mask a missing
    # data/ directory as a clean zero.
    with pytest.raises(RuntimeError, match="DB file not found"):
        score_sample(tmp_path, "nope", "SELECT 1", "SELECT 1")


def test_score_sample_empty_candidate(tmp_path: Path) -> None:
    data_dir = _make_db(tmp_path, "demo")
    correct, err = score_sample(data_dir, "demo", "SELECT 1", "")
    assert correct is False
    assert err == "empty SQL"
