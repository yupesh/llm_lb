#!/usr/bin/env python3
"""Download BIRD dev split sqlite files into tasks/text2sql_bird_exec/data/.

Used by the execution-based BIRD task (text2sql_bird_exec). The dev split is
publicly available — no password, no auth — but the archive is ~200 MB
(~600 MB extracted) so we keep it out of git.

Layout produced:

    tasks/text2sql_bird_exec/data/
        california_schools/california_schools.sqlite
        card_games/card_games.sqlite
        ...

Idempotent — skips DBs that already exist.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Official BIRD dev release (https://bird-bench.github.io). The URL is stable
# but mirrored — if it breaks, check the website for the current location.
DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "tasks" / "text2sql_bird_exec" / "data"

# DBs we actually score on (subset of 11 in BIRD dev). Saves ~half the space.
WANTED_DBS = {
    "california_schools",
    "card_games",
    "codebase_community",
    "debit_card_specializing",
    "financial",
    "student_club",
    "toxicology",
}


def _download(url: str, dst: Path) -> None:
    print(f"downloading {url} -> {dst}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, dst.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    missing = [db for db in WANTED_DBS if not (TARGET_DIR / db / f"{db}.sqlite").exists()]
    if not missing:
        print("all DBs already present, nothing to do", file=sys.stderr)
        return 0
    print(f"missing DBs: {sorted(missing)}", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="bird_dev_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "dev.zip"
        _download(DEV_URL, zip_path)
        size_mb = zip_path.stat().st_size / 1e6
        sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()[:12]
        print(f"got dev.zip {size_mb:.1f} MB sha256={sha}", file=sys.stderr)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path / "extracted")

        # BIRD dev archive layout: dev_20240627/dev_databases/<db_id>/<db_id>.sqlite
        # (or older: dev/dev_databases/...). Find the dev_databases directory.
        candidates = list((tmp_path / "extracted").rglob("dev_databases"))
        if not candidates:
            print("ERROR: dev_databases/ not found in archive", file=sys.stderr)
            return 1
        src_root = candidates[0]
        print(f"extracted DBs at {src_root}", file=sys.stderr)

        copied = 0
        for db in missing:
            src = src_root / db
            if not src.is_dir():
                print(f"WARN: {db} not found in archive", file=sys.stderr)
                continue
            dst = TARGET_DIR / db
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied += 1
            print(f"  copied {db}", file=sys.stderr)

    print(f"done — {copied} DBs in {TARGET_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
