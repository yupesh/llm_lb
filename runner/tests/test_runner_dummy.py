from pathlib import Path

from llm_lb.aggregate import aggregate_all
from llm_lb.models import RunResult
from llm_lb.runner import run

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "tasks" / "text_classification"
MODEL = REPO / "models" / "dummy@local.yaml"


def test_dummy_end_to_end(tmp_path: Path):
    out = run(TASK, MODEL, out_dir=tmp_path)
    assert out.exists()
    result = RunResult.model_validate_json(out.read_text())
    assert result.task_id == "text_classification"
    assert result.model_id == "dummy@local"
    assert result.n_samples == len(result.samples) >= 5
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    # Dummy adapter should get most samples right on the seed task
    assert result.metrics["accuracy"] >= 0.7


def test_aggregate_smoke(tmp_path: Path):
    # Run once into the real results dir so aggregate has something to read
    run(TASK, MODEL)
    idx = aggregate_all(REPO)
    assert any(t["id"] == "text_classification" for t in idx["tasks"])
    assert any(e["model_id"] == "dummy@local" for e in idx["matrix"])


def test_aggregate_idempotent():
    """A second `aggregate_all` call must not touch any files on disk —
    otherwise CI's freshness check would fail on every run."""
    aggregate_all(REPO)
    index_path = REPO / "data" / "index.json"
    lb_path = REPO / "tasks" / "text_classification" / "leaderboard.json"
    before_index = index_path.read_bytes()
    before_lb = lb_path.read_bytes()

    aggregate_all(REPO)

    assert index_path.read_bytes() == before_index
    assert lb_path.read_bytes() == before_lb
