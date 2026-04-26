from __future__ import annotations

import json
from pathlib import Path

import typer

from .aggregate import aggregate_all
from .models import Leaderboard, ModelCard, RunResult, Sample, TaskSpec
from .runner import resume as resume_run
from .runner import run as run_task
from .validate import validate_model_file, validate_task_dir

app = typer.Typer(no_args_is_help=True, add_completion=False, help="LLM Leaderboard CLI")


@app.command()
def validate(path: Path) -> None:
    """Validate a task directory or a model YAML file."""
    if path.is_dir():
        validate_task_dir(path)
        typer.echo(f"OK: task {path}")
    else:
        validate_model_file(path)
        typer.echo(f"OK: model {path}")


@app.command()
def run(
    task: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    model: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    limit: int = typer.Option(
        0,
        "--limit",
        "-n",
        help="Only run the first N samples (smoke-test). 0 = all.",
        min=0,
    ),
    sample_id: list[str] = typer.Option(
        None,
        "--sample-id",
        help="Only run samples with these ids (repeatable). Smoke-test / single-sample re-run.",
    ),
) -> None:
    """Run a model against a task and write results JSON."""
    out = run_task(task, model, limit=limit or None, sample_ids=sample_id or None)
    typer.echo(f"wrote: {out}")


@app.command()
def resume(
    result: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    task: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    model: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Retry error-marked samples in an existing result file, merging in place."""
    path, n = resume_run(result, task, model)
    if n == 0:
        typer.echo(f"no error-marked samples in {path}; nothing to do")
    else:
        typer.echo(f"resumed {n} sample(s) in {path}")


@app.command()
def aggregate(root: Path = typer.Option(Path("."), exists=True, file_okay=False)) -> None:
    """Rebuild per-task leaderboards and the global data/index.json."""
    idx = aggregate_all(root)
    typer.echo(f"aggregated {len(idx['tasks'])} task(s), {len(idx['matrix'])} entries")


@app.command("export-schemas")
def export_schemas(out_dir: Path = typer.Option(Path("schemas"))) -> None:
    """Export pydantic models as JSON Schema files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {
        "task.schema.json": TaskSpec,
        "sample.schema.json": Sample,
        "model.schema.json": ModelCard,
        "result.schema.json": RunResult,
        "leaderboard.schema.json": Leaderboard,
    }
    for name, cls in pairs.items():
        (out_dir / name).write_text(json.dumps(cls.model_json_schema(), indent=2))
        typer.echo(f"wrote {out_dir / name}")


if __name__ == "__main__":
    app()
