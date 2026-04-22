"""Typer CLI entrypoint for llmeval.

Commands are thin wrappers around the core library. Heavy imports are
deferred inside each command so startup time stays fast.

Shell completion
----------------
Enable tab completion for your shell::

    llmeval --install-completion   # install for detected shell
    llmeval --show-completion      # print completion script

Supported shells: bash, zsh, fish, PowerShell.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json as _json
import os
import re
import sys
from enum import IntEnum

import typer
from rich import box
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="llmeval",
    help="LLM evaluation and regression testing framework.",
    no_args_is_help=True,
)
console = Console()
_err = Console(stderr=True)


class ExitCode(IntEnum):
    """Standard exit codes for llmeval commands."""

    OK = 0
    TEST_FAILURE = 1
    USAGE_ERROR = 2
    STORAGE_ERROR = 3
    MODEL_ERROR = 4


def _db_path(db: str | None) -> str:
    return db or os.environ.get("LLMEVAL_DB_PATH", "llmeval.db")


def _abort(msg: str, code: int = ExitCode.USAGE_ERROR) -> None:
    _err.print(f"[bold red]Error:[/bold red] {msg}")
    raise typer.Exit(code=code)


def _parse_labels(raw: list[str]) -> dict[str, str]:
    """Parse ``["key=value", ...]`` into ``{"key": "value", ...}``.

    Aborts with exit 2 if any entry is not in ``key=value`` form.
    """
    result: dict[str, str] = {}
    for kv in raw:
        if "=" not in kv:
            _abort(f"Label {kv!r} must be in key=value format.")
        k, _, v = kv.partition("=")
        result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Global callback — load .env exactly once per CLI invocation
# ---------------------------------------------------------------------------


@app.callback()
def _global_callback() -> None:
    """Load environment variables from .env before any command runs."""
    from dotenv import load_dotenv

    load_dotenv()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed llmeval version."""
    from llmeval import __version__

    console.print(f"llmeval {__version__}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    suite_path: str = typer.Option(
        ..., "--suite", "-s", help="Path to a YAML/JSON test suite file."
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the model in the suite.",
    ),
    tag: list[str] = typer.Option(
        [], "--tag", "-t", help="Only run tests with this tag. Repeatable."
    ),
    db: str | None = typer.Option(
        None, "--db", help="SQLite database path. Overrides LLMEVAL_DB_PATH."
    ),
    no_save: bool = typer.Option(
        False, "--no-save", help="Skip persisting the run to storage."
    ),
    concurrency: int = typer.Option(
        5, "--concurrency", "-c", help="Max simultaneous model API calls."
    ),
    label: list[str] = typer.Option(
        [],
        "--label",
        "-l",
        help=(
            "CI metadata in key=value format "
            "(e.g. --label commit=abc123). Repeatable."
        ),
    ),
    samples: int = typer.Option(
        1,
        "--samples",
        help=(
            "Number of judge calls per test case. When > 1, per-criterion "
            "scores are aggregated as the median and the standard deviation "
            "is recorded."
        ),
    ),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        help=(
            "Sampling temperature for the judge model. Omit to use the "
            "provider default. Values > 0 introduce variance; use with "
            "--samples > 1 to get reliable multi-sample estimates."
        ),
    ),
) -> None:
    """Run a test suite, score outputs with LLM-as-judge, and print results.

    Exits 0 when all tests pass, 1 when any test fails or errors,
    2 on configuration or I/O errors, 3 on storage errors, and 4 on
    model/adapter errors.
    """
    from llmeval.exceptions import (
        ConfigurationError,
        JudgeError,
        RunnerError,
        SchemaValidationError,
        StorageError,
    )
    from llmeval.judge import Judge
    from llmeval.models import create_adapter
    from llmeval.report import CliReporter
    from llmeval.runner import Runner
    from llmeval.schema.results import SuiteRun
    from llmeval.schema.test_suite import load_suite
    from llmeval.storage import SQLiteStorage

    try:
        suite_def = load_suite(suite_path)
    except (SchemaValidationError, ConfigurationError) as exc:
        _abort(str(exc), ExitCode.USAGE_ERROR)
        return

    model_name = model or suite_def.suite.model
    tags = tag or None
    labels = _parse_labels(label)

    try:
        runner_adapter = create_adapter(model_name)
        # Default to temperature=0.0 for the judge to keep scoring deterministic.
        # Users can override with --temperature to introduce variance for --samples > 1.
        judge_temp = temperature if temperature is not None else 0.0
        judge_adapter = create_adapter(
            suite_def.suite.judge_model, temperature=judge_temp
        )
    except ConfigurationError as exc:
        _abort(str(exc), ExitCode.MODEL_ERROR)
        return

    async def _pipeline() -> SuiteRun:
        with console.status(
            f"Running {len(suite_def.tests)} tests against "
            f"[cyan]{model_name}[/cyan]…"
        ):
            runner = Runner(runner_adapter, concurrency=concurrency)
            suite_run = await runner.run(suite_def, tags=tags, suite_path=suite_path)

        with console.status(
            f"Scoring with judge [cyan]{suite_def.suite.judge_model}[/cyan]…"
        ):
            judge = Judge(judge_adapter, concurrency=concurrency, samples=samples)
            suite_run = await judge.score_suite_run(suite_run, suite_def)

        suite_run = suite_run.model_copy(update={"labels": labels})

        if not no_save:
            async with SQLiteStorage(_db_path(db)) as storage:
                await storage.save_run(suite_run)
            console.print(f"[dim]Run saved: {suite_run.run_id}[/dim]")

        return suite_run

    try:
        suite_run = asyncio.run(_pipeline())
    except RunnerError as exc:
        _abort(str(exc), ExitCode.MODEL_ERROR)
        return
    except JudgeError as exc:
        _abort(str(exc), ExitCode.MODEL_ERROR)
        return
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    CliReporter(console).print_run(suite_run)

    if suite_run.failed_tests > 0 or suite_run.errored_tests > 0:
        raise typer.Exit(code=ExitCode.TEST_FAILURE)


# ---------------------------------------------------------------------------
# latest
# ---------------------------------------------------------------------------


@app.command()
def latest(
    suite: str | None = typer.Option(
        None, "--suite", "-s", help="Filter to a specific suite name."
    ),
    status: str | None = typer.Option(
        "completed",
        "--status",
        help=(
            "Only consider runs with this status. Defaults to 'completed' "
            "(last successful run). Pass 'any' to return the absolute latest "
            "regardless of status."
        ),
    ),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Display the most recent run matching the given filters.

    Defaults to the last *completed* run. Use ``--status any`` to return the
    absolute latest run regardless of status (useful for checking whether a
    run is still in progress).
    """
    from llmeval.exceptions import StorageError
    from llmeval.report import CliReporter
    from llmeval.storage import SQLiteStorage

    resolved_status: str | None = None if status == "any" else status

    async def _fetch() -> object:
        async with SQLiteStorage(_db_path(db)) as storage:
            return await storage.get_latest_run(
                suite_name=suite, status=resolved_status
            )

    try:
        run_obj = asyncio.run(_fetch())
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    CliReporter(console).print_run(run_obj)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# rerun
# ---------------------------------------------------------------------------


@app.command()
def rerun(
    run_id: str = typer.Argument(..., help="Run ID (or prefix) to re-run."),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
    no_save: bool = typer.Option(False, "--no-save", help="Skip saving the new run."),
    concurrency: int = typer.Option(
        0, "--concurrency", "-c", help="Override concurrency (0 = use original)."
    ),
    label: list[str] = typer.Option(
        [],
        "--label",
        "-l",
        help="Additional CI metadata in key=value format. Repeatable.",
    ),
    suite: str | None = typer.Option(
        None,
        "--suite",
        "-s",
        help=(
            "Override the suite file path. Use when the original path has moved "
            "and the stored suite_path no longer exists."
        ),
    ),
    samples: int = typer.Option(
        1,
        "--samples",
        help="Number of judge calls per test case for multi-sample scoring.",
    ),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature for the judge model.",
    ),
) -> None:
    """Re-run a suite using the exact configuration of a previous run.

    Loads ``suite_path``, ``model``, and ``tags`` from the stored run, then
    executes a fresh pipeline. Pass ``--suite <path>`` when the original
    suite file has moved since the run was recorded.
    """
    from llmeval.exceptions import (
        ConfigurationError,
        JudgeError,
        RunnerError,
        SchemaValidationError,
        StorageError,
    )
    from llmeval.judge import Judge
    from llmeval.models import create_adapter
    from llmeval.report import CliReporter
    from llmeval.runner import Runner
    from llmeval.schema.results import SuiteRun
    from llmeval.schema.test_suite import load_suite
    from llmeval.storage import SQLiteStorage

    async def _fetch() -> object:
        async with SQLiteStorage(_db_path(db)) as storage:
            return await storage.get_run(run_id)

    try:
        original = asyncio.run(_fetch())
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    original_run: SuiteRun = original  # type: ignore[assignment]

    # Determine suite path: prefer --suite override, fall back to stored value.
    suite_path = suite or original_run.suite_path
    if not suite_path:
        _abort(
            f"Run {run_id!r} has no suite_path stored — cannot rerun. "
            "Use `llmeval run --suite <path>` instead or pass --suite.",
            ExitCode.USAGE_ERROR,
        )
        return

    # If the stored path no longer exists and no override was given, abort helpfully.
    if not suite and not os.path.exists(suite_path):
        _abort(
            f"Suite file {suite_path!r} not found. "
            "Pass --suite <new-path> to override.",
            ExitCode.USAGE_ERROR,
        )
        return

    model_name = original_run.model
    tags = original_run.tags or None
    actual_concurrency = concurrency if concurrency > 0 else original_run.concurrency
    labels = {**original_run.labels, **_parse_labels(label)}

    console.print(
        f"Re-running [bold]{original_run.suite_name}[/bold] "
        f"([cyan]{model_name}[/cyan]) from [dim]{suite_path}[/dim]"
    )

    try:
        suite_def = load_suite(suite_path)
    except (SchemaValidationError, ConfigurationError) as exc:
        _abort(str(exc), ExitCode.USAGE_ERROR)
        return

    try:
        runner_adapter = create_adapter(model_name)
        judge_temp = temperature if temperature is not None else 0.0
        judge_adapter = create_adapter(
            suite_def.suite.judge_model, temperature=judge_temp
        )
    except ConfigurationError as exc:
        _abort(str(exc), ExitCode.MODEL_ERROR)
        return

    async def _pipeline() -> SuiteRun:
        with console.status(
            f"Running {len(suite_def.tests)} tests against "
            f"[cyan]{model_name}[/cyan]…"
        ):
            runner = Runner(runner_adapter, concurrency=actual_concurrency)
            suite_run = await runner.run(suite_def, tags=tags, suite_path=suite_path)

        with console.status(
            f"Scoring with judge [cyan]{suite_def.suite.judge_model}[/cyan]…"
        ):
            judge = Judge(
                judge_adapter, concurrency=actual_concurrency, samples=samples
            )
            suite_run = await judge.score_suite_run(suite_run, suite_def)

        suite_run = suite_run.model_copy(update={"labels": labels})

        if not no_save:
            async with SQLiteStorage(_db_path(db)) as storage:
                await storage.save_run(suite_run)
            console.print(f"[dim]Run saved: {suite_run.run_id}[/dim]")

        return suite_run

    try:
        suite_run = asyncio.run(_pipeline())
    except RunnerError as exc:
        _abort(str(exc), ExitCode.MODEL_ERROR)
        return
    except JudgeError as exc:
        _abort(str(exc), ExitCode.MODEL_ERROR)
        return
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    CliReporter(console).print_run(suite_run)

    if suite_run.failed_tests > 0 or suite_run.errored_tests > 0:
        raise typer.Exit(code=ExitCode.TEST_FAILURE)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command()
def show(
    run_id: str = typer.Argument(..., help="Run ID to display."),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Display a stored run by its run ID or unique prefix.

    A prefix is unambiguous when it matches exactly one stored run.
    The first 8 characters are usually sufficient — use ``llmeval list``
    to browse available run IDs.
    """
    from llmeval.exceptions import StorageError
    from llmeval.report import CliReporter
    from llmeval.storage import SQLiteStorage

    async def _fetch() -> object:
        async with SQLiteStorage(_db_path(db)) as storage:
            return await storage.get_run(run_id)

    try:
        stored_run = asyncio.run(_fetch())
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    CliReporter(console).print_run(stored_run)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@app.command()
def diff(
    run_a: str = typer.Argument(..., help="Run ID of the baseline run (A)."),
    run_b: str | None = typer.Argument(
        None,
        help=(
            "Run ID of the candidate run (B). "
            "Omit to automatically compare against the previous run of the same suite."
        ),
    ),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Compare two stored runs side by side.

    When *run_b* is omitted, the most recent completed run of the same suite
    that predates *run_a* is used automatically.
    """
    from llmeval.exceptions import StorageError
    from llmeval.report import DiffReporter
    from llmeval.schema.results import SuiteRun
    from llmeval.storage import SQLiteStorage

    async def _fetch() -> tuple[SuiteRun, SuiteRun]:
        async with SQLiteStorage(_db_path(db)) as storage:
            a = await storage.get_run(run_a)
            if run_b is not None:
                b = await storage.get_run(run_b)
            else:
                b = await storage.get_previous_run(a.suite_name, a.run_id)
            return a, b

    try:
        run_a_obj, run_b_obj = asyncio.run(_fetch())
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    DiffReporter(console).print_diff(run_a_obj, run_b_obj)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_runs(
    suite: str | None = typer.Option(
        None, "--suite", "-s", help="Filter by exact suite name."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Filter by exact model identifier."
    ),
    status: str | None = typer.Option(
        None,
        "--status",
        help="Filter by status: pending, running, completed, failed.",
    ),
    tag: str | None = typer.Option(None, "--tag", "-t", help="Filter by tag."),
    tag_match: str = typer.Option(
        "exact",
        "--tag-match",
        help="Tag match mode: 'exact' (default) or 'fuzzy' (substring).",
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum runs to show."),
    fmt: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: 'table' (default) or 'json'.",
    ),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """List stored suite runs, most recent first.

    All filters combine with AND — only runs matching every supplied
    filter are shown. Tag filtering defaults to exact-match; use
    ``--tag-match fuzzy`` for substring/prefix search.

    Use ``--format json`` to emit a machine-readable JSON array suitable for
    piping to ``jq`` or CI scripts.
    """
    from llmeval.exceptions import StorageError
    from llmeval.schema.results import SuiteRun
    from llmeval.storage import SQLiteStorage

    if fmt not in ("table", "json"):
        _abort(f"Unknown format {fmt!r}. Use 'table' or 'json'.", ExitCode.USAGE_ERROR)
        return

    async def _list() -> list[SuiteRun]:
        async with SQLiteStorage(_db_path(db)) as storage:
            return await storage.list_runs(
                suite_name=suite,
                model=model,
                status=status,
                tag=tag,
                tag_match=tag_match,
                limit=limit,
            )

    try:
        runs = asyncio.run(_list())
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    if fmt == "json":
        rows = [
            {
                "run_id": r.run_id,
                "suite_name": r.suite_name,
                "model": r.model,
                "status": r.status,
                "total_tests": r.total_tests,
                "passed_tests": r.passed_tests,
                "failed_tests": r.failed_tests,
                "errored_tests": r.errored_tests,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "tags": r.tags,
                "labels": r.labels,
            }
            for r in runs
        ]
        sys.stdout.write(_json.dumps(rows, indent=2) + "\n")
        return

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(box=box.SIMPLE_HEAD, show_footer=False)
    table.add_column("Run ID", style="dim", no_wrap=True)
    table.add_column("Suite")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Pass rate", justify="right")
    table.add_column("Tests", justify="right")
    table.add_column("Started", justify="right")

    for r in runs:
        rate = (
            f"{r.passed_tests / r.total_tests * 100:.1f}%"
            if r.total_tests and r.status == "completed"
            else "—"
        )
        status_style = {
            "completed": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "dim",
        }.get(r.status, "")
        started = r.started_at.strftime("%Y-%m-%d %H:%M")
        table.add_row(
            r.run_id[:8] + "…",
            r.suite_name,
            r.model,
            (
                f"[{status_style}]{r.status}[/{status_style}]"
                if status_style
                else r.status
            ),
            rate,
            str(r.total_tests) if r.status == "completed" else "—",
            started,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@app.command()
def export(
    run_id: str = typer.Argument(..., help="Run ID (or prefix) to export."),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json or csv.",
    ),
    out: str | None = typer.Option(
        None, "--out", "-o", help="Output file path. Defaults to stdout."
    ),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Export a run to JSON or CSV.

    JSON exports the full run including all test results and judge reasoning.
    CSV exports one row per test result, suitable for spreadsheet analysis.

    Examples::

        llmeval export abc123 --format json --out run.json
        llmeval export abc123 --format csv | grep FAIL
    """
    from llmeval.exceptions import StorageError
    from llmeval.schema.results import SuiteRun
    from llmeval.storage import SQLiteStorage

    if format not in ("json", "csv"):
        _abort(f"Unknown format {format!r}. Use 'json' or 'csv'.", ExitCode.USAGE_ERROR)
        return

    async def _fetch() -> SuiteRun:
        async with SQLiteStorage(_db_path(db)) as storage:
            return await storage.get_run(run_id)

    try:
        stored_run = asyncio.run(_fetch())
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    content = _render_export(stored_run, format)

    if out:
        try:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(content)
            console.print(f"[dim]Exported to {out}[/dim]")
        except OSError as exc:
            _abort(f"Could not write to {out!r}: {exc}", ExitCode.USAGE_ERROR)
    else:
        sys.stdout.write(content)


def _render_export(run: object, format: str) -> str:
    """Serialise *run* to *format* (``"json"`` or ``"csv"``)."""
    from llmeval.schema.results import SuiteRun

    suite_run: SuiteRun = run  # type: ignore[assignment]

    if format == "json":
        return suite_run.model_dump_json(indent=2) + "\n"

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "run_id",
            "suite_name",
            "suite_version",
            "model",
            "judge_model",
            "status",
            "started_at",
            "completed_at",
            "test_id",
            "passed",
            "weighted_score",
            "passing_threshold",
            "error",
            "criterion_scores",
        ]
    )
    for result in suite_run.results:
        cs = " | ".join(f"{c.name}:{c.score:.3f}" for c in result.criterion_scores)
        writer.writerow(
            [
                suite_run.run_id,
                suite_run.suite_name,
                suite_run.suite_version,
                suite_run.model,
                suite_run.judge_model,
                suite_run.status,
                suite_run.started_at.isoformat(),
                suite_run.completed_at.isoformat() if suite_run.completed_at else "",
                result.test_id,
                result.passed,
                f"{result.weighted_score:.4f}",
                (
                    f"{result.passing_threshold:.2f}"
                    if result.passing_threshold is not None
                    else ""
                ),
                result.error or "",
                cs,
            ]
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


@app.command()
def cancel(
    run_id: str = typer.Argument(..., help="Run ID (or prefix) to cancel."),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Cancel a pending or running job started via ``llmeval serve``.

    Has no effect on runs executed via ``llmeval run`` (those are synchronous).
    Exits 3 when the run is not found or is already in a terminal state.
    """
    from llmeval.exceptions import StorageError
    from llmeval.storage import SQLiteStorage

    async def _do_cancel() -> None:
        async with SQLiteStorage(_db_path(db)) as storage:
            await storage.cancel_run(run_id)

    try:
        asyncio.run(_do_cancel())
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    console.print(f"[dim]Run {run_id[:8]}… cancelled.[/dim]")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="Baseline run ID (A)."),
    run_b: str | None = typer.Argument(
        None,
        help=(
            "Candidate run ID (B). Omit to compare against the previous run "
            "of the same suite automatically."
        ),
    ),
    fail_on_regression: bool = typer.Option(
        False,
        "--fail-on-regression",
        help="Exit 1 if any regressions are detected. Designed for CI pipelines.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit a machine-readable JSON summary instead of the Rich table.",
    ),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Compare two runs and optionally fail CI when regressions are found.

    When *run_b* is omitted the most recent completed run of the same suite
    that predates *run_a* is used automatically.

    Exit codes:

    - **0** — comparison complete (or no regressions when ``--fail-on-regression``).
    - **1** — regressions detected (only when ``--fail-on-regression``).
    - **3** — storage error.

    JSON output schema (with ``--json``)::

        {
          "run_a": {"run_id": "...", "model": "...", ...},
          "run_b": {"run_id": "...", "model": "...", ...},
          "total": 10,
          "regressions": 1,
          "improvements": 2,
          "unchanged": 7,
          "tests": [{"test_id": "t1", "is_regression": true, "score_delta": -0.3, ...}]
        }
    """
    from llmeval.exceptions import StorageError
    from llmeval.report import DiffReporter
    from llmeval.report.diff_reporter import compute_diff
    from llmeval.schema.results import SuiteRun
    from llmeval.storage import SQLiteStorage

    async def _fetch() -> tuple[SuiteRun, SuiteRun]:
        async with SQLiteStorage(_db_path(db)) as storage:
            a = await storage.get_run(run_a)
            b = (
                await storage.get_run(run_b)
                if run_b is not None
                else await storage.get_previous_run(a.suite_name, a.run_id)
            )
            return a, b

    try:
        run_a_obj, run_b_obj = asyncio.run(_fetch())
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    diffs = compute_diff(run_a_obj, run_b_obj)
    regressions = [d for d in diffs if d.is_regression]
    improvements = [d for d in diffs if d.is_improvement]
    unchanged = [
        d
        for d in diffs
        if not d.is_regression and not d.is_improvement and d.result_a and d.result_b
    ]

    if json_out:
        summary = {
            "run_a": {
                "run_id": run_a_obj.run_id,
                "suite_name": run_a_obj.suite_name,
                "model": run_a_obj.model,
                "total_tests": run_a_obj.total_tests,
                "passed_tests": run_a_obj.passed_tests,
            },
            "run_b": {
                "run_id": run_b_obj.run_id,
                "suite_name": run_b_obj.suite_name,
                "model": run_b_obj.model,
                "total_tests": run_b_obj.total_tests,
                "passed_tests": run_b_obj.passed_tests,
            },
            "total": len(diffs),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "unchanged": len(unchanged),
            "tests": [
                {
                    "test_id": d.test_id,
                    "is_regression": d.is_regression,
                    "is_improvement": d.is_improvement,
                    "score_delta": d.score_delta,
                    "score_a": (
                        d.result_a.weighted_score
                        if d.result_a and not d.result_a.error
                        else None
                    ),
                    "score_b": (
                        d.result_b.weighted_score
                        if d.result_b and not d.result_b.error
                        else None
                    ),
                }
                for d in diffs
            ],
        }
        sys.stdout.write(_json.dumps(summary, indent=2) + "\n")
    else:
        DiffReporter(console).print_diff(run_a_obj, run_b_obj)

    if fail_on_regression and regressions:
        raise typer.Exit(code=ExitCode.TEST_FAILURE)


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+)(d|h|m)$")


def _parse_duration_seconds(value: str) -> int:
    """Parse a duration string like ``30d``, ``12h``, or ``90m`` into seconds.

    Raises :class:`typer.BadParameter` on invalid input.
    """
    m = _DURATION_RE.match(value.strip())
    if not m:
        raise typer.BadParameter(
            f"{value!r} is not a valid duration. Use NNd, NNh, or NNm "
            "(e.g. 30d, 12h, 90m)."
        )
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 86400, "h": 3600, "m": 60}[unit]


@app.command()
def prune(
    older_than: str = typer.Option(
        ...,
        "--older-than",
        help="Delete runs older than this duration (e.g. 30d, 12h, 90m).",
    ),
    status: str | None = typer.Option(
        None,
        "--status",
        help="Limit deletion to runs with this status (e.g. failed, cancelled).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help=(
            "Actually delete. Without this flag, "
            "shows what would be deleted (dry run)."
        ),
    ),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Delete old runs from storage.

    Defaults to a **dry run** — pass ``--yes`` to actually delete. Combine
    ``--status failed`` to only prune runs that never completed successfully.

    Examples::

        llmeval prune --older-than 30d --dry-run
        llmeval prune --older-than 7d --status failed --yes
    """
    import time
    from datetime import UTC, datetime

    from llmeval.exceptions import StorageError
    from llmeval.schema.results import SuiteRun
    from llmeval.storage import SQLiteStorage

    try:
        cutoff_seconds = _parse_duration_seconds(older_than)
    except typer.BadParameter as exc:
        _abort(str(exc), ExitCode.USAGE_ERROR)
        return

    cutoff = datetime.fromtimestamp(time.time() - cutoff_seconds, tz=UTC)

    async def _fetch_and_prune() -> tuple[list[SuiteRun], int]:
        async with SQLiteStorage(_db_path(db)) as storage:
            candidates = await storage.list_runs(
                status=status,
                date_to=cutoff,
                limit=10_000,
            )
            if yes:
                for run in candidates:
                    await storage.delete_run(run.run_id)
            return candidates, len(candidates)

    try:
        candidates, count = asyncio.run(_fetch_and_prune())
    except StorageError as exc:
        _abort(str(exc), ExitCode.STORAGE_ERROR)
        return

    if count == 0:
        console.print("[dim]No runs found matching the criteria.[/dim]")
        return

    if not yes:
        console.print(
            f"[yellow]Dry run:[/yellow] would delete {count} run(s) "
            f"started before {cutoff.strftime('%Y-%m-%d %H:%M UTC')}."
        )
        for r in candidates[:10]:
            console.print(
                f"  [dim]{r.run_id[:8]}…[/dim] {r.suite_name} "
                f"[{r.status}] {r.started_at.strftime('%Y-%m-%d')}"
            )
        if count > 10:
            console.print(f"  … and {count - 10} more.")
        console.print("[dim]Pass --yes to actually delete.[/dim]")
    else:
        console.print(
            f"[green]Deleted {count} run(s)[/green] older than "
            f"{cutoff.strftime('%Y-%m-%d %H:%M UTC')}."
        )


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command(name="init")
def init_project(
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing files without prompting.",
    ),
) -> None:
    """Scaffold a new llmeval project in the current directory.

    Creates ``tests/fixtures/example_suite.yaml`` and ``.env.example``.
    Skips any file that already exists unless ``--force`` is passed.
    """
    import shutil
    from pathlib import Path

    pkg_root = Path(__file__).parent
    fixture_src = pkg_root.parent / "tests" / "fixtures" / "example_suite.yaml"
    env_src = pkg_root.parent / ".env.example"

    targets = [
        (fixture_src, Path("tests/fixtures/example_suite.yaml")),
        (env_src, Path(".env.example")),
    ]

    for src, dst in targets:
        if not src.exists():
            _err.print(f"[dim]Source not found, skipping: {src}[/dim]")
            continue
        if dst.exists() and not force:
            console.print(f"[dim]Already exists, skipping: {dst}[/dim]")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        console.print(f"[green]Created[/green] {dst}")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload (development only)."
    ),
) -> None:
    """Start the dashboard API server.

    The React dashboard (``dashboard/``) proxies ``/api`` requests here.
    Open http://localhost:5173 after starting ``npm run dev`` in the
    ``dashboard/`` directory.

    Note: This server maintains an in-process task registry for cancellation.
    Do not run multiple workers (``--workers`` is intentionally not exposed).
    """
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        _abort("uvicorn is not installed. " "Run: poetry install --with server")
        return

    from llmeval.server.api import create_app

    db_path = _db_path(db)
    console.print(
        f"Starting llmeval API on [cyan]http://{host}:{port}[/cyan] "
        f"— database: [dim]{db_path}[/dim]"
    )
    uvicorn.run(
        create_app(db_path=db_path),
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
