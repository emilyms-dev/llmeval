"""Typer CLI entrypoint for llmeval.

Commands are thin wrappers around the core library. Heavy imports are
deferred inside each command so startup time stays fast.
"""

from __future__ import annotations

import asyncio
import os

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


def _db_path(db: str | None) -> str:
    return db or os.environ.get("LLMEVAL_DB_PATH", "llmeval.db")


def _abort(msg: str) -> None:
    _err.print(f"[bold red]Error:[/bold red] {msg}")
    raise typer.Exit(code=2)


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
        help="Override the model in the suite (e.g. gpt-4o, claude-opus-4-20250514).",
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
) -> None:
    """Run a test suite, score outputs with LLM-as-judge, and print results.

    Exits 0 when all tests pass, 1 when any test fails or errors,
    and 2 on configuration or I/O errors.
    """
    from dotenv import load_dotenv

    load_dotenv()

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
        _abort(str(exc))
        return

    model_name = model or suite_def.suite.model
    tags = tag or None

    try:
        runner_adapter = create_adapter(model_name)
        judge_adapter = create_adapter(suite_def.suite.judge_model)
    except ConfigurationError as exc:
        _abort(str(exc))
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
            judge = Judge(judge_adapter, concurrency=concurrency)
            suite_run = await judge.score_suite_run(suite_run, suite_def)

        if not no_save:
            async with SQLiteStorage(_db_path(db)) as storage:
                await storage.save_run(suite_run)
            console.print(f"[dim]Run saved: {suite_run.run_id}[/dim]")

        return suite_run

    try:
        suite_run = asyncio.run(_pipeline())
    except (RunnerError, JudgeError, StorageError) as exc:
        _abort(str(exc))
        return

    CliReporter(console).print_run(suite_run)

    if suite_run.failed_tests > 0 or suite_run.errored_tests > 0:
        raise typer.Exit(code=1)


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
    from dotenv import load_dotenv

    load_dotenv()

    from llmeval.exceptions import StorageError
    from llmeval.report import CliReporter
    from llmeval.storage import SQLiteStorage

    async def _fetch() -> object:
        async with SQLiteStorage(_db_path(db)) as storage:
            return await storage.get_run(run_id)

    try:
        stored_run = asyncio.run(_fetch())
    except StorageError as exc:
        _abort(str(exc))
        return

    CliReporter(console).print_run(stored_run)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@app.command()
def diff(
    run_a: str = typer.Argument(..., help="Run ID of the baseline run (A)."),
    run_b: str = typer.Argument(..., help="Run ID of the candidate run (B)."),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """Compare two stored runs side by side."""
    from dotenv import load_dotenv

    load_dotenv()

    from llmeval.exceptions import StorageError
    from llmeval.report import DiffReporter
    from llmeval.schema.results import SuiteRun
    from llmeval.storage import SQLiteStorage

    async def _fetch() -> tuple[SuiteRun, SuiteRun]:
        async with SQLiteStorage(_db_path(db)) as storage:
            return await storage.get_run(run_a), await storage.get_run(run_b)

    try:
        run_a_obj, run_b_obj = asyncio.run(_fetch())
    except StorageError as exc:
        _abort(str(exc))
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
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum runs to show."),
    db: str | None = typer.Option(None, "--db", help="SQLite database path."),
) -> None:
    """List stored suite runs, most recent first."""
    from dotenv import load_dotenv

    load_dotenv()

    from llmeval.exceptions import StorageError
    from llmeval.schema.results import SuiteRun
    from llmeval.storage import SQLiteStorage

    async def _list() -> list[SuiteRun]:
        async with SQLiteStorage(_db_path(db)) as storage:
            return await storage.list_runs(suite_name=suite, limit=limit)

    try:
        runs = asyncio.run(_list())
    except StorageError as exc:
        _abort(str(exc))
        return

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(box=box.SIMPLE_HEAD, show_footer=False)
    table.add_column("Run ID", style="dim", no_wrap=True)
    table.add_column("Suite")
    table.add_column("Model")
    table.add_column("Pass rate", justify="right")
    table.add_column("Tests", justify="right")
    table.add_column("Started", justify="right")

    for r in runs:
        rate = f"{r.passed_tests / r.total_tests * 100:.1f}%" if r.total_tests else "—"
        started = r.started_at.strftime("%Y-%m-%d %H:%M")
        table.add_row(
            r.run_id[:8] + "…",
            r.suite_name,
            r.model,
            rate,
            str(r.total_tests),
            started,
        )

    console.print(table)


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
