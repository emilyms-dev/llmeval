"""FastAPI application for the llmeval dashboard backend.

Endpoints:

    GET  /api/suites                            — discover local suite YAML files
    GET  /api/runs                              — paginated run list
    POST /api/runs                              — trigger a new eval run
    GET  /api/runs/{run_id}                     — full run with results
    GET  /api/runs/{run_id_a}/diff/{run_id_b}   — per-test diff

Run ``llmeval serve`` to start the server, or mount ``create_app()`` into
an existing ASGI application.
"""

from __future__ import annotations

import glob
import os
from datetime import UTC, datetime

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llmeval.exceptions import ConfigurationError, SchemaValidationError, StorageError
from llmeval.report.diff_reporter import TestDiff, compute_diff
from llmeval.schema.results import SuiteRun, TestResult
from llmeval.storage.sqlite import SQLiteStorage

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Lightweight run record for the list endpoint."""

    run_id: str
    suite_name: str
    suite_version: str
    model: str
    judge_model: str
    status: str
    suite_path: str | None
    tags: list[str]
    started_at: str
    completed_at: str | None
    total_tests: int
    passed_tests: int
    failed_tests: int
    errored_tests: int
    error_message: str | None


class RunRequest(BaseModel):
    """Body for POST /api/runs."""

    suite_path: str
    model: str | None = None
    tags: list[str] = []
    concurrency: int = 5


class RunStarted(BaseModel):
    """Immediate response from POST /api/runs."""

    run_id: str
    status: str = "running"


class TestDiffOut(BaseModel):
    """Serialisable form of :class:`~llmeval.report.diff_reporter.TestDiff`."""

    test_id: str
    result_a: TestResult | None
    result_b: TestResult | None
    is_regression: bool
    is_improvement: bool
    score_delta: float | None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(db_path: str = "llmeval.db") -> FastAPI:
    """Instantiate and configure the FastAPI application.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A configured :class:`fastapi.FastAPI` instance ready to serve.
    """
    load_dotenv()

    app = FastAPI(
        title="llmeval",
        description="LLM evaluation framework dashboard API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    async def _storage() -> SQLiteStorage:
        async with SQLiteStorage(db_path) as storage:
            yield storage  # type: ignore[misc]

    # -----------------------------------------------------------------------
    # Suite discovery
    # -----------------------------------------------------------------------

    @app.get("/api/suites", response_model=list[str])
    async def list_suites() -> list[str]:
        """Return relative paths to YAML suite files near the working directory."""
        patterns = ["*.yaml", "*.yml", "tests/**/*.yaml", "tests/**/*.yml"]
        found: list[str] = []
        for pat in patterns:
            found.extend(glob.glob(pat, recursive=True))
        # Deduplicate, sort, normalise separators
        return sorted({os.path.normpath(p) for p in found})

    # -----------------------------------------------------------------------
    # Run list & trigger
    # -----------------------------------------------------------------------

    @app.get("/api/runs", response_model=list[RunSummary])
    async def list_runs(
        storage: SQLiteStorage = Depends(_storage),
        suite: str | None = Query(None, description="Filter by suite name"),
        limit: int = Query(20, ge=1, le=200, description="Maximum results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
    ) -> list[RunSummary]:
        """Return a paginated list of suite runs, most recent first."""
        try:
            runs = await storage.list_runs(suite_name=suite, limit=limit, offset=offset)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return [
            RunSummary(
                run_id=r.run_id,
                suite_name=r.suite_name,
                suite_version=r.suite_version,
                model=r.model,
                judge_model=r.judge_model,
                status=r.status,
                suite_path=r.suite_path,
                tags=r.tags,
                started_at=r.started_at.isoformat(),
                completed_at=(r.completed_at.isoformat() if r.completed_at else None),
                total_tests=r.total_tests,
                passed_tests=r.passed_tests,
                failed_tests=r.failed_tests,
                errored_tests=r.errored_tests,
                error_message=r.error_message,
            )
            for r in runs
        ]

    @app.post("/api/runs", response_model=RunStarted, status_code=202)
    async def trigger_run(
        req: RunRequest,
        background_tasks: BackgroundTasks,
        storage: SQLiteStorage = Depends(_storage),
    ) -> RunStarted:
        """Trigger a new eval run. Returns immediately with the run ID.

        The pipeline executes as a background task — poll GET /api/runs/{run_id}
        until ``completed_at`` is set.
        """
        from llmeval.models import create_adapter
        from llmeval.schema.test_suite import load_suite

        try:
            suite_def = load_suite(req.suite_path)
        except (SchemaValidationError, ConfigurationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        model_name = req.model or suite_def.suite.model

        try:
            create_adapter(model_name)
            create_adapter(suite_def.suite.judge_model)
        except ConfigurationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Save a pending run record immediately so the frontend can track it.
        pending = SuiteRun(
            suite_name=suite_def.suite.name,
            suite_version=suite_def.suite.version,
            model=model_name,
            judge_model=suite_def.suite.judge_model,
            status="pending",
            suite_path=req.suite_path,
            tags=req.tags,
            concurrency=req.concurrency,
            started_at=datetime.now(UTC),
        )
        try:
            await storage.save_run(pending)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        background_tasks.add_task(
            _run_pipeline,
            run_id=pending.run_id,
            suite_path=req.suite_path,
            model_name=model_name,
            tags=req.tags or None,
            concurrency=req.concurrency,
            db_path=db_path,
        )

        return RunStarted(run_id=pending.run_id)

    # -----------------------------------------------------------------------
    # Run detail & diff
    # -----------------------------------------------------------------------

    @app.get("/api/runs/{run_id}", response_model=SuiteRun)
    async def get_run(
        run_id: str, storage: SQLiteStorage = Depends(_storage)
    ) -> SuiteRun:
        """Return a single run including all test results."""
        try:
            return await storage.get_run(run_id)
        except StorageError as exc:
            msg = str(exc)
            code = 404 if "not found" in msg.lower() else 500
            raise HTTPException(status_code=code, detail=msg) from exc

    @app.get("/api/runs/{run_id}/previous", response_model=SuiteRun)
    async def get_previous_run(
        run_id: str, storage: SQLiteStorage = Depends(_storage)
    ) -> SuiteRun:
        """Return the most recent completed run of the same suite before *run_id*."""
        try:
            current = await storage.get_run(run_id)
            return await storage.get_previous_run(current.suite_name, current.run_id)
        except StorageError as exc:
            msg = str(exc)
            code = 404 if "not found" in msg.lower() else 500
            raise HTTPException(status_code=code, detail=msg) from exc

    @app.get(
        "/api/runs/{run_id_a}/diff/{run_id_b}",
        response_model=list[TestDiffOut],
    )
    async def diff_runs(
        run_id_a: str,
        run_id_b: str,
        storage: SQLiteStorage = Depends(_storage),
    ) -> list[TestDiffOut]:
        """Return a per-test diff between two runs."""
        try:
            run_a = await storage.get_run(run_id_a)
            run_b = await storage.get_run(run_id_b)
        except StorageError as exc:
            msg = str(exc)
            code = 404 if "not found" in msg.lower() else 500
            raise HTTPException(status_code=code, detail=msg) from exc

        diffs: list[TestDiff] = compute_diff(run_a, run_b)
        return [
            TestDiffOut(
                test_id=d.test_id,
                result_a=d.result_a,
                result_b=d.result_b,
                is_regression=d.is_regression,
                is_improvement=d.is_improvement,
                score_delta=d.score_delta,
            )
            for d in diffs
        ]

    return app


# ---------------------------------------------------------------------------
# Background pipeline (runs outside the request/response cycle)
# ---------------------------------------------------------------------------


async def _run_pipeline(
    run_id: str,
    suite_path: str,
    model_name: str,
    tags: list[str] | None,
    concurrency: int,
    db_path: str,
) -> None:
    """Execute the full run → score → save pipeline for a triggered run."""
    from llmeval.judge import Judge
    from llmeval.models import create_adapter
    from llmeval.runner import Runner
    from llmeval.schema.results import SuiteRun
    from llmeval.schema.test_suite import load_suite

    async def _save(run: SuiteRun) -> None:
        async with SQLiteStorage(db_path) as storage:
            await storage.save_run(run)

    try:
        suite_def = load_suite(suite_path)
        runner_adapter = create_adapter(model_name)
        judge_adapter = create_adapter(suite_def.suite.judge_model)

        # Fetch the pending record so we preserve all its metadata.
        async with SQLiteStorage(db_path) as storage:
            pending = await storage.get_run(run_id)

        # Transition to running.
        await _save(pending.model_copy(update={"status": "running"}))

        runner = Runner(runner_adapter, concurrency=concurrency)
        suite_run = await runner.run(suite_def, tags=tags, suite_path=suite_path)
        # Preserve the run_id and metadata from the pending record.
        suite_run = suite_run.model_copy(
            update={"run_id": run_id, "status": "running"}
        )

        judge = Judge(judge_adapter, concurrency=concurrency)
        suite_run = await judge.score_suite_run(suite_run, suite_def)
        suite_run = suite_run.model_copy(
            update={"run_id": run_id, "status": "completed"}
        )

    except Exception as exc:
        suite_run = SuiteRun(
            run_id=run_id,
            suite_name="(error)",
            suite_version="0",
            model=model_name,
            judge_model="",
            status="failed",
            suite_path=suite_path,
            tags=list(tags) if tags else [],
            concurrency=concurrency,
            completed_at=datetime.now(UTC),
            error_message=str(exc),
            results=[],
        )

    await _save(suite_run)
