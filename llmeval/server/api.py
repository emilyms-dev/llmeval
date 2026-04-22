"""FastAPI application for the llmeval dashboard backend.

Endpoints:

    GET  /api/suites                            — discover local suite YAML files
    GET  /api/runs                              — paginated, filtered run list
    POST /api/runs                              — trigger a new eval run
    GET  /api/runs/{run_id}/status              — lightweight status polling
    POST /api/runs/{run_id}/cancel              — cancel a pending or running job
    GET  /api/runs/{run_id}                     — full run with results
    GET  /api/runs/{run_id}/previous            — previous run of same suite
    GET  /api/runs/{run_id}/export              — download as JSON or CSV
    GET  /api/runs/{run_id_a}/diff/{run_id_b}   — per-test diff

Run ``llmeval serve`` to start the server, or mount ``create_app()`` into
an existing ASGI application.

Security posture
----------------
* Set ``LLMEVAL_API_TOKEN`` to enable bearer-token auth on all mutating
  endpoints (``POST /api/runs``, ``POST /api/runs/{id}/cancel``).
  When unset the server runs in **open mode** and logs a startup warning —
  acceptable for local dev, never for production.
* ``LLMEVAL_CORS_ORIGIN`` (comma-separated, default ``http://localhost:5173``)
  controls the ``Access-Control-Allow-Origin`` allowlist.
* ``LLMEVAL_SUITES_DIR`` (default: CWD) confines suite-path resolution.
  Any ``suite_path`` that resolves outside this directory is rejected with
  422 before reaching ``load_suite``.
* ``POST /api/runs`` is rate-limited to 10 requests/minute per IP.
"""

from __future__ import annotations

import asyncio
import csv
import glob
import io
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from llmeval.exceptions import ConfigurationError, SchemaValidationError, StorageError
from llmeval.report.diff_reporter import TestDiff, compute_diff
from llmeval.schema.results import RunStatus, SuiteRun, TestResult
from llmeval.storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process task registry — maps run_id → active asyncio.Task
# Single-process only; sufficient for the default uvicorn deployment.
# ---------------------------------------------------------------------------

_active_tasks: dict[str, asyncio.Task[None]] = {}
_security = HTTPBearer(auto_error=False)

# Simple in-memory rate limiter for POST /api/runs (10 req/min per IP).
# slowapi is installed (pyproject.toml [server] group) but its @limiter.limit
# decorator rewrites the function signature in a way that breaks FastAPI's
# body-parameter detection in Python 3.13. We implement the same sliding-window
# logic as a Depends to avoid that conflict.
_post_runs_calls: dict[str, list[float]] = defaultdict(list)

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
    labels: dict[str, str]
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
    labels: dict[str, str] = {}
    concurrency: int = 5
    timeout: int = 1800
    samples: int = 1


class RunStarted(BaseModel):
    """Immediate response from POST /api/runs."""

    run_id: str
    status: str = "pending"


class RunStatusOut(BaseModel):
    """Lightweight status response for polling."""

    run_id: str
    status: RunStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    errored_tests: int
    completed_at: str | None


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

    api_token: str | None = os.environ.get("LLMEVAL_API_TOKEN") or None
    if api_token is None:
        logger.warning(
            "LLMEVAL_API_TOKEN is not set. The server is running in open mode — "
            "any client can trigger runs and burn API credits. "
            "Set LLMEVAL_API_TOKEN in your environment for production use."
        )

    raw_cors = os.environ.get("LLMEVAL_CORS_ORIGIN", "http://localhost:5173")
    cors_origins = [o.strip() for o in raw_cors.split(",") if o.strip()]

    suites_dir = Path(os.environ.get("LLMEVAL_SUITES_DIR", ".")).resolve()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
        if workers > 1:
            logger.warning(
                "llmeval detected WEB_CONCURRENCY=%d. The in-process task "
                "registry for run cancellation does not support multiple "
                "workers — cancel operations will silently fail under any "
                "worker that did not start the run. Use a single worker only.",
                workers,
            )
        storage = SQLiteStorage(db_path)
        await storage.initialize()
        _app.state.storage = storage
        yield
        await storage.close()

    app = FastAPI(
        title="llmeval",
        description="LLM evaluation framework dashboard API",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def _get_storage(request: Request) -> SQLiteStorage:
        """Return the shared :class:`SQLiteStorage` initialised at startup."""
        storage: SQLiteStorage = request.app.state.storage
        return storage

    async def require_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(_security),
    ) -> None:
        """FastAPI dependency that enforces bearer-token auth on mutating endpoints.

        A no-op when ``LLMEVAL_API_TOKEN`` is unset (open mode).
        """
        if api_token is None:
            return
        if credentials is None or credentials.credentials != api_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Authorization token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # -----------------------------------------------------------------------
    # Suite discovery
    # -----------------------------------------------------------------------

    @app.get("/api/suites", response_model=list[str])
    async def list_suites() -> list[str]:
        """Return relative paths to YAML suite files inside LLMEVAL_SUITES_DIR."""
        patterns = ["*.yaml", "*.yml", "tests/**/*.yaml", "tests/**/*.yml"]
        found: list[str] = []
        for pat in patterns:
            full_pattern = str(suites_dir / pat)
            for p in glob.glob(full_pattern, recursive=True):
                try:
                    found.append(str(Path(p).relative_to(suites_dir)))
                except ValueError:
                    pass
        return sorted(set(found))

    # -----------------------------------------------------------------------
    # Run list & trigger
    # -----------------------------------------------------------------------

    @app.get("/api/runs", response_model=list[RunSummary])
    async def list_runs(
        storage: SQLiteStorage = Depends(_get_storage),
        suite: str | None = Query(None, description="Filter by suite name"),
        model: str | None = Query(None, description="Filter by model"),
        status_filter: str | None = Query(
            None, alias="status", description="Filter by status"
        ),
        tag: str | None = Query(None, description="Filter by tag"),
        tag_match: str = Query("exact", description="Tag match mode: exact or fuzzy"),
        date_from: str | None = Query(
            None, description="Filter started_at >= date (ISO format)"
        ),
        date_to: str | None = Query(
            None, description="Filter started_at <= date (ISO format)"
        ),
        limit: int = Query(20, ge=1, le=200, description="Maximum results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
    ) -> list[RunSummary]:
        """Return a paginated, filtered list of suite runs, most recent first."""
        df: datetime | None = None
        dt: datetime | None = None
        if date_from:
            try:
                df = datetime.fromisoformat(date_from).replace(tzinfo=UTC)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"Invalid date_from: {date_from!r}"
                ) from exc
        if date_to:
            try:
                dt = datetime.fromisoformat(date_to).replace(tzinfo=UTC)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"Invalid date_to: {date_to!r}"
                ) from exc

        try:
            runs = await storage.list_runs(
                suite_name=suite,
                model=model,
                status=status_filter,
                tag=tag,
                tag_match=tag_match,
                date_from=df,
                date_to=dt,
                limit=limit,
                offset=offset,
            )
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return [_to_summary(r) for r in runs]

    async def check_rate_limit(request: Request) -> None:
        """Sliding-window rate limit: 10 POST /api/runs per IP per minute."""
        key = request.client.host if request.client else "127.0.0.1"
        now = time.monotonic()
        _post_runs_calls[key] = [t for t in _post_runs_calls[key] if now - t < 60]
        if len(_post_runs_calls[key]) >= 10:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: max 10 POST /api/runs per minute.",
            )
        _post_runs_calls[key].append(now)

    @app.post("/api/runs", response_model=RunStarted, status_code=202)
    async def trigger_run(
        req: RunRequest,
        storage: SQLiteStorage = Depends(_get_storage),
        _auth: None = Depends(require_token),
        _rl: None = Depends(check_rate_limit),
    ) -> RunStarted:
        """Trigger a new eval run. Returns immediately with the run ID.

        The pipeline executes as a background task — poll
        ``GET /api/runs/{run_id}/status`` until ``status`` is ``completed``,
        ``failed``, or ``cancelled``.

        Returns 409 if a run for the same suite is already active (pending or
        running). Cancel the existing run first, or wait for it to finish.
        Returns 422 if ``suite_path`` resolves outside ``LLMEVAL_SUITES_DIR``.
        Returns 429 if the rate limit (10 requests/minute) is exceeded.
        """
        from llmeval.models import create_adapter
        from llmeval.schema.test_suite import load_suite

        # Path-traversal guard: suite_path must be inside the configured suites dir.
        try:
            requested_path = (suites_dir / req.suite_path).resolve()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Invalid suite_path.") from exc
        if not requested_path.is_relative_to(suites_dir):
            raise HTTPException(
                status_code=422,
                detail="suite_path must be inside the configured LLMEVAL_SUITES_DIR.",
            )

        try:
            suite_def = load_suite(str(requested_path))
        except (SchemaValidationError, ConfigurationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        model_name = req.model or suite_def.suite.model

        try:
            create_adapter(model_name)
            create_adapter(suite_def.suite.judge_model)
        except ConfigurationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Duplicate-run protection: reject if a run for this suite is already active.
        try:
            active = await storage.list_runs(
                suite_name=suite_def.suite.name, status="running", limit=1
            )
            pending_runs = await storage.list_runs(
                suite_name=suite_def.suite.name, status="pending", limit=1
            )
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if active or pending_runs:
            existing = (active or pending_runs)[0]
            raise HTTPException(
                status_code=409,
                detail=(
                    f"A run for suite {suite_def.suite.name!r} is already active "
                    f"(run_id: {existing.run_id}, status: {existing.status!r}). "
                    "Cancel it first or wait for it to complete."
                ),
            )

        pending = SuiteRun(
            suite_name=suite_def.suite.name,
            suite_version=suite_def.suite.version,
            model=model_name,
            judge_model=suite_def.suite.judge_model,
            status="pending",
            suite_path=req.suite_path,
            tags=req.tags,
            labels=req.labels,
            concurrency=req.concurrency,
            started_at=datetime.now(UTC),
        )
        try:
            await storage.save_run(pending)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        task = asyncio.create_task(
            _run_pipeline(
                run_id=pending.run_id,
                suite_path=str(requested_path),
                model_name=model_name,
                tags=req.tags or None,
                labels=req.labels,
                concurrency=req.concurrency,
                timeout=req.timeout,
                samples=req.samples,
                db_path=db_path,
            )
        )
        _active_tasks[pending.run_id] = task
        task.add_done_callback(lambda _t: _active_tasks.pop(pending.run_id, None))

        return RunStarted(run_id=pending.run_id)

    # -----------------------------------------------------------------------
    # Status polling (lightweight — no full JSON deserialisation)
    # -----------------------------------------------------------------------

    @app.get("/api/runs/{run_id}/status", response_model=RunStatusOut)
    async def get_run_status(
        run_id: str, storage: SQLiteStorage = Depends(_get_storage)
    ) -> RunStatusOut:
        """Return the current status and result counts without the full result payload.

        Reads only indexed columns — no ``run_json`` deserialisation. Designed
        for efficient polling loops in the dashboard and CI scripts.
        """
        try:
            brief = await storage.get_run_brief(run_id)
        except StorageError as exc:
            msg = str(exc)
            code = 404 if "not found" in msg.lower() else 500
            raise HTTPException(status_code=code, detail=msg) from exc

        return RunStatusOut(
            run_id=brief.run_id,
            status=brief.status,
            total_tests=brief.total_tests,
            passed_tests=brief.passed_tests,
            failed_tests=brief.failed_tests,
            errored_tests=brief.errored_tests,
            completed_at=(
                brief.completed_at.isoformat() if brief.completed_at else None
            ),
        )

    # -----------------------------------------------------------------------
    # Cancel
    # -----------------------------------------------------------------------

    @app.post("/api/runs/{run_id}/cancel", response_model=RunStarted, status_code=202)
    async def cancel_run(
        run_id: str,
        storage: SQLiteStorage = Depends(_get_storage),
        _auth: None = Depends(require_token),
    ) -> RunStarted:
        """Cancel a pending or running job.

        Updates the database to ``"cancelled"`` and interrupts the asyncio task
        if it is currently executing in this process. Returns 409 when the run
        is already in a terminal state (``completed``, ``failed``, ``cancelled``).
        """
        try:
            run = await storage.get_run(run_id)
        except StorageError as exc:
            msg = str(exc)
            code = 404 if "not found" in msg.lower() else 500
            raise HTTPException(status_code=code, detail=msg) from exc

        if run.status not in ("pending", "running"):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Run {run_id!r} has status {run.status!r} and cannot be "
                    "cancelled. Only pending or running runs can be cancelled."
                ),
            )

        # Update DB first, then cancel the task.
        try:
            await storage.cancel_run(run_id)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        task = _active_tasks.pop(run_id, None)
        if task and not task.done():
            task.cancel()

        return RunStarted(run_id=run_id, status="cancelled")

    # -----------------------------------------------------------------------
    # Run detail, previous, export, diff
    # -----------------------------------------------------------------------

    @app.get("/api/runs/{run_id}/previous", response_model=SuiteRun)
    async def get_previous_run(
        run_id: str, storage: SQLiteStorage = Depends(_get_storage)
    ) -> SuiteRun:
        """Return the most recent completed run of the same suite before *run_id*."""
        try:
            current = await storage.get_run(run_id)
            return await storage.get_previous_run(current.suite_name, current.run_id)
        except StorageError as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            code = (
                404
                if "not found" in msg_lower or "no previous" in msg_lower
                else 500
            )
            raise HTTPException(status_code=code, detail=msg) from exc

    @app.get("/api/runs/{run_id}/export")
    async def export_run(
        run_id: str,
        format: str = Query("json", description="Export format: json or csv"),
        storage: SQLiteStorage = Depends(_get_storage),
    ) -> Response:
        """Download a run as JSON or CSV.

        JSON contains the full run including all test results and judge
        reasoning. CSV has one row per test result with run metadata repeated.
        """
        if format not in ("json", "csv"):
            raise HTTPException(
                status_code=422, detail=f"Unknown format {format!r}. Use json or csv."
            )
        try:
            run = await storage.get_run(run_id)
        except StorageError as exc:
            msg = str(exc)
            code = 404 if "not found" in msg.lower() else 500
            raise HTTPException(status_code=code, detail=msg) from exc

        slug = run.run_id[:8]
        if format == "csv":
            content = _run_to_csv(run)
            return Response(
                content=content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="run-{slug}.csv"'
                },
            )
        return Response(
            content=run.model_dump_json(indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="run-{slug}.json"'},
        )

    @app.get("/api/runs/{run_id}", response_model=SuiteRun)
    async def get_run(
        run_id: str, storage: SQLiteStorage = Depends(_get_storage)
    ) -> SuiteRun:
        """Return a single run including all test results."""
        try:
            return await storage.get_run(run_id)
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
        storage: SQLiteStorage = Depends(_get_storage),
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
# Helpers
# ---------------------------------------------------------------------------


def _to_summary(r: SuiteRun) -> RunSummary:
    return RunSummary(
        run_id=r.run_id,
        suite_name=r.suite_name,
        suite_version=r.suite_version,
        model=r.model,
        judge_model=r.judge_model,
        status=r.status,
        suite_path=r.suite_path,
        tags=r.tags,
        labels=r.labels,
        started_at=r.started_at.isoformat(),
        completed_at=(r.completed_at.isoformat() if r.completed_at else None),
        total_tests=r.total_tests,
        passed_tests=r.passed_tests,
        failed_tests=r.failed_tests,
        errored_tests=r.errored_tests,
        error_message=r.error_message,
    )


def _run_to_csv(run: SuiteRun) -> str:
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
    for result in run.results:
        cs = " | ".join(f"{c.name}:{c.score:.3f}" for c in result.criterion_scores)
        writer.writerow(
            [
                run.run_id,
                run.suite_name,
                run.suite_version,
                run.model,
                run.judge_model,
                run.status,
                run.started_at.isoformat(),
                run.completed_at.isoformat() if run.completed_at else "",
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
# Background pipeline (runs outside the request/response cycle)
# ---------------------------------------------------------------------------


async def _run_pipeline(
    run_id: str,
    suite_path: str,
    model_name: str,
    tags: list[str] | None,
    labels: dict[str, str],
    concurrency: int,
    timeout: int,
    db_path: str,
    samples: int = 1,
) -> None:
    """Execute the full run → score → save pipeline for a triggered run.

    Wraps the pipeline in ``asyncio.wait_for`` so hung runs are terminated
    after *timeout* seconds. Cancellation is handled gracefully — the DB is
    updated to ``"cancelled"`` before the ``CancelledError`` propagates.
    """
    from llmeval.judge import Judge
    from llmeval.models import create_adapter
    from llmeval.runner import Runner
    from llmeval.schema.results import SuiteRun
    from llmeval.schema.test_suite import load_suite

    async def _save(run: SuiteRun) -> None:
        async with SQLiteStorage(db_path) as storage:
            await storage.save_run(run)

    async def _do_pipeline() -> SuiteRun:
        suite_def = load_suite(suite_path)
        runner_adapter = create_adapter(model_name)
        judge_adapter = create_adapter(suite_def.suite.judge_model, temperature=0.0)

        async with SQLiteStorage(db_path) as storage:
            pending = await storage.get_run(run_id)

        await _save(pending.model_copy(update={"status": "running"}))

        runner = Runner(runner_adapter, concurrency=concurrency)
        suite_run = await runner.run(suite_def, tags=tags, suite_path=suite_path)
        suite_run = suite_run.model_copy(
            update={"run_id": run_id, "status": "running", "labels": labels}
        )

        judge = Judge(judge_adapter, concurrency=concurrency, samples=samples)
        suite_run = await judge.score_suite_run(suite_run, suite_def)
        return suite_run.model_copy(
            update={
                "run_id": run_id,
                "status": "completed",
                "completed_at": datetime.now(UTC),
                "labels": labels,
            }
        )

    async def _error_state(msg: str, terminal_status: str = "failed") -> SuiteRun:
        """Return a terminal SuiteRun that preserves the stored row's metadata.

        Reads the pending/running row so suite_name, judge_model, tags, etc.
        are not replaced with placeholder values — failed runs stay visible
        to suite-filtered queries.
        """
        try:
            async with SQLiteStorage(db_path) as storage:
                current = await storage.get_run(run_id)
            return current.model_copy(
                update={
                    "status": terminal_status,
                    "completed_at": datetime.now(UTC),
                    "error_message": msg,
                }
            )
        except Exception:
            # The pending row is unreadable (very rare). Fall back to minimal
            # metadata but use "unknown" — never "(error)" — so suite filters work.
            logger.warning(
                "Could not read pending row %s when building error state; "
                "falling back to minimal SuiteRun.",
                run_id,
            )
            return SuiteRun(
                run_id=run_id,
                suite_name="unknown",
                suite_version="0",
                model=model_name,
                judge_model="unknown",
                status=terminal_status,  # type: ignore[arg-type]
                suite_path=suite_path,
                tags=list(tags) if tags else [],
                labels=labels,
                concurrency=concurrency,
                completed_at=datetime.now(UTC),
                error_message=msg,
                results=[],
            )

    try:
        suite_run = await asyncio.wait_for(_do_pipeline(), timeout=timeout)
    except asyncio.TimeoutError:
        suite_run = await _error_state(f"Run timed out after {timeout}s.")
    except asyncio.CancelledError:
        # DB was updated by the cancel endpoint; this is a best-effort safety net.
        try:
            async with SQLiteStorage(db_path) as storage:
                existing = await storage.get_run(run_id)
                if existing.status not in ("cancelled",):
                    await storage.save_run(
                        existing.model_copy(
                            update={
                                "status": "cancelled",
                                "completed_at": datetime.now(UTC),
                            }
                        )
                    )
        except Exception:
            pass
        raise
    except Exception as exc:
        suite_run = await _error_state(str(exc))

    await _save(suite_run)
