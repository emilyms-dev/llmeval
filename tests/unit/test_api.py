"""Unit tests for llmeval.server.api.

Uses httpx.AsyncClient with the FastAPI ASGI transport so no real network
or server process is needed. Storage is backed by an in-memory SQLite DB
seeded with fixture data.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from llmeval.schema.results import CriterionScore, SuiteRun, TestResult
from llmeval.server.api import create_app
from llmeval.storage.sqlite import SQLiteStorage

from datetime import UTC, datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(test_id: str, *, passed: bool = True, error: str | None = None) -> TestResult:
    return TestResult(
        test_id=test_id,
        prompt="p",
        model="m",
        raw_output="r" if not error else "",
        criterion_scores=(
            [] if error else [CriterionScore(name="quality", score=0.9 if passed else 0.3, reasoning="ok")]
        ),
        weighted_score=0.9 if passed else 0.3,
        passed=passed,
        error=error,
    )


def _run(**kwargs: object) -> SuiteRun:
    started = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    defaults: dict[str, object] = dict(
        suite_name="API Test Suite",
        suite_version="1.0.0",
        model="claude-sonnet-4-20250514",
        judge_model="claude-sonnet-4-20250514",
        started_at=started,
        completed_at=started + timedelta(seconds=3),
        results=[_result("t1"), _result("t2")],
    )
    defaults.update(kwargs)
    return SuiteRun(**defaults)  # type: ignore[arg-type]


async def _seed(db_path: str, *runs: SuiteRun) -> None:
    async with SQLiteStorage(db_path) as storage:
        for run in runs:
            await storage.save_run(run)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def seeded_app(tmp_path: object) -> tuple[object, SuiteRun]:
    """Returns (FastAPI app, seeded SuiteRun) backed by a tmp SQLite file."""
    import tempfile, os
    db_file = os.path.join(str(tmp_path), "test.db")
    run = _run()
    await _seed(db_file, run)
    return create_app(db_path=db_file), run


# ---------------------------------------------------------------------------
# GET /api/runs
# ---------------------------------------------------------------------------


class TestListRuns:
    @pytest.mark.asyncio
    async def test_returns_200(self, seeded_app: tuple) -> None:
        app, _ = seeded_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/runs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_run_fields(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            data = (await client.get("/api/runs")).json()
        assert len(data) == 1
        assert data[0]["run_id"] == run.run_id
        assert data[0]["suite_name"] == "API Test Suite"
        assert data[0]["total_tests"] == 2
        assert data[0]["passed_tests"] == 2

    @pytest.mark.asyncio
    async def test_suite_filter(self, tmp_path: object) -> None:
        import os
        db_file = os.path.join(str(tmp_path), "filter.db")
        run_a = _run(suite_name="Suite A")
        run_b = _run(suite_name="Suite B")
        await _seed(db_file, run_a, run_b)
        app = create_app(db_path=db_file)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            data = (await client.get("/api/runs?suite=Suite+A")).json()
        assert len(data) == 1
        assert data[0]["suite_name"] == "Suite A"

    @pytest.mark.asyncio
    async def test_empty_returns_empty_list(self, tmp_path: object) -> None:
        import os
        db_file = os.path.join(str(tmp_path), "empty.db")
        app = create_app(db_path=db_file)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            data = (await client.get("/api/runs")).json()
        assert data == []


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}
# ---------------------------------------------------------------------------


class TestGetRun:
    @pytest.mark.asyncio
    async def test_returns_full_run(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            data = (await client.get(f"/api/runs/{run.run_id}")).json()
        assert data["run_id"] == run.run_id
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_prefix_lookup_works(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run.run_id[:8]}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, seeded_app: tuple) -> None:
        app, _ = seeded_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/runs/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id_a}/diff/{run_id_b}
# ---------------------------------------------------------------------------


class TestDiffRuns:
    @pytest.mark.asyncio
    async def test_returns_diff_list(self, tmp_path: object) -> None:
        import os
        db_file = os.path.join(str(tmp_path), "diff.db")
        run_a = _run(results=[_result("t1", passed=True), _result("t2", passed=True)])
        run_b = _run(results=[_result("t1", passed=False), _result("t2", passed=True)])
        await _seed(db_file, run_a, run_b)
        app = create_app(db_path=db_file)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            data = (await client.get(f"/api/runs/{run_a.run_id}/diff/{run_b.run_id}")).json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_regression_flagged(self, tmp_path: object) -> None:
        import os
        db_file = os.path.join(str(tmp_path), "reg.db")
        run_a = _run(results=[_result("t1", passed=True)])
        run_b = _run(results=[_result("t1", passed=False)])
        await _seed(db_file, run_a, run_b)
        app = create_app(db_path=db_file)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            data = (await client.get(f"/api/runs/{run_a.run_id}/diff/{run_b.run_id}")).json()
        assert data[0]["is_regression"] is True
        assert data[0]["is_improvement"] is False

    @pytest.mark.asyncio
    async def test_missing_run_returns_404(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run.run_id}/diff/no-such-run")
        assert resp.status_code == 404
