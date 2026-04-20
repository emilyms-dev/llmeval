"""Unit tests for llmeval.server.api.

Uses httpx.AsyncClient with the FastAPI ASGI transport so no real network
or server process is needed. Storage is backed by an in-memory SQLite DB
seeded with fixture data.
"""

from __future__ import annotations

import os
import tempfile  # noqa: F401 (imported for clarity; used indirectly)
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from llmeval.schema.results import CriterionScore, SuiteRun, TestResult
from llmeval.server.api import create_app
from llmeval.storage.sqlite import SQLiteStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    test_id: str, *, passed: bool = True, error: str | None = None
) -> TestResult:
    return TestResult(
        test_id=test_id,
        prompt="p",
        model="m",
        raw_output="r" if not error else "",
        criterion_scores=(
            []
            if error
            else [
                CriterionScore(
                    name="quality", score=0.9 if passed else 0.3, reasoning="ok"
                )
            ]
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


def _client(app: object) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def seeded_app(tmp_path: object) -> tuple[object, SuiteRun]:
    """Returns (FastAPI app, seeded SuiteRun) backed by a tmp SQLite file."""
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
        async with _client(app) as client:
            resp = await client.get("/api/runs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_run_fields(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            data = (await client.get("/api/runs")).json()
        assert len(data) == 1
        assert data[0]["run_id"] == run.run_id
        assert data[0]["suite_name"] == "API Test Suite"
        assert data[0]["total_tests"] == 2
        assert data[0]["passed_tests"] == 2

    @pytest.mark.asyncio
    async def test_suite_filter(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "filter.db")
        run_a = _run(suite_name="Suite A")
        run_b = _run(suite_name="Suite B")
        await _seed(db_file, run_a, run_b)
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            data = (await client.get("/api/runs?suite=Suite+A")).json()
        assert len(data) == 1
        assert data[0]["suite_name"] == "Suite A"

    @pytest.mark.asyncio
    async def test_empty_returns_empty_list(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "empty.db")
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            data = (await client.get("/api/runs")).json()
        assert data == []


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}
# ---------------------------------------------------------------------------


class TestGetRun:
    @pytest.mark.asyncio
    async def test_returns_full_run(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            data = (await client.get(f"/api/runs/{run.run_id}")).json()
        assert data["run_id"] == run.run_id
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_prefix_lookup_works(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            resp = await client.get(f"/api/runs/{run.run_id[:8]}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, seeded_app: tuple) -> None:
        app, _ = seeded_app
        async with _client(app) as client:
            resp = await client.get("/api/runs/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id_a}/diff/{run_id_b}
# ---------------------------------------------------------------------------


class TestDiffRuns:
    @pytest.mark.asyncio
    async def test_returns_diff_list(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "diff.db")
        run_a = _run(results=[_result("t1", passed=True), _result("t2", passed=True)])
        run_b = _run(results=[_result("t1", passed=False), _result("t2", passed=True)])
        await _seed(db_file, run_a, run_b)
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            url = f"/api/runs/{run_a.run_id}/diff/{run_b.run_id}"
            data = (await client.get(url)).json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_regression_flagged(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "reg.db")
        run_a = _run(results=[_result("t1", passed=True)])
        run_b = _run(results=[_result("t1", passed=False)])
        await _seed(db_file, run_a, run_b)
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            url = f"/api/runs/{run_a.run_id}/diff/{run_b.run_id}"
            data = (await client.get(url)).json()
        assert data[0]["is_regression"] is True
        assert data[0]["is_improvement"] is False

    @pytest.mark.asyncio
    async def test_missing_run_returns_404(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            resp = await client.get(f"/api/runs/{run.run_id}/diff/no-such-run")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/runs — filters
# ---------------------------------------------------------------------------


class TestListRunsFiltered:
    @pytest.mark.asyncio
    async def test_filter_by_status(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "filt.db")
        run_a = _run()
        run_b = _run()
        run_b = run_b.model_copy(update={"status": "failed", "completed_at": None})
        await _seed(db_file, run_a, run_b)
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            data = (await client.get("/api/runs?status=failed")).json()
        assert len(data) == 1
        assert data[0]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_filter_by_model(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "model.db")
        run_a = _run(model="gpt-4o")
        run_b = _run()
        await _seed(db_file, run_a, run_b)
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            data = (await client.get("/api/runs?model=gpt-4o")).json()
        assert len(data) == 1
        assert data[0]["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_invalid_date_from_returns_422(self, seeded_app: tuple) -> None:
        app, _ = seeded_app
        async with _client(app) as client:
            resp = await client.get("/api/runs?date_from=not-a-date")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}/export
# ---------------------------------------------------------------------------


class TestExportRun:
    @pytest.mark.asyncio
    async def test_json_export_returns_200(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            resp = await client.get(f"/api/runs/{run.run_id}/export?format=json")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert data["run_id"] == run.run_id

    @pytest.mark.asyncio
    async def test_csv_export_returns_200(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            resp = await client.get(f"/api/runs/{run.run_id}/export?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        lines = resp.text.strip().splitlines()
        assert lines[0].startswith("run_id")
        assert len(lines) == 3  # header + 2 results

    @pytest.mark.asyncio
    async def test_invalid_format_returns_422(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            resp = await client.get(f"/api/runs/{run.run_id}/export?format=xml")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, seeded_app: tuple) -> None:
        app, _ = seeded_app
        async with _client(app) as client:
            resp = await client.get("/api/runs/no-such-run/export")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}/previous
# ---------------------------------------------------------------------------


class TestGetPreviousRun:
    @pytest.mark.asyncio
    async def test_returns_previous_run(self, tmp_path: object) -> None:
        from datetime import timedelta

        db_file = os.path.join(str(tmp_path), "prev.db")
        started = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        older = _run()
        older = older.model_copy(
            update={
                "started_at": started,
                "completed_at": started + timedelta(seconds=3),
            }
        )
        newer = _run()
        newer = newer.model_copy(
            update={
                "started_at": started + timedelta(hours=1),
                "completed_at": started + timedelta(hours=1, seconds=3),
            }
        )
        await _seed(db_file, older, newer)
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            resp = await client.get(f"/api/runs/{newer.run_id}/previous")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == older.run_id

    @pytest.mark.asyncio
    async def test_no_previous_returns_404(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            resp = await client.get(f"/api/runs/{run.run_id}/previous")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/suites
# ---------------------------------------------------------------------------


class TestListSuites:
    @pytest.mark.asyncio
    async def test_returns_list(self, seeded_app: tuple) -> None:
        app, _ = seeded_app
        async with _client(app) as client:
            resp = await client.get("/api/suites")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

# ===========================================================================
# GET /api/runs/{run_id}/status
# ===========================================================================


class TestGetRunStatusEndpoint:
    @pytest.mark.asyncio
    async def test_returns_status_and_counts(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            resp = await client.get(f"/api/runs/{run.run_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run.run_id
        assert data["status"] == "completed"
        assert data["total_tests"] == 2
        assert data["passed_tests"] == 2

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, seeded_app: tuple) -> None:
        app, _ = seeded_app
        async with _client(app) as client:
            resp = await client.get("/api/runs/no-such-run/status")
        assert resp.status_code == 404


# ===========================================================================
# POST /api/runs/{run_id}/cancel
# ===========================================================================


class TestCancelRunEndpoint:
    @pytest.mark.asyncio
    async def test_cancel_pending_returns_202(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "cancel.db")
        run = _run()
        pending = run.model_copy(update={"status": "pending", "completed_at": None})
        await _seed(db_file, pending)
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            resp = await client.post(f"/api/runs/{pending.run_id}/cancel")
        assert resp.status_code == 202
        assert resp.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_completed_returns_409(self, seeded_app: tuple) -> None:
        app, run = seeded_app
        async with _client(app) as client:
            resp = await client.post(f"/api/runs/{run.run_id}/cancel")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_cancel_not_found_returns_404(self, seeded_app: tuple) -> None:
        app, _ = seeded_app
        async with _client(app) as client:
            resp = await client.post("/api/runs/no-such-run/cancel")
        assert resp.status_code == 404


# ===========================================================================
# Duplicate-run protection (POST /api/runs)
# ===========================================================================


class TestDuplicateRunProtection:
    @pytest.mark.asyncio
    async def test_active_run_returns_409(self, tmp_path: object) -> None:
        from unittest.mock import MagicMock, patch

        db_file = os.path.join(str(tmp_path), "dup.db")
        run = _run()
        running = run.model_copy(
            update={"status": "running", "suite_name": "Dup Suite"}
        )
        await _seed(db_file, running)
        app = create_app(db_path=db_file)

        suite_mock = MagicMock()
        suite_mock.suite.name = "Dup Suite"
        suite_mock.suite.model = "m"
        suite_mock.suite.judge_model = "j"
        suite_mock.suite.version = "1.0.0"

        with (
            patch("llmeval.schema.test_suite.load_suite", return_value=suite_mock),
            patch("llmeval.models.create_adapter"),
        ):
            async with _client(app) as client:
                resp = await client.post("/api/runs", json={"suite_path": "s.yaml"})
        assert resp.status_code == 409
        assert "already active" in resp.json()["detail"]


# ===========================================================================
# Labels in runs
# ===========================================================================


class TestLabelsInApi:
    @pytest.mark.asyncio
    async def test_labels_appear_in_list(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "labels.db")
        run = _run()
        labelled = run.model_copy(
            update={"labels": {"commit": "abc123", "branch": "main"}}
        )
        await _seed(db_file, labelled)
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            data = (await client.get("/api/runs")).json()
        assert data[0]["labels"] == {"commit": "abc123", "branch": "main"}

    @pytest.mark.asyncio
    async def test_labels_appear_in_detail(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "labels2.db")
        run = _run()
        labelled = run.model_copy(update={"labels": {"pr": "99"}})
        await _seed(db_file, labelled)
        app = create_app(db_path=db_file)
        async with _client(app) as client:
            resp = await client.get(f"/api/runs/{labelled.run_id}")
        assert resp.status_code == 200
        assert resp.json()["labels"] == {"pr": "99"}


# ===========================================================================
# Security: auth, CORS, path traversal, rate limit
# ===========================================================================


class TestAuthToken:
    """POST endpoints require a bearer token when LLMEVAL_API_TOKEN is set."""

    @pytest.mark.asyncio
    async def test_post_without_token_returns_401(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "auth.db")
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("LLMEVAL_API_TOKEN", "secret")
            secured_app = create_app(db_path=db_file)
        async with _client(secured_app) as client:
            resp = await client.post("/api/runs", json={"suite_path": "s.yaml"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_post_with_wrong_token_returns_401(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "auth2.db")
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("LLMEVAL_API_TOKEN", "correct-token")
            secured_app = create_app(db_path=db_file)
        async with _client(secured_app) as client:
            resp = await client.post(
                "/api/runs",
                json={"suite_path": "s.yaml"},
                headers={"Authorization": "Bearer wrong-token"},
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_post_with_correct_token_proceeds(self, tmp_path: object) -> None:
        from unittest.mock import MagicMock, patch

        db_file = os.path.join(str(tmp_path), "auth3.db")
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("LLMEVAL_API_TOKEN", "mytoken")
            secured_app = create_app(db_path=db_file)

        suite_mock = MagicMock()
        suite_mock.suite.name = "Auth Suite"
        suite_mock.suite.model = "m"
        suite_mock.suite.judge_model = "j"
        suite_mock.suite.version = "1.0"

        with (
            patch("llmeval.schema.test_suite.load_suite", return_value=suite_mock),
            patch("llmeval.models.create_adapter"),
            patch("asyncio.create_task"),
        ):
            async with _client(secured_app) as client:
                resp = await client.post(
                    "/api/runs",
                    json={"suite_path": "s.yaml"},
                    headers={"Authorization": "Bearer mytoken"},
                )
        # 202 Accepted (or 409 if a run is already pending — either way not 401/403)
        assert resp.status_code in (202, 409)

    @pytest.mark.asyncio
    async def test_get_endpoints_open_regardless_of_token(
        self, seeded_app: tuple
    ) -> None:
        app, _ = seeded_app
        async with _client(app) as client:
            resp = await client.get("/api/runs")
        assert resp.status_code == 200


class TestPathTraversal:
    """suite_path must not escape LLMEVAL_SUITES_DIR."""

    @pytest.mark.asyncio
    async def test_traversal_path_returns_422(self, tmp_path: object) -> None:
        db_file = os.path.join(str(tmp_path), "pt.db")
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("LLMEVAL_SUITES_DIR", str(tmp_path))
            app = create_app(db_path=db_file)
        async with _client(app) as client:
            resp = await client.post(
                "/api/runs", json={"suite_path": "../../etc/passwd.yaml"}
            )
        assert resp.status_code == 422
        # Error must not reveal whether the file exists
        assert "passwd" not in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_valid_path_inside_dir_proceeds_to_load(
        self, tmp_path: object
    ) -> None:
        from unittest.mock import MagicMock, patch

        db_file = os.path.join(str(tmp_path), "pt2.db")
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text("placeholder")

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("LLMEVAL_SUITES_DIR", str(tmp_path))
            app = create_app(db_path=db_file)

        suite_mock = MagicMock()
        suite_mock.suite.name = "PT Suite"
        suite_mock.suite.model = "m"
        suite_mock.suite.judge_model = "j"
        suite_mock.suite.version = "1.0"

        with (
            patch("llmeval.schema.test_suite.load_suite", return_value=suite_mock),
            patch("llmeval.models.create_adapter"),
            patch("asyncio.create_task"),
        ):
            async with _client(app) as client:
                resp = await client.post(
                    "/api/runs", json={"suite_path": "suite.yaml"}
                )
        assert resp.status_code == 202


class TestErrorStateSuiteMetadata:
    """Failed pipeline runs must preserve the original suite_name in storage."""

    @pytest.mark.asyncio
    async def test_pipeline_failure_preserves_suite_name(
        self, tmp_path: object
    ) -> None:
        from unittest.mock import patch

        from llmeval.server.api import _run_pipeline

        db_file = os.path.join(str(tmp_path), "errstate.db")
        run = _run(suite_name="My Real Suite", status="pending")
        await _seed(db_file, run)

        with patch(
            "llmeval.schema.test_suite.load_suite",
            side_effect=RuntimeError("load failed"),
        ):
            await _run_pipeline(
                run_id=run.run_id,
                suite_path="suite.yaml",
                model_name="m",
                tags=None,
                labels={},
                concurrency=5,
                timeout=60,
                db_path=db_file,
            )

        async with SQLiteStorage(db_file) as storage:
            saved = await storage.get_run(run.run_id)

        assert saved.suite_name == "My Real Suite"
        assert saved.status == "failed"
        assert saved.error_message is not None
        assert "load failed" in saved.error_message


class TestRateLimit:
    """POST /api/runs is limited to 10 requests per minute per IP."""

    @pytest.mark.asyncio
    async def test_eleventh_request_returns_429(self, tmp_path: object) -> None:
        from unittest.mock import MagicMock, patch
        from llmeval.server import api as api_module

        db_file = os.path.join(str(tmp_path), "rl.db")
        app = create_app(db_path=db_file)

        # Clear rate-limit state from other tests.
        api_module._post_runs_calls.clear()

        suite_mock = MagicMock()
        suite_mock.suite.name = "RL Suite"
        suite_mock.suite.model = "m"
        suite_mock.suite.judge_model = "j"
        suite_mock.suite.version = "1.0"

        with (
            patch("llmeval.schema.test_suite.load_suite", return_value=suite_mock),
            patch("llmeval.models.create_adapter"),
            patch("asyncio.create_task"),
        ):
            async with _client(app) as client:
                for _ in range(10):
                    suite_mock.suite.name = f"RL Suite {_}"
                    await client.post("/api/runs", json={"suite_path": "s.yaml"})
                resp = await client.post("/api/runs", json={"suite_path": "s.yaml"})

        assert resp.status_code == 429
