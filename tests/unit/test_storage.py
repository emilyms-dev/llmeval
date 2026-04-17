"""Unit tests for llmeval.storage (SQLite backend).

Uses an in-memory SQLite database (:memory:) for speed and isolation.
No filesystem side-effects.

Coverage targets:
- StorageBackend ABC: not directly instantiable, context manager delegation
- SQLiteStorage.initialize: creates table and indexes, idempotent
- SQLiteStorage.close: safe when uninitialized
- SQLiteStorage._db property: raises before initialize()
- SQLiteStorage.save_run: insert, round-trip fidelity, upsert on duplicate
- SQLiteStorage.get_run: happy path, not-found raises StorageError
- SQLiteStorage.list_runs: all, suite_name filter, limit, offset, ordering
- SQLiteStorage.delete_run: removes row, not-found raises StorageError
- _deserialise: corrupt JSON raises StorageError
- Async context manager: __aenter__/__aexit__
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from llmeval.exceptions import StorageError
from llmeval.schema.results import CriterionScore, SuiteRun, TestResult
from llmeval.storage import SQLiteStorage, StorageBackend
from llmeval.storage.sqlite import _deserialise

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    suite_name: str = "My Suite",
    suite_version: str = "1.0.0",
    model: str = "claude-sonnet-4-20250514",
    results: list[TestResult] | None = None,
    started_at: datetime | None = None,
) -> SuiteRun:
    now = started_at or datetime.now(UTC)
    return SuiteRun(
        suite_name=suite_name,
        suite_version=suite_version,
        model=model,
        judge_model="claude-sonnet-4-20250514",
        started_at=now,
        completed_at=now + timedelta(seconds=5),
        results=results or [],
    )


def _result(
    test_id: str = "t1",
    passed: bool = True,
    score: float = 0.9,
) -> TestResult:
    return TestResult(
        test_id=test_id,
        prompt="Hello",
        model="claude-sonnet-4-20250514",
        raw_output="Hi there",
        criterion_scores=[
            CriterionScore(name="quality", score=score, reasoning="Good.")
        ],
        weighted_score=score,
        passed=passed,
    )


async def _storage() -> SQLiteStorage:
    """Return an initialized in-memory SQLiteStorage."""
    s = SQLiteStorage(":memory:")
    await s.initialize()
    return s


# ===========================================================================
# StorageBackend ABC
# ===========================================================================


class TestStorageBackendABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            StorageBackend()  # type: ignore[abstract]

    def test_concrete_subclass_missing_methods_is_abstract(self) -> None:
        class Partial(StorageBackend):
            async def initialize(self) -> None:
                pass

        with pytest.raises(TypeError, match="abstract"):
            Partial()  # type: ignore[abstract]


# ===========================================================================
# SQLiteStorage lifecycle
# ===========================================================================


class TestSQLiteStorageLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_creates_table(self) -> None:
        storage = await _storage()
        async with storage._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='suite_runs'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        await storage.close()

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self) -> None:
        storage = SQLiteStorage(":memory:")
        await storage.initialize()
        await storage.initialize()  # second call is a no-op
        await storage.close()

    @pytest.mark.asyncio
    async def test_close_when_not_initialized_is_safe(self) -> None:
        storage = SQLiteStorage(":memory:")
        await storage.close()  # should not raise

    @pytest.mark.asyncio
    async def test_db_property_raises_before_initialize(self) -> None:
        storage = SQLiteStorage(":memory:")
        with pytest.raises(StorageError, match="not initialised"):
            _ = storage._db

    @pytest.mark.asyncio
    async def test_context_manager_initializes_and_closes(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            assert storage._conn is not None
        assert storage._conn is None

    @pytest.mark.asyncio
    async def test_initialize_wraps_connect_error(self) -> None:
        with patch(
            "llmeval.storage.sqlite.aiosqlite.connect",
            side_effect=OSError("no disk"),
        ):
            storage = SQLiteStorage("bad_path.db")
            with pytest.raises(StorageError, match="Failed to initialise"):
                await storage.initialize()

    @pytest.mark.asyncio
    async def test_context_manager_aenter_returns_self(self) -> None:
        storage = SQLiteStorage(":memory:")
        result = await storage.__aenter__()
        assert result is storage
        await storage.close()


# ===========================================================================
# save_run
# ===========================================================================


class TestSaveRun:
    @pytest.mark.asyncio
    async def test_save_inserts_row(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            run = _run()
            await storage.save_run(run)
            async with storage._db.execute(
                "SELECT run_id FROM suite_runs WHERE run_id = ?", (run.run_id,)
            ) as cur:
                row = await cur.fetchone()
            assert row is not None

    @pytest.mark.asyncio
    async def test_save_upserts_on_duplicate_run_id(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            run = _run()
            await storage.save_run(run)
            updated = run.model_copy(
                update={"results": [_result("t1", passed=True)]}
            )
            await storage.save_run(updated)
            fetched = await storage.get_run(run.run_id)
            assert fetched.total_tests == 1

    @pytest.mark.asyncio
    async def test_save_raises_before_initialize(self) -> None:
        storage = SQLiteStorage(":memory:")
        with pytest.raises(StorageError):
            await storage.save_run(_run())

    @pytest.mark.asyncio
    async def test_save_wraps_db_error_in_storage_error(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            storage._conn.execute = MagicMock(side_effect=OSError("disk full"))  # type: ignore[union-attr]
            with pytest.raises(StorageError, match="Failed to save run"):
                await storage.save_run(_run())


# ===========================================================================
# get_run — round-trip fidelity
# ===========================================================================


class TestGetRun:
    @pytest.mark.asyncio
    async def test_get_returns_matching_run(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            run = _run(suite_name="Fidelity Suite")
            await storage.save_run(run)
            fetched = await storage.get_run(run.run_id)
            assert fetched.run_id == run.run_id
            assert fetched.suite_name == "Fidelity Suite"

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_results(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            run = _run(results=[_result("t1", score=0.85)])
            await storage.save_run(run)
            fetched = await storage.get_run(run.run_id)
            assert len(fetched.results) == 1
            assert fetched.results[0].test_id == "t1"
            assert fetched.results[0].weighted_score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_criterion_scores(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            run = _run(results=[_result("t1")])
            await storage.save_run(run)
            fetched = await storage.get_run(run.run_id)
            score = fetched.results[0].criterion_scores[0]
            assert score.name == "quality"
            assert score.reasoning == "Good."

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_timestamps(self) -> None:
        fixed = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        async with SQLiteStorage(":memory:") as storage:
            run = _run(started_at=fixed)
            await storage.save_run(run)
            fetched = await storage.get_run(run.run_id)
            assert fetched.started_at == fixed
            assert fetched.completed_at == fixed + timedelta(seconds=5)

    @pytest.mark.asyncio
    async def test_roundtrip_null_completed_at(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            run = SuiteRun(
                suite_name="In Progress",
                suite_version="1.0",
                model="gpt-4o",
                judge_model="gpt-4o",
                completed_at=None,
            )
            await storage.save_run(run)
            fetched = await storage.get_run(run.run_id)
            assert fetched.completed_at is None

    @pytest.mark.asyncio
    async def test_get_missing_run_raises_storage_error(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            with pytest.raises(StorageError, match="not found"):
                await storage.get_run("nonexistent-uuid")

    @pytest.mark.asyncio
    async def test_get_raises_before_initialize(self) -> None:
        storage = SQLiteStorage(":memory:")
        with pytest.raises(StorageError, match="not initialised"):
            await storage.get_run("any-id")

    @pytest.mark.asyncio
    async def test_get_wraps_db_error_in_storage_error(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            storage._conn.execute = MagicMock(side_effect=OSError("io error"))  # type: ignore[union-attr]
            with pytest.raises(StorageError, match="Failed to fetch run"):
                await storage.get_run("any-id")

    @pytest.mark.asyncio
    async def test_roundtrip_error_result(self) -> None:
        errored = TestResult(
            test_id="t-err",
            prompt="p",
            model="m",
            raw_output="",
            error="API timed out",
        )
        async with SQLiteStorage(":memory:") as storage:
            run = _run(results=[errored])
            await storage.save_run(run)
            fetched = await storage.get_run(run.run_id)
            assert fetched.results[0].error == "API timed out"


# ===========================================================================
# list_runs
# ===========================================================================


class TestListRuns:
    @pytest.mark.asyncio
    async def test_list_returns_all_runs(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            for i in range(3):
                await storage.save_run(_run(suite_name=f"Suite {i}"))
            runs = await storage.list_runs()
            assert len(runs) == 3

    @pytest.mark.asyncio
    async def test_list_empty_returns_empty_list(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            assert await storage.list_runs() == []

    @pytest.mark.asyncio
    async def test_list_ordered_most_recent_first(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        async with SQLiteStorage(":memory:") as storage:
            for i in range(3):
                await storage.save_run(
                    _run(started_at=base + timedelta(hours=i))
                )
            runs = await storage.list_runs()
            times = [r.started_at for r in runs]
            assert times == sorted(times, reverse=True)

    @pytest.mark.asyncio
    async def test_list_filters_by_suite_name(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            await storage.save_run(_run(suite_name="Alpha"))
            await storage.save_run(_run(suite_name="Alpha"))
            await storage.save_run(_run(suite_name="Beta"))
            runs = await storage.list_runs(suite_name="Alpha")
            assert len(runs) == 2
            assert all(r.suite_name == "Alpha" for r in runs)

    @pytest.mark.asyncio
    async def test_list_suite_name_no_match_returns_empty(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            await storage.save_run(_run(suite_name="Alpha"))
            runs = await storage.list_runs(suite_name="Nonexistent")
            assert runs == []

    @pytest.mark.asyncio
    async def test_list_raises_before_initialize(self) -> None:
        storage = SQLiteStorage(":memory:")
        with pytest.raises(StorageError, match="not initialised"):
            await storage.list_runs()

    @pytest.mark.asyncio
    async def test_list_wraps_db_error_in_storage_error(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            storage._conn.execute = MagicMock(side_effect=OSError("io error"))  # type: ignore[union-attr]
            with pytest.raises(StorageError, match="Failed to list runs"):
                await storage.list_runs()

    @pytest.mark.asyncio
    async def test_list_respects_limit(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            for _ in range(5):
                await storage.save_run(_run())
            runs = await storage.list_runs(limit=3)
            assert len(runs) == 3

    @pytest.mark.asyncio
    async def test_list_respects_offset(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        async with SQLiteStorage(":memory:") as storage:
            ids = []
            for i in range(4):
                run = _run(started_at=base + timedelta(hours=i))
                ids.append(run.run_id)
                await storage.save_run(run)
            # Most recent first: ids[3], ids[2], ids[1], ids[0]
            page1 = await storage.list_runs(limit=2, offset=0)
            page2 = await storage.list_runs(limit=2, offset=2)
            all_ids = [r.run_id for r in page1 + page2]
            assert len(all_ids) == 4
            assert len(set(all_ids)) == 4  # no duplicates across pages


# ===========================================================================
# delete_run
# ===========================================================================


class TestDeleteRun:
    @pytest.mark.asyncio
    async def test_delete_raises_before_initialize(self) -> None:
        storage = SQLiteStorage(":memory:")
        with pytest.raises(StorageError, match="not initialised"):
            await storage.delete_run("any-id")

    @pytest.mark.asyncio
    async def test_delete_wraps_db_error_in_storage_error(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            storage._conn.execute = MagicMock(side_effect=OSError("io error"))  # type: ignore[union-attr]
            with pytest.raises(StorageError, match="Failed to delete run"):
                await storage.delete_run("any-id")

    @pytest.mark.asyncio
    async def test_delete_removes_run(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            run = _run()
            await storage.save_run(run)
            await storage.delete_run(run.run_id)
            with pytest.raises(StorageError, match="not found"):
                await storage.get_run(run.run_id)

    @pytest.mark.asyncio
    async def test_delete_missing_raises_storage_error(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            with pytest.raises(StorageError, match="not found"):
                await storage.delete_run("does-not-exist")

    @pytest.mark.asyncio
    async def test_delete_only_removes_target_run(self) -> None:
        async with SQLiteStorage(":memory:") as storage:
            run_a = _run(suite_name="A")
            run_b = _run(suite_name="B")
            await storage.save_run(run_a)
            await storage.save_run(run_b)
            await storage.delete_run(run_a.run_id)
            remaining = await storage.list_runs()
            assert len(remaining) == 1
            assert remaining[0].run_id == run_b.run_id


# ===========================================================================
# _deserialise
# ===========================================================================


class TestDeserialise:
    def test_valid_json_roundtrips(self) -> None:
        run = _run(suite_name="Test")
        serialised = run.model_dump_json()
        restored = _deserialise(serialised)
        assert restored.suite_name == "Test"
        assert restored.run_id == run.run_id

    def test_corrupt_json_raises_storage_error(self) -> None:
        with pytest.raises(StorageError, match="Failed to deserialise"):
            _deserialise("{not valid json")

    def test_invalid_schema_raises_storage_error(self) -> None:
        with pytest.raises(StorageError, match="Failed to deserialise"):
            _deserialise('{"run_id": "x"}')  # missing required fields
