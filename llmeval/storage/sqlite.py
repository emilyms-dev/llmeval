"""SQLite storage backend.

Uses ``aiosqlite`` for non-blocking I/O. A single persistent connection is
held for the lifetime of the backend instance (opened in :meth:`initialize`,
closed in :meth:`close`). This makes ``:memory:`` databases usable in tests
and avoids connection-per-operation overhead in production.

Table layout
------------
One table — ``suite_runs`` — stores indexed metadata columns (for fast
queries and sorting without deserialisation) and a full JSON blob so the
complete :class:`~llmeval.schema.results.SuiteRun` round-trips without loss::

    suite_runs
    ├── run_id        TEXT PK
    ├── suite_name    TEXT  (indexed)
    ├── suite_version TEXT
    ├── model         TEXT
    ├── judge_model   TEXT
    ├── status        TEXT  (indexed)
    ├── started_at    TEXT  ISO-8601 UTC, indexed for sorted listing
    ├── completed_at  TEXT  nullable
    ├── total_tests   INTEGER
    ├── passed_tests  INTEGER
    ├── failed_tests  INTEGER
    ├── errored_tests INTEGER
    └── run_json      TEXT  full model_dump_json() blob
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from llmeval.exceptions import StorageError
from llmeval.schema.results import SuiteRun
from llmeval.storage.base import StorageBackend

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS suite_runs (
    run_id        TEXT PRIMARY KEY,
    suite_name    TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    model         TEXT NOT NULL,
    judge_model   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'completed',
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    total_tests   INTEGER NOT NULL DEFAULT 0,
    passed_tests  INTEGER NOT NULL DEFAULT 0,
    failed_tests  INTEGER NOT NULL DEFAULT 0,
    errored_tests INTEGER NOT NULL DEFAULT 0,
    run_json      TEXT NOT NULL
);
"""

_CREATE_IDX_SUITE_NAME = """
CREATE INDEX IF NOT EXISTS idx_suite_runs_suite_name
    ON suite_runs (suite_name);
"""

_CREATE_IDX_STARTED_AT = """
CREATE INDEX IF NOT EXISTS idx_suite_runs_started_at
    ON suite_runs (started_at);
"""

_CREATE_IDX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_suite_runs_status
    ON suite_runs (status);
"""

# Adds the status column to databases created before this column existed.
# DEFAULT 'completed' gives existing rows a sensible value.
_MIGRATE_ADD_STATUS = """
ALTER TABLE suite_runs ADD COLUMN status TEXT NOT NULL DEFAULT 'completed';
"""


class SQLiteStorage(StorageBackend):
    """SQLite-backed storage for suite runs.

    Args:
        db_path: Filesystem path to the SQLite database file, or the special
            string ``":memory:"`` for an in-process in-memory database (useful
            in tests). Relative paths are resolved from the current working
            directory.

    Example::

        async with SQLiteStorage("llmeval.db") as storage:
            await storage.save_run(suite_run)
            runs = await storage.list_runs(suite_name="My Suite")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the database connection and create/migrate schema.

        Idempotent — calling more than once has no effect if a connection is
        already open. Automatically adds any new columns introduced in later
        versions so existing databases are upgraded in place.

        Raises:
            StorageError: If the database file cannot be opened or the schema
                cannot be created.
        """
        if self._conn is not None:
            return
        try:
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute(_CREATE_TABLE)
            await self._conn.execute(_CREATE_IDX_SUITE_NAME)
            await self._conn.execute(_CREATE_IDX_STARTED_AT)
            await self._conn.execute(_CREATE_IDX_STATUS)
            await _migrate(self._conn)
            await self._conn.commit()
        except Exception as exc:
            raise StorageError(
                f"Failed to initialise SQLite database at {self._db_path!r}: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the database connection.

        Safe to call when no connection is open.
        """
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise StorageError(
                "Storage is not initialised. "
                "Call initialize() or use as an async context manager."
            )
        return self._conn

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def save_run(self, suite_run: SuiteRun) -> None:
        """Insert or replace *suite_run* in the database.

        Uses ``INSERT OR REPLACE`` so calling this a second time with the same
        ``run_id`` overwrites the existing record (useful when a run is
        persisted in-progress and again after scoring).

        Args:
            suite_run: Run to persist.

        Raises:
            StorageError: If the write fails.
        """
        started_at = suite_run.started_at.isoformat()
        completed_at = (
            suite_run.completed_at.isoformat()
            if suite_run.completed_at is not None
            else None
        )
        run_json = suite_run.model_dump_json()

        try:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO suite_runs (
                    run_id, suite_name, suite_version, model, judge_model,
                    status, started_at, completed_at,
                    total_tests, passed_tests, failed_tests, errored_tests,
                    run_json
                ) VALUES (
                    :run_id, :suite_name, :suite_version, :model, :judge_model,
                    :status, :started_at, :completed_at,
                    :total_tests, :passed_tests, :failed_tests, :errored_tests,
                    :run_json
                )
                """,
                {
                    "run_id": suite_run.run_id,
                    "suite_name": suite_run.suite_name,
                    "suite_version": suite_run.suite_version,
                    "model": suite_run.model,
                    "judge_model": suite_run.judge_model,
                    "status": suite_run.status,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "total_tests": suite_run.total_tests,
                    "passed_tests": suite_run.passed_tests,
                    "failed_tests": suite_run.failed_tests,
                    "errored_tests": suite_run.errored_tests,
                    "run_json": run_json,
                },
            )
            await self._db.commit()
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                f"Failed to save run {suite_run.run_id!r}: {exc}"
            ) from exc

    async def get_run(self, run_id: str) -> SuiteRun:
        """Fetch a single run by UUID or unambiguous prefix.

        A prefix is unambiguous when exactly one stored run ID starts with it.

        Args:
            run_id: Full UUID or a unique prefix (minimum 8 hex characters
                recommended).

        Returns:
            Fully deserialised :class:`~llmeval.schema.results.SuiteRun`.

        Raises:
            StorageError: If *run_id* matches zero or more than one run, or if
                deserialisation fails.
        """
        try:
            async with self._db.execute(
                "SELECT run_json FROM suite_runs WHERE run_id LIKE :prefix",
                {"prefix": run_id + "%"},
            ) as cursor:
                rows = await cursor.fetchall()
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Failed to fetch run {run_id!r}: {exc}") from exc

        if not rows:
            raise StorageError(f"Run {run_id!r} not found.")
        if len(rows) > 1:
            raise StorageError(
                f"Prefix {run_id!r} is ambiguous ({len(rows)} matches). "
                "Provide more characters."
            )

        return _deserialise(rows[0]["run_json"])

    async def get_previous_run(self, suite_name: str, before_run_id: str) -> SuiteRun:
        """Return the most recent completed run of *suite_name* before *before_run_id*.

        Useful for "compare with previous" functionality.

        Args:
            suite_name: Exact suite name to match.
            before_run_id: UUID of the reference run. The returned run will
                have an earlier ``started_at``.

        Returns:
            The previous completed :class:`~llmeval.schema.results.SuiteRun`.

        Raises:
            StorageError: If no previous run exists or the query fails.
        """
        try:
            async with self._db.execute(
                """
                SELECT run_json FROM suite_runs
                WHERE suite_name = :suite_name
                  AND run_id != :run_id
                  AND status = 'completed'
                  AND started_at < (
                      SELECT started_at FROM suite_runs WHERE run_id = :run_id
                  )
                ORDER BY started_at DESC
                LIMIT 1
                """,
                {"suite_name": suite_name, "run_id": before_run_id},
            ) as cursor:
                row = await cursor.fetchone()
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                f"Failed to fetch previous run for {suite_name!r}: {exc}"
            ) from exc

        if row is None:
            raise StorageError(
                f"No previous completed run found for suite {suite_name!r}."
            )

        return _deserialise(row["run_json"])

    async def list_runs(
        self,
        *,
        suite_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SuiteRun]:
        """Return runs ordered by ``started_at`` descending (most recent first).

        Args:
            suite_name: Exact-match filter on suite name. ``None`` returns all.
            limit: Maximum number of runs to return.
            offset: Rows to skip for pagination.

        Returns:
            List of deserialised :class:`~llmeval.schema.results.SuiteRun`.

        Raises:
            StorageError: If the query fails.
        """
        if suite_name is not None:
            query = """
                SELECT run_json FROM suite_runs
                WHERE suite_name = :suite_name
                ORDER BY started_at DESC
                LIMIT :limit OFFSET :offset
            """
            params: dict[str, object] = {
                "suite_name": suite_name,
                "limit": limit,
                "offset": offset,
            }
        else:
            query = """
                SELECT run_json FROM suite_runs
                ORDER BY started_at DESC
                LIMIT :limit OFFSET :offset
            """
            params = {"limit": limit, "offset": offset}

        try:
            async with self._db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Failed to list runs: {exc}") from exc

        return [_deserialise(row["run_json"]) for row in rows]

    async def delete_run(self, run_id: str) -> None:
        """Delete a run from storage.

        Args:
            run_id: UUID of the run to remove.

        Raises:
            StorageError: If *run_id* is not found or the delete fails.
        """
        try:
            async with self._db.execute(
                "DELETE FROM suite_runs WHERE run_id = :run_id",
                {"run_id": run_id},
            ) as cursor:
                deleted = cursor.rowcount
            await self._db.commit()
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Failed to delete run {run_id!r}: {exc}") from exc

        if deleted == 0:
            raise StorageError(f"Run {run_id!r} not found; nothing deleted.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _migrate(conn: aiosqlite.Connection) -> None:
    """Apply any schema migrations needed for databases created before the
    current schema version.

    Each migration is idempotent — it checks whether the change is needed
    before applying it, so running against an already-current database is safe.
    """
    async with conn.execute("PRAGMA table_info(suite_runs)") as cursor:
        columns = {row["name"] async for row in cursor}

    if "status" not in columns:
        await conn.execute(_MIGRATE_ADD_STATUS)


def _deserialise(run_json: str) -> SuiteRun:
    """Reconstruct a :class:`~llmeval.schema.results.SuiteRun` from JSON.

    Args:
        run_json: JSON string produced by ``SuiteRun.model_dump_json()``.

    Returns:
        Validated :class:`~llmeval.schema.results.SuiteRun` instance.

    Raises:
        StorageError: If the JSON is corrupt or fails schema validation.
    """
    try:
        return SuiteRun.model_validate(json.loads(run_json))
    except Exception as exc:
        raise StorageError(
            f"Failed to deserialise SuiteRun from storage: {exc}"
        ) from exc
