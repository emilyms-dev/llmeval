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
    ├── model         TEXT  (indexed)
    ├── judge_model   TEXT
    ├── status        TEXT  (indexed)
    ├── tags_text     TEXT  comma-joined tags, indexed for substring search
    ├── labels_json   TEXT  JSON object of CI labels (e.g. commit, branch)
    ├── started_at    TEXT  ISO-8601 UTC, indexed for sorted listing
    ├── completed_at  TEXT  nullable
    ├── total_tests   INTEGER
    ├── passed_tests  INTEGER
    ├── failed_tests  INTEGER
    ├── errored_tests INTEGER
    └── run_json      TEXT  full model_dump_json() blob
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import aiosqlite

from llmeval.exceptions import StorageError
from llmeval.schema.results import RunStatus, SuiteRun
from llmeval.storage._errors import storage_op
from llmeval.storage.base import RunBrief, StorageBackend

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS suite_runs (
    run_id        TEXT PRIMARY KEY,
    suite_name    TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    model         TEXT NOT NULL,
    judge_model   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'completed',
    tags_text     TEXT NOT NULL DEFAULT '',
    labels_json   TEXT NOT NULL DEFAULT '{}',
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

_CREATE_IDX_MODEL = """
CREATE INDEX IF NOT EXISTS idx_suite_runs_model
    ON suite_runs (model);
"""

_CREATE_IDX_STARTED_AT = """
CREATE INDEX IF NOT EXISTS idx_suite_runs_started_at
    ON suite_runs (started_at);
"""

_CREATE_IDX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_suite_runs_status
    ON suite_runs (status);
"""

_CREATE_IDX_TAGS_TEXT = """
CREATE INDEX IF NOT EXISTS idx_suite_runs_tags_text
    ON suite_runs (tags_text);
"""

# Migration statements — each is idempotent via PRAGMA check.
_MIGRATE_ADD_STATUS = """
ALTER TABLE suite_runs ADD COLUMN status TEXT NOT NULL DEFAULT 'completed';
"""
_MIGRATE_ADD_TAGS_TEXT = """
ALTER TABLE suite_runs ADD COLUMN tags_text TEXT NOT NULL DEFAULT '';
"""
_MIGRATE_ADD_LABELS_JSON = """
ALTER TABLE suite_runs ADD COLUMN labels_json TEXT NOT NULL DEFAULT '{}';
"""

# Data migration: reformat tags_text from "tag1, tag2" to "|tag1|tag2|"
# so that exact-tag lookups are possible via LIKE '%|tag|%'.
# Safe to run on already-migrated databases (rows starting with | are skipped).
_MIGRATE_TAGS_TEXT_FORMAT = """
UPDATE suite_runs
SET tags_text = '|' || REPLACE(tags_text, ', ', '|') || '|'
WHERE length(tags_text) > 0
  AND substr(tags_text, 1, 1) != '|';
"""
_MIGRATE_TAGS_TEXT_EMPTY = """
UPDATE suite_runs SET tags_text = '||' WHERE tags_text = '';
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
            runs = await storage.list_runs(suite_name="My Suite", status="completed")
            await storage.cancel_run(suite_run.run_id)
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
            # WAL mode allows concurrent readers and a background pipeline writer
            # without hitting "database is locked" when the request-handler
            # connection and the per-pipeline connection overlap.
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute(_CREATE_TABLE)
            await self._conn.execute(_CREATE_IDX_SUITE_NAME)
            await self._conn.execute(_CREATE_IDX_MODEL)
            await self._conn.execute(_CREATE_IDX_STARTED_AT)
            await self._conn.execute(_CREATE_IDX_STATUS)
            await self._conn.execute(_CREATE_IDX_TAGS_TEXT)
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
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_run_id(self, run_id: str) -> str:
        """Resolve a full UUID or unambiguous prefix to a canonical run ID.

        Args:
            run_id: Full UUID or unique prefix (minimum 8 hex characters
                recommended).

        Returns:
            The full run ID string stored in the database.

        Raises:
            StorageError: If *run_id* matches zero or more than one run.
        """
        async with (
            storage_op(f"Failed to resolve run ID {run_id!r}"),
            self._db.execute(
                "SELECT run_id FROM suite_runs WHERE run_id LIKE :prefix",
                {"prefix": run_id + "%"},
            ) as cursor,
        ):
            rows = list(await cursor.fetchall())
        if not rows:
            raise StorageError(f"Run {run_id!r} not found.")
        if len(rows) > 1:
            raise StorageError(
                f"Prefix {run_id!r} is ambiguous ({len(rows)} matches). "
                "Provide more characters."
            )
        return cast(str, rows[0]["run_id"])

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
        tags_text = "|" + "|".join(suite_run.tags) + "|" if suite_run.tags else "||"
        run_json = suite_run.model_dump_json()

        async with storage_op(f"Failed to save run {suite_run.run_id!r}"):
            await self._db.execute(
                """
                INSERT OR REPLACE INTO suite_runs (
                    run_id, suite_name, suite_version, model, judge_model,
                    status, tags_text, labels_json, started_at, completed_at,
                    total_tests, passed_tests, failed_tests, errored_tests,
                    run_json
                ) VALUES (
                    :run_id, :suite_name, :suite_version, :model, :judge_model,
                    :status, :tags_text, :labels_json, :started_at, :completed_at,
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
                    "tags_text": tags_text,
                    "labels_json": json.dumps(suite_run.labels),
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
        resolved = await self._resolve_run_id(run_id)
        async with (
            storage_op(f"Failed to fetch run {run_id!r}"),
            self._db.execute(
                "SELECT run_json FROM suite_runs WHERE run_id = :run_id",
                {"run_id": resolved},
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            raise StorageError(f"Run {run_id!r} not found.")
        return _deserialise(row["run_json"])

    async def get_run_brief(self, run_id: str) -> RunBrief:
        """Return only indexed metadata columns for *run_id*.

        Reads ``status``, counts, and ``completed_at`` from the indexed
        columns without touching ``run_json``. Use this for polling loops
        where deserialising a large result blob would be wasteful.

        Args:
            run_id: Full UUID of the run.

        Returns:
            A :class:`~llmeval.storage.base.RunBrief` with current counters.

        Raises:
            StorageError: If the run is not found or the query fails.
        """
        resolved = await self._resolve_run_id(run_id)
        async with (
            storage_op(f"Failed to fetch brief for run {run_id!r}"),
            self._db.execute(
                """
                SELECT run_id, status, total_tests, passed_tests,
                       failed_tests, errored_tests, completed_at
                FROM suite_runs WHERE run_id = :run_id
                """,
                {"run_id": resolved},
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            raise StorageError(f"Run {run_id!r} not found.")

        completed_at: datetime | None = None
        if row["completed_at"]:
            with contextlib.suppress(ValueError):
                completed_at = datetime.fromisoformat(row["completed_at"])

        return RunBrief(
            run_id=row["run_id"],
            status=row["status"],
            total_tests=row["total_tests"],
            passed_tests=row["passed_tests"],
            failed_tests=row["failed_tests"],
            errored_tests=row["errored_tests"],
            completed_at=completed_at,
        )

    async def get_latest_run(
        self,
        suite_name: str | None = None,
        status: str | None = "completed",
    ) -> SuiteRun:
        """Return the most recent run matching *suite_name* and *status*.

        Args:
            suite_name: When provided, only runs for this exact suite name are
                considered.
            status: Restrict to runs with this status. Defaults to
                ``"completed"`` (last successful run). Pass ``None`` to return
                the absolute latest run regardless of status.

        Returns:
            The latest matching :class:`~llmeval.schema.results.SuiteRun`.

        Raises:
            StorageError: If no matching run exists or the query fails.
        """
        conditions: list[str] = []
        params: dict[str, object] = {}
        if status is not None:
            conditions.append("status = :status")
            params["status"] = status
        if suite_name is not None:
            conditions.append("suite_name = :suite_name")
            params["suite_name"] = suite_name
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            f"SELECT run_json FROM suite_runs {where} "
            "ORDER BY started_at DESC LIMIT 1"
        )
        async with (
            storage_op("Failed to fetch latest run"),
            self._db.execute(query, params) as cursor,
        ):
            row = await cursor.fetchone()

        if row is None:
            parts = []
            if status:
                parts.append(f"status={status!r}")
            if suite_name:
                parts.append(f"suite={suite_name!r}")
            label = f" ({', '.join(parts)})" if parts else ""
            raise StorageError(f"No run found{label}.")
        return _deserialise(row["run_json"])

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
        async with (
            storage_op(f"Failed to fetch previous run for {suite_name!r}"),
            self._db.execute(
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
            ) as cursor,
        ):
            row = await cursor.fetchone()

        if row is None:
            raise StorageError(
                f"No previous completed run found for suite {suite_name!r}."
            )

        return _deserialise(row["run_json"])

    async def cancel_run(self, run_id: str) -> None:
        """Mark *run_id* as cancelled.

        Idempotent when the run is already ``"cancelled"``. Raises for any
        other terminal status (``"completed"`` or ``"failed"``).

        Args:
            run_id: Full UUID of the run to cancel.

        Raises:
            StorageError: If the run is not found, already terminal, or the
                update fails.
        """
        run = await self.get_run(run_id)
        if run.status == "cancelled":
            return
        if run.status not in ("pending", "running"):
            raise StorageError(
                f"Run {run_id!r} cannot be cancelled (status: {run.status!r})."
            )
        cancelled = run.model_copy(
            update={"status": "cancelled", "completed_at": datetime.now(UTC)}
        )
        await self.save_run(cancelled)

    async def get_run_status(self, run_id: str) -> RunStatus:
        """Return only the status column for *run_id* without full deserialisation.

        Args:
            run_id: Full UUID of the run.

        Returns:
            The current :data:`~llmeval.schema.results.RunStatus` value.

        Raises:
            StorageError: If the run is not found or the query fails.
        """
        resolved = await self._resolve_run_id(run_id)
        async with (
            storage_op(f"Failed to fetch status for run {run_id!r}"),
            self._db.execute(
                "SELECT status FROM suite_runs WHERE run_id = :run_id",
                {"run_id": resolved},
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            raise StorageError(f"Run {run_id!r} not found.")
        return cast(RunStatus, row["status"])

    async def list_runs(
        self,
        *,
        suite_name: str | None = None,
        model: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        tag_match: str = "exact",
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SuiteRun]:
        """Return runs ordered by ``started_at`` descending (most recent first).

        All supplied filters are combined with AND.

        Args:
            suite_name: Exact-match filter on suite name.
            model: Exact-match filter on model identifier.
            status: Exact-match filter on run status.
            tag: Filter on tags. Interpretation depends on *tag_match*.
            tag_match: ``"exact"`` (default) — *tag* must be a complete tag
                token in the pipe-delimited list. ``"fuzzy"`` — any tag must
                contain *tag* as a substring.
            date_from: Include only runs started on or after this timestamp.
                Timezone-aware datetimes are converted to UTC ISO-8601.
            date_to: Include only runs started on or before this timestamp.
                Timezone-aware datetimes are converted to UTC ISO-8601.
            limit: Maximum number of runs to return.
            offset: Rows to skip for pagination.

        Returns:
            List of deserialised :class:`~llmeval.schema.results.SuiteRun`.

        Raises:
            StorageError: If the query fails.
        """
        conditions: list[str] = []
        params: dict[str, object] = {}

        if suite_name is not None:
            conditions.append("suite_name = :suite_name")
            params["suite_name"] = suite_name
        if model is not None:
            conditions.append("model = :model")
            params["model"] = model
        if status is not None:
            conditions.append("status = :status")
            params["status"] = status
        if tag is not None:
            conditions.append("tags_text LIKE :tag")
            if tag_match == "fuzzy":
                params["tag"] = f"%{tag}%"
            else:
                params["tag"] = f"%|{tag}|%"
        if date_from is not None:
            conditions.append("started_at >= :date_from")
            params["date_from"] = date_from.astimezone(UTC).isoformat()
        if date_to is not None:
            conditions.append("started_at <= :date_to")
            params["date_to"] = date_to.astimezone(UTC).isoformat()

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            f"SELECT run_json FROM suite_runs {where} "
            "ORDER BY started_at DESC "
            "LIMIT :limit OFFSET :offset"
        )
        params["limit"] = limit
        params["offset"] = offset

        async with (
            storage_op("Failed to list runs"),
            self._db.execute(query, params) as cursor,
        ):
            rows = await cursor.fetchall()
        return [_deserialise(row["run_json"]) for row in rows]

    async def delete_run(self, run_id: str) -> None:
        """Delete a run from storage.

        Args:
            run_id: UUID of the run to remove.

        Raises:
            StorageError: If *run_id* is not found or the delete fails.
        """
        async with storage_op(f"Failed to delete run {run_id!r}"):
            async with self._db.execute(
                "DELETE FROM suite_runs WHERE run_id = :run_id",
                {"run_id": run_id},
            ) as cursor:
                deleted = cursor.rowcount
            await self._db.commit()
        if deleted == 0:
            raise StorageError(f"Run {run_id!r} not found; nothing deleted.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _migrate(conn: aiosqlite.Connection) -> None:
    """Apply schema migrations needed for databases created before the current
    schema version.

    Each migration checks whether the change is needed before applying it,
    making the function safe to call against an already-current database.
    """
    async with conn.execute("PRAGMA table_info(suite_runs)") as cursor:
        columns = {row["name"] async for row in cursor}

    if "status" not in columns:
        await conn.execute(_MIGRATE_ADD_STATUS)
    if "tags_text" not in columns:
        await conn.execute(_MIGRATE_ADD_TAGS_TEXT)
    if "labels_json" not in columns:
        await conn.execute(_MIGRATE_ADD_LABELS_JSON)
    # Reformat tags_text to pipe-delimited for exact-tag matching.
    await conn.execute(_MIGRATE_TAGS_TEXT_FORMAT)
    await conn.execute(_MIGRATE_TAGS_TEXT_EMPTY)


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
