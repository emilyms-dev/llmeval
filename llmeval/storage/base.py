"""Abstract storage backend.

All storage implementations must subclass :class:`StorageBackend` and
implement its abstract methods. The base class also provides async context
manager support so callers can write::

    async with SQLiteStorage("llmeval.db") as storage:
        await storage.save_run(suite_run)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Self

from llmeval.schema.results import RunStatus, SuiteRun


@dataclass(frozen=True)
class RunBrief:
    """Lightweight run metadata read from indexed columns only.

    Returned by :meth:`StorageBackend.get_run_brief` to avoid deserialising
    the full JSON blob when only status and counts are needed (e.g. polling).

    Args:
        run_id: Unique run identifier.
        status: Current lifecycle state.
        total_tests: Total test results recorded.
        passed_tests: Tests that passed with no error.
        failed_tests: Tests that completed scoring but did not pass.
        errored_tests: Tests that could not be scored due to an error.
        completed_at: UTC timestamp when the pipeline finished, or ``None``.
    """

    run_id: str
    status: RunStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    errored_tests: int
    completed_at: datetime | None


class StorageBackend(ABC):
    """Provider-agnostic interface for persisting and retrieving suite runs.

    Concrete subclasses must implement all abstract methods. The default
    ``__aenter__`` / ``__aexit__`` implementations delegate to
    :meth:`initialize` and :meth:`close` respectively, so subclasses only
    need to implement those two lifecycle methods.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Set up the backend (create tables, open connections, etc.).

        Must be called before any read/write operation. Calling it more than
        once on the same instance must be idempotent.

        Raises:
            StorageError: If the backend cannot be initialised.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the backend (connections, files).

        Safe to call even if :meth:`initialize` was never called.
        """

    @abstractmethod
    async def save_run(self, suite_run: SuiteRun) -> None:
        """Persist *suite_run*, inserting or replacing any existing record.

        Args:
            suite_run: Completed (or in-progress) run to store.

        Raises:
            StorageError: If the write fails.
        """

    @abstractmethod
    async def get_run(self, run_id: str) -> SuiteRun:
        """Fetch a single run by its UUID or unambiguous prefix.

        Args:
            run_id: Full UUID or a unique prefix.

        Returns:
            The matching :class:`~llmeval.schema.results.SuiteRun`.

        Raises:
            StorageError: If *run_id* is not found or the read fails.
        """

    @abstractmethod
    async def get_run_brief(self, run_id: str) -> RunBrief:
        """Return indexed metadata for *run_id* without deserialising JSON.

        Reads only the indexed columns (status, counts, timestamps), making
        this suitable for high-frequency status polling.

        Args:
            run_id: Full UUID of the run.

        Returns:
            A :class:`RunBrief` with status and result counts.

        Raises:
            StorageError: If the run is not found or the query fails.
        """

    @abstractmethod
    async def get_latest_run(
        self,
        suite_name: str | None = None,
        status: str | None = "completed",
    ) -> SuiteRun:
        """Return the most recent run matching the given filters.

        Args:
            suite_name: When provided, only runs whose
                :attr:`~llmeval.schema.results.SuiteRun.suite_name` matches
                exactly are considered.
            status: Restrict to runs with this status. Defaults to
                ``"completed"`` (the last *successful* run). Pass ``None``
                to return the absolute latest run regardless of status.

        Returns:
            The latest matching :class:`~llmeval.schema.results.SuiteRun`.

        Raises:
            StorageError: If no matching run exists or the query fails.
        """

    @abstractmethod
    async def get_previous_run(self, suite_name: str, before_run_id: str) -> SuiteRun:
        """Return the most recent completed run of *suite_name* before *before_run_id*.

        Args:
            suite_name: Exact suite name to match.
            before_run_id: UUID or resolved run ID that acts as the upper bound.

        Returns:
            The previous completed :class:`~llmeval.schema.results.SuiteRun`.

        Raises:
            StorageError: If no matching run exists or the query fails.
        """

    @abstractmethod
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

        All filters are combined with AND — only runs matching every supplied
        filter are returned. Omitted filters are not applied.

        Args:
            suite_name: Exact-match filter on suite name.
            model: Exact-match filter on model identifier.
            status: Exact-match filter on run status
                (``"pending"``, ``"running"``, ``"completed"``, ``"failed"``,
                ``"cancelled"``).
            tag: Filter on the tags list.
            tag_match: How *tag* is matched. ``"exact"`` (default) checks
                whether *tag* is one of the run's tags verbatim.
                ``"fuzzy"`` checks whether any tag contains *tag* as a
                substring (useful for prefix/infix searches).
            date_from: Include only runs whose ``started_at`` is on or after
                this timestamp (UTC).
            date_to: Include only runs whose ``started_at`` is on or before
                this timestamp (UTC).
            limit: Maximum number of runs to return. Defaults to 50.
            offset: Number of runs to skip (for pagination). Defaults to 0.

        Returns:
            List of :class:`~llmeval.schema.results.SuiteRun` objects.
            Returns an empty list when no runs match.

        Raises:
            StorageError: If the read fails.
        """

    @abstractmethod
    async def cancel_run(self, run_id: str) -> None:
        """Mark a run as cancelled.

        Succeeds when the run is ``"pending"`` or ``"running"``.
        Is a no-op when the run is already ``"cancelled"``.
        Raises when the run is ``"completed"`` or ``"failed"``.

        Args:
            run_id: The run to cancel.

        Raises:
            StorageError: If the run is not found, already terminal, or the
                update fails.
        """

    @abstractmethod
    async def get_run_status(self, run_id: str) -> RunStatus:
        """Return only the status of a run without deserialising the full JSON.

        Intended for lightweight polling loops.

        Args:
            run_id: Full UUID of the run.

        Returns:
            The current :data:`~llmeval.schema.results.RunStatus`.

        Raises:
            StorageError: If the run is not found or the query fails.
        """

    @abstractmethod
    async def delete_run(self, run_id: str) -> None:
        """Remove a run and all its results from storage.

        Args:
            run_id: The :attr:`~llmeval.schema.results.SuiteRun.run_id` to
                delete.

        Raises:
            StorageError: If *run_id* is not found or the delete fails.
        """

    # ------------------------------------------------------------------
    # Async context manager — delegates to initialize() / close()
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        """Call :meth:`initialize` and return ``self``."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Call :meth:`close` unconditionally."""
        await self.close()
