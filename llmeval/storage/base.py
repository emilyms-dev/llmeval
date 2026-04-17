"""Abstract storage backend.

All storage implementations must subclass :class:`StorageBackend` and
implement its abstract methods. The base class also provides async context
manager support so callers can write::

    async with SQLiteStorage("llmeval.db") as storage:
        await storage.save_run(suite_run)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llmeval.schema.results import SuiteRun


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
        """Fetch a single run by its UUID.

        Args:
            run_id: The :attr:`~llmeval.schema.results.SuiteRun.run_id` to
                look up.

        Returns:
            The matching :class:`~llmeval.schema.results.SuiteRun`.

        Raises:
            StorageError: If *run_id* is not found or the read fails.
        """

    @abstractmethod
    async def list_runs(
        self,
        *,
        suite_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SuiteRun]:
        """Return runs ordered by ``started_at`` descending (most recent first).

        Args:
            suite_name: When provided, only runs whose
                :attr:`~llmeval.schema.results.SuiteRun.suite_name` matches
                exactly are returned.
            limit: Maximum number of runs to return. Defaults to 50.
            offset: Number of runs to skip (for pagination). Defaults to 0.

        Returns:
            List of :class:`~llmeval.schema.results.SuiteRun` objects.
            Returns an empty list when no runs match.

        Raises:
            StorageError: If the read fails.
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

    async def __aenter__(self) -> StorageBackend:
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
