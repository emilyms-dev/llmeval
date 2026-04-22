"""Storage error handling utilities."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from llmeval.exceptions import StorageError


@asynccontextmanager
async def storage_op(context: str) -> AsyncGenerator[None, None]:
    """Async context manager that wraps DB operations in a uniform error boundary.

    Re-raises :class:`~llmeval.exceptions.StorageError` unchanged; converts
    any other exception into a :class:`~llmeval.exceptions.StorageError` with
    *context* as the message prefix.

    Args:
        context: Human-readable description of the operation (used as the
            error message prefix when a non-StorageError is caught).
    """
    try:
        yield
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"{context}: {exc}") from exc
