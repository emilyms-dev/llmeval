"""Test suite execution engine.

The :class:`Runner` executes a :class:`~llmeval.schema.test_suite.TestSuite`
against a :class:`~llmeval.models.base.ModelAdapter`, collecting raw model
outputs into a :class:`~llmeval.schema.results.SuiteRun`.

Scoring (LLM-as-judge) is handled separately by ``judge.py``. The runner
only calls the model and records whatever it returns — or whatever error it
raises — so the judge receives clean, self-contained ``TestResult`` objects.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from llmeval.exceptions import ModelAdapterError, RunnerError
from llmeval.models.base import ModelAdapter
from llmeval.schema.results import SuiteRun, TestResult
from llmeval.schema.test_suite import TestCase, TestSuite

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 5


class Runner:
    """Executes a test suite against a model adapter, collecting raw outputs.

    Tests are fanned out concurrently, bounded by a semaphore. Individual test
    failures are captured in the result object rather than aborting the run, so
    a single flaky API call never kills an entire suite execution.

    Args:
        adapter: Model adapter to call for each test case.
        concurrency: Maximum number of simultaneous in-flight model calls.
            Defaults to 5. Use 1 to serialise execution.

    Raises:
        RunnerError: If ``concurrency`` is less than 1.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        if concurrency < 1:
            raise RunnerError(f"concurrency must be >= 1, got {concurrency!r}")
        self._adapter = adapter
        self._concurrency = concurrency

    async def run(
        self,
        suite: TestSuite,
        *,
        tags: list[str] | None = None,
        suite_path: str | None = None,
    ) -> SuiteRun:
        """Run all (or a tag-filtered subset of) test cases in *suite*.

        Args:
            suite: Validated :class:`~llmeval.schema.test_suite.TestSuite`.
            tags: When provided, only test cases whose
                :attr:`~llmeval.schema.test_suite.TestCase.tags` list
                includes *at least one* of the given tags are executed.
                Pass ``None`` (default) to run every test.

        Returns:
            A :class:`~llmeval.schema.results.SuiteRun` with one
            :class:`~llmeval.schema.results.TestResult` per executed test.
            Results preserve the order of the suite definition (guaranteed by
            ``asyncio.gather``). ``completed_at`` is always set before this
            coroutine returns.

        Raises:
            RunnerError: If no tests match the provided tag filter.
        """
        tests = _filter_tests(suite.tests, tags)
        if not tests:
            raise RunnerError(
                f"No tests matched the tag filter {tags!r}. "
                "Check tag names against the suite definition."
            )

        started_at = datetime.now(UTC)
        semaphore = asyncio.Semaphore(self._concurrency)
        tasks = [self._run_test(test, semaphore) for test in tests]
        results: list[TestResult] = list(await asyncio.gather(*tasks))
        completed_at = datetime.now(UTC)

        suite_run = SuiteRun(
            suite_name=suite.suite.name,
            suite_version=suite.suite.version,
            model=self._adapter.model_id,
            judge_model=suite.suite.judge_model,
            status="completed",
            suite_path=suite_path,
            tags=list(tags) if tags else [],
            concurrency=self._concurrency,
            started_at=started_at,
            completed_at=completed_at,
            results=results,
        )

        logger.info(
            "Suite %r finished: %d/%d passed (%d errored)",
            suite.suite.name,
            suite_run.passed_tests,
            suite_run.total_tests,
            suite_run.errored_tests,
        )
        return suite_run

    async def _run_test(
        self,
        test: TestCase,
        semaphore: asyncio.Semaphore,
    ) -> TestResult:
        """Call the adapter for one test case, respecting the concurrency limit.

        Any exception from ``adapter.complete()`` is caught and surfaced as a
        :class:`~llmeval.schema.results.TestResult` with ``error`` set, so the
        rest of the run continues uninterrupted. Non-
        :class:`~llmeval.exceptions.ModelAdapterError` exceptions are wrapped
        in :class:`~llmeval.exceptions.RunnerError` before recording.

        Args:
            test: Test case to execute.
            semaphore: Shared concurrency limiter.

        Returns:
            A populated :class:`~llmeval.schema.results.TestResult`.
        """
        async with semaphore:
            try:
                logger.debug(
                    "Running test %r against %r", test.id, self._adapter.model_id
                )
                raw_output = await self._adapter.complete(
                    test.prompt,
                    system_prompt=test.system_prompt,
                )
                return TestResult(
                    test_id=test.id,
                    prompt=test.prompt,
                    model=self._adapter.model_id,
                    raw_output=raw_output,
                )
            except ModelAdapterError as exc:
                logger.warning("Test %r errored (adapter): %s", test.id, exc)
                return TestResult(
                    test_id=test.id,
                    prompt=test.prompt,
                    model=self._adapter.model_id,
                    raw_output="",
                    error=str(exc),
                )
            except Exception as exc:
                wrapped = RunnerError(
                    f"Unexpected error running test {test.id!r}: {exc}"
                )
                logger.warning("Test %r errored (unexpected): %s", test.id, exc)
                return TestResult(
                    test_id=test.id,
                    prompt=test.prompt,
                    model=self._adapter.model_id,
                    raw_output="",
                    error=str(wrapped),
                )


def _filter_tests(
    tests: list[TestCase],
    tags: list[str] | None,
) -> list[TestCase]:
    """Return tests that match any of *tags*, or all tests when *tags* is None.

    An empty list is treated identically to ``None`` — both mean "no filter,
    run everything". Only a non-empty list restricts which tests are selected.

    Args:
        tests: Full list of test cases from the suite.
        tags: Tag names to filter on. ``None`` or ``[]`` means no filter.

    Returns:
        Filtered (or complete) list of test cases.
    """
    if not tags:
        return tests
    tag_set = frozenset(tags)
    return [t for t in tests if tag_set.intersection(t.tags)]
