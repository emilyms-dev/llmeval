"""Unit tests for llmeval.runner.

All model calls are mocked — no real API requests are made.

Coverage targets:
- Runner construction: valid and invalid concurrency
- _filter_tests: no filter, matching tags, non-matching tags, partial match
- Runner.run: happy path, partial error, all-error, tag filtering
- SuiteRun fields: names, timing, result ordering
- RunnerError raised when tag filter matches nothing
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmeval.exceptions import ModelAdapterError, RunnerError
from llmeval.models.base import ModelAdapter, ModelResponse
from llmeval.runner import Runner, _filter_tests
from llmeval.schema.results import SuiteRun
from llmeval.schema.test_suite import (
    Criterion,
    Rubric,
    SuiteConfig,
    TestCase,
    TestSuite,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_criterion(name: str = "quality", weight: float = 1.0) -> Criterion:
    return Criterion(name=name, description="Test criterion", weight=weight)


def _make_rubric(threshold: float = 0.75) -> Rubric:
    return Rubric(criteria=[_make_criterion()], passing_threshold=threshold)


def _make_test(
    test_id: str = "test-001",
    prompt: str = "Hello",
    tags: list[str] | None = None,
    system_prompt: str | None = None,
) -> TestCase:
    return TestCase(
        id=test_id,
        description="A test case",
        prompt=prompt,
        rubric=_make_rubric(),
        tags=tags or [],
        system_prompt=system_prompt,
    )


def _make_suite(tests: list[TestCase] | None = None) -> TestSuite:
    if tests is None:
        tests = [_make_test()]
    return TestSuite(
        suite=SuiteConfig(
            name="Test Suite",
            version="1.0.0",
            model="claude-sonnet-4-20250514",
            judge_model="claude-sonnet-4-20250514",
        ),
        tests=tests,
    )


def _make_adapter(model_id: str = "claude-sonnet-4-20250514") -> MagicMock:
    """Return a mock ModelAdapter that returns 'mocked response' by default."""
    adapter = MagicMock(spec=ModelAdapter)
    adapter.model_id = model_id
    adapter.complete = AsyncMock(return_value=ModelResponse(text="mocked response"))
    return adapter


# ===========================================================================
# _filter_tests
# ===========================================================================


class TestFilterTests:
    def test_none_tags_returns_all_tests(self) -> None:
        tests = [_make_test("t1"), _make_test("t2"), _make_test("t3")]
        assert _filter_tests(tests, None) == tests

    def test_matching_tag_returns_matching_subset(self) -> None:
        a = _make_test("t1", tags=["alpha", "shared"])
        b = _make_test("t2", tags=["beta"])
        c = _make_test("t3", tags=["shared"])
        result = _filter_tests([a, b, c], ["shared"])
        assert result == [a, c]

    def test_no_matching_tag_returns_empty_list(self) -> None:
        tests = [_make_test("t1", tags=["foo"]), _make_test("t2", tags=["bar"])]
        assert _filter_tests(tests, ["baz"]) == []

    def test_multiple_filter_tags_union_semantics(self) -> None:
        """A test matching ANY of the filter tags must be included."""
        a = _make_test("t1", tags=["alpha"])
        b = _make_test("t2", tags=["beta"])
        c = _make_test("t3", tags=["gamma"])
        result = _filter_tests([a, b, c], ["alpha", "beta"])
        assert result == [a, b]

    def test_test_with_no_tags_never_matches(self) -> None:
        a = _make_test("t1", tags=[])
        assert _filter_tests([a], ["any"]) == []

    def test_empty_tags_list_returns_all_tests(self) -> None:
        """An empty list is treated as no filter — returns all tests."""
        tests = [_make_test("t1", tags=["foo"]), _make_test("t2", tags=[])]
        assert _filter_tests(tests, []) == tests


# ===========================================================================
# Runner construction
# ===========================================================================


class TestRunnerConstruction:
    def test_valid_concurrency_constructs_ok(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter, concurrency=3)
        assert runner._concurrency == 3

    def test_default_concurrency_is_five(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        assert runner._concurrency == 5

    def test_concurrency_one_is_valid(self) -> None:
        Runner(_make_adapter(), concurrency=1)  # should not raise

    def test_zero_concurrency_raises_runner_error(self) -> None:
        with pytest.raises(RunnerError, match="concurrency must be >= 1"):
            Runner(_make_adapter(), concurrency=0)

    def test_negative_concurrency_raises_runner_error(self) -> None:
        with pytest.raises(RunnerError, match="concurrency must be >= 1"):
            Runner(_make_adapter(), concurrency=-1)


# ===========================================================================
# Runner.run — happy path
# ===========================================================================


class TestRunnerHappyPath:
    @pytest.mark.asyncio
    async def test_run_returns_suite_run(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        suite = _make_suite()
        result = await runner.run(suite)
        assert isinstance(result, SuiteRun)

    @pytest.mark.asyncio
    async def test_run_one_test_produces_one_result(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        suite = _make_suite([_make_test("t1", prompt="Say hello")])
        suite_run = await runner.run(suite)
        assert suite_run.total_tests == 1
        assert suite_run.results[0].test_id == "t1"

    @pytest.mark.asyncio
    async def test_run_multiple_tests_all_recorded(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        tests = [_make_test(f"t{i}") for i in range(4)]
        suite_run = await runner.run(_make_suite(tests))
        assert suite_run.total_tests == 4
        ids = {r.test_id for r in suite_run.results}
        assert ids == {"t0", "t1", "t2", "t3"}

    @pytest.mark.asyncio
    async def test_raw_output_stored_correctly(self) -> None:
        adapter = _make_adapter()
        adapter.complete = AsyncMock(
            return_value=ModelResponse(text="The answer is 42.")
        )
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite([_make_test("t1")]))
        assert suite_run.results[0].raw_output == "The answer is 42."

    @pytest.mark.asyncio
    async def test_result_model_matches_adapter_model_id(self) -> None:
        adapter = _make_adapter(model_id="gpt-4o")
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite())
        assert suite_run.results[0].model == "gpt-4o"
        assert suite_run.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_suite_run_metadata_copied_from_suite(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        suite = _make_suite()
        suite_run = await runner.run(suite)
        assert suite_run.suite_name == suite.suite.name
        assert suite_run.suite_version == suite.suite.version
        assert suite_run.judge_model == suite.suite.judge_model

    @pytest.mark.asyncio
    async def test_completed_at_is_set_after_run(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite())
        assert suite_run.completed_at is not None

    @pytest.mark.asyncio
    async def test_completed_at_not_before_started_at(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite())
        assert suite_run.completed_at >= suite_run.started_at  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_system_prompt_forwarded_to_adapter(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        test = _make_test("t1", prompt="Hello", system_prompt="Be terse.")
        await runner.run(_make_suite([test]))
        adapter.complete.assert_awaited_once_with("Hello", system_prompt="Be terse.")

    @pytest.mark.asyncio
    async def test_no_system_prompt_passes_none(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        test = _make_test("t1", system_prompt=None)
        await runner.run(_make_suite([test]))
        adapter.complete.assert_awaited_once_with(test.prompt, system_prompt=None)

    @pytest.mark.asyncio
    async def test_successful_result_has_no_error(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite())
        assert suite_run.results[0].error is None

    @pytest.mark.asyncio
    async def test_errored_tests_zero_on_clean_run(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite([_make_test("t1"), _make_test("t2")]))
        assert suite_run.errored_tests == 0


# ===========================================================================
# Runner.run — error handling
# ===========================================================================


class TestRunnerErrorHandling:
    @pytest.mark.asyncio
    async def test_model_adapter_error_is_captured_not_raised(self) -> None:
        adapter = _make_adapter()
        adapter.complete = AsyncMock(side_effect=ModelAdapterError("API call failed"))
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite([_make_test("t1")]))
        # Should not raise; error is recorded
        assert suite_run.errored_tests == 1

    @pytest.mark.asyncio
    async def test_errored_result_has_empty_raw_output(self) -> None:
        adapter = _make_adapter()
        adapter.complete = AsyncMock(side_effect=ModelAdapterError("boom"))
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite([_make_test("t1")]))
        assert suite_run.results[0].raw_output == ""

    @pytest.mark.asyncio
    async def test_errored_result_stores_error_message(self) -> None:
        adapter = _make_adapter()
        adapter.complete = AsyncMock(
            side_effect=ModelAdapterError("rate limit exceeded")
        )
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite([_make_test("t1")]))
        assert "rate limit exceeded" in (suite_run.results[0].error or "")

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_abort_other_tests(self) -> None:
        """One failing test must not prevent others from running."""
        call_count = 0

        async def flaky(prompt: str, system_prompt: str | None = None) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if prompt == "fail me":
                raise ModelAdapterError("forced failure")
            return ModelResponse(text="ok")

        adapter = _make_adapter()
        adapter.complete = flaky
        tests = [
            _make_test("t1", prompt="hello"),
            _make_test("t2", prompt="fail me"),
            _make_test("t3", prompt="world"),
        ]
        runner = Runner(adapter, concurrency=1)
        suite_run = await runner.run(_make_suite(tests))

        assert call_count == 3
        assert suite_run.errored_tests == 1
        succeeded = [r for r in suite_run.results if r.error is None]
        assert len(succeeded) == 2

    @pytest.mark.asyncio
    async def test_all_tests_fail_still_returns_suite_run(self) -> None:
        adapter = _make_adapter()
        adapter.complete = AsyncMock(side_effect=ModelAdapterError("always fails"))
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite([_make_test("t1"), _make_test("t2")]))
        assert suite_run.total_tests == 2
        assert suite_run.errored_tests == 2
        assert suite_run.completed_at is not None

    @pytest.mark.asyncio
    async def test_unexpected_exception_wrapped_in_runner_error(self) -> None:
        """Non-ModelAdapterError exceptions are captured and wrapped."""
        adapter = _make_adapter()
        adapter.complete = AsyncMock(side_effect=ValueError("unexpected"))
        runner = Runner(adapter)
        suite_run = await runner.run(_make_suite([_make_test("t1")]))
        assert suite_run.errored_tests == 1
        assert "Unexpected error" in (suite_run.results[0].error or "")

    @pytest.mark.asyncio
    async def test_unexpected_exception_does_not_abort_run(self) -> None:
        """An unexpected exception from one test must not abort others."""
        call_count = 0

        async def raises_on_first(
            prompt: str, system_prompt: str | None = None
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("one-time failure")
            return ModelResponse(text="ok")

        adapter = _make_adapter()
        adapter.complete = raises_on_first
        tests = [_make_test("t1"), _make_test("t2"), _make_test("t3")]
        runner = Runner(adapter, concurrency=1)
        suite_run = await runner.run(_make_suite(tests))
        assert call_count == 3
        assert suite_run.errored_tests == 1


# ===========================================================================
# Runner.run — tag filtering
# ===========================================================================


class TestRunnerTagFiltering:
    @pytest.mark.asyncio
    async def test_tag_filter_runs_only_matching_tests(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        tests = [
            _make_test("t1", tags=["smoke"]),
            _make_test("t2", tags=["regression"]),
            _make_test("t3", tags=["smoke", "regression"]),
        ]
        suite_run = await runner.run(_make_suite(tests), tags=["smoke"])
        ids = {r.test_id for r in suite_run.results}
        assert ids == {"t1", "t3"}

    @pytest.mark.asyncio
    async def test_no_matching_tags_raises_runner_error(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        suite = _make_suite([_make_test("t1", tags=["foo"])])
        with pytest.raises(RunnerError, match="No tests matched"):
            await runner.run(suite, tags=["nonexistent"])

    @pytest.mark.asyncio
    async def test_none_tags_runs_all_tests(self) -> None:
        adapter = _make_adapter()
        runner = Runner(adapter)
        tests = [_make_test("t1"), _make_test("t2")]
        suite_run = await runner.run(_make_suite(tests), tags=None)
        assert suite_run.total_tests == 2

    @pytest.mark.asyncio
    async def test_empty_tags_list_runs_all_tests(self) -> None:
        """tags=[] is treated as no filter, identical to tags=None."""
        adapter = _make_adapter()
        runner = Runner(adapter)
        tests = [_make_test("t1"), _make_test("t2")]
        suite_run = await runner.run(_make_suite(tests), tags=[])
        assert suite_run.total_tests == 2


# ===========================================================================
# Concurrency
# ===========================================================================


class TestRunnerConcurrency:
    @pytest.mark.asyncio
    async def test_concurrency_one_serialises_calls(self) -> None:
        """With concurrency=1 each call must start after the previous one."""
        max_concurrent = 0
        current_concurrent = 0

        async def controlled_complete(
            prompt: str, system_prompt: str | None = None
        ) -> ModelResponse:
            nonlocal current_concurrent, max_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0)  # yield to allow other tasks to run
            current_concurrent -= 1
            return ModelResponse(text="ok")

        adapter = _make_adapter()
        adapter.complete = controlled_complete
        tests = [_make_test(f"t{i}") for i in range(4)]
        runner = Runner(adapter, concurrency=1)
        await runner.run(_make_suite(tests))
        # With concurrency=1 peak simultaneous calls must never exceed 1
        assert max_concurrent == 1

    @pytest.mark.asyncio
    async def test_concurrency_higher_than_test_count_all_run(self) -> None:
        """Setting concurrency higher than the number of tests is safe."""
        adapter = _make_adapter()
        runner = Runner(adapter, concurrency=100)
        tests = [_make_test(f"t{i}") for i in range(3)]
        suite_run = await runner.run(_make_suite(tests))
        assert suite_run.total_tests == 3
