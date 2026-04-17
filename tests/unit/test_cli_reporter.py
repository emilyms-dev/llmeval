"""Unit tests for llmeval.report.cli_reporter.

Uses an in-memory Rich Console (file=io.StringIO()) to capture rendered
output and assert on its text content without mocking Rich internals.

Coverage targets:
- CliReporter.print_run: header, results table, summary rendered in sequence
- Header: suite name, version, model, judge model, run_id, timestamps, duration
- Header: in-progress run (completed_at=None) shows "(in progress)"
- Results table: PASS / FAIL / ERROR rows rendered with correct labels and scores
- Results table: omitted when run has no results
- Results table: criterion scores formatted correctly
- Results table: results with no criterion scores render empty detail column
- Summary: correct counts and pass rate rendered
- Summary: pass rate style — all pass (green), any fail (red), any error (yellow)
- Summary: zero-test run does not divide by zero
- _format_criteria: helper tested independently
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

from rich.console import Console

from llmeval.report.cli_reporter import CliReporter, _format_criteria
from llmeval.schema.results import CriterionScore, SuiteRun, TestResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, highlight=False, markup=True), buf


def _make_run(
    results: list[TestResult] | None = None,
    completed_at_offset: float | None = 5.0,
) -> SuiteRun:
    started = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    completed = (
        started + timedelta(seconds=completed_at_offset)
        if completed_at_offset is not None
        else None
    )
    return SuiteRun(
        suite_name="My Suite",
        suite_version="1.2.3",
        model="claude-sonnet-4-20250514",
        judge_model="claude-opus-4-20250514",
        started_at=started,
        completed_at=completed,
        results=results or [],
    )


def _passing_result(test_id: str = "t1", score: float = 0.9) -> TestResult:
    return TestResult(
        test_id=test_id,
        prompt="Hello",
        model="claude-sonnet-4-20250514",
        raw_output="Hi there",
        criterion_scores=[
            CriterionScore(name="quality", score=score, reasoning="Good."),
            CriterionScore(name="tone", score=0.8, reasoning="Friendly."),
        ],
        weighted_score=score,
        passed=True,
    )


def _failing_result(test_id: str = "t2", score: float = 0.4) -> TestResult:
    return TestResult(
        test_id=test_id,
        prompt="Hello",
        model="claude-sonnet-4-20250514",
        raw_output="meh",
        criterion_scores=[
            CriterionScore(name="quality", score=score, reasoning="Poor."),
        ],
        weighted_score=score,
        passed=False,
    )


def _errored_result(test_id: str = "t3", message: str = "API timed out") -> TestResult:
    return TestResult(
        test_id=test_id,
        prompt="Hello",
        model="claude-sonnet-4-20250514",
        raw_output="",
        error=message,
    )


# ===========================================================================
# Header
# ===========================================================================


class TestHeader:
    def test_suite_name_in_header(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run())
        assert "My Suite" in buf.getvalue()

    def test_suite_version_in_header(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run())
        assert "1.2.3" in buf.getvalue()

    def test_model_in_header(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run())
        assert "claude-sonnet-4-20250514" in buf.getvalue()

    def test_judge_model_in_header(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run())
        assert "claude-opus-4-20250514" in buf.getvalue()

    def test_run_id_in_header(self) -> None:
        console, buf = _console()
        run = _make_run()
        CliReporter(console).print_run(run)
        assert run.run_id in buf.getvalue()

    def test_started_at_in_header(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run())
        assert "2024-06-01 12:00:00 UTC" in buf.getvalue()

    def test_duration_shown_when_completed(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(completed_at_offset=7.3))
        assert "7.3s" in buf.getvalue()

    def test_in_progress_when_no_completed_at(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(completed_at_offset=None))
        assert "in progress" in buf.getvalue()


# ===========================================================================
# Results table
# ===========================================================================


class TestResultsTable:
    def test_pass_row_shows_pass_label(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[_passing_result()]))
        assert "PASS" in buf.getvalue()

    def test_fail_row_shows_fail_label(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[_failing_result()]))
        assert "FAIL" in buf.getvalue()

    def test_error_row_shows_error_label(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[_errored_result()]))
        assert "ERROR" in buf.getvalue()

    def test_error_row_shows_error_message(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(
            _make_run(results=[_errored_result(message="Network failure")])
        )
        assert "Network failure" in buf.getvalue()

    def test_pass_row_shows_test_id(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[_passing_result("my-test-id")]))
        assert "my-test-id" in buf.getvalue()

    def test_pass_row_shows_score(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[_passing_result(score=0.87)]))
        assert "0.87" in buf.getvalue()

    def test_pass_row_shows_criterion_scores(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[_passing_result()]))
        output = buf.getvalue()
        assert "quality" in output
        assert "tone" in output

    def test_table_omitted_when_no_results(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[]))
        output = buf.getvalue()
        assert "Test ID" not in output

    def test_multiple_results_all_appear(self) -> None:
        console, buf = _console()
        results = [
            _passing_result("t1"),
            _failing_result("t2"),
            _errored_result("t3"),
        ]
        CliReporter(console).print_run(_make_run(results=results))
        output = buf.getvalue()
        assert "t1" in output
        assert "t2" in output
        assert "t3" in output

    def test_result_with_no_criterion_scores(self) -> None:
        result = TestResult(
            test_id="bare",
            prompt="p",
            model="m",
            raw_output="r",
            weighted_score=0.5,
            passed=False,
        )
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[result]))
        assert "bare" in buf.getvalue()


# ===========================================================================
# Summary panel
# ===========================================================================


class TestSummary:
    def test_total_count_in_summary(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(
            _make_run(results=[_passing_result(), _failing_result()])
        )
        assert "Total: 2" in buf.getvalue()

    def test_passed_count_in_summary(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(
            _make_run(results=[_passing_result(), _failing_result()])
        )
        assert "Passed: 1" in buf.getvalue()

    def test_failed_count_in_summary(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(
            _make_run(results=[_passing_result(), _failing_result()])
        )
        assert "Failed: 1" in buf.getvalue()

    def test_errored_count_in_summary(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(
            _make_run(results=[_errored_result()])
        )
        assert "Errored: 1" in buf.getvalue()

    def test_pass_rate_all_pass(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(
            _make_run(results=[_passing_result("t1"), _passing_result("t2")])
        )
        assert "100.0%" in buf.getvalue()

    def test_pass_rate_mixed(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(
            _make_run(results=[_passing_result("t1"), _failing_result("t2")])
        )
        assert "50.0%" in buf.getvalue()

    def test_pass_rate_zero_results(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[]))
        assert "0.0%" in buf.getvalue()

    def test_summary_renders_without_results(self) -> None:
        console, buf = _console()
        CliReporter(console).print_run(_make_run(results=[]))
        assert "Total: 0" in buf.getvalue()


# ===========================================================================
# _format_criteria helper
# ===========================================================================


class TestFormatCriteria:
    def test_formats_single_criterion(self) -> None:
        result = TestResult(
            test_id="t",
            prompt="p",
            model="m",
            raw_output="r",
            criterion_scores=[
                CriterionScore(name="quality", score=0.75, reasoning="ok")
            ],
        )
        assert _format_criteria(result) == "quality: 0.75"

    def test_formats_multiple_criteria_with_separator(self) -> None:
        result = TestResult(
            test_id="t",
            prompt="p",
            model="m",
            raw_output="r",
            criterion_scores=[
                CriterionScore(name="quality", score=0.8, reasoning="ok"),
                CriterionScore(name="tone", score=0.6, reasoning="meh"),
            ],
        )
        formatted = _format_criteria(result)
        assert "quality: 0.80" in formatted
        assert "tone: 0.60" in formatted
        assert "  ·  " in formatted

    def test_empty_criterion_scores_returns_empty_string(self) -> None:
        result = TestResult(
            test_id="t", prompt="p", model="m", raw_output="r"
        )
        assert _format_criteria(result) == ""
