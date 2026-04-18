"""Unit tests for llmeval.report.diff_reporter.

Uses an in-memory Rich Console (file=io.StringIO()) for output assertions.

Coverage targets:
- compute_diff: ordering, only-in-A, only-in-B, empty runs, both empty
- TestDiff.is_regression: PASS→FAIL, PASS→ERROR, FAIL→FAIL (not regression),
  missing result_a, missing result_b
- TestDiff.is_improvement: FAIL→PASS, ERROR→PASS, PASS→PASS (not improvement),
  missing result_a or result_b
- TestDiff.score_delta: normal, positive, negative, near-zero, missing result,
  errored result
- DiffReporter.print_diff: header content, table content, summary counts
- Cell helpers: _status_cell, _change_indicator, _delta_cell
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

from rich.console import Console

from llmeval.report.diff_reporter import (
    DiffReporter,
    TestDiff,
    _change_indicator,
    _delta_cell,
    _status_cell,
    compute_diff,
)
from llmeval.schema.results import CriterionScore, SuiteRun, TestResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, highlight=False, markup=True), buf


def _make_run(
    results: list[TestResult] | None = None,
    model: str = "claude-sonnet-4-20250514",
    suite_name: str = "My Suite",
) -> SuiteRun:
    started = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    return SuiteRun(
        suite_name=suite_name,
        suite_version="1.0.0",
        model=model,
        judge_model="claude-sonnet-4-20250514",
        started_at=started,
        completed_at=started + timedelta(seconds=5),
        results=results or [],
    )


def _pass(test_id: str, score: float = 0.9) -> TestResult:
    return TestResult(
        test_id=test_id,
        prompt="p",
        model="m",
        raw_output="r",
        criterion_scores=[CriterionScore(name="q", score=score, reasoning="ok")],
        weighted_score=score,
        passed=True,
    )


def _fail(test_id: str, score: float = 0.4) -> TestResult:
    return TestResult(
        test_id=test_id,
        prompt="p",
        model="m",
        raw_output="r",
        criterion_scores=[CriterionScore(name="q", score=score, reasoning="poor")],
        weighted_score=score,
        passed=False,
    )


def _error(test_id: str, msg: str = "timeout") -> TestResult:
    return TestResult(test_id=test_id, prompt="p", model="m", raw_output="", error=msg)


# ===========================================================================
# compute_diff
# ===========================================================================


class TestComputeDiff:
    def test_same_test_ids_produce_one_diff_each(self) -> None:
        run_a = _make_run([_pass("t1"), _fail("t2")])
        run_b = _make_run([_pass("t1"), _pass("t2")])
        diffs = compute_diff(run_a, run_b)
        assert len(diffs) == 2
        assert {d.test_id for d in diffs} == {"t1", "t2"}

    def test_preserves_run_a_ordering(self) -> None:
        run_a = _make_run([_pass("c"), _pass("a"), _pass("b")])
        run_b = _make_run([_pass("a"), _pass("b"), _pass("c")])
        diffs = compute_diff(run_a, run_b)
        assert [d.test_id for d in diffs] == ["c", "a", "b"]

    def test_test_only_in_a_has_none_result_b(self) -> None:
        run_a = _make_run([_pass("t1"), _pass("t2")])
        run_b = _make_run([_pass("t1")])
        diffs = compute_diff(run_a, run_b)
        diff_t2 = next(d for d in diffs if d.test_id == "t2")
        assert diff_t2.result_a is not None
        assert diff_t2.result_b is None

    def test_test_only_in_b_has_none_result_a(self) -> None:
        run_a = _make_run([_pass("t1")])
        run_b = _make_run([_pass("t1"), _pass("t-new")])
        diffs = compute_diff(run_a, run_b)
        diff_new = next(d for d in diffs if d.test_id == "t-new")
        assert diff_new.result_a is None
        assert diff_new.result_b is not None

    def test_only_in_b_tests_appended_after_run_a_ids(self) -> None:
        run_a = _make_run([_pass("t1")])
        run_b = _make_run([_pass("t-new"), _pass("t1")])
        diffs = compute_diff(run_a, run_b)
        assert diffs[0].test_id == "t1"
        assert diffs[1].test_id == "t-new"

    def test_both_empty_returns_empty_list(self) -> None:
        assert compute_diff(_make_run(), _make_run()) == []

    def test_run_a_empty_all_diffs_have_none_result_a(self) -> None:
        run_b = _make_run([_pass("t1"), _pass("t2")])
        diffs = compute_diff(_make_run(), run_b)
        assert all(d.result_a is None for d in diffs)

    def test_run_b_empty_all_diffs_have_none_result_b(self) -> None:
        run_a = _make_run([_pass("t1"), _pass("t2")])
        diffs = compute_diff(run_a, _make_run())
        assert all(d.result_b is None for d in diffs)


# ===========================================================================
# TestDiff.is_regression
# ===========================================================================


class TestIsRegression:
    def test_pass_to_fail_is_regression(self) -> None:
        d = TestDiff("t", _pass("t"), _fail("t"))
        assert d.is_regression is True

    def test_pass_to_error_is_regression(self) -> None:
        d = TestDiff("t", _pass("t"), _error("t"))
        assert d.is_regression is True

    def test_fail_to_fail_is_not_regression(self) -> None:
        d = TestDiff("t", _fail("t"), _fail("t"))
        assert d.is_regression is False

    def test_fail_to_pass_is_not_regression(self) -> None:
        d = TestDiff("t", _fail("t"), _pass("t"))
        assert d.is_regression is False

    def test_pass_to_pass_is_not_regression(self) -> None:
        d = TestDiff("t", _pass("t"), _pass("t"))
        assert d.is_regression is False

    def test_missing_result_a_is_not_regression(self) -> None:
        d = TestDiff("t", None, _fail("t"))
        assert d.is_regression is False

    def test_missing_result_b_is_not_regression(self) -> None:
        d = TestDiff("t", _pass("t"), None)
        assert d.is_regression is False

    def test_error_to_fail_is_not_regression(self) -> None:
        d = TestDiff("t", _error("t"), _fail("t"))
        assert d.is_regression is False


# ===========================================================================
# TestDiff.is_improvement
# ===========================================================================


class TestIsImprovement:
    def test_fail_to_pass_is_improvement(self) -> None:
        d = TestDiff("t", _fail("t"), _pass("t"))
        assert d.is_improvement is True

    def test_error_to_pass_is_improvement(self) -> None:
        d = TestDiff("t", _error("t"), _pass("t"))
        assert d.is_improvement is True

    def test_pass_to_pass_is_not_improvement(self) -> None:
        d = TestDiff("t", _pass("t"), _pass("t"))
        assert d.is_improvement is False

    def test_pass_to_fail_is_not_improvement(self) -> None:
        d = TestDiff("t", _pass("t"), _fail("t"))
        assert d.is_improvement is False

    def test_fail_to_fail_is_not_improvement(self) -> None:
        d = TestDiff("t", _fail("t"), _fail("t"))
        assert d.is_improvement is False

    def test_missing_result_a_is_not_improvement(self) -> None:
        d = TestDiff("t", None, _pass("t"))
        assert d.is_improvement is False

    def test_missing_result_b_is_not_improvement(self) -> None:
        d = TestDiff("t", _fail("t"), None)
        assert d.is_improvement is False

    def test_fail_to_error_is_not_improvement(self) -> None:
        d = TestDiff("t", _fail("t"), _error("t"))
        assert d.is_improvement is False


# ===========================================================================
# TestDiff.score_delta
# ===========================================================================


class TestScoreDelta:
    def test_positive_delta(self) -> None:
        d = TestDiff("t", _fail("t", score=0.4), _pass("t", score=0.9))
        assert d.score_delta == pytest.approx(0.5)

    def test_negative_delta(self) -> None:
        d = TestDiff("t", _pass("t", score=0.9), _fail("t", score=0.4))
        assert d.score_delta == pytest.approx(-0.5)

    def test_zero_delta(self) -> None:
        d = TestDiff("t", _pass("t", score=0.8), _pass("t", score=0.8))
        assert d.score_delta == pytest.approx(0.0)

    def test_none_when_result_a_missing(self) -> None:
        d = TestDiff("t", None, _pass("t"))
        assert d.score_delta is None

    def test_none_when_result_b_missing(self) -> None:
        d = TestDiff("t", _pass("t"), None)
        assert d.score_delta is None

    def test_none_when_result_a_errored(self) -> None:
        d = TestDiff("t", _error("t"), _pass("t"))
        assert d.score_delta is None

    def test_none_when_result_b_errored(self) -> None:
        d = TestDiff("t", _pass("t"), _error("t"))
        assert d.score_delta is None


# ===========================================================================
# Cell helpers
# ===========================================================================


class TestStatusCell:
    def test_none_result_returns_dash(self) -> None:
        text, score = _status_cell(None)
        assert "—" in text.plain
        assert "—" in score

    def test_passing_result(self) -> None:
        text, score = _status_cell(_pass("t", score=0.85))
        assert "PASS" in text.plain
        assert "0.85" in score

    def test_failing_result(self) -> None:
        text, score = _status_cell(_fail("t", score=0.3))
        assert "FAIL" in text.plain
        assert "0.30" in score

    def test_errored_result_empty_score(self) -> None:
        text, score = _status_cell(_error("t"))
        assert "ERROR" in text.plain
        assert score == ""


class TestChangeIndicator:
    def test_regression_shows_down_arrow(self) -> None:
        d = TestDiff("t", _pass("t"), _fail("t"))
        assert "▼" in _change_indicator(d).plain

    def test_improvement_shows_up_arrow(self) -> None:
        d = TestDiff("t", _fail("t"), _pass("t"))
        assert "▲" in _change_indicator(d).plain

    def test_unchanged_shows_right_arrow(self) -> None:
        d = TestDiff("t", _pass("t"), _pass("t"))
        assert "→" in _change_indicator(d).plain

    def test_only_in_b_shows_plus(self) -> None:
        d = TestDiff("t", None, _pass("t"))
        assert "+" in _change_indicator(d).plain

    def test_only_in_a_shows_minus(self) -> None:
        d = TestDiff("t", _pass("t"), None)
        assert "−" in _change_indicator(d).plain


class TestDeltaCell:
    def test_positive_delta_has_plus_sign(self) -> None:
        d = TestDiff("t", _fail("t", 0.4), _pass("t", 0.9))
        assert "+0.50" in _delta_cell(d)

    def test_negative_delta_has_minus_sign(self) -> None:
        d = TestDiff("t", _pass("t", 0.9), _fail("t", 0.4))
        assert "-0.50" in _delta_cell(d)

    def test_near_zero_delta(self) -> None:
        d = TestDiff("t", _pass("t", 0.8), _pass("t", 0.8))
        assert "±0.00" in _delta_cell(d)

    def test_none_delta_returns_empty_string(self) -> None:
        d = TestDiff("t", None, _pass("t"))
        assert _delta_cell(d) == ""


# ===========================================================================
# DiffReporter.print_diff — integration
# ===========================================================================


import pytest  # noqa: E402  (after helpers so fixtures/marks are available)


class TestDiffReporterOutput:
    def test_header_contains_suite_name(self) -> None:
        console, buf = _console()
        run_a = _make_run(suite_name="Support Suite")
        run_b = _make_run(suite_name="Support Suite")
        DiffReporter(console).print_diff(run_a, run_b)
        assert "Support Suite" in buf.getvalue()

    def test_header_contains_both_models(self) -> None:
        console, buf = _console()
        run_a = _make_run(model="model-a")
        run_b = _make_run(model="model-b")
        DiffReporter(console).print_diff(run_a, run_b)
        output = buf.getvalue()
        assert "model-a" in output
        assert "model-b" in output

    def test_header_contains_truncated_run_ids(self) -> None:
        console, buf = _console()
        run_a = _make_run()
        run_b = _make_run()
        DiffReporter(console).print_diff(run_a, run_b)
        output = buf.getvalue()
        assert run_a.run_id[:8] in output
        assert run_b.run_id[:8] in output

    def test_header_shows_pass_rates(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_pass("t1"), _fail("t2")])
        run_b = _make_run(results=[_pass("t1"), _pass("t2")])
        DiffReporter(console).print_diff(run_a, run_b)
        output = buf.getvalue()
        assert "50.0%" in output
        assert "100.0%" in output

    def test_table_contains_test_ids(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_pass("tone-001"), _fail("tone-002")])
        run_b = _make_run(results=[_pass("tone-001"), _pass("tone-002")])
        DiffReporter(console).print_diff(run_a, run_b)
        output = buf.getvalue()
        assert "tone-001" in output
        assert "tone-002" in output

    def test_table_omitted_when_both_runs_empty(self) -> None:
        console, buf = _console()
        DiffReporter(console).print_diff(_make_run(), _make_run())
        assert "Test ID" not in buf.getvalue()

    def test_summary_regression_count(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_pass("t1"), _pass("t2")])
        run_b = _make_run(results=[_fail("t1"), _pass("t2")])
        DiffReporter(console).print_diff(run_a, run_b)
        assert "Regressions: 1" in buf.getvalue()

    def test_summary_improvement_count(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_fail("t1")])
        run_b = _make_run(results=[_pass("t1")])
        DiffReporter(console).print_diff(run_a, run_b)
        assert "Improvements: 1" in buf.getvalue()

    def test_summary_unchanged_count(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_pass("t1"), _fail("t2")])
        run_b = _make_run(results=[_pass("t1"), _fail("t2")])
        DiffReporter(console).print_diff(run_a, run_b)
        assert "Unchanged: 2" in buf.getvalue()

    def test_summary_shows_only_in_a_and_b_when_asymmetric(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_pass("t1"), _pass("removed")])
        run_b = _make_run(results=[_pass("t1"), _pass("added")])
        DiffReporter(console).print_diff(run_a, run_b)
        output = buf.getvalue()
        assert "Only in A: 1" in output
        assert "Only in B: 1" in output

    def test_summary_hides_only_in_line_when_symmetric(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_pass("t1")])
        run_b = _make_run(results=[_pass("t1")])
        DiffReporter(console).print_diff(run_a, run_b)
        assert "Only in" not in buf.getvalue()

    def test_summary_positive_pass_rate_delta(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_fail("t1")])
        run_b = _make_run(results=[_pass("t1")])
        DiffReporter(console).print_diff(run_a, run_b)
        assert "+100.0pp" in buf.getvalue()

    def test_summary_negative_pass_rate_delta(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_pass("t1")])
        run_b = _make_run(results=[_fail("t1")])
        DiffReporter(console).print_diff(run_a, run_b)
        assert "-100.0pp" in buf.getvalue()

    def test_summary_zero_pass_rate_delta(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_pass("t1")])
        run_b = _make_run(results=[_pass("t1")])
        DiffReporter(console).print_diff(run_a, run_b)
        assert "±0.0pp" in buf.getvalue()

    def test_empty_run_pass_rate_shows_dash(self) -> None:
        console, buf = _console()
        DiffReporter(console).print_diff(_make_run(), _make_run())
        assert "—" in buf.getvalue()

    def test_error_result_appears_in_table(self) -> None:
        console, buf = _console()
        run_a = _make_run(results=[_pass("t1")])
        run_b = _make_run(results=[_error("t1", "timeout")])
        DiffReporter(console).print_diff(run_a, run_b)
        assert "ERROR" in buf.getvalue()
