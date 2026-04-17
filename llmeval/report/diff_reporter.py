"""Side-by-side diff reporter for comparing two suite runs.

Compares two :class:`~llmeval.schema.results.SuiteRun` objects keyed by
``test_id`` and renders a Rich-formatted terminal report showing regressions,
improvements, score deltas, and tests that appear in only one run.

Example::

    from llmeval.report import DiffReporter

    reporter = DiffReporter()
    reporter.print_diff(baseline_run, candidate_run)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmeval.schema.results import SuiteRun, TestResult

_PASS_STYLE: Final[str] = "bold green"
_FAIL_STYLE: Final[str] = "bold red"
_ERROR_STYLE: Final[str] = "bold yellow"
_REGRESSION_STYLE: Final[str] = "bold red"
_IMPROVEMENT_STYLE: Final[str] = "bold green"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestDiff:
    """Comparison of a single test across two suite runs.

    Args:
        test_id: The test identifier shared by both runs.
        result_a: Result from run A (baseline). ``None`` if the test was not
            present in run A.
        result_b: Result from run B (candidate). ``None`` if the test was not
            present in run B.
    """

    test_id: str
    result_a: TestResult | None
    result_b: TestResult | None

    @property
    def is_regression(self) -> bool:
        """``True`` when A passed and B did not (or errored)."""
        return (
            self.result_a is not None
            and self.result_a.error is None
            and self.result_a.passed
            and self.result_b is not None
            and (not self.result_b.passed or self.result_b.error is not None)
        )

    @property
    def is_improvement(self) -> bool:
        """``True`` when A did not pass (or errored) and B passed."""
        return (
            self.result_a is not None
            and (not self.result_a.passed or self.result_a.error is not None)
            and self.result_b is not None
            and self.result_b.error is None
            and self.result_b.passed
        )

    @property
    def score_delta(self) -> float | None:
        """``weighted_score`` change (B − A).

        Returns ``None`` when either result is absent or carries an error,
        since those results have no meaningful score.
        """
        if (
            self.result_a is None
            or self.result_b is None
            or self.result_a.error is not None
            or self.result_b.error is not None
        ):
            return None
        return self.result_b.weighted_score - self.result_a.weighted_score


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_diff(run_a: SuiteRun, run_b: SuiteRun) -> list[TestDiff]:
    """Compare two suite runs by ``test_id``.

    Test IDs from *run_a* appear first (preserving *run_a*'s ordering),
    followed by any IDs that are only in *run_b*.

    Args:
        run_a: Baseline / reference run.
        run_b: Candidate / comparison run.

    Returns:
        One :class:`TestDiff` per unique ``test_id`` across both runs.
    """
    map_a = {r.test_id: r for r in run_a.results}
    map_b = {r.test_id: r for r in run_b.results}
    all_ids = list(map_a) + [k for k in map_b if k not in map_a]
    return [TestDiff(tid, map_a.get(tid), map_b.get(tid)) for tid in all_ids]


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class DiffReporter:
    """Rich-powered terminal reporter for side-by-side suite run comparison.

    Args:
        console: Rich :class:`~rich.console.Console` to write to.
            Defaults to a standard console when omitted.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def print_diff(self, run_a: SuiteRun, run_b: SuiteRun) -> None:
        """Print a formatted diff between *run_a* (baseline) and *run_b*.

        Renders a metadata header, a per-test comparison table, and a
        summary panel with regression/improvement counts and pass-rate delta.

        Args:
            run_a: Baseline / reference run (labelled **A**).
            run_b: Candidate / comparison run (labelled **B**).
        """
        diffs = compute_diff(run_a, run_b)
        self._print_header(run_a, run_b)
        if diffs:
            self._print_diff_table(diffs)
        self._print_summary(run_a, run_b, diffs)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _print_header(self, run_a: SuiteRun, run_b: SuiteRun) -> None:
        def _rate(run: SuiteRun) -> str:
            if run.total_tests == 0:
                return "—"
            return f"{run.passed_tests / run.total_tests * 100:.1f}%"

        lines = [
            f"[bold]{run_a.suite_name}[/bold]  [dim]v{run_a.suite_version}[/dim]",
            "",
            (
                f"  [dim]A[/dim]  model: [cyan]{run_a.model}[/cyan]  "
                f"run: [dim]{run_a.run_id[:8]}…[/dim]  "
                f"pass rate: {_rate(run_a)}"
            ),
            (
                f"  [dim]B[/dim]  model: [cyan]{run_b.model}[/cyan]  "
                f"run: [dim]{run_b.run_id[:8]}…[/dim]  "
                f"pass rate: {_rate(run_b)}"
            ),
        ]
        self._console.print(
            Panel("\n".join(lines), title="[bold]llmeval diff[/bold]", expand=False)
        )

    def _print_diff_table(self, diffs: list[TestDiff]) -> None:
        table = Table(box=box.SIMPLE_HEAD, expand=False, show_footer=False)
        table.add_column("Test ID", style="bold", no_wrap=True)
        table.add_column("A Status", justify="center", no_wrap=True)
        table.add_column("A Score", justify="right", no_wrap=True)
        table.add_column(" ", justify="center", no_wrap=True)  # change indicator
        table.add_column("B Status", justify="center", no_wrap=True)
        table.add_column("B Score", justify="right", no_wrap=True)
        table.add_column("Δ Score", justify="right", no_wrap=True)

        for diff in diffs:
            a_status, a_score = _status_cell(diff.result_a)
            b_status, b_score = _status_cell(diff.result_b)
            indicator = _change_indicator(diff)
            delta = _delta_cell(diff)
            table.add_row(
                diff.test_id, a_status, a_score, indicator, b_status, b_score, delta
            )

        self._console.print(table)

    def _print_summary(
        self, run_a: SuiteRun, run_b: SuiteRun, diffs: list[TestDiff]
    ) -> None:
        regressions = sum(1 for d in diffs if d.is_regression)
        improvements = sum(1 for d in diffs if d.is_improvement)
        unchanged = sum(
            1
            for d in diffs
            if not d.is_regression
            and not d.is_improvement
            and d.result_a is not None
            and d.result_b is not None
        )
        only_a = sum(1 for d in diffs if d.result_b is None)
        only_b = sum(1 for d in diffs if d.result_a is None)

        def _rate(run: SuiteRun) -> float:
            return run.passed_tests / run.total_tests if run.total_tests > 0 else 0.0

        delta_pp = (_rate(run_b) - _rate(run_a)) * 100
        if delta_pp > 0:
            delta_str = (
                f"[{_IMPROVEMENT_STYLE}]{delta_pp:+.1f}pp[/{_IMPROVEMENT_STYLE}]"
            )
        elif delta_pp < 0:
            delta_str = (
                f"[{_REGRESSION_STYLE}]{delta_pp:+.1f}pp[/{_REGRESSION_STYLE}]"
            )
        else:
            delta_str = "[dim]±0.0pp[/dim]"

        reg = f"[bold red]▼ Regressions: {regressions}[/bold red]"
        imp = f"[bold green]▲ Improvements: {improvements}[/bold green]"
        lines = [f"{reg}   {imp}   = Unchanged: {unchanged}"]
        if only_a or only_b:
            lines.append(
                f"[dim]Only in A: {only_a}   Only in B: {only_b}[/dim]"
            )
        lines.append(f"Pass-rate delta (B − A): {delta_str}")

        self._console.print(Panel("\n".join(lines), expand=False))


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------


def _status_cell(result: TestResult | None) -> tuple[Text, str]:
    """Return a (status Text, score string) pair for a single result cell."""
    if result is None:
        return Text("—", style="dim"), "[dim]—[/dim]"
    if result.error is not None:
        return Text("⚠ ERROR", style=_ERROR_STYLE), ""
    score = f"{result.weighted_score:.2f}"
    if result.passed:
        return Text("✓ PASS", style=_PASS_STYLE), score
    return Text("✗ FAIL", style=_FAIL_STYLE), score


def _change_indicator(diff: TestDiff) -> Text:
    """Return a single-character change indicator for the middle column."""
    if diff.result_a is None:
        return Text("+", style="bold cyan")
    if diff.result_b is None:
        return Text("−", style="dim")
    if diff.is_regression:
        return Text("▼", style=_REGRESSION_STYLE)
    if diff.is_improvement:
        return Text("▲", style=_IMPROVEMENT_STYLE)
    return Text("→", style="dim")


def _delta_cell(diff: TestDiff) -> str:
    """Return a formatted score-delta string with colour."""
    delta = diff.score_delta
    if delta is None:
        return ""
    if delta > 0.001:
        return f"[{_IMPROVEMENT_STYLE}]{delta:+.2f}[/{_IMPROVEMENT_STYLE}]"
    if delta < -0.001:
        return f"[{_REGRESSION_STYLE}]{delta:+.2f}[/{_REGRESSION_STYLE}]"
    return "[dim]±0.00[/dim]"
