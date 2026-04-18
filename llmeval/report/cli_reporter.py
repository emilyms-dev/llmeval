"""Rich-powered CLI report renderer for suite run results.

Displays a formatted suite run report with a metadata header, per-test
results table, a failure detail section (with judge reasoning), and a
pass/fail summary panel.

Example::

    from rich.console import Console
    from llmeval.report import CliReporter

    reporter = CliReporter()
    reporter.print_run(suite_run)
"""

from __future__ import annotations

from typing import Final

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from llmeval.schema.results import SuiteRun, TestResult

_PASS_STYLE: Final[str] = "bold green"
_FAIL_STYLE: Final[str] = "bold red"
_ERROR_STYLE: Final[str] = "bold yellow"
_DIM: Final[str] = "dim"


class CliReporter:
    """Rich-powered terminal reporter for suite run results.

    Args:
        console: Rich :class:`~rich.console.Console` instance to write to.
            Defaults to a standard stderr-backed console when omitted.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_run(self, run: SuiteRun) -> None:
        """Print a complete formatted report for *run*.

        Renders four sections: metadata header, results table, failure detail
        (only when failures or errors exist), and summary panel.

        Args:
            run: The :class:`~llmeval.schema.results.SuiteRun` to render.
        """
        self._print_header(run)
        if run.results:
            self._print_results_table(run)
        if run.failed_tests > 0 or run.errored_tests > 0:
            self._print_failure_detail(run)
        self._print_summary(run)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _print_header(self, run: SuiteRun) -> None:
        lines = [
            f"[bold]{run.suite_name}[/bold]  [{_DIM}]v{run.suite_version}[/{_DIM}]",
            f"Model:  [cyan]{run.model}[/cyan]",
            f"Judge:  [cyan]{run.judge_model}[/cyan]",
        ]
        if run.suite_path:
            lines.append(f"Suite:  [{_DIM}]{run.suite_path}[/{_DIM}]")
        if run.tags:
            lines.append(f"Tags:   [{_DIM}]{', '.join(run.tags)}[/{_DIM}]")
        lines.append(f"Run ID: [{_DIM}]{run.run_id}[/{_DIM}]")

        started = run.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if run.completed_at is not None:
            duration = (run.completed_at - run.started_at).total_seconds()
            lines.append(f"Time:   {started}  [{_DIM}]({duration:.1f}s)[/{_DIM}]")
        else:
            status_label = {
                "pending": "[yellow](pending)[/yellow]",
                "running": "[yellow](running…)[/yellow]",
                "failed": "[red](failed)[/red]",
            }.get(run.status, "[yellow](in progress)[/yellow]")
            lines.append(f"Time:   {started}  {status_label}")

        if run.error_message:
            lines.append(f"Error:  [red]{run.error_message}[/red]")

        self._console.print(
            Panel("\n".join(lines), title="[bold]llmeval run[/bold]", expand=False)
        )

    def _print_results_table(self, run: SuiteRun) -> None:
        table = Table(box=box.SIMPLE_HEAD, show_footer=False, expand=False)
        table.add_column("Test ID", style="bold", no_wrap=True)
        table.add_column("Status", justify="center", no_wrap=True)
        table.add_column("Score", justify="right", no_wrap=True)
        table.add_column("Threshold", justify="right", no_wrap=True)
        table.add_column("Criteria")

        for result in run.results:
            test_id, status, score, threshold, detail = self._result_row(result)
            table.add_row(test_id, status, score, threshold, detail)

        self._console.print(table)

    def _print_failure_detail(self, run: SuiteRun) -> None:
        """Print a section showing judge reasoning for every failing test."""
        failing = [r for r in run.results if r.error is not None or not r.passed]
        # Sort: errors last (they have no scores to reason about), then by score asc.
        failing.sort(key=lambda r: (r.error is not None, r.weighted_score))

        self._console.print(Rule("[bold red]Failure detail[/bold red]"))

        for result in failing:
            if result.error is not None:
                self._console.print(
                    f"  [bold]{result.test_id}[/bold]  "
                    f"[{_ERROR_STYLE}]⚠ Error[/{_ERROR_STYLE}]  "
                    f"[yellow]{result.error}[/yellow]"
                )
                self._console.print()
                continue

            threshold_str = (
                f"  [dim](threshold {result.passing_threshold:.2f})[/dim]"
                if result.passing_threshold is not None
                else ""
            )
            self._console.print(
                f"  [bold]{result.test_id}[/bold]  "
                f"[{_FAIL_STYLE}]✗ {result.weighted_score:.2f}[/{_FAIL_STYLE}]"
                f"{threshold_str}"
            )

            # Show all criteria, highlighting the ones dragging the score down.
            for cs in sorted(result.criterion_scores, key=lambda c: c.score):
                score_style = _FAIL_STYLE if cs.score < 0.6 else _DIM  # noqa: PLR2004
                self._console.print(
                    f"    [{score_style}]{cs.name}: {cs.score:.2f}[/{score_style}]"
                    f"  [dim]{cs.reasoning}[/dim]"
                )
            self._console.print()

    def _print_summary(self, run: SuiteRun) -> None:
        total = run.total_tests
        passed = run.passed_tests
        failed = run.failed_tests
        errored = run.errored_tests
        rate = (passed / total * 100) if total > 0 else 0.0

        if errored > 0:
            rate_style = _ERROR_STYLE
        elif failed > 0:
            rate_style = _FAIL_STYLE
        else:
            rate_style = _PASS_STYLE

        summary = (
            f"Total: [bold]{total}[/bold]   "
            f"[{_PASS_STYLE}]✓ Passed: {passed}[/{_PASS_STYLE}]   "
            f"[{_FAIL_STYLE}]✗ Failed: {failed}[/{_FAIL_STYLE}]   "
            f"[{_ERROR_STYLE}]⚠ Errored: {errored}[/{_ERROR_STYLE}]   "
            f"Pass rate: [{rate_style}]{rate:.1f}%[/{rate_style}]"
        )
        self._console.print(Panel(summary, expand=False))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _result_row(self, result: TestResult) -> tuple[str, Text, str, str, str]:
        threshold_str = (
            f"{result.passing_threshold:.2f}"
            if result.passing_threshold is not None
            else "—"
        )
        if result.error is not None:
            status = Text("⚠ ERROR", style=_ERROR_STYLE)
            return result.test_id, status, "—", "—", f"[yellow]{result.error}[/yellow]"

        score_str = f"{result.weighted_score:.2f}"
        criteria_str = _format_criteria(result)
        if result.passed:
            return (
                result.test_id,
                Text("✓ PASS", style=_PASS_STYLE),
                score_str,
                threshold_str,
                criteria_str,
            )
        return (
            result.test_id,
            Text("✗ FAIL", style=_FAIL_STYLE),
            score_str,
            threshold_str,
            criteria_str,
        )


def _format_criteria(result: TestResult) -> str:
    """Return a compact inline summary of criterion scores for *result*."""
    if not result.criterion_scores:
        return ""
    return "  ·  ".join(f"{s.name}: {s.score:.2f}" for s in result.criterion_scores)
