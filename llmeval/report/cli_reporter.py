"""Rich-powered CLI report renderer for suite run results.

Displays a formatted suite run report with a metadata header, per-test
results table, and a pass/fail summary panel. Designed to be called
directly after a run completes or when replaying a stored run.

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
from rich.table import Table
from rich.text import Text

from llmeval.schema.results import SuiteRun, TestResult

_PASS_STYLE: Final[str] = "bold green"
_FAIL_STYLE: Final[str] = "bold red"
_ERROR_STYLE: Final[str] = "bold yellow"


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

        Renders three sections in order: metadata header panel, results
        table (omitted when the run has no results), and summary panel.

        Args:
            run: The :class:`~llmeval.schema.results.SuiteRun` to render.
        """
        self._print_header(run)
        if run.results:
            self._print_results_table(run)
        self._print_summary(run)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _print_header(self, run: SuiteRun) -> None:
        lines = [
            f"[bold]{run.suite_name}[/bold]  [dim]v{run.suite_version}[/dim]",
            f"Model:  [cyan]{run.model}[/cyan]",
            f"Judge:  [cyan]{run.judge_model}[/cyan]",
            f"Run ID: [dim]{run.run_id}[/dim]",
        ]
        started = run.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if run.completed_at is not None:
            duration = (run.completed_at - run.started_at).total_seconds()
            lines.append(f"Time:   {started}  [dim]({duration:.1f}s)[/dim]")
        else:
            lines.append(f"Time:   {started}  [yellow](in progress)[/yellow]")

        self._console.print(
            Panel("\n".join(lines), title="[bold]llmeval run[/bold]", expand=False)
        )

    def _print_results_table(self, run: SuiteRun) -> None:
        table = Table(box=box.SIMPLE_HEAD, show_footer=False, expand=False)
        table.add_column("Test ID", style="bold", no_wrap=True)
        table.add_column("Status", justify="center", no_wrap=True)
        table.add_column("Score", justify="right", no_wrap=True)
        table.add_column("Criteria / Error")

        for result in run.results:
            test_id, status, score, detail = self._result_row(result)
            table.add_row(test_id, status, score, detail)

        self._console.print(table)

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

    def _result_row(self, result: TestResult) -> tuple[str, Text, str, str]:
        if result.error is not None:
            status = Text("⚠ ERROR", style=_ERROR_STYLE)
            return result.test_id, status, "", f"[yellow]{result.error}[/yellow]"

        score_str = f"{result.weighted_score:.2f}"
        criteria_str = _format_criteria(result)
        if result.passed:
            return (
                result.test_id,
                Text("✓ PASS", style=_PASS_STYLE),
                score_str,
                criteria_str,
            )
        return (
            result.test_id,
            Text("✗ FAIL", style=_FAIL_STYLE),
            score_str,
            criteria_str,
        )


def _format_criteria(result: TestResult) -> str:
    """Return a compact inline summary of criterion scores for *result*."""
    if not result.criterion_scores:
        return ""
    return "  ·  ".join(
        f"{s.name}: {s.score:.2f}" for s in result.criterion_scores
    )
