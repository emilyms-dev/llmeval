"""Report rendering package.

Public surface::

    from llmeval.report import CliReporter, DiffReporter
"""

from llmeval.report.cli_reporter import CliReporter
from llmeval.report.diff_reporter import DiffReporter, TestDiff, compute_diff

__all__ = ["CliReporter", "DiffReporter", "TestDiff", "compute_diff"]
