"""llmeval — LLM evaluation and regression testing framework.

Package surface convention
--------------------------
- Schema/data types (``TestSuite``, ``SuiteRun``, ``Rubric``, etc.) live in
  ``llmeval.schema`` and are imported from there directly.
- Execution classes (``Runner``, ``Judge``) are re-exported here so callers
  can do ``from llmeval import Runner, Judge`` without knowing the sub-module.
"""

from llmeval.judge import Judge
from llmeval.report import CliReporter, DiffReporter
from llmeval.report.diff_reporter import TestDiff, compute_diff
from llmeval.runner import Runner
from llmeval.storage import SQLiteStorage, StorageBackend

__version__ = "0.1.0"

__all__ = [
    "CliReporter",
    "DiffReporter",
    "Judge",
    "Runner",
    "SQLiteStorage",
    "StorageBackend",
    "TestDiff",
    "compute_diff",
    "__version__",
]
