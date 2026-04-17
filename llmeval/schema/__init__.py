"""Schema package — Pydantic models for test suite definitions and run results.

Public surface::

    from llmeval.schema import (
        # Test suite (input)
        Criterion, Rubric, TestCase, SuiteConfig, TestSuite, load_suite,
        # Run results (output)
        CriterionScore, TestResult, SuiteRun,
    )
"""

from llmeval.schema.results import CriterionScore, SuiteRun, TestResult
from llmeval.schema.test_suite import (
    Criterion,
    Rubric,
    SuiteConfig,
    TestCase,
    TestSuite,
    load_suite,
)

__all__ = [
    # test_suite
    "Criterion",
    "Rubric",
    "SuiteConfig",
    "TestCase",
    "TestSuite",
    "load_suite",
    # results
    "CriterionScore",
    "SuiteRun",
    "TestResult",
]
