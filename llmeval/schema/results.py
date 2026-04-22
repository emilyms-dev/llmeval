"""Pydantic v2 models for test run results and scores.

These models represent the *output* side of the evaluation pipeline.
The runner populates :class:`TestResult` with raw model output; the judge
fills in :class:`CriterionScore` objects and computes the weighted score.
:class:`SuiteRun` aggregates a full execution pass over a test suite.

Hierarchy::

    SuiteRun
    └── results: list[TestResult]
                 └── criterion_scores: list[CriterionScore]
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import BaseModel, Field, computed_field, model_validator

RunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class CriterionScore(BaseModel):
    """Judge-assigned score for a single rubric criterion.

    Args:
        name: Must match a :attr:`~llmeval.schema.test_suite.Criterion.name`
            in the corresponding :class:`~llmeval.schema.test_suite.Rubric`.
        score: Normalised score from ``0.0`` (completely fails the criterion)
            to ``1.0`` (fully satisfies it). When ``samples > 1`` this is the
            median across all samples.
        reasoning: One- or two-sentence rationale from the judge explaining
            why this score was assigned.
        score_stddev: Standard deviation of scores across judge samples.
            ``None`` when only a single sample was taken (``samples=1``).
    """

    name: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)
    score_stddev: float | None = Field(default=None, ge=0.0)


class TestResult(BaseModel):
    """Result for a single test case execution.

    The runner sets ``test_id``, ``prompt``, ``model``, and ``raw_output``
    (or ``error``). The judge subsequently populates ``criterion_scores``,
    ``weighted_score``, and ``passed``.

    Args:
        test_id: Matches :attr:`~llmeval.schema.test_suite.TestCase.id`.
        prompt: The prompt that was sent (denormalised for self-contained
            storage and display).
        model: The model identifier that produced ``raw_output``.
        raw_output: Verbatim text response from the model. Empty string if
            ``error`` is set.
        criterion_scores: One score per rubric criterion, in rubric order.
        weighted_score: ``sum(c.weight * s.score)`` over all criteria, as
            computed by the judge. Range ``[0.0, 1.0]``.
        passed: ``True`` when ``weighted_score >=`` the rubric's
            ``passing_threshold``.
        passing_threshold: The rubric's passing threshold, copied from the
            suite definition so results are self-contained.
        error: Human-readable error description if the model call or judge
            scoring failed. ``None`` on success.
    """

    test_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    raw_output: str
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    weighted_score: float = Field(default=0.0, ge=0.0, le=1.0)
    passed: bool = False
    passing_threshold: float | None = None
    error: str | None = None
    judge_tokens: dict[str, int] | None = None


class SuiteRun(BaseModel):
    """Aggregated record of a complete test suite execution.

    Args:
        run_id: UUID string, auto-generated on construction.
        suite_name: Copied from :attr:`~llmeval.schema.test_suite.SuiteConfig.name`.
        suite_version: Copied from
            :attr:`~llmeval.schema.test_suite.SuiteConfig.version`.
        model: Model under test (may differ from suite default if overridden
            via CLI).
        judge_model: Model used for LLM-as-judge scoring.
        status: Lifecycle state of the run.

            - ``"pending"`` — saved to storage, pipeline not yet started.
            - ``"running"`` — pipeline is actively executing.
            - ``"completed"`` — pipeline finished (tests may still have failed).
            - ``"failed"`` — pipeline itself encountered an unrecoverable error.
            - ``"cancelled"`` — stopped by user before or during execution.

        suite_path: Filesystem path to the suite YAML/JSON that was loaded.
            ``None`` when the run was created programmatically without a file.
        tags: Tag filter applied when the run was triggered. Empty list means
            no filter (all tests ran).
        concurrency: Semaphore width used for concurrent API calls.
        error_message: Top-level error description when ``status="failed"``.
            ``None`` for all other statuses.
        started_at: UTC timestamp set when the run begins.
        completed_at: UTC timestamp set when the pipeline finishes. ``None``
            while the run is in progress.
        results: Ordered list of :class:`TestResult` objects, one per test
            case.
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    suite_name: str = Field(..., min_length=1)
    suite_version: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    judge_model: str = Field(..., min_length=1)
    status: RunStatus = "completed"
    suite_path: str | None = None
    tags: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    concurrency: int = Field(default=5, ge=1)
    error_message: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    results: list[TestResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def completed_after_started(self) -> Self:
        """Ensure ``completed_at`` is not earlier than ``started_at``."""
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not be earlier than started_at")
        return self

    # ------------------------------------------------------------------
    # Computed summaries — derived from results, included in model_dump()
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tests(self) -> int:
        """Total number of test results recorded so far."""
        return len(self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed_tests(self) -> int:
        """Number of tests that passed (``passed=True``, no error)."""
        return sum(1 for r in self.results if r.passed and r.error is None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed_tests(self) -> int:
        """Number of tests that completed scoring but did not pass."""
        return sum(1 for r in self.results if not r.passed and r.error is None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def errored_tests(self) -> int:
        """Number of tests that could not be scored due to an error."""
        return sum(1 for r in self.results if r.error is not None)
