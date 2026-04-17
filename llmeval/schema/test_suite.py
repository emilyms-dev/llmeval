"""Pydantic v2 models for the YAML/JSON test suite schema.

The top-level entry point for users is :func:`load_suite`, which reads a
``.yaml``, ``.yml``, or ``.json`` file and returns a validated
:class:`TestSuite` instance.

Hierarchy::

    TestSuite
    ├── suite: SuiteConfig
    └── tests: list[TestCase]
                └── rubric: Rubric
                            └── criteria: list[Criterion]
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from llmeval.exceptions import ConfigurationError, SchemaValidationError


class Criterion(BaseModel):
    """A single scoring criterion within a rubric.

    Args:
        name: Short identifier for the criterion (e.g. ``"empathy"``).
        description: Human-readable description of what the criterion measures.
        weight: Fractional contribution to the overall score. All weights in a
            :class:`Rubric` must sum to ``1.0``.
    """

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0.0, le=1.0)


class Rubric(BaseModel):
    """Scoring rubric for a single test case.

    Args:
        criteria: One or more :class:`Criterion` objects. Names must be unique.
            Weights are automatically normalised to sum to ``1.0`` if they do
            not already; a :class:`UserWarning` is emitted when normalisation
            is applied so that suite authors are aware.
        passing_threshold: Minimum weighted score (``0.0``–``1.0``) required
            for a test to be considered passing.
    """

    criteria: list[Criterion] = Field(..., min_length=1)
    passing_threshold: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_rubric_invariants(self) -> Self:
        """Enforce unique criterion names; normalise weights to sum to 1.0.

        If the weights do not already sum to 1.0 (within 1e-6 tolerance) they
        are rescaled proportionally and a :class:`UserWarning` is emitted.
        This keeps suite files forgiving while surfacing the discrepancy so
        authors can fix it intentionally.
        """
        seen: set[str] = set()
        for criterion in self.criteria:
            if criterion.name in seen:
                raise ValueError(
                    f"Duplicate criterion name: {criterion.name!r}"
                )
            seen.add(criterion.name)

        total = sum(c.weight for c in self.criteria)
        if abs(total - 1.0) > 1e-6:
            warnings.warn(
                f"Criterion weights sum to {total:.6f} instead of 1.0. "
                "Normalising automatically — update your suite file to silence "
                "this warning.",
                UserWarning,
                stacklevel=2,
            )
            self.criteria = [
                Criterion(
                    name=c.name,
                    description=c.description,
                    weight=c.weight / total,
                )
                for c in self.criteria
            ]

        return self


class TestCase(BaseModel):
    """A single behavioral test case.

    Args:
        id: Unique identifier for this test within the suite
            (e.g. ``"tone-empathy-001"``).
        description: One-sentence summary of what is being tested.
        prompt: The user message sent to the model under test.
        rubric: Scoring rubric applied to the model's response.
        tags: Optional list of labels for filtering / grouping tests.
        system_prompt: Optional system prompt override. When omitted the
            model adapter's default system prompt (if any) is used.
    """

    id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    rubric: Rubric
    tags: list[str] = Field(default_factory=list)
    system_prompt: str | None = None

    @field_validator("tags")
    @classmethod
    def tags_are_unique(cls, v: list[str]) -> list[str]:
        """Reject duplicate tags within a single test case."""
        seen: set[str] = set()
        for tag in v:
            if tag in seen:
                raise ValueError(f"Duplicate tag: {tag!r}")
            seen.add(tag)
        return v


class SuiteConfig(BaseModel):
    """Suite-level metadata and defaults.

    Args:
        name: Human-readable name for the test suite.
        version: Semantic version string for the suite definition.
        model: Default model identifier used when running tests
            (e.g. ``"claude-sonnet-4-20250514"``).
        judge_model: Model used for LLM-as-judge scoring. May differ from
            ``model`` to keep judge costs predictable.
    """

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    judge_model: str = Field(..., min_length=1)


class TestSuite(BaseModel):
    """Top-level test suite definition, parsed from a YAML or JSON file.

    Args:
        suite: Suite-level configuration metadata.
        tests: One or more :class:`TestCase` objects. Test IDs must be unique.
    """

    suite: SuiteConfig
    tests: list[TestCase] = Field(..., min_length=1)

    @model_validator(mode="after")
    def test_ids_are_unique(self) -> Self:
        """Reject suites where two or more tests share the same ID."""
        seen: set[str] = set()
        for test in self.tests:
            if test.id in seen:
                raise ValueError(
                    f"Duplicate test ID: {test.id!r}. "
                    "Each test must have a unique id within the suite."
                )
            seen.add(test.id)
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_suite(path: str | Path) -> TestSuite:
    """Load and validate a test suite from a YAML or JSON file.

    Supports ``.yaml``, ``.yml``, and ``.json`` extensions. Raises specific
    exceptions so CLI error messages are actionable.

    Args:
        path: Filesystem path to the test suite definition file.

    Returns:
        A fully validated :class:`TestSuite` instance.

    Raises:
        SchemaValidationError: If the file cannot be read, fails to parse, or
            does not satisfy the schema constraints.
        ConfigurationError: If the file extension is not ``.yaml``, ``.yml``,
            or ``.json``.
    """
    resolved = Path(path)

    # Validate extension before touching the filesystem so callers get a
    # ConfigurationError (not a SchemaValidationError) for bad extensions,
    # even when the file doesn't exist.
    suffix = resolved.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ConfigurationError(
            f"Unsupported file extension {resolved.suffix!r}. "
            "Test suite files must use .yaml, .yml, or .json."
        )

    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaValidationError(
            f"Cannot read suite file {resolved}: {exc}"
        ) from exc

    match suffix:
        case ".yaml" | ".yml":
            try:
                data = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise SchemaValidationError(
                    f"Invalid YAML in {resolved}: {exc}"
                ) from exc
        case ".json":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    f"Invalid JSON in {resolved}: {exc}"
                ) from exc

    if not isinstance(data, dict):
        raise SchemaValidationError(
            f"Suite file {resolved} must contain a YAML/JSON object at the "
            f"top level, got {type(data).__name__}."
        )

    try:
        return TestSuite.model_validate(data)
    except Exception as exc:
        raise SchemaValidationError(
            f"Suite validation failed for {resolved}: {exc}"
        ) from exc
