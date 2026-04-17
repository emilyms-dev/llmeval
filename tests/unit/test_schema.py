"""Unit tests for llmeval.schema.

Coverage targets:
- test_suite.py — Criterion, Rubric, TestCase, SuiteConfig, TestSuite, load_suite
- results.py    — CriterionScore, TestResult, SuiteRun and its computed properties
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmeval.exceptions import ConfigurationError, SchemaValidationError
from llmeval.schema import (
    Criterion,
    CriterionScore,
    Rubric,
    SuiteRun,
    TestCase,
    TestResult,
    TestSuite,
    load_suite,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"

# ---------------------------------------------------------------------------
# Helpers — minimal valid objects
# ---------------------------------------------------------------------------


def _criterion(name: str = "clarity", weight: float = 1.0) -> dict:
    return {"name": name, "description": f"Tests {name}", "weight": weight}


def _rubric(criteria: list[dict] | None = None) -> dict:
    if criteria is None:
        criteria = [_criterion()]
    return {"criteria": criteria, "passing_threshold": 0.75}


def _test_case(test_id: str = "tc-001") -> dict:
    return {
        "id": test_id,
        "description": "A test case",
        "prompt": "Hello, world!",
        "rubric": _rubric(),
    }


def _suite(tests: list[dict] | None = None) -> dict:
    if tests is None:
        tests = [_test_case()]
    return {
        "suite": {
            "name": "Test Suite",
            "version": "1.0.0",
            "model": "claude-sonnet-4-20250514",
            "judge_model": "claude-sonnet-4-20250514",
        },
        "tests": tests,
    }


# ===========================================================================
# Criterion
# ===========================================================================


class TestCriterion:
    def test_valid(self) -> None:
        c = Criterion(
            name="empathy", description="Acknowledges frustration", weight=0.5
        )
        assert c.name == "empathy"
        assert c.weight == 0.5

    def test_name_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            Criterion(name="", description="desc", weight=0.5)

    def test_weight_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Criterion(name="c", description="d", weight=0.0)

    def test_weight_cannot_exceed_one(self) -> None:
        with pytest.raises(ValidationError):
            Criterion(name="c", description="d", weight=1.1)


# ===========================================================================
# Rubric
# ===========================================================================


class TestRubric:
    def test_valid_single_criterion(self) -> None:
        r = Rubric.model_validate(_rubric())
        assert r.passing_threshold == 0.75
        assert len(r.criteria) == 1

    def test_valid_multi_criterion(self) -> None:
        criteria = [
            _criterion("empathy", 0.4),
            _criterion("actionability", 0.4),
            _criterion("tone", 0.2),
        ]
        r = Rubric.model_validate(_rubric(criteria))
        assert len(r.criteria) == 3

    def test_weights_normalised_automatically(self) -> None:
        """Non-unit weights are rescaled and a UserWarning is emitted."""
        criteria = [_criterion("a", 0.3), _criterion("b", 0.3)]
        with pytest.warns(UserWarning, match="Normalising automatically"):
            rubric = Rubric.model_validate(_rubric(criteria))
        total = sum(c.weight for c in rubric.criteria)
        assert abs(total - 1.0) < 1e-9
        # Each weight should be 0.5 (0.3 / 0.6)
        assert abs(rubric.criteria[0].weight - 0.5) < 1e-9
        assert abs(rubric.criteria[1].weight - 0.5) < 1e-9

    def test_duplicate_criterion_names_rejected(self) -> None:
        criteria = [_criterion("tone", 0.5), _criterion("tone", 0.5)]
        with pytest.raises(Exception, match="Duplicate criterion"):
            Rubric.model_validate(_rubric(criteria))

    def test_passing_threshold_below_zero_rejected(self) -> None:
        data = _rubric()
        data["passing_threshold"] = -0.1
        with pytest.raises(ValidationError):
            Rubric.model_validate(data)

    def test_passing_threshold_above_one_rejected(self) -> None:
        data = _rubric()
        data["passing_threshold"] = 1.1
        with pytest.raises(ValidationError):
            Rubric.model_validate(data)

    def test_empty_criteria_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Rubric.model_validate({"criteria": [], "passing_threshold": 0.5})

    def test_float_precision_tolerance(self) -> None:
        """Floating-point arithmetic near 1.0 must be accepted."""
        # 0.1 + 0.2 + 0.3 + 0.4 == 1.0 in exact arithmetic but may drift
        criteria = [
            _criterion("a", 0.1),
            _criterion("b", 0.2),
            _criterion("c", 0.3),
            _criterion("d", 0.4),
        ]
        r = Rubric.model_validate(_rubric(criteria))
        assert len(r.criteria) == 4


# ===========================================================================
# TestCase
# ===========================================================================


class TestTestCase:
    def test_valid_minimal(self) -> None:
        tc = TestCase.model_validate(_test_case())
        assert tc.id == "tc-001"
        assert tc.tags == []
        assert tc.system_prompt is None

    def test_valid_with_tags_and_system_prompt(self) -> None:
        data = _test_case()
        data["tags"] = ["tone", "regression-critical"]
        data["system_prompt"] = "You are a helpful assistant."
        tc = TestCase.model_validate(data)
        assert tc.system_prompt == "You are a helpful assistant."
        assert "tone" in tc.tags

    def test_empty_id_rejected(self) -> None:
        data = _test_case()
        data["id"] = ""
        with pytest.raises(ValidationError):
            TestCase.model_validate(data)

    def test_empty_prompt_rejected(self) -> None:
        data = _test_case()
        data["prompt"] = ""
        with pytest.raises(ValidationError):
            TestCase.model_validate(data)

    def test_duplicate_tags_rejected(self) -> None:
        data = _test_case()
        data["tags"] = ["tone", "tone"]
        with pytest.raises(Exception, match="Duplicate tag"):
            TestCase.model_validate(data)


# ===========================================================================
# TestSuite
# ===========================================================================


class TestTestSuite:
    def test_valid(self) -> None:
        s = TestSuite.model_validate(_suite())
        assert s.suite.name == "Test Suite"
        assert len(s.tests) == 1

    def test_duplicate_test_ids_rejected(self) -> None:
        tests = [_test_case("dup-001"), _test_case("dup-001")]
        with pytest.raises(Exception, match="Duplicate test ID"):
            TestSuite.model_validate(_suite(tests))

    def test_empty_tests_rejected(self) -> None:
        data = _suite([])
        with pytest.raises(ValidationError):
            TestSuite.model_validate(data)

    def test_model_dump_roundtrip(self) -> None:
        s = TestSuite.model_validate(_suite())
        restored = TestSuite.model_validate(s.model_dump())
        assert restored.suite.name == s.suite.name
        assert restored.tests[0].id == s.tests[0].id


# ===========================================================================
# load_suite
# ===========================================================================


class TestLoadSuite:
    def test_load_yaml_fixture(self) -> None:
        suite = load_suite(FIXTURES / "example_suite.yaml")
        assert suite.suite.name == "Customer Support Bot - Tone Tests"
        assert len(suite.tests) == 2

    def test_load_json_fixture(self) -> None:
        suite = load_suite(FIXTURES / "example_suite.json")
        assert suite.suite.model == "gpt-4o"
        assert suite.tests[0].id == "json-tone-001"

    def test_missing_file_raises_schema_error(self) -> None:
        with pytest.raises(SchemaValidationError, match="Cannot read"):
            load_suite("/nonexistent/path/suite.yaml")

    def test_unsupported_extension_raises_config_error(self) -> None:
        with pytest.raises(ConfigurationError, match="Unsupported file extension"):
            load_suite(FIXTURES / "example_suite.yaml.bak")

    def test_invalid_yaml_raises_schema_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("suite: [\nnot valid yaml: {{{", encoding="utf-8")
        with pytest.raises(SchemaValidationError, match="Invalid YAML"):
            load_suite(bad)

    def test_invalid_json_raises_schema_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not: valid json", encoding="utf-8")
        with pytest.raises(SchemaValidationError, match="Invalid JSON"):
            load_suite(bad)

    def test_schema_validation_failure_raises_schema_error(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("suite:\n  name: missing required fields\n", encoding="utf-8")
        with pytest.raises(SchemaValidationError, match="Suite validation failed"):
            load_suite(bad)

    def test_invalid_weights_fixture_loads_with_warning(self) -> None:
        """A suite with non-unit weights loads successfully after normalisation."""
        with pytest.warns(UserWarning, match="Normalising automatically"):
            suite = load_suite(FIXTURES / "invalid_weights.yaml")
        total = sum(c.weight for c in suite.tests[0].rubric.criteria)
        assert abs(total - 1.0) < 1e-9

    def test_top_level_non_dict_raises_schema_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(SchemaValidationError, match="object at the top level"):
            load_suite(bad)

    def test_load_suite_accepts_string_path(self) -> None:
        suite = load_suite(str(FIXTURES / "example_suite.yaml"))
        assert suite.suite.version == "1.0.0"

    def test_load_json_via_json_loads(self, tmp_path: Path) -> None:
        """Valid JSON file should parse identically to YAML equivalent."""
        data = _suite()
        p = tmp_path / "suite.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        suite = load_suite(p)
        assert suite.tests[0].id == "tc-001"


# ===========================================================================
# CriterionScore
# ===========================================================================


class TestCriterionScore:
    def test_valid(self) -> None:
        cs = CriterionScore(name="empathy", score=0.8, reasoning="Good.")
        assert cs.score == 0.8

    def test_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CriterionScore(name="a", score=-0.1, reasoning="r")

    def test_score_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CriterionScore(name="a", score=1.1, reasoning="r")

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CriterionScore(name="", score=0.5, reasoning="r")

    def test_empty_reasoning_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CriterionScore(name="a", score=0.5, reasoning="")


# ===========================================================================
# TestResult
# ===========================================================================


class TestTestResult:
    def test_valid_success(self) -> None:
        r = TestResult(
            test_id="tc-001",
            prompt="Hello",
            model="gpt-4o",
            raw_output="Hi there!",
            criterion_scores=[
                CriterionScore(name="tone", score=0.9, reasoning="Polite.")
            ],
            weighted_score=0.9,
            passed=True,
        )
        assert r.error is None
        assert r.passed is True

    def test_valid_error_result(self) -> None:
        r = TestResult(
            test_id="tc-001",
            prompt="Hello",
            model="gpt-4o",
            raw_output="",
            error="Rate limit exceeded",
        )
        assert r.error == "Rate limit exceeded"
        assert r.passed is False

    def test_weighted_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TestResult(
                test_id="tc-001",
                prompt="p",
                model="m",
                raw_output="o",
                weighted_score=1.5,
                passed=True,
            )


# ===========================================================================
# SuiteRun
# ===========================================================================


class TestSuiteRun:
    def _make_run(self, **kwargs: object) -> SuiteRun:
        defaults = dict(
            suite_name="My Suite",
            suite_version="1.0.0",
            model="claude-sonnet-4-20250514",
            judge_model="claude-sonnet-4-20250514",
        )
        defaults.update(kwargs)
        return SuiteRun(**defaults)

    def test_run_id_auto_generated(self) -> None:
        run = self._make_run()
        assert len(run.run_id) == 36  # UUID4 string
        assert run.run_id != self._make_run().run_id  # unique each time

    def test_started_at_is_utc(self) -> None:
        run = self._make_run()
        assert run.started_at.tzinfo is not None

    def test_completed_at_defaults_none(self) -> None:
        run = self._make_run()
        assert run.completed_at is None

    def test_completed_before_started_rejected(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(Exception, match="completed_at must not be earlier"):
            self._make_run(
                started_at=now,
                completed_at=now - timedelta(seconds=1),
            )

    def test_computed_properties_empty_run(self) -> None:
        run = self._make_run()
        assert run.total_tests == 0
        assert run.passed_tests == 0
        assert run.failed_tests == 0
        assert run.errored_tests == 0

    def test_computed_fields_present_in_model_dump(self) -> None:
        """Computed summary fields must appear in model_dump() output."""
        run = self._make_run()
        dumped = run.model_dump()
        for key in ("total_tests", "passed_tests", "failed_tests", "errored_tests"):
            assert key in dumped, f"{key!r} missing from model_dump()"
        assert dumped["total_tests"] == 0

    def test_computed_properties_with_results(self) -> None:
        def _r(test_id: str, passed: bool, error: str | None = None) -> TestResult:
            return TestResult(
                test_id=test_id,
                prompt="p",
                model="m",
                raw_output="" if error else "output",
                weighted_score=0.9 if passed else 0.3,
                passed=passed,
                error=error,
            )

        run = self._make_run()
        run.results.extend([
            _r("t1", passed=True),
            _r("t2", passed=False),
            _r("t3", passed=False, error="timeout"),
        ])
        assert run.total_tests == 3
        assert run.passed_tests == 1
        assert run.failed_tests == 1
        assert run.errored_tests == 1
