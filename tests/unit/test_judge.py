"""Unit tests for llmeval.judge (LLM-as-judge).

All judge model calls are mocked — no real API requests are made.

Coverage targets:
- _extract_json: clean JSON, markdown-fenced JSON, no JSON found
- _parse_scores: valid response, missing/duplicate/unknown criteria,
                 malformed entries, out-of-range score clamping
- _build_prompt: criteria block content, prompt/response injection
- _compute_weighted_score: single and multi-criterion weighting
- Judge.__init__: valid concurrency, invalid concurrency
- Judge.score: happy path, pre-errored result pass-through,
               JudgeError captured, unexpected exception captured
- Judge.score_suite_run: all results scored, order preserved,
                          unknown test_id raises, errored results unchanged
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmeval.exceptions import JudgeError
from llmeval.judge import (
    Judge,
    _build_prompt,
    _compute_weighted_score,
    _extract_json,
    _parse_scores,
)
from llmeval.models.base import ModelAdapter
from llmeval.schema.results import CriterionScore, SuiteRun, TestResult
from llmeval.schema.test_suite import (
    Criterion,
    Rubric,
    SuiteConfig,
    TestCase,
    TestSuite,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _criterion(name: str, weight: float = 1.0) -> Criterion:
    return Criterion(name=name, description=f"Tests {name}", weight=weight)


def _rubric(
    criteria: list[Criterion] | None = None,
    threshold: float = 0.75,
) -> Rubric:
    if criteria is None:
        criteria = [_criterion("quality")]
    return Rubric(criteria=criteria, passing_threshold=threshold)


def _result(
    test_id: str = "t1",
    raw_output: str = "The model response.",
    error: str | None = None,
) -> TestResult:
    return TestResult(
        test_id=test_id,
        prompt="Test prompt",
        model="claude-sonnet-4-20250514",
        raw_output=raw_output,
        error=error,
    )


def _make_adapter(response: str = "") -> MagicMock:
    adapter = MagicMock(spec=ModelAdapter)
    adapter.model_id = "claude-sonnet-4-20250514"
    adapter.complete = AsyncMock(return_value=response)
    return adapter


def _judge_json(*scores: tuple[str, float, str]) -> str:
    """Build a valid judge JSON response string."""
    items = [
        {"name": name, "score": score, "reasoning": reasoning}
        for name, score, reasoning in scores
    ]
    return json.dumps({"scores": items})


def _make_suite(tests: list[TestCase] | None = None) -> TestSuite:
    if tests is None:
        tests = [
            TestCase(
                id="t1",
                description="A test",
                prompt="Test prompt",
                rubric=_rubric(),
            )
        ]
    return TestSuite(
        suite=SuiteConfig(
            name="Test Suite",
            version="1.0.0",
            model="claude-sonnet-4-20250514",
            judge_model="claude-sonnet-4-20250514",
        ),
        tests=tests,
    )


def _make_suite_run(results: list[TestResult]) -> SuiteRun:
    return SuiteRun(
        suite_name="Test Suite",
        suite_version="1.0.0",
        model="claude-sonnet-4-20250514",
        judge_model="claude-sonnet-4-20250514",
        results=results,
    )


# ===========================================================================
# _extract_json
# ===========================================================================


class TestExtractJson:
    def test_clean_json_object(self) -> None:
        assert _extract_json('{"scores": []}') == '{"scores": []}'

    def test_json_with_leading_prose(self) -> None:
        text = 'Sure, here is the evaluation: {"scores": []}'
        assert _extract_json(text) == '{"scores": []}'

    def test_json_with_trailing_prose(self) -> None:
        text = '{"scores": []} Hope that helps!'
        assert _extract_json(text) == '{"scores": []}'

    def test_strips_json_markdown_fence(self) -> None:
        text = '```json\n{"scores": []}\n```'
        assert _extract_json(text) == '{"scores": []}'

    def test_strips_plain_markdown_fence(self) -> None:
        text = '```\n{"scores": []}\n```'
        assert _extract_json(text) == '{"scores": []}'

    def test_multiline_json(self) -> None:
        text = (
            '{\n  "scores": [\n'
            '    {"name": "a", "score": 1.0, "reasoning": "r"}\n'
            "  ]\n}"
        )
        result = _extract_json(text)
        parsed = json.loads(result)
        assert "scores" in parsed

    def test_no_json_raises_judge_error(self) -> None:
        with pytest.raises(JudgeError, match="No JSON object found"):
            _extract_json("There is no JSON here at all.")

    def test_empty_string_raises_judge_error(self) -> None:
        with pytest.raises(JudgeError, match="No JSON object found"):
            _extract_json("")


# ===========================================================================
# _parse_scores
# ===========================================================================


class TestParseScores:
    def test_valid_single_criterion(self) -> None:
        rubric = _rubric([_criterion("empathy")])
        raw = _judge_json(("empathy", 0.8, "Good empathy shown."))
        scores = _parse_scores(raw, rubric)
        assert len(scores) == 1
        assert scores[0].name == "empathy"
        assert scores[0].score == pytest.approx(0.8)
        assert scores[0].reasoning == "Good empathy shown."

    def test_valid_multi_criterion(self) -> None:
        rubric = _rubric([_criterion("a", 0.5), _criterion("b", 0.5)])
        raw = _judge_json(("a", 0.9, "r1"), ("b", 0.6, "r2"))
        scores = _parse_scores(raw, rubric)
        names = {s.name for s in scores}
        assert names == {"a", "b"}

    def test_score_above_one_is_clamped(self) -> None:
        rubric = _rubric([_criterion("q")])
        raw = _judge_json(("q", 1.05, "Slightly over."))
        scores = _parse_scores(raw, rubric)
        assert scores[0].score == pytest.approx(1.0)

    def test_score_below_zero_is_clamped(self) -> None:
        rubric = _rubric([_criterion("q")])
        raw = _judge_json(("q", -0.1, "Slightly under."))
        scores = _parse_scores(raw, rubric)
        assert scores[0].score == pytest.approx(0.0)

    def test_integer_score_is_accepted(self) -> None:
        rubric = _rubric([_criterion("q")])
        raw = json.dumps({"scores": [{"name": "q", "score": 1, "reasoning": "r"}]})
        scores = _parse_scores(raw, rubric)
        assert scores[0].score == pytest.approx(1.0)

    def test_missing_criterion_raises_judge_error(self) -> None:
        rubric = _rubric([_criterion("a"), _criterion("b")])
        raw = _judge_json(("a", 0.8, "r"))  # b missing
        with pytest.raises(JudgeError, match="Missing"):
            _parse_scores(raw, rubric)

    def test_unknown_criterion_raises_judge_error(self) -> None:
        rubric = _rubric([_criterion("a")])
        raw = _judge_json(("unknown", 0.5, "r"))
        with pytest.raises(JudgeError, match="unknown criterion"):
            _parse_scores(raw, rubric)

    def test_duplicate_criterion_raises_judge_error(self) -> None:
        rubric = _rubric([_criterion("a"), _criterion("b")])
        raw = _judge_json(("a", 0.8, "r"), ("a", 0.5, "r2"), ("b", 0.7, "r3"))
        with pytest.raises(JudgeError, match="duplicate"):
            _parse_scores(raw, rubric)

    def test_missing_scores_key_raises_judge_error(self) -> None:
        rubric = _rubric()
        raw = json.dumps({"result": []})
        with pytest.raises(JudgeError, match="'scores' key"):
            _parse_scores(raw, rubric)

    def test_scores_not_a_list_raises_judge_error(self) -> None:
        rubric = _rubric()
        raw = json.dumps({"scores": "wrong"})
        with pytest.raises(JudgeError, match="must be a list"):
            _parse_scores(raw, rubric)

    def test_score_entry_not_a_dict_raises_judge_error(self) -> None:
        rubric = _rubric([_criterion("q")])
        raw = json.dumps({"scores": ["not a dict"]})
        with pytest.raises(JudgeError, match="JSON object"):
            _parse_scores(raw, rubric)

    def test_non_numeric_score_raises_judge_error(self) -> None:
        rubric = _rubric([_criterion("q")])
        raw = json.dumps({"scores": [{"name": "q", "score": "high", "reasoning": "r"}]})
        with pytest.raises(JudgeError, match="must be a number"):
            _parse_scores(raw, rubric)

    def test_boolean_score_raises_judge_error(self) -> None:
        """bool is a subclass of int in Python — must be explicitly rejected."""
        rubric = _rubric([_criterion("q")])
        raw = json.dumps({"scores": [{"name": "q", "score": True, "reasoning": "r"}]})
        with pytest.raises(JudgeError, match="must be a number"):
            _parse_scores(raw, rubric)

    def test_empty_reasoning_raises_judge_error(self) -> None:
        rubric = _rubric([_criterion("q")])
        raw = json.dumps({"scores": [{"name": "q", "score": 0.5, "reasoning": ""}]})
        with pytest.raises(JudgeError, match="reasoning"):
            _parse_scores(raw, rubric)

    def test_whitespace_only_reasoning_raises_judge_error(self) -> None:
        rubric = _rubric([_criterion("q")])
        raw = json.dumps({"scores": [{"name": "q", "score": 0.5, "reasoning": "   "}]})
        with pytest.raises(JudgeError, match="reasoning"):
            _parse_scores(raw, rubric)

    def test_reasoning_is_stripped(self) -> None:
        rubric = _rubric([_criterion("q")])
        raw = json.dumps(
            {"scores": [{"name": "q", "score": 0.5, "reasoning": "  good  "}]}
        )
        scores = _parse_scores(raw, rubric)
        assert scores[0].reasoning == "good"

    def test_invalid_json_raises_judge_error(self) -> None:
        rubric = _rubric()
        with pytest.raises(JudgeError, match="invalid JSON"):
            _parse_scores("{not json}", rubric)


# ===========================================================================
# _build_prompt
# ===========================================================================


class TestBuildPrompt:
    def test_contains_original_prompt(self) -> None:
        rubric = _rubric()
        result = _build_prompt("User message", "Model output", rubric)
        assert "User message" in result

    def test_contains_model_response(self) -> None:
        rubric = _rubric()
        result = _build_prompt("p", "Model response here", rubric)
        assert "Model response here" in result

    def test_contains_criterion_names(self) -> None:
        rubric = _rubric([_criterion("empathy", 0.6), _criterion("tone", 0.4)])
        result = _build_prompt("p", "r", rubric)
        assert "empathy" in result
        assert "tone" in result

    def test_contains_criterion_weights(self) -> None:
        rubric = _rubric([_criterion("a", 0.4), _criterion("b", 0.6)])
        result = _build_prompt("p", "r", rubric)
        assert "0.40" in result
        assert "0.60" in result

    def test_contains_criterion_descriptions(self) -> None:
        rubric = _rubric([_criterion("empathy")])
        result = _build_prompt("p", "r", rubric)
        assert "Tests empathy" in result


# ===========================================================================
# _compute_weighted_score
# ===========================================================================


class TestComputeWeightedScore:
    def test_single_criterion_full_score(self) -> None:
        rubric = _rubric([_criterion("q", 1.0)])
        scores = [CriterionScore(name="q", score=1.0, reasoning="r")]
        assert _compute_weighted_score(scores, rubric) == pytest.approx(1.0)

    def test_single_criterion_zero_score(self) -> None:
        rubric = _rubric([_criterion("q", 1.0)])
        scores = [CriterionScore(name="q", score=0.0, reasoning="r")]
        assert _compute_weighted_score(scores, rubric) == pytest.approx(0.0)

    def test_multi_criterion_weighted_average(self) -> None:
        rubric = _rubric([_criterion("a", 0.4), _criterion("b", 0.6)])
        scores = [
            CriterionScore(name="a", score=1.0, reasoning="r"),
            CriterionScore(name="b", score=0.0, reasoning="r"),
        ]
        # 0.4*1.0 + 0.6*0.0 = 0.4
        assert _compute_weighted_score(scores, rubric) == pytest.approx(0.4)

    def test_all_criteria_half_score(self) -> None:
        rubric = _rubric([_criterion("a", 0.3), _criterion("b", 0.7)])
        scores = [
            CriterionScore(name="a", score=0.5, reasoning="r"),
            CriterionScore(name="b", score=0.5, reasoning="r"),
        ]
        assert _compute_weighted_score(scores, rubric) == pytest.approx(0.5)


# ===========================================================================
# Judge construction
# ===========================================================================


class TestJudgeConstruction:
    def test_valid_concurrency(self) -> None:
        judge = Judge(_make_adapter(), concurrency=3)
        assert judge._concurrency == 3

    def test_default_concurrency_is_five(self) -> None:
        judge = Judge(_make_adapter())
        assert judge._concurrency == 5

    def test_zero_concurrency_raises_judge_error(self) -> None:
        with pytest.raises(JudgeError, match="concurrency must be >= 1"):
            Judge(_make_adapter(), concurrency=0)

    def test_negative_concurrency_raises_judge_error(self) -> None:
        with pytest.raises(JudgeError, match="concurrency must be >= 1"):
            Judge(_make_adapter(), concurrency=-1)


# ===========================================================================
# Judge.score — happy path
# ===========================================================================


class TestJudgeScoreHappyPath:
    @pytest.mark.asyncio
    async def test_returns_updated_test_result(self) -> None:
        rubric = _rubric([_criterion("quality")])
        adapter = _make_adapter(_judge_json(("quality", 0.9, "Very good.")))
        judge = Judge(adapter)
        scored = await judge.score(_result(), rubric)
        assert isinstance(scored, TestResult)

    @pytest.mark.asyncio
    async def test_criterion_scores_populated(self) -> None:
        rubric = _rubric([_criterion("quality")])
        adapter = _make_adapter(_judge_json(("quality", 0.9, "Very good.")))
        judge = Judge(adapter)
        scored = await judge.score(_result(), rubric)
        assert len(scored.criterion_scores) == 1
        assert scored.criterion_scores[0].name == "quality"
        assert scored.criterion_scores[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_weighted_score_computed(self) -> None:
        rubric = _rubric([_criterion("a", 0.4), _criterion("b", 0.6)])
        adapter = _make_adapter(_judge_json(("a", 1.0, "r"), ("b", 0.5, "r")))
        judge = Judge(adapter)
        scored = await judge.score(_result(), rubric)
        # 0.4*1.0 + 0.6*0.5 = 0.7
        assert scored.weighted_score == pytest.approx(0.7, abs=1e-5)

    @pytest.mark.asyncio
    async def test_passes_when_above_threshold(self) -> None:
        rubric = _rubric([_criterion("q")], threshold=0.75)
        adapter = _make_adapter(_judge_json(("q", 0.9, "r")))
        scored = await Judge(adapter).score(_result(), rubric)
        assert scored.passed is True

    @pytest.mark.asyncio
    async def test_fails_when_below_threshold(self) -> None:
        rubric = _rubric([_criterion("q")], threshold=0.75)
        adapter = _make_adapter(_judge_json(("q", 0.5, "r")))
        scored = await Judge(adapter).score(_result(), rubric)
        assert scored.passed is False

    @pytest.mark.asyncio
    async def test_passes_when_exactly_at_threshold(self) -> None:
        rubric = _rubric([_criterion("q")], threshold=0.75)
        adapter = _make_adapter(_judge_json(("q", 0.75, "r")))
        scored = await Judge(adapter).score(_result(), rubric)
        assert scored.passed is True

    @pytest.mark.asyncio
    async def test_original_fields_preserved(self) -> None:
        rubric = _rubric([_criterion("q")])
        adapter = _make_adapter(_judge_json(("q", 0.8, "r")))
        result = _result(test_id="orig-001", raw_output="original output")
        scored = await Judge(adapter).score(result, rubric)
        assert scored.test_id == "orig-001"
        assert scored.raw_output == "original output"
        assert scored.error is None

    @pytest.mark.asyncio
    async def test_judge_called_with_system_prompt(self) -> None:
        rubric = _rubric([_criterion("q")])
        adapter = _make_adapter(_judge_json(("q", 0.8, "r")))
        await Judge(adapter).score(_result(), rubric)
        system_prompt = adapter.complete.call_args.kwargs.get("system_prompt")
        assert system_prompt is not None
        assert "objective evaluator" in system_prompt

    @pytest.mark.asyncio
    async def test_judge_prompt_contains_raw_output(self) -> None:
        rubric = _rubric([_criterion("q")])
        adapter = _make_adapter(_judge_json(("q", 0.8, "r")))
        result = _result(raw_output="unique model output XYZ")
        await Judge(adapter).score(result, rubric)
        prompt_arg = adapter.complete.call_args.args[0]
        assert "unique model output XYZ" in prompt_arg


# ===========================================================================
# Judge.score — error handling
# ===========================================================================


class TestJudgeScoreErrors:
    @pytest.mark.asyncio
    async def test_pre_errored_result_returned_unchanged(self) -> None:
        adapter = _make_adapter()
        judge = Judge(adapter)
        errored = _result(error="runner failed")
        result = await judge.score(errored, _rubric())
        assert result.error == "runner failed"
        adapter.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_judge_error_stored_not_raised(self) -> None:
        adapter = _make_adapter("not valid json at all")
        judge = Judge(adapter)
        scored = await judge.score(_result(), _rubric())
        assert scored.error is not None
        assert scored.criterion_scores == []

    @pytest.mark.asyncio
    async def test_judge_error_message_in_result(self) -> None:
        adapter = _make_adapter('{"scores": []}')  # missing criteria
        judge = Judge(adapter)
        rubric = _rubric([_criterion("quality")])
        scored = await judge.score(_result(), rubric)
        assert scored.error is not None
        assert "Missing" in (scored.error or "")

    @pytest.mark.asyncio
    async def test_unexpected_exception_captured(self) -> None:
        adapter = _make_adapter()
        adapter.complete = AsyncMock(side_effect=RuntimeError("network error"))
        judge = Judge(adapter)
        scored = await judge.score(_result(), _rubric())
        assert scored.error is not None
        assert "Unexpected judge error" in (scored.error or "")

    @pytest.mark.asyncio
    async def test_model_adapter_error_captured(self) -> None:
        from llmeval.exceptions import ModelAdapterError

        adapter = _make_adapter()
        adapter.complete = AsyncMock(side_effect=ModelAdapterError("API down"))
        judge = Judge(adapter)
        scored = await judge.score(_result(), _rubric())
        assert scored.error is not None

    @pytest.mark.asyncio
    async def test_error_result_has_zero_weighted_score(self) -> None:
        adapter = _make_adapter("garbage")
        judge = Judge(adapter)
        scored = await judge.score(_result(), _rubric())
        assert scored.weighted_score == pytest.approx(0.0)


# ===========================================================================
# Judge.score_suite_run
# ===========================================================================


class TestJudgeScoreSuiteRun:
    def _make_test_case(
        self,
        test_id: str,
        criteria: list[Criterion] | None = None,
    ) -> TestCase:
        return TestCase(
            id=test_id,
            description="A test",
            prompt="Test prompt",
            rubric=_rubric(criteria),
        )

    @pytest.mark.asyncio
    async def test_returns_suite_run(self) -> None:
        suite = _make_suite()
        run = _make_suite_run([_result("t1")])
        adapter = _make_adapter(_judge_json(("quality", 0.8, "r")))
        result = await Judge(adapter).score_suite_run(run, suite)
        assert isinstance(result, SuiteRun)

    @pytest.mark.asyncio
    async def test_all_results_scored(self) -> None:
        criteria = [_criterion("quality")]
        tests = [
            self._make_test_case("t1", criteria),
            self._make_test_case("t2", criteria),
        ]
        suite = _make_suite(tests)
        run = _make_suite_run([_result("t1"), _result("t2")])
        adapter = _make_adapter(_judge_json(("quality", 0.8, "r")))
        scored_run = await Judge(adapter).score_suite_run(run, suite)
        for r in scored_run.results:
            assert r.criterion_scores != []

    @pytest.mark.asyncio
    async def test_result_order_preserved(self) -> None:
        criteria = [_criterion("quality")]
        tests = [self._make_test_case(f"t{i}", criteria) for i in range(4)]
        suite = _make_suite(tests)
        run = _make_suite_run([_result(f"t{i}") for i in range(4)])
        adapter = _make_adapter(_judge_json(("quality", 0.8, "r")))
        scored_run = await Judge(adapter).score_suite_run(run, suite)
        ids = [r.test_id for r in scored_run.results]
        assert ids == ["t0", "t1", "t2", "t3"]

    @pytest.mark.asyncio
    async def test_pre_errored_results_pass_through(self) -> None:
        suite = _make_suite()
        errored = _result("t1", error="adapter failed")
        run = _make_suite_run([errored])
        adapter = _make_adapter()
        scored_run = await Judge(adapter).score_suite_run(run, suite)
        assert scored_run.results[0].error == "adapter failed"
        adapter.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_test_id_raises_judge_error(self) -> None:
        suite = _make_suite()  # only has "t1"
        run = _make_suite_run([_result("nonexistent-id")])
        adapter = _make_adapter()
        with pytest.raises(JudgeError, match="not found in suite"):
            await Judge(adapter).score_suite_run(run, suite)

    @pytest.mark.asyncio
    async def test_suite_run_metadata_unchanged(self) -> None:
        suite = _make_suite()
        run = _make_suite_run([_result("t1")])
        adapter = _make_adapter(_judge_json(("quality", 0.8, "r")))
        scored_run = await Judge(adapter).score_suite_run(run, suite)
        assert scored_run.run_id == run.run_id
        assert scored_run.suite_name == run.suite_name
        assert scored_run.model == run.model
