"""LLM-as-judge scoring logic.

The :class:`Judge` calls a :class:`~llmeval.models.base.ModelAdapter` (the
"judge model") to evaluate raw model responses against rubric criteria, then
updates each :class:`~llmeval.schema.results.TestResult` with per-criterion
scores, a weighted aggregate, and a pass/fail determination.

Flow::

    Runner produces TestResult (raw_output set, scores empty)
        ↓
    Judge.score(result, rubric)
        ↓
    Build structured prompt → call judge model → parse JSON response
        ↓
    TestResult (criterion_scores, weighted_score, passed populated)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Final

from llmeval.exceptions import JudgeError
from llmeval.models.base import ModelAdapter
from llmeval.schema.results import CriterionScore, SuiteRun, TestResult
from llmeval.schema.test_suite import Rubric, TestSuite

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY: Final[int] = 5

_SYSTEM_PROMPT: Final[str] = (
    "You are an objective evaluator assessing an AI assistant's response "
    "quality. Score the response against each rubric criterion. "
    "Return ONLY valid JSON — no prose, no markdown fences, no extra text."
)

# Criteria block is injected at call time via _build_prompt().
_USER_PROMPT_TEMPLATE: Final[str] = """\
## Original Prompt
{prompt}

## Model Response
{response}

## Scoring Rubric
Score each criterion from 0.0 (completely fails) to 1.0 (fully satisfies).
Use the full range — reserve 1.0 for genuinely excellent responses and 0.0
for complete failures.

Criteria:
{criteria_block}

## Required Output Format
Return ONLY this JSON — no text before or after it:
{{
  "scores": [
    {{"name": "<name>", "score": <float 0.0-1.0>, "reasoning": "<one sentence>"}},
    ...one entry per criterion in the rubric...
  ]
}}
"""


class Judge:
    """Scores model outputs against rubric criteria using an LLM as evaluator.

    Scoring is done concurrently, bounded by a semaphore. Individual scoring
    failures are captured on the :class:`~llmeval.schema.results.TestResult`
    rather than aborting the whole suite.

    Args:
        adapter: Model adapter for the judge model (may differ from the model
            under test).
        concurrency: Maximum simultaneous judge API calls. Defaults to 5.

    Raises:
        JudgeError: If ``concurrency`` is less than 1.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        if concurrency < 1:
            raise JudgeError(f"concurrency must be >= 1, got {concurrency!r}")
        self._adapter = adapter
        self._concurrency = concurrency

    async def score(
        self,
        result: TestResult,
        rubric: Rubric,
    ) -> TestResult:
        """Score a single test result against its rubric.

        Results that already have ``error`` set (from the runner) are returned
        unchanged — there is no output to score.

        Args:
            result: Runner-produced result with ``raw_output`` populated.
            rubric: Rubric from the corresponding
                :class:`~llmeval.schema.test_suite.TestCase`.

        Returns:
            A new :class:`~llmeval.schema.results.TestResult` with
            ``criterion_scores``, ``weighted_score``, and ``passed``
            populated. On judge failure the result is returned with ``error``
            set and scores empty.
        """
        if result.error is not None:
            return result

        try:
            prompt = _build_prompt(result.prompt, result.raw_output, rubric)
            raw_response = await self._adapter.complete(
                prompt, system_prompt=_SYSTEM_PROMPT
            )
            criterion_scores = _parse_scores(raw_response, rubric)
            weighted = _compute_weighted_score(criterion_scores, rubric)
            passed = weighted >= rubric.passing_threshold
            return result.model_copy(
                update={
                    "criterion_scores": criterion_scores,
                    "weighted_score": round(weighted, 6),
                    "passed": passed,
                }
            )
        except JudgeError as exc:
            logger.warning(
                "Judge scoring failed for test %r: %s", result.test_id, exc
            )
            return result.model_copy(update={"error": str(exc)})
        except Exception as exc:
            wrapped = JudgeError(
                f"Unexpected judge error for test {result.test_id!r}: {exc}"
            )
            logger.warning(
                "Unexpected judge error for test %r: %s", result.test_id, exc
            )
            return result.model_copy(update={"error": str(wrapped)})

    async def score_suite_run(
        self,
        suite_run: SuiteRun,
        suite: TestSuite,
    ) -> SuiteRun:
        """Score all results in *suite_run* concurrently.

        Pre-errored results (``error`` already set by the runner) are passed
        through unchanged. Result order from the runner is preserved.

        Args:
            suite_run: Completed :class:`~llmeval.schema.results.SuiteRun`
                from the runner.
            suite: Original :class:`~llmeval.schema.test_suite.TestSuite`
                that produced the run (needed for rubric lookup).

        Returns:
            A new :class:`~llmeval.schema.results.SuiteRun` with all results
            scored.

        Raises:
            JudgeError: If a result's ``test_id`` cannot be found in *suite*.
                This indicates a mismatch between the run and the suite and
                should not occur in normal usage.
        """
        test_map = {t.id: t for t in suite.tests}
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _score_one(result: TestResult) -> TestResult:
            if result.test_id not in test_map:
                raise JudgeError(
                    f"Result test_id {result.test_id!r} not found in suite "
                    f"{suite.suite.name!r}. Ensure the SuiteRun and TestSuite "
                    "originate from the same evaluation run."
                )
            async with semaphore:
                return await self.score(result, test_map[result.test_id].rubric)

        scored: list[TestResult] = list(
            await asyncio.gather(*[_score_one(r) for r in suite_run.results])
        )

        logger.info(
            "Judge finished suite %r: %d scored, %d errored",
            suite_run.suite_name,
            sum(1 for r in scored if r.error is None),
            sum(1 for r in scored if r.error is not None),
        )
        return suite_run.model_copy(update={"results": scored})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_prompt(prompt: str, response: str, rubric: Rubric) -> str:
    """Render the judge user-turn prompt for a single test case.

    Args:
        prompt: The original user prompt sent to the model under test.
        response: The raw text response from the model under test.
        rubric: Rubric whose criteria the judge will evaluate.

    Returns:
        Formatted prompt string ready to pass to the judge adapter.
    """
    criteria_lines = "\n".join(
        f"- name: {c.name!r} | description: {c.description!r}"
        f" | weight: {c.weight:.2f}"
        for c in rubric.criteria
    )
    return _USER_PROMPT_TEMPLATE.format(
        prompt=prompt,
        response=response,
        criteria_block=criteria_lines,
    )


def _extract_json(text: str) -> str:
    """Extract the first JSON object from *text*, stripping markdown fences.

    The judge model may wrap its response in triple-backtick fences. This
    function strips them and returns the raw JSON substring.

    Args:
        text: Raw text response from the judge model.

    Returns:
        The first ``{...}`` substring found in *text*.

    Raises:
        JudgeError: If no JSON object can be found in *text*.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise JudgeError(
            f"No JSON object found in judge response: {text[:300]!r}"
        )
    return match.group()


def _parse_scores(raw: str, rubric: Rubric) -> list[CriterionScore]:
    """Parse and validate the judge's JSON response into criterion scores.

    Validates that:
    - The response contains parseable JSON with a ``"scores"`` list.
    - Every expected criterion is scored exactly once.
    - No unknown criteria are present.
    - Each score value is a number; out-of-range values are clamped to
      ``[0.0, 1.0]`` rather than rejected, since minor overshoot (e.g.
      ``1.001``) is a common LLM formatting artifact.

    Args:
        raw: Raw text response from the judge model.
        rubric: Rubric whose criteria the judge was asked to score.

    Returns:
        List of :class:`~llmeval.schema.results.CriterionScore` objects,
        one per criterion, in the order the judge returned them.

    Raises:
        JudgeError: On any structural or content validation failure.
    """
    json_text = _extract_json(raw)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise JudgeError(
            f"Judge returned invalid JSON: {exc}. "
            f"Raw response (truncated): {raw[:300]!r}"
        ) from exc

    if not isinstance(data, dict) or "scores" not in data:
        got = (
            list(data.keys()) if isinstance(data, dict) else type(data).__name__
        )
        raise JudgeError(
            f"Judge JSON missing top-level 'scores' key. Got: {got!r}"
        )

    raw_scores = data["scores"]
    if not isinstance(raw_scores, list):
        raise JudgeError(
            f"Judge 'scores' must be a list, got {type(raw_scores).__name__!r}"
        )

    criterion_names = {c.name for c in rubric.criteria}
    seen: set[str] = set()
    scores: list[CriterionScore] = []

    for item in raw_scores:
        if not isinstance(item, dict):
            raise JudgeError(
                f"Each score entry must be a JSON object, got {item!r}"
            )

        name = item.get("name")
        score_val = item.get("score")
        reasoning = item.get("reasoning")

        if not isinstance(name, str) or not name:
            raise JudgeError(
                f"Score entry missing valid 'name' field: {item!r}"
            )
        if not isinstance(score_val, int | float) or isinstance(score_val, bool):
            raise JudgeError(
                f"Criterion {name!r} 'score' must be a number, "
                f"got {type(score_val).__name__!r}: {item!r}"
            )
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise JudgeError(
                f"Criterion {name!r} missing non-empty 'reasoning': {item!r}"
            )
        if name not in criterion_names:
            raise JudgeError(
                f"Judge scored unknown criterion {name!r}. "
                f"Expected one of: {sorted(criterion_names)}"
            )
        if name in seen:
            raise JudgeError(
                f"Judge returned duplicate score for criterion {name!r}"
            )
        seen.add(name)

        clamped = max(0.0, min(1.0, float(score_val)))
        scores.append(
            CriterionScore(
                name=name,
                score=clamped,
                reasoning=reasoning.strip(),
            )
        )

    missing = criterion_names - seen
    if missing:
        raise JudgeError(
            f"Judge did not score all criteria. "
            f"Missing: {sorted(missing)}"
        )

    return scores


def _compute_weighted_score(
    scores: list[CriterionScore],
    rubric: Rubric,
) -> float:
    """Compute the weighted aggregate score from criterion scores.

    Args:
        scores: Judge-assigned scores, one per rubric criterion. Every
            ``score.name`` must be present in ``rubric.criteria``; this is
            guaranteed when *scores* comes from :func:`_parse_scores`.
        rubric: Rubric containing criterion weights.

    Returns:
        Weighted sum in ``[0.0, 1.0]``.
    """
    weight_map = {c.name: c.weight for c in rubric.criteria}
    return sum(weight_map[s.name] * s.score for s in scores)
