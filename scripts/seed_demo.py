"""Seed the local database with realistic demo data for dashboard development.

Usage:
    poetry run python scripts/seed_demo.py
    poetry run python scripts/seed_demo.py --db /path/to/other.db
"""

import argparse
import asyncio
import os
from datetime import UTC, datetime, timedelta

from llmeval.schema.results import CriterionScore, SuiteRun, TestResult
from llmeval.storage.sqlite import SQLiteStorage


def _score(name: str, score: float, reasoning: str) -> CriterionScore:
    return CriterionScore(name=name, score=score, reasoning=reasoning)


def _result(
    test_id: str,
    prompt: str,
    output: str,
    model: str,
    scores: list[CriterionScore],
    threshold: float = 0.75,
    error: str | None = None,
) -> TestResult:
    weighted = (
        sum(s.score for s in scores) / len(scores) if scores and not error else 0.0
    )
    return TestResult(
        test_id=test_id,
        prompt=prompt,
        model=model,
        raw_output=output if not error else "",
        criterion_scores=scores if not error else [],
        weighted_score=weighted,
        passed=weighted >= threshold and error is None,
        error=error,
    )


def make_runs() -> list[SuiteRun]:
    model_v1 = "claude-sonnet-4-20250514"
    model_v2 = "claude-opus-4-20250514"
    judge = "claude-sonnet-4-20250514"

    t0 = datetime(2024, 10, 1, 9, 0, 0, tzinfo=UTC)

    # -----------------------------------------------------------------------
    # Run 1 — baseline (v1, mixed results)
    # -----------------------------------------------------------------------
    run1_results = [
        _result(
            "tone-empathy-001",
            "I've been waiting 3 weeks for my order and nobody will help me!",
            (
                "I'm really sorry to hear that — three weeks is far too long. "
                "Let me pull up your order right now and escalate this to our "
                "fulfillment team."
            ),
            model_v1,
            [
                _score(
                    "empathy",
                    0.95,
                    "Directly acknowledges frustration and validates the wait time.",
                ),
                _score(
                    "actionability",
                    0.90,
                    "Offers a concrete next step with escalation path.",
                ),
                _score(
                    "tone",
                    0.88,
                    "Professional and warm without being defensive.",
                ),
            ],
        ),
        _result(
            "tone-boundaries-002",
            "Can you write me a poem about my cat?",
            (
                "That sounds like a fun request! While poem-writing is outside "
                "what I can help with here, I'm happy to assist with any order "
                "questions or account issues."
            ),
            model_v1,
            [
                _score(
                    "scope_adherence",
                    0.80,
                    "Politely declines and redirects to relevant scope.",
                ),
                _score("tone", 0.92, "Friendly and not dismissive."),
            ],
            threshold=0.70,
        ),
        _result(
            "accuracy-refund-003",
            "My item arrived broken. What are my options?",
            (
                "We stand behind every order. For a damaged item you have three "
                "options: full refund, replacement, or store credit. Reply with "
                "your preference and I'll get it sorted today."
            ),
            model_v1,
            [
                _score(
                    "accuracy",
                    0.93,
                    "Correctly describes all three resolution paths.",
                ),
                _score(
                    "empathy",
                    0.85,
                    "Acknowledges inconvenience without over-apologizing.",
                ),
                _score(
                    "actionability",
                    0.96,
                    "Clear call to action with timeline commitment.",
                ),
            ],
            threshold=0.80,
        ),
        _result(
            "accuracy-policy-004",
            "Do you offer price matching?",
            (
                "Yes! We match prices from any authorised retailer within 14 days "
                "of purchase. Just send us a link to the lower price and we'll "
                "apply the difference as a refund."
            ),
            model_v1,
            [
                _score(
                    "accuracy",
                    0.40,
                    "Policy is actually 7 days, not 14 — factual error.",
                ),
                _score("tone", 0.90, "Helpful and clear phrasing."),
            ],
            threshold=0.70,
        ),
        _result(
            "tone-escalation-005",
            "This is completely unacceptable. I want to speak to a manager NOW.",
            (
                "Absolutely understood. I'll connect you with a senior support "
                "manager right away. While I transfer you, could I get your order "
                "number so they have full context?"
            ),
            model_v1,
            [
                _score(
                    "empathy",
                    0.88,
                    "Validates urgency without being confrontational.",
                ),
                _score(
                    "actionability",
                    0.94,
                    "Immediately agrees to escalate and bridges to manager.",
                ),
                _score("tone", 0.82, "Calm and professional."),
            ],
        ),
        _result(
            "accuracy-shipping-006",
            "When will my package arrive?",
            "",
            model_v1,
            [],
            error="API timeout after 30s",
        ),
    ]

    run1 = SuiteRun(
        suite_name="Customer Support Bot — Tone Tests",
        suite_version="1.0.0",
        model=model_v1,
        judge_model=judge,
        started_at=t0,
        completed_at=t0 + timedelta(seconds=47),
        results=run1_results,
    )

    # -----------------------------------------------------------------------
    # Run 2 — candidate (v2, improved accuracy, one regression)
    # -----------------------------------------------------------------------
    t1 = t0 + timedelta(hours=3)

    run2_results = [
        _result(
            "tone-empathy-001",
            "I've been waiting 3 weeks for my order and nobody will help me!",
            (
                "Three weeks without a resolution is completely unacceptable and "
                "I sincerely apologise. I'm escalating this to our fulfillment "
                "manager right now and will send you a tracking update within the hour."
            ),
            model_v2,
            [
                _score(
                    "empathy",
                    0.98,
                    "Strong acknowledgement with personal apology.",
                ),
                _score(
                    "actionability",
                    0.96,
                    "Specific timeline commitment with escalation.",
                ),
                _score("tone", 0.94, "Warm and accountable."),
            ],
        ),
        _result(
            "tone-boundaries-002",
            "Can you write me a poem about my cat?",
            (
                "What a sweet idea! I'm focused on support questions today, but "
                "I'd love to help with anything related to your orders or account."
            ),
            model_v2,
            [
                _score(
                    "scope_adherence",
                    0.55,
                    "Complimenting the request before declining may encourage "
                    "off-topic conversation.",
                ),
                _score(
                    "tone",
                    0.88,
                    "Friendly, but slightly too encouraging of the out-of-scope "
                    "request.",
                ),
            ],
            threshold=0.70,
        ),
        _result(
            "accuracy-refund-003",
            "My item arrived broken. What are my options?",
            (
                "I'm sorry to hear your item arrived damaged. You can choose from "
                "a full refund, a free replacement, or store credit — whichever "
                "works best for you. Just let me know and I'll process it immediately."
            ),
            model_v2,
            [
                _score(
                    "accuracy", 0.95, "All three resolution paths correctly stated."
                ),
                _score("empathy", 0.92, "Leads with empathy before options."),
                _score(
                    "actionability",
                    0.97,
                    "Immediate action with customer choice.",
                ),
            ],
            threshold=0.80,
        ),
        _result(
            "accuracy-policy-004",
            "Do you offer price matching?",
            (
                "Yes, we offer price matching within 7 days of purchase for any "
                "lower price from an authorised retailer. Share the link and "
                "we'll refund the difference."
            ),
            model_v2,
            [
                _score("accuracy", 0.97, "Correct 7-day window stated clearly."),
                _score("tone", 0.93, "Concise and confident."),
            ],
            threshold=0.70,
        ),
        _result(
            "tone-escalation-005",
            "This is completely unacceptable. I want to speak to a manager NOW.",
            (
                "Of course. Transferring you to a senior manager immediately — "
                "your satisfaction is our top priority."
            ),
            model_v2,
            [
                _score(
                    "empathy",
                    0.65,
                    "Too brief — does not sufficiently acknowledge the customer's "
                    "frustration.",
                ),
                _score("actionability", 0.90, "Agrees to escalate promptly."),
                _score("tone", 0.70, "Professional but slightly cold."),
            ],
        ),
        _result(
            "accuracy-shipping-006",
            "When will my package arrive?",
            (
                "Your estimated delivery date is shown in your confirmation email. "
                "If you share your order number I can look up the live tracking "
                "status for you right now."
            ),
            model_v2,
            [
                _score(
                    "accuracy",
                    0.88,
                    "Correct guidance on tracking, offers live lookup.",
                ),
                _score(
                    "actionability",
                    0.91,
                    "Prompts for order number to provide specific help.",
                ),
            ],
            threshold=0.75,
        ),
    ]

    run2 = SuiteRun(
        suite_name="Customer Support Bot — Tone Tests",
        suite_version="1.0.0",
        model=model_v2,
        judge_model=judge,
        started_at=t1,
        completed_at=t1 + timedelta(seconds=39),
        results=run2_results,
    )

    return [run1, run2]


async def seed(db_path: str) -> None:
    runs = make_runs()
    async with SQLiteStorage(db_path) as storage:
        for run in runs:
            await storage.save_run(run)
    print(f"Seeded {len(runs)} runs into {db_path}")
    for r in runs:
        print(f"  {r.run_id[:8]}…  {r.model}  {r.passed_tests}/{r.total_tests} passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.environ.get("LLMEVAL_DB_PATH", "llmeval.db"))
    args = parser.parse_args()
    asyncio.run(seed(args.db))
