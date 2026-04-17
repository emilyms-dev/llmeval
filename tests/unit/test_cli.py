"""Unit tests for llmeval.cli.

Uses Typer's CliRunner for all command invocations. External I/O (model API
calls, storage, suite loading) is mocked at the component level rather than
patching asyncio.run, which Typer uses internally and must remain unpatched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from llmeval.cli import app
from llmeval.schema.results import CriterionScore, SuiteRun, TestResult

cli_runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(passed: int = 2, failed: int = 0, errored: int = 0) -> SuiteRun:
    started = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    results = (
        [
            TestResult(
                test_id=f"t{i}",
                prompt="p",
                model="claude-sonnet-4-20250514",
                raw_output="r",
                criterion_scores=[
                    CriterionScore(name="q", score=0.9, reasoning="ok")
                ],
                weighted_score=0.9,
                passed=True,
            )
            for i in range(passed)
        ]
        + [
            TestResult(
                test_id=f"f{i}",
                prompt="p",
                model="claude-sonnet-4-20250514",
                raw_output="r",
                criterion_scores=[
                    CriterionScore(name="q", score=0.3, reasoning="poor")
                ],
                weighted_score=0.3,
                passed=False,
            )
            for i in range(failed)
        ]
        + [
            TestResult(
                test_id=f"e{i}",
                prompt="p",
                model="claude-sonnet-4-20250514",
                raw_output="",
                error="timeout",
            )
            for i in range(errored)
        ]
    )
    return SuiteRun(
        suite_name="Test Suite",
        suite_version="1.0.0",
        model="claude-sonnet-4-20250514",
        judge_model="claude-sonnet-4-20250514",
        started_at=started,
        completed_at=started + timedelta(seconds=5),
        results=results,
    )


def _mock_suite(model: str = "claude-sonnet-4-20250514") -> MagicMock:
    """Return a mock TestSuite with sensible defaults."""
    s = MagicMock()
    s.suite.model = model
    s.suite.judge_model = "claude-sonnet-4-20250514"
    s.tests = [MagicMock()]
    return s


# Shared patch targets
_LOAD_SUITE = "llmeval.schema.test_suite.load_suite"
_CREATE_ADAPTER = "llmeval.models.create_adapter"
_RUNNER_RUN = "llmeval.runner.Runner.run"
_JUDGE_SCORE = "llmeval.judge.Judge.score_suite_run"
_DB = "llmeval.storage.sqlite.SQLiteStorage"
_DB_INIT = f"{_DB}.initialize"
_DB_CLOSE = f"{_DB}.close"
_DB_SAVE = f"{_DB}.save_run"
_DB_GET = f"{_DB}.get_run"
_DB_LIST = f"{_DB}.list_runs"


# ===========================================================================
# version
# ===========================================================================


class TestVersionCommand:
    def test_exits_zero(self) -> None:
        result = cli_runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_prints_version(self) -> None:
        from llmeval import __version__

        result = cli_runner.invoke(app, ["version"])
        assert __version__ in result.output


# ===========================================================================
# run
# ===========================================================================


class TestRunCommand:
    def test_missing_suite_option_exits_nonzero(self) -> None:
        result = cli_runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_invalid_suite_path_exits_2(self) -> None:
        result = cli_runner.invoke(app, ["run", "--suite", "nonexistent.yaml"])
        assert result.exit_code == 2

    def test_unsupported_model_prefix_exits_2(self) -> None:
        with patch(_LOAD_SUITE, return_value=_mock_suite()):
            result = cli_runner.invoke(
                app, ["run", "--suite", "fake.yaml", "--model", "unknown-xyz"]
            )
        assert result.exit_code == 2

    def test_all_pass_exits_0(self) -> None:
        suite_run = _make_run(passed=2, failed=0, errored=0)
        with (
            patch(_LOAD_SUITE, return_value=_mock_suite()),
            patch(_CREATE_ADAPTER),
            patch(_RUNNER_RUN, new_callable=AsyncMock, return_value=suite_run),
            patch(_JUDGE_SCORE, new_callable=AsyncMock, return_value=suite_run),
        ):
            result = cli_runner.invoke(
                app, ["run", "--suite", "fake.yaml", "--no-save"]
            )
        assert result.exit_code == 0

    def test_any_failure_exits_1(self) -> None:
        suite_run = _make_run(passed=1, failed=1, errored=0)
        with (
            patch(_LOAD_SUITE, return_value=_mock_suite()),
            patch(_CREATE_ADAPTER),
            patch(_RUNNER_RUN, new_callable=AsyncMock, return_value=suite_run),
            patch(_JUDGE_SCORE, new_callable=AsyncMock, return_value=suite_run),
        ):
            result = cli_runner.invoke(
                app, ["run", "--suite", "fake.yaml", "--no-save"]
            )
        assert result.exit_code == 1

    def test_any_error_exits_1(self) -> None:
        suite_run = _make_run(passed=1, failed=0, errored=1)
        with (
            patch(_LOAD_SUITE, return_value=_mock_suite()),
            patch(_CREATE_ADAPTER),
            patch(_RUNNER_RUN, new_callable=AsyncMock, return_value=suite_run),
            patch(_JUDGE_SCORE, new_callable=AsyncMock, return_value=suite_run),
        ):
            result = cli_runner.invoke(
                app, ["run", "--suite", "fake.yaml", "--no-save"]
            )
        assert result.exit_code == 1

    def test_runner_error_exits_2(self) -> None:
        from llmeval.exceptions import RunnerError

        with (
            patch(_LOAD_SUITE, return_value=_mock_suite()),
            patch(_CREATE_ADAPTER),
            patch(
                _RUNNER_RUN,
                new_callable=AsyncMock,
                side_effect=RunnerError("no match"),
            ),
        ):
            result = cli_runner.invoke(
                app, ["run", "--suite", "fake.yaml", "--no-save"]
            )
        assert result.exit_code == 2

    def test_output_contains_pass_rate(self) -> None:
        suite_run = _make_run(passed=2, failed=0)
        with (
            patch(_LOAD_SUITE, return_value=_mock_suite()),
            patch(_CREATE_ADAPTER),
            patch(_RUNNER_RUN, new_callable=AsyncMock, return_value=suite_run),
            patch(_JUDGE_SCORE, new_callable=AsyncMock, return_value=suite_run),
        ):
            result = cli_runner.invoke(
                app, ["run", "--suite", "fake.yaml", "--no-save"]
            )
        assert "100.0%" in result.output

    def test_output_contains_suite_name(self) -> None:
        suite_run = _make_run(passed=1)
        with (
            patch(_LOAD_SUITE, return_value=_mock_suite()),
            patch(_CREATE_ADAPTER),
            patch(_RUNNER_RUN, new_callable=AsyncMock, return_value=suite_run),
            patch(_JUDGE_SCORE, new_callable=AsyncMock, return_value=suite_run),
        ):
            result = cli_runner.invoke(
                app, ["run", "--suite", "fake.yaml", "--no-save"]
            )
        assert "Test Suite" in result.output

    def test_save_calls_storage(self) -> None:
        suite_run = _make_run(passed=1)
        mock_save = AsyncMock()
        with (
            patch(_LOAD_SUITE, return_value=_mock_suite()),
            patch(_CREATE_ADAPTER),
            patch(_RUNNER_RUN, new_callable=AsyncMock, return_value=suite_run),
            patch(_JUDGE_SCORE, new_callable=AsyncMock, return_value=suite_run),
            patch(_DB_INIT, new_callable=AsyncMock),
            patch(_DB_SAVE, mock_save),
            patch(_DB_CLOSE, new_callable=AsyncMock),
        ):
            cli_runner.invoke(app, ["run", "--suite", "fake.yaml"])
        mock_save.assert_called_once()


# ===========================================================================
# list
# ===========================================================================


class TestListCommand:
    def test_no_runs_prints_message(self) -> None:
        with (
            patch(_DB_INIT, new_callable=AsyncMock),
            patch(_DB_LIST, new_callable=AsyncMock, return_value=[]),
            patch(_DB_CLOSE, new_callable=AsyncMock),
        ):
            result = cli_runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_runs_shown_in_table(self) -> None:
        runs = [_make_run(passed=2)]
        with (
            patch(_DB_INIT, new_callable=AsyncMock),
            patch(_DB_LIST, new_callable=AsyncMock, return_value=runs),
            patch(_DB_CLOSE, new_callable=AsyncMock),
        ):
            result = cli_runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Test Suite" in result.output

    def test_storage_error_exits_2(self) -> None:
        from llmeval.exceptions import StorageError

        with (
            patch(_DB_INIT, new_callable=AsyncMock),
            patch(
                "llmeval.storage.sqlite.SQLiteStorage.list_runs",
                new_callable=AsyncMock,
                side_effect=StorageError("db locked"),
            ),
            patch(_DB_CLOSE, new_callable=AsyncMock),
        ):
            result = cli_runner.invoke(app, ["list"])
        assert result.exit_code == 2


# ===========================================================================
# show
# ===========================================================================


class TestShowCommand:
    def test_shows_run_output(self) -> None:
        suite_run = _make_run(passed=1)
        with (
            patch(_DB_INIT, new_callable=AsyncMock),
            patch(_DB_GET, new_callable=AsyncMock, return_value=suite_run),
            patch(_DB_CLOSE, new_callable=AsyncMock),
        ):
            result = cli_runner.invoke(app, ["show", suite_run.run_id])
        assert result.exit_code == 0
        assert "Test Suite" in result.output

    def test_not_found_exits_2(self) -> None:
        from llmeval.exceptions import StorageError

        with (
            patch(_DB_INIT, new_callable=AsyncMock),
            patch(
                "llmeval.storage.sqlite.SQLiteStorage.get_run",
                new_callable=AsyncMock,
                side_effect=StorageError("not found"),
            ),
            patch(_DB_CLOSE, new_callable=AsyncMock),
        ):
            result = cli_runner.invoke(app, ["show", "bad-id"])
        assert result.exit_code == 2


# ===========================================================================
# diff
# ===========================================================================


class TestDiffCommand:
    def test_diff_output_contains_suite_name(self) -> None:
        run_a = _make_run(passed=1, failed=1)
        run_b = _make_run(passed=2, failed=0)
        with (
            patch(_DB_INIT, new_callable=AsyncMock),
            patch(
                "llmeval.storage.sqlite.SQLiteStorage.get_run",
                new_callable=AsyncMock,
                side_effect=[run_a, run_b],
            ),
            patch(_DB_CLOSE, new_callable=AsyncMock),
        ):
            result = cli_runner.invoke(app, ["diff", "id-a", "id-b"])
        assert result.exit_code == 0
        assert "Test Suite" in result.output

    def test_missing_run_exits_2(self) -> None:
        from llmeval.exceptions import StorageError

        with (
            patch(_DB_INIT, new_callable=AsyncMock),
            patch(
                "llmeval.storage.sqlite.SQLiteStorage.get_run",
                new_callable=AsyncMock,
                side_effect=StorageError("not found"),
            ),
            patch(_DB_CLOSE, new_callable=AsyncMock),
        ):
            result = cli_runner.invoke(app, ["diff", "id-a", "id-b"])
        assert result.exit_code == 2
