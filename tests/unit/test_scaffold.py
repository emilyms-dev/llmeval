"""Smoke tests for the project scaffold.

Verifies that the package structure is correct and all top-level modules
are importable without errors.
"""

import importlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Package import tests
# ---------------------------------------------------------------------------

EXPECTED_MODULES = [
    "llmeval",
    "llmeval.exceptions",
    "llmeval.cli",
    "llmeval.runner",
    "llmeval.judge",
    "llmeval.models",
    "llmeval.models.base",
    "llmeval.models.openai_adapter",
    "llmeval.models.anthropic_adapter",
    "llmeval.schema",
    "llmeval.schema.test_suite",
    "llmeval.schema.results",
    "llmeval.storage",
    "llmeval.storage.base",
    "llmeval.storage.sqlite",
    "llmeval.report",
    "llmeval.report.cli_reporter",
]


@pytest.mark.parametrize("module_path", EXPECTED_MODULES)
def test_module_is_importable(module_path: str) -> None:
    """Each expected module must be importable without raising an exception."""
    module = importlib.import_module(module_path)
    assert module is not None


def test_package_version_is_set() -> None:
    """The package must expose a non-empty __version__ string."""
    import llmeval

    assert isinstance(llmeval.__version__, str)
    assert llmeval.__version__ != ""


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


def test_all_exceptions_inherit_from_base() -> None:
    """Every custom exception must be a subclass of LLMEvalError."""
    from llmeval.exceptions import (
        ConfigurationError,
        JudgeError,
        LLMEvalError,
        ModelAdapterError,
        RunnerError,
        SchemaValidationError,
        StorageError,
    )

    subclasses = [
        ConfigurationError,
        SchemaValidationError,
        ModelAdapterError,
        JudgeError,
        StorageError,
        RunnerError,
    ]
    for exc_class in subclasses:
        assert issubclass(
            exc_class, LLMEvalError
        ), f"{exc_class.__name__} must inherit from LLMEvalError"


def test_exceptions_are_raiseable() -> None:
    """Exceptions must be instantiatable and raiseable with a message."""
    from llmeval.exceptions import ConfigurationError, LLMEvalError

    with pytest.raises(ConfigurationError, match="missing key"):
        raise ConfigurationError("missing key")

    with pytest.raises(LLMEvalError):
        raise ConfigurationError("also caught by base class")


# ---------------------------------------------------------------------------
# File structure tests
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent


def test_env_example_exists() -> None:
    """.env.example must be present so contributors know what vars to set."""
    assert (REPO_ROOT / ".env.example").is_file()


def test_gitignore_excludes_dotenv() -> None:
    """.gitignore must exclude .env to prevent accidental key commits."""
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert ".env" in gitignore


def test_example_suite_yaml_exists() -> None:
    """The fixture YAML used in docs and integration tests must exist."""
    assert (REPO_ROOT / "tests" / "fixtures" / "example_suite.yaml").is_file()


def test_example_suite_yaml_is_valid_yaml() -> None:
    """The fixture YAML must parse without errors."""
    import yaml

    path = REPO_ROOT / "tests" / "fixtures" / "example_suite.yaml"
    data = yaml.safe_load(path.read_text())
    assert "suite" in data
    assert "tests" in data
    assert len(data["tests"]) > 0


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_cli_version_command() -> None:
    """``llmeval version`` must print the package version and exit 0."""
    from typer.testing import CliRunner

    from llmeval import __version__
    from llmeval.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_run_command_exits_nonzero_without_implementation() -> None:
    """``llmeval run`` must exit non-zero until the runner is implemented."""
    from typer.testing import CliRunner

    from llmeval.cli import app

    result = CliRunner().invoke(app, ["run", "--suite", "fake.yaml"])
    assert result.exit_code != 0
