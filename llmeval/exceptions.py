"""Custom exception hierarchy for llmeval.

All exceptions raised by the library descend from LLMEvalError so that
callers can catch the entire family with a single except clause.
"""


class LLMEvalError(Exception):
    """Base exception for all llmeval errors."""


class ConfigurationError(LLMEvalError):
    """Raised when required configuration (env vars, file paths) is missing or invalid."""  # noqa: E501


class SchemaValidationError(LLMEvalError):
    """Raised when a test-suite YAML/JSON file fails schema validation."""


class ModelAdapterError(LLMEvalError):
    """Raised when a model adapter encounters an unrecoverable error."""


class JudgeError(LLMEvalError):
    """Raised when the LLM-as-judge returns an unparseable or invalid response."""


class StorageError(LLMEvalError):
    """Raised when a storage backend operation fails."""


class RunnerError(LLMEvalError):
    """Raised when the test runner encounters an error executing a suite."""
