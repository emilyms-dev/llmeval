"""Model adapter package.

Public surface::

    from llmeval.models import ModelAdapter, OpenAIAdapter, AnthropicAdapter
    from llmeval.models import create_adapter   # provider-routing factory

The :func:`create_adapter` factory infers the provider from the model name
so callers don't need to import the concrete adapter classes directly.
"""

from typing import Any

from llmeval.models.anthropic_adapter import AnthropicAdapter
from llmeval.models.base import ModelAdapter, ModelResponse
from llmeval.models.openai_adapter import OpenAIAdapter

# Model name prefixes that identify each provider.  Keep in sync with the
# providers actually supported by concrete adapter classes above.
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "text-davinci")
_ANTHROPIC_PREFIXES = ("claude-",)


def create_adapter(
    model: str,
    api_key: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> ModelAdapter:
    """Instantiate the appropriate :class:`ModelAdapter` for *model*.

    Provider is inferred from the model name prefix:

    - ``gpt-*``, ``o1*``, ``o3*``  →  :class:`OpenAIAdapter`
    - ``claude-*``                  →  :class:`AnthropicAdapter`

    Pass ``api_key`` to override the default environment-variable lookup.
    Any additional keyword arguments are forwarded to the adapter constructor
    (e.g. ``max_tokens`` for :class:`AnthropicAdapter`).

    Args:
        model: Model identifier string (e.g. ``"gpt-4o"`` or
            ``"claude-sonnet-4-20250514"``).
        api_key: Optional API key override.
        **kwargs: Extra keyword arguments forwarded to the adapter.

    Returns:
        A concrete :class:`ModelAdapter` instance ready to call.

    Raises:
        ConfigurationError: If the provider cannot be inferred from *model*,
            or if the resolved adapter cannot find an API key.
    """
    from llmeval.exceptions import ConfigurationError

    if model.startswith(_OPENAI_PREFIXES):
        return OpenAIAdapter(model=model, api_key=api_key, **kwargs)
    if model.startswith(_ANTHROPIC_PREFIXES):
        return AnthropicAdapter(model=model, api_key=api_key, **kwargs)
    raise ConfigurationError(
        f"Cannot infer provider for model {model!r}. "
        "Use OpenAIAdapter() or AnthropicAdapter() directly, or open an "
        "issue to request support for this model prefix."
    )


__all__ = [
    "ModelAdapter",
    "ModelResponse",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "create_adapter",
]
