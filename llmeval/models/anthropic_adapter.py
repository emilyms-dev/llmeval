"""Anthropic model adapter.

Wraps the ``anthropic`` async client so the runner can call Claude models
through the same :class:`~llmeval.models.base.ModelAdapter` interface as
every other provider.

Anthropic's Messages API differs from OpenAI's in two notable ways:

- ``max_tokens`` is **required** — the adapter exposes it as a constructor
  parameter with a sensible default.
- The ``system`` prompt is a top-level parameter, not a ``role: system``
  message in the messages list.

API key resolution order:

1. ``api_key`` constructor argument.
2. ``ANTHROPIC_API_KEY`` environment variable (loaded from ``.env`` via the
   global ``load_dotenv()`` call in :mod:`llmeval.cli`).

Raises :class:`~llmeval.exceptions.ConfigurationError` at construction time
if no key is available.
"""

from __future__ import annotations

import os

from anthropic import APIError as AnthropicAPIError
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from llmeval.exceptions import ConfigurationError, ModelAdapterError
from llmeval.models.base import ModelAdapter, ModelResponse

# Anthropic requires an explicit max_tokens; 4096 is a reasonable default
# that fits most evaluation responses without truncation.
_DEFAULT_MAX_TOKENS = 4096


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models.

    Args:
        model: Anthropic model identifier
            (e.g. ``"claude-sonnet-4-20250514"``).
        api_key: Anthropic API key. When ``None`` the value of the
            ``ANTHROPIC_API_KEY`` environment variable is used.
        max_tokens: Maximum tokens to generate per response. Defaults to
            ``4096``.
        temperature: Sampling temperature passed to the API. ``None`` omits
            the parameter so the model's server-side default applies.

    Raises:
        ConfigurationError: If no API key can be found.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ConfigurationError(
                "No Anthropic API key found. Set the ANTHROPIC_API_KEY "
                "environment variable or pass api_key= to AnthropicAdapter()."
            )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = AsyncAnthropic(api_key=resolved_key)

    @property
    def model_id(self) -> str:
        """Anthropic model identifier passed at construction."""
        return self._model

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Call the Anthropic Messages API and return a structured response.

        Concatenates all ``text``-type content blocks in the response. In
        practice Claude returns a single block, but the API allows multiple.

        Args:
            prompt: The user-turn message.
            system_prompt: Optional system prompt passed as the top-level
                ``system`` parameter (not as a message role).

        Returns:
            A :class:`~llmeval.models.base.ModelResponse` with the assistant's
            text reply and token-usage counters from the API response.

        Raises:
            ModelAdapterError: Wraps any :class:`anthropic.APIError`.
        """
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        try:
            if self._temperature is None and system_prompt is None:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=messages,
                )
            elif self._temperature is None:
                assert system_prompt is not None
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=messages,
                    system=system_prompt,
                )
            elif system_prompt is None:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=messages,
                    temperature=self._temperature,
                )
            else:
                assert system_prompt is not None
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=messages,
                    system=system_prompt,
                    temperature=self._temperature,
                )
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            usage: dict[str, int] | None = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                }
            return ModelResponse(text=text, usage=usage)
        except AnthropicAPIError as exc:
            raise ModelAdapterError(
                f"Anthropic API call failed for model {self._model!r}: {exc}"
            ) from exc
