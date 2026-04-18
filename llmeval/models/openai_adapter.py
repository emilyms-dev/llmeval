"""OpenAI model adapter.

Wraps the ``openai`` async client so the runner can call OpenAI chat models
(GPT-4o, o1, o3, etc.) through the same :class:`~llmeval.models.base.ModelAdapter`
interface as every other provider.

API key resolution order:

1. ``api_key`` constructor argument (useful in tests / programmatic use).
2. ``OPENAI_API_KEY`` environment variable (loaded from ``.env`` via
   ``python-dotenv``).

Raises :class:`~llmeval.exceptions.ConfigurationError` at construction time
if no key is available, so misconfiguration is caught before any API call.
"""

import os

from dotenv import load_dotenv
from openai import APIError as OpenAIAPIError
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from llmeval.exceptions import ConfigurationError, ModelAdapterError
from llmeval.models.base import ModelAdapter

load_dotenv()


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI chat-completion models.

    Args:
        model: OpenAI model identifier (e.g. ``"gpt-4o"``, ``"o3-mini"``).
        api_key: OpenAI API key. When ``None`` the value of the
            ``OPENAI_API_KEY`` environment variable is used.

    Raises:
        ConfigurationError: If no API key can be found.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ConfigurationError(
                "No OpenAI API key found. Set the OPENAI_API_KEY environment "
                "variable or pass api_key= to OpenAIAdapter()."
            )
        self._model = model
        self._client = AsyncOpenAI(api_key=resolved_key)

    @property
    def model_id(self) -> str:
        """OpenAI model identifier passed at construction."""
        return self._model

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Call the OpenAI chat completions API and return the text response.

        Args:
            prompt: The user-turn message.
            system_prompt: Optional system message prepended to the
                conversation.

        Returns:
            The assistant's text reply, or an empty string if the API
            returns ``None`` content.

        Raises:
            ModelAdapterError: Wraps any :class:`openai.APIError`.
        """
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except OpenAIAPIError as exc:
            raise ModelAdapterError(
                f"OpenAI API call failed for model {self._model!r}: {exc}"
            ) from exc
