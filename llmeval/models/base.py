"""Abstract model adapter base class.

All provider-specific adapters inherit from :class:`ModelAdapter`. The
contract is intentionally minimal: receive a prompt (and optional system
prompt), return a :class:`ModelResponse` with the model's text and token usage.
This keeps the runner and judge decoupled from any provider SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    """Structured return value from a model adapter call.

    Args:
        text: The model's text reply. Never ``None`` — empty string if the
            provider returns no content.
        usage: Optional token-usage counters keyed by provider convention
            (e.g. ``{"prompt_tokens": 42, "completion_tokens": 18}``).
            ``None`` when the provider does not return usage data.
    """

    text: str
    usage: dict[str, int] | None = field(default=None)


class ModelAdapter(ABC):
    """Provider-agnostic interface for sending prompts to an LLM.

    Concrete subclasses must implement :meth:`complete` and :attr:`model_id`.
    All network I/O must be async so the runner can fan out test cases
    concurrently without blocking.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The model identifier string used when calling the provider API.

        Examples: ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``.
        """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Send a prompt to the model and return a structured response.

        Args:
            prompt: The user-turn message to send.
            system_prompt: Optional system / instruction prefix. When
                ``None`` the adapter's provider default (if any) is used.

        Returns:
            A :class:`ModelResponse` with the model's text reply and optional
            token-usage counters. ``text`` is never ``None`` — returns an empty
            string if the provider returns no content.

        Raises:
            ModelAdapterError: If the provider API call fails for any reason
                (auth error, rate limit, timeout, malformed response, etc.).
        """
