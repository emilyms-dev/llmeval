"""Abstract model adapter base class.

All provider-specific adapters inherit from :class:`ModelAdapter`. The
contract is intentionally minimal: receive a prompt (and optional system
prompt), return the model's text response as a plain string. This keeps the
runner and judge decoupled from any provider SDK.
"""

from abc import ABC, abstractmethod


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
    ) -> str:
        """Send a prompt to the model and return the text response.

        Args:
            prompt: The user-turn message to send.
            system_prompt: Optional system / instruction prefix. When
                ``None`` the adapter's provider default (if any) is used.

        Returns:
            The model's text response. Never ``None`` — returns an empty
            string if the provider returns no content.

        Raises:
            ModelAdapterError: If the provider API call fails for any reason
                (auth error, rate limit, timeout, malformed response, etc.).
        """
