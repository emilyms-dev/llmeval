"""Unit tests for llmeval.models.

All API calls are mocked — no real network requests are made.

Coverage targets:
- base.py        — ModelAdapter is abstract and un-instantiable
- openai_adapter — construction, model_id, complete(), error handling
- anthropic_adapter — construction, model_id, complete(), error handling
- __init__.py    — create_adapter() routing and unknown-model error
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmeval.exceptions import ConfigurationError, ModelAdapterError
from llmeval.models import AnthropicAdapter, ModelAdapter, OpenAIAdapter, create_adapter
from llmeval.models.base import ModelAdapter as ModelAdapterBase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_OPENAI_KEY = "sk-test-openai-key"
FAKE_ANTHROPIC_KEY = "sk-ant-test-anthropic-key"


def _openai_response(content: str | None) -> MagicMock:
    """Build a minimal mock of openai.ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


def _anthropic_response(*texts: str) -> MagicMock:
    """Build a minimal mock of anthropic.Message response with text blocks."""
    blocks = []
    for text in texts:
        block = MagicMock()
        block.type = "text"
        block.text = text
        blocks.append(block)
    resp = MagicMock()
    resp.content = blocks
    resp.usage = None
    return resp


# ===========================================================================
# ModelAdapter (abstract base)
# ===========================================================================


class TestModelAdapterBase:
    def test_cannot_be_instantiated_directly(self) -> None:
        """ModelAdapter is abstract and must not be instantiable."""
        with pytest.raises(TypeError, match="abstract"):
            ModelAdapterBase()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_complete(self) -> None:
        """A subclass that omits complete() is also un-instantiable."""

        class Incomplete(ModelAdapterBase):
            @property
            def model_id(self) -> str:
                return "fake"

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_model_id(self) -> None:
        """A subclass that omits model_id is also un-instantiable."""

        class Incomplete(ModelAdapterBase):
            async def complete(
                self, prompt: str, system_prompt: str | None = None
            ) -> str:
                return ""

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_full_concrete_subclass_is_instantiable(self) -> None:
        """A class implementing all abstract members can be instantiated."""

        class Concrete(ModelAdapterBase):
            @property
            def model_id(self) -> str:
                return "test-model"

            async def complete(
                self, prompt: str, system_prompt: str | None = None
            ) -> str:
                return "response"

        adapter = Concrete()
        assert adapter.model_id == "test-model"


# ===========================================================================
# OpenAIAdapter
# ===========================================================================


class TestOpenAIAdapter:
    def _make(self, model: str = "gpt-4o", **kwargs: object) -> OpenAIAdapter:
        with patch("llmeval.models.openai_adapter.AsyncOpenAI"):
            return OpenAIAdapter(model=model, api_key=FAKE_OPENAI_KEY, **kwargs)  # type: ignore[arg-type]

    # --- construction -------------------------------------------------------

    def test_model_id(self) -> None:
        adapter = self._make(model="gpt-4o")
        assert adapter.model_id == "gpt-4o"

    def test_raises_config_error_without_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Ensure OPENAI_API_KEY is absent
            os.environ.pop("OPENAI_API_KEY", None)
            with (
                patch("llmeval.models.openai_adapter.AsyncOpenAI"),
                pytest.raises(ConfigurationError, match="OPENAI_API_KEY"),
            ):
                OpenAIAdapter(model="gpt-4o")

    def test_env_var_key_is_used_when_no_explicit_key(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": FAKE_OPENAI_KEY}),
            patch("llmeval.models.openai_adapter.AsyncOpenAI") as mock_client,
        ):
            OpenAIAdapter(model="gpt-4o")
            mock_client.assert_called_once_with(api_key=FAKE_OPENAI_KEY)

    # --- complete() ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_complete_without_system_prompt(self) -> None:
        adapter = self._make()
        mock_create = AsyncMock(return_value=_openai_response("Hello!"))
        adapter._client.chat.completions.create = mock_create

        result = await adapter.complete("Say hi")

        assert result.text == "Hello!"
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        # No system message — only user message
        assert call_kwargs["messages"] == [{"role": "user", "content": "Say hi"}]

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self) -> None:
        adapter = self._make()
        mock_create = AsyncMock(return_value=_openai_response("Hi"))
        adapter._client.chat.completions.create = mock_create

        await adapter.complete("Say hi", system_prompt="You are helpful.")

        messages = mock_create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Say hi"}

    @pytest.mark.asyncio
    async def test_complete_returns_empty_string_on_none_content(self) -> None:
        adapter = self._make()
        adapter._client.chat.completions.create = AsyncMock(
            return_value=_openai_response(None)
        )
        result = await adapter.complete("prompt")
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_complete_wraps_api_error_in_model_adapter_error(self) -> None:
        from openai import APIError as OpenAIAPIError

        adapter = self._make()
        # Use __new__ to bypass the SDK's complex constructor while still
        # producing a real OpenAIAPIError subclass instance.
        adapter._client.chat.completions.create = AsyncMock(
            side_effect=OpenAIAPIError.__new__(OpenAIAPIError)
        )
        with pytest.raises(ModelAdapterError, match="OpenAI API call failed"):
            await adapter.complete("prompt")

    @pytest.mark.asyncio
    async def test_model_adapter_error_chains_original_exception(self) -> None:
        from openai import APIError as OpenAIAPIError

        adapter = self._make()
        original = OpenAIAPIError.__new__(OpenAIAPIError)
        adapter._client.chat.completions.create = AsyncMock(side_effect=original)

        with pytest.raises(ModelAdapterError) as exc_info:
            await adapter.complete("prompt")
        assert exc_info.value.__cause__ is original


# ===========================================================================
# AnthropicAdapter
# ===========================================================================


class TestAnthropicAdapter:
    def _make(
        self, model: str = "claude-sonnet-4-20250514", **kwargs: object
    ) -> AnthropicAdapter:
        with patch("llmeval.models.anthropic_adapter.AsyncAnthropic"):
            return AnthropicAdapter(model=model, api_key=FAKE_ANTHROPIC_KEY, **kwargs)  # type: ignore[arg-type]

    # --- construction -------------------------------------------------------

    def test_model_id(self) -> None:
        adapter = self._make()
        assert adapter.model_id == "claude-sonnet-4-20250514"

    def test_default_max_tokens(self) -> None:
        adapter = self._make()
        assert adapter._max_tokens == 4096

    def test_custom_max_tokens(self) -> None:
        adapter = self._make(max_tokens=1024)
        assert adapter._max_tokens == 1024

    def test_raises_config_error_without_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with (
                patch("llmeval.models.anthropic_adapter.AsyncAnthropic"),
                pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"),
            ):
                AnthropicAdapter(model="claude-sonnet-4-20250514")

    def test_env_var_key_is_used_when_no_explicit_key(self) -> None:
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY}),
            patch("llmeval.models.anthropic_adapter.AsyncAnthropic") as mock_client,
        ):
            AnthropicAdapter(model="claude-sonnet-4-20250514")
            mock_client.assert_called_once_with(api_key=FAKE_ANTHROPIC_KEY)

    # --- complete() ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_complete_without_system_prompt(self) -> None:
        adapter = self._make()
        mock_create = AsyncMock(return_value=_anthropic_response("Hello!"))
        adapter._client.messages.create = mock_create

        result = await adapter.complete("Say hi")

        assert result.text == "Hello!"
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Say hi"}]
        assert "system" not in call_kwargs

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self) -> None:
        adapter = self._make()
        mock_create = AsyncMock(return_value=_anthropic_response("Hi"))
        adapter._client.messages.create = mock_create

        await adapter.complete("Say hi", system_prompt="You are helpful.")

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."
        # System is NOT in the messages list for Anthropic
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    @pytest.mark.asyncio
    async def test_complete_concatenates_multiple_text_blocks(self) -> None:
        """Multiple text content blocks should be joined into one string."""
        adapter = self._make()
        adapter._client.messages.create = AsyncMock(
            return_value=_anthropic_response("Hello", " world")
        )
        result = await adapter.complete("Say hi")
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_complete_skips_non_text_blocks(self) -> None:
        """Tool-use or other non-text blocks must not appear in the output."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.text = "should not appear"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "actual text"

        resp = MagicMock()
        resp.content = [tool_block, text_block]
        resp.usage = None

        adapter = self._make()
        adapter._client.messages.create = AsyncMock(return_value=resp)
        result = await adapter.complete("prompt")
        assert result.text == "actual text"

    @pytest.mark.asyncio
    async def test_complete_returns_empty_string_on_no_text_blocks(self) -> None:
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        resp = MagicMock()
        resp.content = [tool_block]
        resp.usage = None

        adapter = self._make()
        adapter._client.messages.create = AsyncMock(return_value=resp)
        result = await adapter.complete("prompt")
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_complete_passes_max_tokens(self) -> None:
        adapter = self._make(max_tokens=512)
        mock_create = AsyncMock(return_value=_anthropic_response("ok"))
        adapter._client.messages.create = mock_create

        await adapter.complete("prompt")

        assert mock_create.call_args.kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_complete_wraps_api_error_in_model_adapter_error(self) -> None:
        from anthropic import APIError as AnthropicAPIError

        adapter = self._make()
        adapter._client.messages.create = AsyncMock(
            side_effect=AnthropicAPIError.__new__(AnthropicAPIError)
        )
        with pytest.raises(ModelAdapterError, match="Anthropic API call failed"):
            await adapter.complete("prompt")

    @pytest.mark.asyncio
    async def test_model_adapter_error_chains_original_exception(self) -> None:
        from anthropic import APIError as AnthropicAPIError

        adapter = self._make()
        original = AnthropicAPIError.__new__(AnthropicAPIError)
        adapter._client.messages.create = AsyncMock(side_effect=original)

        with pytest.raises(ModelAdapterError) as exc_info:
            await adapter.complete("prompt")
        assert exc_info.value.__cause__ is original


# ===========================================================================
# create_adapter factory
# ===========================================================================


class TestCreateAdapter:
    def _patched(self, model: str) -> ModelAdapter:
        with (
            patch("llmeval.models.openai_adapter.AsyncOpenAI"),
            patch("llmeval.models.anthropic_adapter.AsyncAnthropic"),
        ):
            return create_adapter(model=model, api_key="fake-key")

    def test_gpt_model_returns_openai_adapter(self) -> None:
        assert isinstance(self._patched("gpt-4o"), OpenAIAdapter)

    def test_gpt4_turbo_returns_openai_adapter(self) -> None:
        assert isinstance(self._patched("gpt-4-turbo"), OpenAIAdapter)

    def test_o1_model_returns_openai_adapter(self) -> None:
        assert isinstance(self._patched("o1-mini"), OpenAIAdapter)

    def test_o3_model_returns_openai_adapter(self) -> None:
        assert isinstance(self._patched("o3-mini"), OpenAIAdapter)

    def test_claude_model_returns_anthropic_adapter(self) -> None:
        assert isinstance(self._patched("claude-sonnet-4-20250514"), AnthropicAdapter)

    def test_claude_opus_returns_anthropic_adapter(self) -> None:
        assert isinstance(self._patched("claude-opus-4-20250514"), AnthropicAdapter)

    def test_unknown_model_raises_config_error(self) -> None:
        with pytest.raises(ConfigurationError, match="Cannot infer provider"):
            create_adapter(model="llama-3-70b", api_key="fake")

    def test_extra_kwargs_forwarded_to_anthropic(self) -> None:
        with patch("llmeval.models.anthropic_adapter.AsyncAnthropic"):
            adapter = create_adapter(
                model="claude-sonnet-4-20250514",
                api_key=FAKE_ANTHROPIC_KEY,
                max_tokens=256,
            )
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter._max_tokens == 256
