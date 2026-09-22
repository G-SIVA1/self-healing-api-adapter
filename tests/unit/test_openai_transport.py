from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from self_healing_agent.agent.llm_client import (
    LLMRequestError,
    OpenAITransport,
)


@pytest.mark.asyncio
async def test_openai_transport_returns_output_text() -> None:
    response = MagicMock()

    response.output_text = (
        '{"explanation":"repair",'
        '"confidence":0.95,'
        '"replacement_code":"fixed"}'
    )

    client = MagicMock()

    client.responses.create = AsyncMock(
        return_value=response,
    )

    with patch(
        "self_healing_agent.agent.llm_client.AsyncOpenAI",
        return_value=client,
    ):
        transport = OpenAITransport(
            api_key="test-key",
        )

        result = await transport.send(
            prompt="repair this",
            model="test-model",
        )

    assert result == response.output_text

    client.responses.create.assert_awaited_once_with(
        model="test-model",
        input="repair this",
    )


@pytest.mark.asyncio
async def test_openai_transport_wraps_provider_error() -> None:
    client = MagicMock()

    client.responses.create = AsyncMock(
        side_effect=RuntimeError(
            "provider unavailable"
        ),
    )

    with patch(
        "self_healing_agent.agent.llm_client.AsyncOpenAI",
        return_value=client,
    ):
        transport = OpenAITransport(
            api_key="test-key",
        )

        with pytest.raises(
            LLMRequestError,
            match="OpenAI request failed",
        ):
            await transport.send(
                prompt="repair this",
                model="test-model",
            )


@pytest.mark.asyncio
async def test_openai_transport_rejects_empty_output() -> None:
    response = MagicMock()

    response.output_text = ""

    client = MagicMock()

    client.responses.create = AsyncMock(
        return_value=response,
    )

    with patch(
        "self_healing_agent.agent.llm_client.AsyncOpenAI",
        return_value=client,
    ):
        transport = OpenAITransport(
            api_key="test-key",
        )

        with pytest.raises(
            LLMRequestError,
            match="empty response",
        ):
            await transport.send(
                prompt="repair this",
                model="test-model",
            )