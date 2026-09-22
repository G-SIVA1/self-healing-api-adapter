from __future__ import annotations

import asyncio

import pytest

from self_healing_agent.agent.llm_client import (
    AsyncLLMClient,
    LLMRequestError,
    LLMTimeoutError,
)


class SuccessfulTransport:
    """Transport that always succeeds."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.models: list[str] = []

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.models.append(model)

        return '{"status": "ok"}'


class FailingTransport:
    """Transport that always fails."""

    def __init__(self) -> None:
        self.calls = 0

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        self.calls += 1

        raise RuntimeError(
            "temporary provider failure"
        )


class RetryThenSuccessTransport:
    """Fail a fixed number of times, then succeed."""

    def __init__(
        self,
        failures_before_success: int,
    ) -> None:
        self.calls = 0
        self.failures_before_success = (
            failures_before_success
        )

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        self.calls += 1

        if (
            self.calls
            <= self.failures_before_success
        ):
            raise RuntimeError(
                "temporary provider failure"
            )

        return '{"status": "recovered"}'


class SlowTransport:
    """Transport that deliberately exceeds the timeout."""

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        await asyncio.sleep(1)

        return '{"status": "too late"}'


@pytest.mark.asyncio
async def test_llm_client_returns_response() -> None:
    transport = SuccessfulTransport()

    client = AsyncLLMClient(
        transport=transport,
        model="test-model",
        timeout_seconds=1,
        max_retries=0,
        base_delay_seconds=0,
    )

    result = await client.generate(
        "repair this API",
    )

    assert result == '{"status": "ok"}'

    assert transport.calls == 1

    assert transport.prompts == [
        "repair this API"
    ]

    assert transport.models == [
        "test-model"
    ]


@pytest.mark.asyncio
async def test_llm_client_retries_then_succeeds() -> None:
    transport = RetryThenSuccessTransport(
        failures_before_success=2,
    )

    client = AsyncLLMClient(
        transport=transport,
        model="test-model",
        timeout_seconds=1,
        max_retries=2,
        base_delay_seconds=0,
    )

    result = await client.generate(
        "repair this API",
    )

    assert result == '{"status": "recovered"}'

    assert transport.calls == 3


@pytest.mark.asyncio
async def test_llm_client_raises_after_retries() -> None:
    transport = FailingTransport()

    client = AsyncLLMClient(
        transport=transport,
        model="test-model",
        timeout_seconds=1,
        max_retries=2,
        base_delay_seconds=0,
    )

    with pytest.raises(
        LLMRequestError,
        match="after 3 attempts",
    ):
        await client.generate(
            "repair this API",
        )

    assert transport.calls == 3


@pytest.mark.asyncio
async def test_llm_client_handles_timeout() -> None:
    transport = SlowTransport()

    client = AsyncLLMClient(
        transport=transport,
        model="test-model",
        timeout_seconds=0.01,
        max_retries=0,
        base_delay_seconds=0,
    )

    with pytest.raises(
        LLMTimeoutError,
        match="timed out",
    ):
        await client.generate(
            "repair this API",
        )


@pytest.mark.asyncio
async def test_llm_client_retries_timeout() -> None:
    transport = SlowTransport()

    client = AsyncLLMClient(
        transport=transport,
        model="test-model",
        timeout_seconds=0.01,
        max_retries=2,
        base_delay_seconds=0,
    )

    with pytest.raises(
        LLMTimeoutError,
        match="timed out",
    ):
        await client.generate(
            "repair this API",
        )


@pytest.mark.asyncio
async def test_llm_client_rejects_empty_prompt() -> None:
    transport = SuccessfulTransport()

    client = AsyncLLMClient(
        transport=transport,
        model="test-model",
    )

    with pytest.raises(
        ValueError,
        match="prompt cannot be empty",
    ):
        await client.generate("")


def test_llm_client_rejects_invalid_configuration() -> None:
    transport = SuccessfulTransport()

    with pytest.raises(
        ValueError,
        match="model cannot be empty",
    ):
        AsyncLLMClient(
            transport=transport,
            model="",
        )

    with pytest.raises(
        ValueError,
        match="timeout_seconds",
    ):
        AsyncLLMClient(
            transport=transport,
            model="test-model",
            timeout_seconds=0,
        )

    with pytest.raises(
        ValueError,
        match="max_retries",
    ):
        AsyncLLMClient(
            transport=transport,
            model="test-model",
            max_retries=-1,
        )