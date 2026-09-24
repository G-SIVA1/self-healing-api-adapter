from __future__ import annotations

import asyncio

import pytest

from self_healing_agent.agent.llm_client import (
    AsyncLLMClient,
    LLMRequestError,
)
from self_healing_agent.agent.reasoner import (
    LLMRepairReasoner,
)


class FakeTransport:
    def __init__(
        self,
        responses: dict[str, str | Exception],
    ) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        self.calls.append(model)

        response = self.responses[model]

        if isinstance(response, Exception):
            raise response

        return response


class FailingTransport:
    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        raise LLMRequestError(
            "HTTP 503 Service Unavailable"
        )


@pytest.mark.asyncio
async def test_fallback_model_is_used_after_primary_failure() -> None:
    transport = FakeTransport(
        responses={
            "primary-model": LLMRequestError(
                "HTTP 503 Service Unavailable"
            ),
            "fallback-model": "successful response",
        }
    )

    client = AsyncLLMClient(
        transport=transport,
        model="primary-model",
        fallback_models=("fallback-model",),
        max_retries=0,
        base_delay_seconds=0,
    )

    result = await client.generate(
        "test prompt"
    )

    assert result == "successful response"

    assert transport.calls == [
        "primary-model",
        "fallback-model",
    ]


@pytest.mark.asyncio
async def test_last_model_records_successful_fallback() -> None:
    transport = FakeTransport(
        responses={
            "primary-model": LLMRequestError(
                "HTTP 503 Service Unavailable"
            ),
            "fallback-model": "successful response",
        }
    )

    client = AsyncLLMClient(
        transport=transport,
        model="primary-model",
        fallback_models=("fallback-model",),
        max_retries=0,
        base_delay_seconds=0,
    )

    await client.generate(
        "test prompt"
    )

    assert client.last_model == "fallback-model"


@pytest.mark.asyncio
async def test_last_model_records_primary_success() -> None:
    transport = FakeTransport(
        responses={
            "primary-model": "successful response",
            "fallback-model": "fallback response",
        }
    )

    client = AsyncLLMClient(
        transport=transport,
        model="primary-model",
        fallback_models=("fallback-model",),
        max_retries=0,
        base_delay_seconds=0,
    )

    result = await client.generate(
        "test prompt"
    )

    assert result == "successful response"
    assert client.last_model == "primary-model"

    assert transport.calls == [
        "primary-model",
    ]


@pytest.mark.asyncio
async def test_last_model_resets_when_new_request_fails() -> None:
    transport = FakeTransport(
        responses={
            "primary-model": "successful response",
        }
    )

    client = AsyncLLMClient(
        transport=transport,
        model="primary-model",
        max_retries=0,
        base_delay_seconds=0,
    )

    await client.generate(
        "first request"
    )

    assert client.last_model == "primary-model"

    transport.responses["primary-model"] = LLMRequestError(
        "HTTP 503 Service Unavailable"
    )

    with pytest.raises(LLMRequestError):
        await client.generate(
            "second request"
        )

    assert client.last_model is None


@pytest.mark.asyncio
async def test_timeout_can_fail_over_to_next_model() -> None:
    class TimeoutTransport:
        async def send(
            self,
            prompt: str,
            model: str,
        ) -> str:
            if model == "primary-model":
                await asyncio.sleep(0.05)

            return "fallback response"

    client = AsyncLLMClient(
        transport=TimeoutTransport(),
        model="primary-model",
        fallback_models=("fallback-model",),
        timeout_seconds=0.01,
        max_retries=0,
        base_delay_seconds=0,
    )

    result = await client.generate(
        "test prompt"
    )

    assert result == "fallback response"
    assert client.last_model == "fallback-model"


@pytest.mark.asyncio
async def test_non_retryable_error_does_not_use_fallback() -> None:
    transport = FakeTransport(
        responses={
            "primary-model": LLMRequestError(
                "Gemini request failed "
                "(HTTP 400). "
                "Retryable=False, "
                "FallbackAllowed=False."
            ),
            "fallback-model": "should not be used",
        }
    )

    client = AsyncLLMClient(
        transport=transport,
        model="primary-model",
        fallback_models=("fallback-model",),
        max_retries=0,
        base_delay_seconds=0,
    )

    with pytest.raises(LLMRequestError):
        await client.generate(
            "test prompt"
        )

    assert transport.calls == [
        "primary-model",
    ]

    assert client.last_model is None