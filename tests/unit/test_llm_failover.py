from __future__ import annotations

import asyncio

import pytest

from self_healing_agent.agent.llm_client import (
    AsyncLLMClient,
    LLMRequestError,
    LLMTimeoutError,
)


class SequenceTransport:
    """Transport that returns predefined responses/errors."""

    def __init__(
        self,
        outcomes: list[str | Exception],
    ) -> None:
        self._outcomes = outcomes
        self.calls: list[str] = []

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        self.calls.append(model)

        if not self._outcomes:
            raise RuntimeError(
                "No more configured transport outcomes"
            )

        outcome = self._outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


class TimeoutTransport:
    """Transport that always times out."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        self.calls.append(model)
        await asyncio.sleep(1)
        return "never reached"


@pytest.mark.asyncio
async def test_503_retries_then_uses_fallback() -> None:
    transport = SequenceTransport(
        [
            LLMRequestError(
                "Gemini request failed for model "
                "'primary-model' (HTTP 503). "
                "Retryable=True, "
                "FallbackAllowed=True."
            ),
            LLMRequestError(
                "Gemini request failed for model "
                "'primary-model' (HTTP 503). "
                "Retryable=True, "
                "FallbackAllowed=True."
            ),
            "successful fallback response",
        ]
    )

    client = AsyncLLMClient(
        transport=transport,
        model="primary-model",
        fallback_models=("fallback-model",),
        timeout_seconds=1.0,
        max_retries=1,
        base_delay_seconds=0,
    )

    result = await client.generate(
        "repair this API",
    )

    assert result == "successful fallback response"

    assert transport.calls == [
        "primary-model",
        "primary-model",
        "fallback-model",
    ]


@pytest.mark.asyncio
async def test_404_model_is_skipped_without_retry() -> None:
    transport = SequenceTransport(
        [
            LLMRequestError(
                "Gemini request failed for model "
                "'unavailable-model' (HTTP 404). "
                "Retryable=False, "
                "FallbackAllowed=True."
            ),
            "fallback response",
        ]
    )

    client = AsyncLLMClient(
        transport=transport,
        model="unavailable-model",
        fallback_models=("fallback-model",),
        timeout_seconds=1.0,
        max_retries=2,
        base_delay_seconds=0,
    )

    result = await client.generate(
        "repair this API",
    )

    assert result == "fallback response"

    assert transport.calls == [
        "unavailable-model",
        "fallback-model",
    ]


@pytest.mark.asyncio
async def test_timeout_retries_then_uses_fallback() -> None:
    transport = SequenceTransport(
        [
            LLMTimeoutError(
                "LLM request timed out for model "
                "'primary-model'"
            ),
            "fallback response",
        ]
    )

    client = AsyncLLMClient(
        transport=transport,
        model="primary-model",
        fallback_models=("fallback-model",),
        timeout_seconds=1.0,
        max_retries=0,
        base_delay_seconds=0,
    )

    result = await client.generate(
        "repair this API",
    )

    assert result == "fallback response"

    assert transport.calls == [
        "primary-model",
        "fallback-model",
    ]


@pytest.mark.asyncio
async def test_429_daily_quota_uses_fallback_without_retry() -> None:
    transport = SequenceTransport(
        [
            LLMRequestError(
                "Gemini request failed for model "
                "'primary-model' (HTTP 429). "
                "Retryable=False, "
                "FallbackAllowed=True. "
                "Provider message: "
                "Quota exceeded for metric: "
                "generativelanguage.googleapis.com/"
                "generate_content_free_tier_requests"
            ),
            "fallback response",
        ]
    )

    client = AsyncLLMClient(
        transport=transport,
        model="primary-model",
        fallback_models=("fallback-model",),
        timeout_seconds=1.0,
        max_retries=2,
        base_delay_seconds=0,
    )

    result = await client.generate(
        "repair this API",
    )

    assert result == "fallback response"

    assert transport.calls == [
        "primary-model",
        "fallback-model",
    ]


@pytest.mark.asyncio
async def test_all_models_failure_contains_failure_details() -> None:
    transport = SequenceTransport(
        [
            LLMRequestError(
                "Gemini request failed for model "
                "'primary-model' (HTTP 503). "
                "Retryable=True, "
                "FallbackAllowed=True."
            ),
            LLMRequestError(
                "Gemini request failed for model "
                "'fallback-model' (HTTP 503). "
                "Retryable=True, "
                "FallbackAllowed=True."
            ),
        ]
    )

    client = AsyncLLMClient(
        transport=transport,
        model="primary-model",
        fallback_models=("fallback-model",),
        timeout_seconds=1.0,
        max_retries=0,
        base_delay_seconds=0,
    )

    with pytest.raises(
        LLMRequestError,
        match="All configured LLM models failed",
    ) as exc_info:
        await client.generate(
            "repair this API",
        )

    error_message = str(exc_info.value)

    assert "primary-model" in error_message
    assert "fallback-model" in error_message
    assert "HTTP 503" in error_message


@pytest.mark.asyncio
async def test_timeout_without_fallback_preserves_timeout_error() -> None:
    transport = TimeoutTransport()

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

    assert transport.calls == [
        "test-model",
    ]