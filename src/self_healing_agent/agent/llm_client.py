from __future__ import annotations

import asyncio
from typing import Protocol

from google import genai
from google.genai import types
from openai import AsyncOpenAI


class LLMClientError(Exception):
    """Base exception for LLM client failures."""


class LLMTimeoutError(LLMClientError):
    """Raised when an LLM request times out."""


class LLMRequestError(LLMClientError):
    """Raised when an LLM request fails."""


class LLMTransport(Protocol):
    """Low-level asynchronous transport used by the LLM client."""

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        ...


class OpenAITransport:
    """OpenAI implementation of the LLM transport."""

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        if api_key is not None and not api_key.strip():
            raise ValueError(
                "api_key cannot be empty"
            )

        self._client = AsyncOpenAI(
            api_key=api_key,
            max_retries=0,
        )

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        if not prompt.strip():
            raise ValueError(
                "prompt cannot be empty"
            )

        if not model.strip():
            raise ValueError(
                "model cannot be empty"
            )

        try:
            response = await self._client.responses.create(
                model=model,
                input=prompt,
            )
        except Exception as exc:
            raise LLMRequestError(
                "OpenAI request failed"
            ) from exc

        output = response.output_text

        if not output.strip():
            raise LLMRequestError(
                "OpenAI returned an empty response"
            )

        return output


class GeminiTransport:
    """Google Gemini implementation of the LLM transport."""

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        if api_key is not None and not api_key.strip():
            raise ValueError(
                "api_key cannot be empty"
            )

        try:
            self._client = genai.Client(
                api_key=api_key,
            )
        except Exception as exc:
            raise LLMRequestError(
                "Unable to initialize Gemini client"
            ) from exc

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        if not prompt.strip():
            raise ValueError(
                "prompt cannot be empty"
            )

        if not model.strip():
            raise ValueError(
                "model cannot be empty"
            )

        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
        except Exception as exc:
            raise LLMRequestError(
                "Gemini request failed"
            ) from exc

        output = response.text

        if not output or not output.strip():
            raise LLMRequestError(
                "Gemini returned an empty response"
            )

        return output


class AsyncLLMClient:
    """Provider-independent LLM client."""

    def __init__(
        self,
        transport: LLMTransport,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        base_delay_seconds: float = 0.5,
    ) -> None:
        if not model.strip():
            raise ValueError(
                "model cannot be empty"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero"
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries cannot be negative"
            )

        if base_delay_seconds < 0:
            raise ValueError(
                "base_delay_seconds cannot be negative"
            )

        self._transport = transport
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._base_delay_seconds = base_delay_seconds

    async def generate(
        self,
        prompt: str,
    ) -> str:
        if not prompt.strip():
            raise ValueError(
                "prompt cannot be empty"
            )

        last_error: Exception | None = None

        for attempt in range(
            self._max_retries + 1
        ):
            try:
                return await asyncio.wait_for(
                    self._transport.send(
                        prompt=prompt,
                        model=self._model,
                    ),
                    timeout=self._timeout_seconds,
                )

            except asyncio.TimeoutError as exc:
                last_error = LLMTimeoutError(
                    "LLM request timed out"
                )

                if attempt >= self._max_retries:
                    raise last_error from exc

            except Exception as exc:
                last_error = exc

                if attempt >= self._max_retries:
                    raise LLMRequestError(
                        "LLM request failed after "
                        f"{self._max_retries + 1} attempts"
                    ) from exc

            delay = (
                self._base_delay_seconds
                * (2**attempt)
            )

            if delay > 0:
                await asyncio.sleep(delay)

        raise LLMRequestError(
            "LLM request failed unexpectedly"
        ) from last_error