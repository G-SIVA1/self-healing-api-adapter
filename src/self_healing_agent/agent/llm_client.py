from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from google import genai
from google.genai import types
from openai import AsyncOpenAI


class LLMClientError(Exception):
    """Base exception for LLM client failures."""


class LLMTimeoutError(LLMClientError):
    """Raised when an LLM request exceeds the configured timeout."""


class LLMRequestError(LLMClientError):
    """Raised when an LLM provider request fails."""


@dataclass(frozen=True, slots=True)
class ProviderError:
    """Normalized provider error information."""

    status_code: int | None
    message: str
    retryable: bool
    fallback_allowed: bool


class LLMTransport(Protocol):
    async def send(self, prompt: str, model: str) -> str:
        """Send a prompt to an LLM provider."""


class OpenAITransport:
    """Async transport for OpenAI's Responses API."""

    def __init__(self, api_key: str | None = None) -> None:
        if api_key is not None and not api_key.strip():
            raise ValueError("api_key cannot be empty")

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
            raise ValueError("prompt cannot be empty")

        if not model.strip():
            raise ValueError("model cannot be empty")

        try:
            response = await self._client.responses.create(
                model=model,
                input=prompt,
            )
        except Exception as exc:
            raise LLMRequestError(
                self._format_provider_error(
                    provider="OpenAI",
                    error=exc,
                )
            ) from exc

        output = response.output_text

        if not output.strip():
            raise LLMRequestError(
                "OpenAI returned an empty response"
            )

        return output

    @staticmethod
    def _format_provider_error(
        provider: str,
        error: Exception,
    ) -> str:
        status_code = getattr(
            error,
            "status_code",
            None,
        )

        if status_code is not None:
            return (
                f"{provider} request failed with HTTP "
                f"{status_code}: {error}"
            )

        return f"{provider} request failed: {error}"


class GeminiTransport:
    """Async transport for Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        if api_key is not None and not api_key.strip():
            raise ValueError("api_key cannot be empty")

        try:
            self._client = genai.Client(
                api_key=api_key,
            )
        except Exception as exc:
            raise LLMRequestError(
                f"Unable to initialize Gemini client: {exc}"
            ) from exc

    async def send(
        self,
        prompt: str,
        model: str,
    ) -> str:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if not model.strip():
            raise ValueError("model cannot be empty")

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
            provider_error = self._classify_error(exc)

            raise LLMRequestError(
                self._format_error(
                    model=model,
                    provider_error=provider_error,
                )
            ) from exc

        output = response.text

        if not output or not output.strip():
            raise LLMRequestError(
                f"Gemini model '{model}' returned "
                "an empty response"
            )

        return output

    @classmethod
    def _classify_error(
        cls,
        error: Exception,
    ) -> ProviderError:
        status_code = cls._extract_status_code(error)
        message = str(error).strip()

        if status_code in {
            408,
            429,
            500,
            502,
            503,
            504,
        }:
            return ProviderError(
                status_code=status_code,
                message=message,
                retryable=True,
                fallback_allowed=True,
            )

        if status_code in {
            400,
            401,
            403,
            404,
        }:
            return ProviderError(
                status_code=status_code,
                message=message,
                retryable=False,
                fallback_allowed=status_code == 404,
            )

        return ProviderError(
            status_code=status_code,
            message=message,
            retryable=True,
            fallback_allowed=True,
        )

    @staticmethod
    def _extract_status_code(
        error: Exception,
    ) -> int | None:
        status_code = getattr(
            error,
            "status_code",
            None,
        )

        if isinstance(status_code, int):
            return status_code

        response = getattr(
            error,
            "response",
            None,
        )

        if response is not None:
            response_status = getattr(
                response,
                "status_code",
                None,
            )

            if isinstance(response_status, int):
                return response_status

        return None

    @staticmethod
    def _format_error(
        model: str,
        provider_error: ProviderError,
    ) -> str:
        if provider_error.status_code is not None:
            status = (
                f"HTTP {provider_error.status_code}"
            )
        else:
            status = "unknown status"

        return (
            f"Gemini request failed for model "
            f"'{model}' ({status}). "
            f"Retryable={provider_error.retryable}, "
            f"FallbackAllowed="
            f"{provider_error.fallback_allowed}. "
            f"Provider message: "
            f"{provider_error.message}"
        )


class AsyncLLMClient:
    """
    Resilient asynchronous LLM client.

    Supports:
    - request timeouts
    - retry with exponential backoff
    - model failover
    - permanent error detection
    - detailed provider errors
    - successful model provenance
    """

    def __init__(
        self,
        transport: LLMTransport,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        base_delay_seconds: float = 2.0,
        fallback_models: tuple[str, ...] = (),
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")

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

        normalized_fallback_models = tuple(
            fallback_model.strip()
            for fallback_model in fallback_models
            if fallback_model.strip()
            and fallback_model.strip() != model
        )

        self._transport = transport
        self._model = model
        self._models = (
            model,
            *normalized_fallback_models,
        )
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._base_delay_seconds = base_delay_seconds
        self._last_model: str | None = None

    @property
    def last_model(self) -> str | None:
        """Return the model that produced the latest successful response."""

        return self._last_model

    @property
    def configured_models(self) -> tuple[str, ...]:
        """Return the primary model followed by fallback models."""

        return self._models

    async def generate(
        self,
        prompt: str,
    ) -> str:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        has_fallback_models = len(self._models) > 1
        failures: list[str] = []

        self._last_model = None

        for model in self._models:
            try:
                response = await self._generate_with_model(
                    prompt=prompt,
                    model=model,
                    allow_timeout_failover=has_fallback_models,
                )

                self._last_model = model

                return response

            except LLMTimeoutError as exc:
                if not has_fallback_models:
                    raise

                failures.append(
                    f"{model}: {exc}"
                )

                continue

            except LLMRequestError as exc:
                if self._should_stop_for_error(exc):
                    raise

                failures.append(
                    f"{model}: {exc}"
                )

                continue

        failure_summary = "\n".join(
            f"- {failure}"
            for failure in failures
        )

        raise LLMRequestError(
            "All configured LLM models failed.\n"
            f"{failure_summary}"
        )

    async def _generate_with_model(
        self,
        prompt: str,
        model: str,
        allow_timeout_failover: bool,
    ) -> str:
        last_error: Exception | None = None

        for attempt in range(
            self._max_retries + 1
        ):
            try:
                return await asyncio.wait_for(
                    self._transport.send(
                        prompt=prompt,
                        model=model,
                    ),
                    timeout=self._timeout_seconds,
                )

            except asyncio.TimeoutError as exc:
                timeout_error = LLMTimeoutError(
                    f"LLM request timed out for model "
                    f"'{model}' after "
                    f"{self._timeout_seconds:.1f} seconds"
                )

                last_error = timeout_error

                if attempt >= self._max_retries:
                    if allow_timeout_failover:
                        raise timeout_error from exc

                    raise timeout_error from exc

                await self._sleep_before_retry(
                    attempt=attempt,
                )

            except LLMRequestError as exc:
                last_error = exc

                if not self._is_retryable_error(exc):
                    raise

                if attempt >= self._max_retries:
                    raise LLMRequestError(
                        f"LLM request failed for model "
                        f"'{model}' after "
                        f"{self._max_retries + 1} attempts: "
                        f"{exc}"
                    ) from exc

                await self._sleep_before_retry(
                    attempt=attempt,
                )

            except Exception as exc:
                last_error = exc

                if attempt >= self._max_retries:
                    raise LLMRequestError(
                        f"LLM request failed for model "
                        f"'{model}' after "
                        f"{self._max_retries + 1} attempts: "
                        f"{exc}"
                    ) from exc

                await self._sleep_before_retry(
                    attempt=attempt,
                )

        raise LLMRequestError(
            f"LLM request failed unexpectedly for model "
            f"'{model}'"
        ) from last_error

    async def _sleep_before_retry(
        self,
        attempt: int,
    ) -> None:
        delay = self._calculate_backoff_delay(
            attempt=attempt,
        )

        if delay > 0:
            await asyncio.sleep(delay)

    def _calculate_backoff_delay(
        self,
        attempt: int,
    ) -> float:
        return self._base_delay_seconds * (
            2**attempt
        )

    @staticmethod
    def _is_retryable_error(
        error: LLMRequestError,
    ) -> bool:
        message = str(error)

        if "Retryable=False" in message:
            return False

        return True

    @staticmethod
    def _should_stop_for_error(
        error: LLMRequestError,
    ) -> bool:
        """
        Stop immediately for errors that are not useful
        to retry or hide behind another model.
        """

        message = str(error)

        if "HTTP 400" in message:
            return True

        if "HTTP 401" in message:
            return True

        if "HTTP 403" in message:
            return True

        if "FallbackAllowed=False" in message:
            return True

        return False