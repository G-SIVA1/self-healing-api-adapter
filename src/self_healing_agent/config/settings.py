from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(Exception):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Application configuration."""

    llm_provider: str
    llm_model: str
    llm_timeout_seconds: float
    llm_max_retries: int

    @classmethod
    def from_environment(cls) -> Settings:
        """Create settings from environment variables."""

        provider = os.getenv(
            "LLM_PROVIDER",
            "openai",
        ).strip()

        model = os.getenv(
            "LLM_MODEL",
            "",
        ).strip()

        timeout_raw = os.getenv(
            "LLM_TIMEOUT_SECONDS",
            "30",
        ).strip()

        retries_raw = os.getenv(
            "LLM_MAX_RETRIES",
            "2",
        ).strip()

        if not provider:
            raise SettingsError(
                "LLM_PROVIDER cannot be empty"
            )

        if not model:
            raise SettingsError(
                "LLM_MODEL cannot be empty"
            )

        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise SettingsError(
                "LLM_TIMEOUT_SECONDS must be a number"
            ) from exc

        if timeout_seconds <= 0:
            raise SettingsError(
                "LLM_TIMEOUT_SECONDS must be greater than zero"
            )

        try:
            max_retries = int(retries_raw)
        except ValueError as exc:
            raise SettingsError(
                "LLM_MAX_RETRIES must be an integer"
            ) from exc

        if max_retries < 0:
            raise SettingsError(
                "LLM_MAX_RETRIES cannot be negative"
            )

        return cls(
            llm_provider=provider,
            llm_model=model,
            llm_timeout_seconds=timeout_seconds,
            llm_max_retries=max_retries,
        )