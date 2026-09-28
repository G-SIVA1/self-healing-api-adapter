from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(Exception):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated application configuration."""

    llm_provider: str
    llm_model: str
    llm_timeout_seconds: float
    llm_max_retries: int
    llm_fallback_models: tuple[str, ...]

    github_token: str | None = None
    github_repository: str | None = None

    @classmethod
    def from_environment(cls) -> Settings:
        """Create validated settings from environment variables."""

        provider = os.getenv("LLM_PROVIDER", "openai").strip()
        model = os.getenv("LLM_MODEL", "").strip()

        fallback_models_raw = os.getenv(
            "LLM_FALLBACK_MODELS",
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

        github_token_raw = os.getenv(
            "GITHUB_TOKEN",
            "",
        ).strip()

        github_repository_raw = os.getenv(
            "GITHUB_REPOSITORY",
            "",
        ).strip()

        if not provider:
            raise SettingsError(
                "LLM_PROVIDER cannot be empty"
            )

        if not model:
            raise SettingsError(
                "LLM_MODEL cannot be empty"
            )

        fallback_models = cls._parse_fallback_models(
            fallback_models_raw=fallback_models_raw,
            primary_model=model,
        )

        timeout_seconds = cls._parse_timeout(
            timeout_raw=timeout_raw,
        )

        max_retries = cls._parse_max_retries(
            retries_raw=retries_raw,
        )

        github_token = (
            github_token_raw
            if github_token_raw
            else None
        )

        github_repository = cls._parse_github_repository(
            github_repository_raw=github_repository_raw,
        )

        return cls(
            llm_provider=provider,
            llm_model=model,
            llm_timeout_seconds=timeout_seconds,
            llm_max_retries=max_retries,
            llm_fallback_models=fallback_models,
            github_token=github_token,
            github_repository=github_repository,
        )

    @staticmethod
    def _parse_fallback_models(
        fallback_models_raw: str,
        primary_model: str,
    ) -> tuple[str, ...]:
        """Parse and validate fallback LLM models."""

        if not fallback_models_raw:
            return ()

        fallback_models = tuple(
            model_name.strip()
            for model_name in fallback_models_raw.split(",")
            if model_name.strip()
        )

        return tuple(
            model_name
            for model_name in fallback_models
            if model_name != primary_model
        )

    @staticmethod
    def _parse_timeout(
        timeout_raw: str,
    ) -> float:
        """Parse and validate the LLM timeout."""

        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise SettingsError(
                "LLM_TIMEOUT_SECONDS must be a number"
            ) from exc

        if not timeout_seconds > 0:
            raise SettingsError(
                "LLM_TIMEOUT_SECONDS must be greater than zero"
            )

        return timeout_seconds

    @staticmethod
    def _parse_max_retries(
        retries_raw: str,
    ) -> int:
        """Parse and validate the maximum retry count."""

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

        return max_retries

    @staticmethod
    def _parse_github_repository(
        github_repository_raw: str,
    ) -> str | None:
        """Parse and validate the GitHub repository identifier."""

        if not github_repository_raw:
            return None

        repository = github_repository_raw.strip()

        repository_parts = repository.split("/")

        if (
            len(repository_parts) != 2
            or not repository_parts[0].strip()
            or not repository_parts[1].strip()
        ):
            raise SettingsError(
                "GITHUB_REPOSITORY must use the format 'owner/repository'"
            )

        owner = repository_parts[0].strip()
        repository_name = repository_parts[1].strip()

        if any(
            character.isspace()
            for character in owner
        ):
            raise SettingsError(
                "GITHUB_REPOSITORY owner cannot contain whitespace"
            )

        if any(
            character.isspace()
            for character in repository_name
        ):
            raise SettingsError(
                "GITHUB_REPOSITORY repository cannot contain whitespace"
            )

        return f"{owner}/{repository_name}"