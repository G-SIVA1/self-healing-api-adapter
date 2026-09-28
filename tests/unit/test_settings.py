from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from self_healing_agent.config.settings import (
    Settings,
    SettingsError,
)


def test_settings_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv(
        "LLM_FALLBACK_MODELS",
        "gemini-3.7,gemini-3.8",
    )
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")

    settings = Settings.from_environment()

    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-3.6-flash"
    assert settings.llm_fallback_models == (
        "gemini-3.7",
        "gemini-3.8",
    )
    assert settings.llm_timeout_seconds == 45.0
    assert settings.llm_max_retries == 3


def test_settings_use_default_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")

    monkeypatch.delenv(
        "LLM_TIMEOUT_SECONDS",
        raising=False,
    )

    settings = Settings.from_environment()

    assert settings.llm_timeout_seconds == 30.0


def test_settings_reject_empty_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")

    with pytest.raises(
        SettingsError,
        match="LLM_PROVIDER cannot be empty",
    ):
        Settings.from_environment()


def test_settings_reject_empty_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "")

    with pytest.raises(
        SettingsError,
        match="LLM_MODEL cannot be empty",
    ):
        Settings.from_environment()


def test_settings_reject_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "invalid")

    with pytest.raises(
        SettingsError,
        match="LLM_TIMEOUT_SECONDS must be a number",
    ):
        Settings.from_environment()


def test_settings_reject_negative_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("LLM_MAX_RETRIES", "-1")

    with pytest.raises(
        SettingsError,
        match="LLM_MAX_RETRIES cannot be negative",
    ):
        Settings.from_environment()


def test_settings_allow_missing_github_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")

    monkeypatch.delenv(
        "GITHUB_TOKEN",
        raising=False,
    )
    monkeypatch.delenv(
        "GITHUB_REPOSITORY",
        raising=False,
    )

    settings = Settings.from_environment()

    assert settings.github_token is None
    assert settings.github_repository is None


def test_settings_allow_blank_github_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv(
        "GITHUB_REPOSITORY",
        "   ",
    )

    settings = Settings.from_environment()

    assert settings.github_repository is None


def test_settings_load_valid_github_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "  github-secret-token  ",
    )
    monkeypatch.setenv(
        "GITHUB_REPOSITORY",
        "  G-SIVA1/self-healing-api-adapter  ",
    )

    settings = Settings.from_environment()

    assert settings.github_token == "github-secret-token"
    assert (
        settings.github_repository
        == "G-SIVA1/self-healing-api-adapter"
    )


@pytest.mark.parametrize(
    "repository",
    [
        "owner",
        "owner/",
        "/repository",
        "owner/repository/extra",
        "owner repository/name",
        "owner/name repository",
    ],
)
def test_settings_reject_invalid_github_repository(
    monkeypatch: pytest.MonkeyPatch,
    repository: str,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv(
        "GITHUB_REPOSITORY",
        repository,
    )

    with pytest.raises(
        SettingsError,
        match="GITHUB_REPOSITORY",
    ):
        Settings.from_environment()


def test_settings_strip_github_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "   secret-token-123   ",
    )

    settings = Settings.from_environment()

    assert settings.github_token == "secret-token-123"


def test_settings_error_does_not_expose_github_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_token = "super-secret-github-token"

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv(
        "GITHUB_TOKEN",
        secret_token,
    )
    monkeypatch.setenv(
        "GITHUB_REPOSITORY",
        "invalid repository/name",
    )

    with pytest.raises(SettingsError) as exc_info:
        Settings.from_environment()

    assert secret_token not in str(exc_info.value)


def test_settings_are_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")

    settings = Settings.from_environment()

    with pytest.raises(FrozenInstanceError):
        settings.llm_model = "different-model"


def test_settings_cannot_change_github_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv(
        "GITHUB_TOKEN",
        "secret-token",
    )

    settings = Settings.from_environment()

    with pytest.raises(FrozenInstanceError):
        settings.github_token = "new-secret-token"