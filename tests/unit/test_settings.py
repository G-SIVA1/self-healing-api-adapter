from __future__ import annotations

import pytest

from self_healing_agent.config.settings import (
    Settings,
    SettingsError,
)


def test_settings_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openai",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "test-model",
    )

    monkeypatch.setenv(
        "LLM_TIMEOUT_SECONDS",
        "45",
    )

    monkeypatch.setenv(
        "LLM_MAX_RETRIES",
        "3",
    )

    settings = Settings.from_environment()

    assert settings.llm_provider == "openai"

    assert settings.llm_model == "test-model"

    assert settings.llm_timeout_seconds == 45.0

    assert settings.llm_max_retries == 3


def test_settings_use_default_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openai",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "test-model",
    )

    monkeypatch.delenv(
        "LLM_TIMEOUT_SECONDS",
        raising=False,
    )

    monkeypatch.delenv(
        "LLM_MAX_RETRIES",
        raising=False,
    )

    settings = Settings.from_environment()

    assert settings.llm_timeout_seconds == 30.0

    assert settings.llm_max_retries == 2


def test_settings_reject_empty_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "test-model",
    )

    with pytest.raises(
        SettingsError,
        match="LLM_PROVIDER",
    ):
        Settings.from_environment()


def test_settings_reject_empty_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openai",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "",
    )

    with pytest.raises(
        SettingsError,
        match="LLM_MODEL",
    ):
        Settings.from_environment()


def test_settings_reject_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openai",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "test-model",
    )

    monkeypatch.setenv(
        "LLM_TIMEOUT_SECONDS",
        "invalid",
    )

    with pytest.raises(
        SettingsError,
        match="LLM_TIMEOUT_SECONDS",
    ):
        Settings.from_environment()


def test_settings_reject_negative_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openai",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "test-model",
    )

    monkeypatch.setenv(
        "LLM_MAX_RETRIES",
        "-1",
    )

    with pytest.raises(
        SettingsError,
        match="LLM_MAX_RETRIES",
    ):
        Settings.from_environment()