from __future__ import annotations

from unittest.mock import patch

from self_healing_agent.main import (
    load_settings,
    main,
)


def test_load_settings_uses_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "gemini",
    )
    monkeypatch.setenv(
        "LLM_MODEL",
        "gemini-3.6-flash",
    )

    settings = load_settings()

    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-3.6-flash"


def test_main_returns_zero_for_valid_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "gemini",
    )
    monkeypatch.setenv(
        "LLM_MODEL",
        "gemini-3.6-flash",
    )

    result = main()

    assert result == 0


def test_main_returns_one_for_invalid_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "",
    )
    monkeypatch.setenv(
        "LLM_MODEL",
        "gemini-3.6-flash",
    )

    result = main()

    assert result == 1


def test_main_does_not_expose_github_token(
    monkeypatch,
    capsys,
) -> None:
    secret = "super-secret-github-token"

    monkeypatch.setenv(
        "LLM_PROVIDER",
        "",
    )
    monkeypatch.setenv(
        "LLM_MODEL",
        "gemini-3.6-flash",
    )
    monkeypatch.setenv(
        "GITHUB_TOKEN",
        secret,
    )

    main()

    captured = capsys.readouterr()

    assert secret not in captured.out
    assert secret not in captured.err


def test_main_uses_load_settings(
    monkeypatch,
) -> None:
    with patch(
        "self_healing_agent.main.load_settings"
    ) as mocked_load_settings:
        mocked_load_settings.return_value.llm_provider = (
            "gemini"
        )
        mocked_load_settings.return_value.llm_model = (
            "gemini-3.6-flash"
        )

        result = main()

    assert result == 0
    mocked_load_settings.assert_called_once()