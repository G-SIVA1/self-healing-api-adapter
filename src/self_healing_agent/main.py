from __future__ import annotations

import sys

from self_healing_agent.config.settings import (
    Settings,
    SettingsError,
)


def load_settings() -> Settings:
    """Load and validate application settings."""

    return Settings.from_environment()


def main() -> int:
    """Start the self-healing agent application."""

    try:
        settings = load_settings()
    except SettingsError as exc:
        print(
            f"Configuration error: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        "Self-Healing API Adapter Agent started "
        f"with provider '{settings.llm_provider}' "
        f"and model '{settings.llm_model}'."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())