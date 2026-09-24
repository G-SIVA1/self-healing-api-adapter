from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class RepairAttempt:
    """Immutable record describing one repair attempt."""

    iteration: int
    timestamp: datetime
    target_file: str
    exception_type: str
    error_message: str

    model: str | None
    explanation: str | None
    confidence: float | None

    original_code: str | None
    generated_code: str | None

    validation_command: tuple[str, ...]
    test_passed: bool
    test_output: str | None

    failure_type: str | None
    failure_summary: str | None
    failure_details: str | None

    status: str

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(
                "iteration cannot be negative"
            )

        if not self.target_file.strip():
            raise ValueError(
                "target_file cannot be empty"
            )

        if not self.exception_type.strip():
            raise ValueError(
                "exception_type cannot be empty"
            )

        if not self.error_message.strip():
            raise ValueError(
                "error_message cannot be empty"
            )

        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0 and 1"
                )

        if not self.status.strip():
            raise ValueError(
                "status cannot be empty"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the repair attempt into JSON-compatible data."""

        data = asdict(self)

        data["timestamp"] = self.timestamp.isoformat()

        data["validation_command"] = list(
            self.validation_command
        )

        return data


class RepairHistory:
    """In-memory history of repair attempts."""

    def __init__(self) -> None:
        self._attempts: list[RepairAttempt] = []

    def add(self, attempt: RepairAttempt) -> None:
        """Add a completed repair attempt."""

        self._attempts.append(attempt)

    @property
    def attempts(self) -> tuple[RepairAttempt, ...]:
        """Return an immutable view of recorded attempts."""

        return tuple(self._attempts)

    @property
    def count(self) -> int:
        """Return the number of recorded attempts."""

        return len(self._attempts)

    @property
    def latest(self) -> RepairAttempt | None:
        """Return the most recent attempt."""

        if not self._attempts:
            return None

        return self._attempts[-1]

    @property
    def succeeded(self) -> bool:
        """Return whether the latest attempt succeeded."""

        latest = self.latest

        if latest is None:
            return False

        return latest.test_passed

    def clear(self) -> None:
        """Clear all recorded attempts."""

        self._attempts.clear()

    def to_list(self) -> list[dict[str, Any]]:
        """Convert the complete history into serializable data."""

        return [
            attempt.to_dict()
            for attempt in self._attempts
        ]

    def total_successes(self) -> int:
        """Return the number of successful attempts."""

        return sum(
            attempt.test_passed
            for attempt in self._attempts
        )

    def total_failures(self) -> int:
        """Return the number of failed attempts."""

        return sum(
            not attempt.test_passed
            for attempt in self._attempts
        )


def create_attempt_timestamp() -> datetime:
    """Create a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)