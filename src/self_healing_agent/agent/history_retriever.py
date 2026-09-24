from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.agent.history import RepairAttempt
from self_healing_agent.agent.history_store import RepairHistoryStore


@dataclass(frozen=True, slots=True)
class HistoricalRepair:
    """A previously recorded repair relevant to a current failure."""

    attempt: RepairAttempt
    relevance_reason: str


class RepairHistoryRetriever:
    """Retrieve previous repair attempts relevant to an error."""

    def __init__(
        self,
        history_store: RepairHistoryStore,
    ) -> None:
        self._history_store = history_store

    def find_by_target_file(
        self,
        target_file: str,
        limit: int = 5,
    ) -> list[HistoricalRepair]:
        """Find previous repairs for the same target file."""

        if not target_file.strip():
            raise ValueError(
                "target_file cannot be empty"
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        attempts = self._history_store.list_attempts()

        matching_attempts = [
            attempt
            for attempt in attempts
            if attempt.target_file == target_file
        ]

        matching_attempts = matching_attempts[-limit:]

        return [
            HistoricalRepair(
                attempt=attempt,
                relevance_reason=(
                    "Previous repair targeted the same file."
                ),
            )
            for attempt in reversed(matching_attempts)
        ]

    def find_by_exception(
        self,
        exception_type: str,
        limit: int = 5,
    ) -> list[HistoricalRepair]:
        """Find previous repairs with the same exception type."""

        if not exception_type.strip():
            raise ValueError(
                "exception_type cannot be empty"
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        attempts = self._history_store.list_attempts()

        matching_attempts = [
            attempt
            for attempt in attempts
            if attempt.exception_type
            == exception_type
        ]

        matching_attempts = matching_attempts[-limit:]

        return [
            HistoricalRepair(
                attempt=attempt,
                relevance_reason=(
                    "Previous repair encountered the "
                    "same exception type."
                ),
            )
            for attempt in reversed(matching_attempts)
        ]

    def find_similar(
        self,
        target_file: str,
        exception_type: str,
        limit: int = 5,
    ) -> list[HistoricalRepair]:
        """Find previous repairs matching file and exception."""

        if not target_file.strip():
            raise ValueError(
                "target_file cannot be empty"
            )

        if not exception_type.strip():
            raise ValueError(
                "exception_type cannot be empty"
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        attempts = self._history_store.list_attempts()

        exact_matches = [
            attempt
            for attempt in attempts
            if (
                attempt.target_file == target_file
                and attempt.exception_type
                == exception_type
            )
        ]

        exact_matches = exact_matches[-limit:]

        return [
            HistoricalRepair(
                attempt=attempt,
                relevance_reason=(
                    "Previous repair matches both the "
                    "target file and exception type."
                ),
            )
            for attempt in reversed(exact_matches)
        ]

    def find_successful(
        self,
        target_file: str,
        exception_type: str,
        limit: int = 5,
    ) -> list[HistoricalRepair]:
        """Find successful previous repairs for the same failure."""

        results = self.find_similar(
            target_file=target_file,
            exception_type=exception_type,
            limit=limit,
        )

        return [
            historical_repair
            for historical_repair in results
            if historical_repair.attempt.test_passed
        ]