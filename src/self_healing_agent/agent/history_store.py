from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from self_healing_agent.agent.history import RepairAttempt


class RepairHistoryStoreError(Exception):
    """Raised when repair history persistence fails."""


class RepairHistoryStore:
    """Persist and retrieve repair attempts using SQLite."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

        try:
            self._database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self._initialize_database()
        except OSError as exc:
            raise RepairHistoryStoreError(
                f"Unable to initialize history database directory: "
                f"{self._database_path}"
            ) from exc
        except sqlite3.Error as exc:
            raise RepairHistoryStoreError(
                f"Unable to initialize history database: "
                f"{self._database_path}"
            ) from exc

    @property
    def database_path(self) -> Path:
        """Return the configured SQLite database path."""
        return self._database_path

    def save(self, attempt: RepairAttempt) -> int:
        """Persist one repair attempt and return its database ID."""

        payload = attempt.to_dict()

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO repair_attempts (
                        iteration,
                        timestamp,
                        target_file,
                        exception_type,
                        error_message,
                        model,
                        explanation,
                        confidence,
                        original_code,
                        generated_code,
                        validation_command,
                        test_passed,
                        test_output,
                        failure_type,
                        failure_summary,
                        failure_details,
                        status
                    )
                    VALUES (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        payload["iteration"],
                        payload["timestamp"],
                        payload["target_file"],
                        payload["exception_type"],
                        payload["error_message"],
                        payload["model"],
                        payload["explanation"],
                        payload["confidence"],
                        payload["original_code"],
                        payload["generated_code"],
                        json.dumps(
                            payload["validation_command"]
                        ),
                        int(payload["test_passed"]),
                        payload["test_output"],
                        payload["failure_type"],
                        payload["failure_summary"],
                        payload["failure_details"],
                        payload["status"],
                    ),
                )

                attempt_id = cursor.lastrowid

                if attempt_id is None:
                    raise RepairHistoryStoreError(
                        "SQLite did not return an attempt ID"
                    )

                return int(attempt_id)

        except sqlite3.Error as exc:
            raise RepairHistoryStoreError(
                "Unable to save repair attempt"
            ) from exc

    def get(self, attempt_id: int) -> RepairAttempt | None:
        """Retrieve one repair attempt by database ID."""

        if attempt_id <= 0:
            raise ValueError(
                "attempt_id must be greater than zero"
            )

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT
                        iteration,
                        timestamp,
                        target_file,
                        exception_type,
                        error_message,
                        model,
                        explanation,
                        confidence,
                        original_code,
                        generated_code,
                        validation_command,
                        test_passed,
                        test_output,
                        failure_type,
                        failure_summary,
                        failure_details,
                        status
                    FROM repair_attempts
                    WHERE id = ?
                    """,
                    (attempt_id,),
                ).fetchone()

        except sqlite3.Error as exc:
            raise RepairHistoryStoreError(
                f"Unable to retrieve repair attempt {attempt_id}"
            ) from exc

        if row is None:
            return None

        return self._row_to_attempt(row)

    def list_attempts(self) -> list[RepairAttempt]:
        """Return all repair attempts ordered by insertion."""

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        iteration,
                        timestamp,
                        target_file,
                        exception_type,
                        error_message,
                        model,
                        explanation,
                        confidence,
                        original_code,
                        generated_code,
                        validation_command,
                        test_passed,
                        test_output,
                        failure_type,
                        failure_summary,
                        failure_details,
                        status
                    FROM repair_attempts
                    ORDER BY id ASC
                    """
                ).fetchall()

        except sqlite3.Error as exc:
            raise RepairHistoryStoreError(
                "Unable to retrieve repair history"
            ) from exc

        return [
            self._row_to_attempt(row)
            for row in rows
        ]

    def count(self) -> int:
        """Return the total number of persisted repair attempts."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM repair_attempts
                    """
                ).fetchone()

        except sqlite3.Error as exc:
            raise RepairHistoryStoreError(
                "Unable to count repair attempts"
            ) from exc

        if row is None:
            return 0

        return int(row[0])

    def clear(self) -> None:
        """Delete all persisted repair attempts."""

        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM repair_attempts"
                )

        except sqlite3.Error as exc:
            raise RepairHistoryStoreError(
                "Unable to clear repair history"
            ) from exc

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    target_file TEXT NOT NULL,
                    exception_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    model TEXT,
                    explanation TEXT,
                    confidence REAL,
                    original_code TEXT,
                    generated_code TEXT,
                    validation_command TEXT NOT NULL,
                    test_passed INTEGER NOT NULL,
                    test_output TEXT,
                    failure_type TEXT,
                    failure_summary TEXT,
                    failure_details TEXT,
                    status TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path,
            timeout=10.0,
        )

        connection.row_factory = sqlite3.Row

        return connection

    @staticmethod
    def _row_to_attempt(
        row: sqlite3.Row,
    ) -> RepairAttempt:
        try:
            validation_command_raw: Any = json.loads(
                row["validation_command"]
            )

            if not isinstance(
                validation_command_raw,
                list,
            ):
                raise RepairHistoryStoreError(
                    "Stored validation command is invalid"
                )

            validation_command = tuple(
                str(item)
                for item in validation_command_raw
            )

            from datetime import datetime

            timestamp = datetime.fromisoformat(
                row["timestamp"]
            )

            return RepairAttempt(
                iteration=int(row["iteration"]),
                timestamp=timestamp,
                target_file=str(row["target_file"]),
                exception_type=str(
                    row["exception_type"]
                ),
                error_message=str(
                    row["error_message"]
                ),
                model=row["model"],
                explanation=row["explanation"],
                confidence=(
                    float(row["confidence"])
                    if row["confidence"] is not None
                    else None
                ),
                original_code=row["original_code"],
                generated_code=row["generated_code"],
                validation_command=validation_command,
                test_passed=bool(row["test_passed"]),
                test_output=row["test_output"],
                failure_type=row["failure_type"],
                failure_summary=row["failure_summary"],
                failure_details=row["failure_details"],
                status=str(row["status"]),
            )

        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RepairHistoryStoreError(
                "Stored repair attempt contains invalid data"
            ) from exc