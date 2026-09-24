from __future__ import annotations

from datetime import datetime, timezone

import pytest

from self_healing_agent.agent.history import RepairAttempt
from self_healing_agent.agent.history_store import (
    RepairHistoryStore,
    RepairHistoryStoreError,
)


def create_attempt(
    iteration: int = 0,
    test_passed: bool = True,
    status: str = "success",
) -> RepairAttempt:
    return RepairAttempt(
        iteration=iteration,
        timestamp=datetime.now(timezone.utc),
        target_file="examples/broken_api_client.py",
        exception_type="AttributeError",
        error_message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
        model="gemini-3.6-flash",
        explanation="Replace deprecated Stripe API.",
        confidence=0.95,
        original_code=(
            "stripe.Customer.create(email=email)"
        ),
        generated_code=(
            "stripe.customers.create(email=email)"
        ),
        validation_command=(
            "python",
            "-m",
            "pytest",
            "-q",
        ),
        test_passed=test_passed,
        test_output=(
            "1 passed"
            if test_passed
            else "1 failed"
        ),
        failure_type=(
            None
            if test_passed
            else "assertion"
        ),
        failure_summary=(
            None
            if test_passed
            else "Validation failed"
        ),
        failure_details=(
            None
            if test_passed
            else "Expected output did not match"
        ),
        status=status,
    )


def test_store_creates_database(tmp_path) -> None:
    database_path = tmp_path / "history.db"

    store = RepairHistoryStore(database_path)

    assert store.database_path == database_path
    assert database_path.exists()


def test_save_returns_database_id(tmp_path) -> None:
    store = RepairHistoryStore(
        tmp_path / "history.db"
    )

    attempt = create_attempt()

    attempt_id = store.save(attempt)

    assert attempt_id == 1
    assert store.count() == 1


def test_get_returns_saved_attempt(tmp_path) -> None:
    store = RepairHistoryStore(
        tmp_path / "history.db"
    )

    attempt = create_attempt()

    attempt_id = store.save(attempt)

    restored = store.get(attempt_id)

    assert restored is not None

    assert restored.iteration == attempt.iteration
    assert restored.timestamp == attempt.timestamp
    assert restored.target_file == attempt.target_file
    assert restored.exception_type == attempt.exception_type
    assert restored.error_message == attempt.error_message
    assert restored.model == attempt.model
    assert restored.explanation == attempt.explanation
    assert restored.confidence == attempt.confidence
    assert restored.original_code == attempt.original_code
    assert restored.generated_code == attempt.generated_code
    assert (
        restored.validation_command
        == attempt.validation_command
    )
    assert restored.test_passed == attempt.test_passed
    assert restored.test_output == attempt.test_output
    assert restored.failure_type == attempt.failure_type
    assert (
        restored.failure_summary
        == attempt.failure_summary
    )
    assert (
        restored.failure_details
        == attempt.failure_details
    )
    assert restored.status == attempt.status


def test_get_returns_none_for_unknown_id(tmp_path) -> None:
    store = RepairHistoryStore(
        tmp_path / "history.db"
    )

    assert store.get(999) is None


def test_get_rejects_invalid_id(tmp_path) -> None:
    store = RepairHistoryStore(
        tmp_path / "history.db"
    )

    with pytest.raises(
        ValueError,
        match="attempt_id must be greater than zero",
    ):
        store.get(0)


def test_list_attempts_preserves_insertion_order(
    tmp_path,
) -> None:
    store = RepairHistoryStore(
        tmp_path / "history.db"
    )

    first = create_attempt(iteration=0)
    second = create_attempt(
        iteration=1,
        test_passed=False,
        status="failed",
    )
    third = create_attempt(iteration=2)

    store.save(first)
    store.save(second)
    store.save(third)

    attempts = store.list_attempts()

    assert len(attempts) == 3
    assert attempts[0].iteration == 0
    assert attempts[1].iteration == 1
    assert attempts[2].iteration == 2


def test_count_returns_zero_for_empty_store(
    tmp_path,
) -> None:
    store = RepairHistoryStore(
        tmp_path / "history.db"
    )

    assert store.count() == 0


def test_clear_removes_all_attempts(tmp_path) -> None:
    store = RepairHistoryStore(
        tmp_path / "history.db"
    )

    store.save(create_attempt())
    store.save(create_attempt(iteration=1))

    assert store.count() == 2

    store.clear()

    assert store.count() == 0
    assert store.list_attempts() == []


def test_failed_attempt_is_persisted_correctly(
    tmp_path,
) -> None:
    store = RepairHistoryStore(
        tmp_path / "history.db"
    )

    attempt = create_attempt(
        iteration=2,
        test_passed=False,
        status="failed",
    )

    attempt_id = store.save(attempt)

    restored = store.get(attempt_id)

    assert restored is not None
    assert restored.test_passed is False
    assert restored.status == "failed"
    assert restored.failure_type == "assertion"
    assert restored.failure_summary == "Validation failed"
    assert (
        restored.failure_details
        == "Expected output did not match"
    )


def test_history_survives_store_recreation(
    tmp_path,
) -> None:
    database_path = tmp_path / "history.db"

    first_store = RepairHistoryStore(
        database_path
    )

    attempt = create_attempt()

    attempt_id = first_store.save(attempt)

    second_store = RepairHistoryStore(
        database_path
    )

    restored = second_store.get(attempt_id)

    assert restored is not None
    assert restored.target_file == attempt.target_file
    assert restored.test_passed is True
    assert second_store.count() == 1


def test_multiple_stores_share_same_database(
    tmp_path,
) -> None:
    database_path = tmp_path / "history.db"

    first_store = RepairHistoryStore(
        database_path
    )

    second_store = RepairHistoryStore(
        database_path
    )

    first_store.save(create_attempt())

    assert second_store.count() == 1

    second_store.save(
        create_attempt(iteration=1)
    )

    assert first_store.count() == 2


def test_store_error_type_exists() -> None:
    assert issubclass(
        RepairHistoryStoreError,
        Exception,
    )