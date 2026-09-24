from __future__ import annotations

from datetime import datetime, timezone

import pytest

from self_healing_agent.agent.history import (
    RepairAttempt,
    RepairHistory,
    create_attempt_timestamp,
)


def create_attempt(
    *,
    iteration: int = 1,
    test_passed: bool = True,
    status: str = "success",
) -> RepairAttempt:
    return RepairAttempt(
        iteration=iteration,
        timestamp=datetime(
            2026,
            9,
            23,
            10,
            30,
            tzinfo=timezone.utc,
        ),
        target_file="examples/broken_api_client.py",
        exception_type="AttributeError",
        error_message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
        model="gemini-3.6-flash",
        explanation=(
            "Replace the deprecated Stripe Customer API."
        ),
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
            else "Expected repaired API call"
        ),
        status=status,
    )


def test_repair_attempt_accepts_valid_data() -> None:
    attempt = create_attempt()

    assert attempt.iteration == 1
    assert (
        attempt.target_file
        == "examples/broken_api_client.py"
    )
    assert attempt.model == "gemini-3.6-flash"
    assert attempt.test_passed is True
    assert attempt.status == "success"


def test_repair_attempt_is_immutable() -> None:
    attempt = create_attempt()

    with pytest.raises(AttributeError):
        attempt.status = "failed"  # type: ignore[misc]


def test_repair_attempt_rejects_negative_iteration() -> None:
    with pytest.raises(
        ValueError,
        match="iteration cannot be negative",
    ):
        create_attempt(iteration=-1)


def test_repair_attempt_rejects_empty_target_file() -> None:
    with pytest.raises(
        ValueError,
        match="target_file cannot be empty",
    ):
        RepairAttempt(
            iteration=1,
            timestamp=create_attempt_timestamp(),
            target_file="",
            exception_type="AttributeError",
            error_message="API failure",
            model=None,
            explanation=None,
            confidence=None,
            original_code=None,
            generated_code=None,
            validation_command=(
                "python",
                "-m",
                "pytest",
            ),
            test_passed=False,
            test_output=None,
            failure_type=None,
            failure_summary=None,
            failure_details=None,
            status="failed",
        )


def test_repair_attempt_rejects_invalid_confidence() -> None:
    with pytest.raises(
        ValueError,
        match="confidence must be between 0 and 1",
    ):
        RepairAttempt(
            iteration=1,
            timestamp=create_attempt_timestamp(),
            target_file="example.py",
            exception_type="ValueError",
            error_message="invalid value",
            model="test-model",
            explanation="repair",
            confidence=1.5,
            original_code="old",
            generated_code="new",
            validation_command=("pytest",),
            test_passed=False,
            test_output="failed",
            failure_type="assertion",
            failure_summary="failed",
            failure_details="details",
            status="failed",
        )


def test_repair_attempt_to_dict_serializes_timestamp() -> None:
    attempt = create_attempt()

    data = attempt.to_dict()

    assert isinstance(data, dict)
    assert data["timestamp"] == (
        "2026-09-23T10:30:00+00:00"
    )
    assert data["validation_command"] == [
        "python",
        "-m",
        "pytest",
        "-q",
    ]
    assert data["test_passed"] is True


def test_empty_history_has_expected_state() -> None:
    history = RepairHistory()

    assert history.count == 0
    assert history.latest is None
    assert history.succeeded is False
    assert history.attempts == ()
    assert history.total_successes() == 0
    assert history.total_failures() == 0
    assert history.to_list() == []


def test_history_adds_attempts() -> None:
    history = RepairHistory()

    first = create_attempt(
        iteration=1,
        test_passed=False,
        status="failed",
    )
    second = create_attempt(
        iteration=2,
        test_passed=True,
        status="success",
    )

    history.add(first)
    history.add(second)

    assert history.count == 2
    assert history.attempts == (
        first,
        second,
    )
    assert history.latest == second


def test_history_tracks_successes_and_failures() -> None:
    history = RepairHistory()

    history.add(
        create_attempt(
            iteration=1,
            test_passed=False,
            status="failed",
        )
    )

    history.add(
        create_attempt(
            iteration=2,
            test_passed=False,
            status="failed",
        )
    )

    history.add(
        create_attempt(
            iteration=3,
            test_passed=True,
            status="success",
        )
    )

    assert history.count == 3
    assert history.total_successes() == 1
    assert history.total_failures() == 2
    assert history.succeeded is True


def test_history_to_list_returns_serializable_records() -> None:
    history = RepairHistory()

    first = create_attempt(
        iteration=1,
        test_passed=False,
        status="failed",
    )
    second = create_attempt(
        iteration=2,
        test_passed=True,
        status="success",
    )

    history.add(first)
    history.add(second)

    records = history.to_list()

    assert len(records) == 2
    assert records[0]["iteration"] == 1
    assert records[0]["test_passed"] is False
    assert records[1]["iteration"] == 2
    assert records[1]["test_passed"] is True


def test_history_clear_removes_all_attempts() -> None:
    history = RepairHistory()

    history.add(create_attempt())

    assert history.count == 1

    history.clear()

    assert history.count == 0
    assert history.latest is None
    assert history.attempts == ()


def test_create_attempt_timestamp_is_timezone_aware() -> None:
    timestamp = create_attempt_timestamp()

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() is not None