from __future__ import annotations

from datetime import datetime, timezone

from self_healing_agent.agent.context import (
    ReasoningContextBuilder,
)
from self_healing_agent.agent.history import RepairAttempt
from self_healing_agent.agent.state import AgentState
from self_healing_agent.models.events import ErrorEvent


def create_attempt(
    iteration: int = 0,
    test_passed: bool = True,
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
            else "Test failed"
        ),
        status=(
            "success"
            if test_passed
            else "failed"
        ),
    )


def create_state() -> AgentState:
    event = ErrorEvent(
        timestamp=datetime.now(timezone.utc),
        exception_type="AttributeError",
        message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
        file_path="examples/broken_api_client.py",
        line_number=5,
        function_name="create_customer",
        api_service="stripe",
        api_call="stripe.Customer.create",
        raw_log=(
            "AttributeError: module 'stripe' "
            "has no attribute 'Customer'"
        ),
    )

    return AgentState(
        error_event=event,
        max_iterations=3,
    )


def test_context_accepts_historical_repairs() -> None:
    builder = ReasoningContextBuilder()

    historical_repairs = [
        create_attempt(iteration=0),
        create_attempt(iteration=1),
    ]

    formatted_history = (
        builder.format_historical_repairs(
            tuple(historical_repairs)
        )
    )

    assert "Historical Repair 1" in formatted_history
    assert "Historical Repair 2" in formatted_history
    assert "AttributeError" in formatted_history
    assert (
        "examples/broken_api_client.py"
        in formatted_history
    )
    assert "stripe.Customer.create" in formatted_history
    assert (
        "stripe.customers.create"
        in formatted_history
    )


def test_context_formats_failed_historical_repair() -> None:
    builder = ReasoningContextBuilder()

    attempt = create_attempt(
        iteration=2,
        test_passed=False,
    )

    formatted_history = (
        builder.format_historical_repairs(
            (attempt,)
        )
    )

    assert "Historical Repair 1" in formatted_history
    assert "Status: failed" in formatted_history
    assert "Test passed: False" in formatted_history
    assert "1 failed" in formatted_history


def test_context_formats_empty_history() -> None:
    builder = ReasoningContextBuilder()

    formatted_history = (
        builder.format_historical_repairs(())
    )

    assert (
        formatted_history
        == "No previous repair history available."
    )


def test_context_builds_with_historical_repairs() -> None:
    builder = ReasoningContextBuilder()
    state = create_state()

    historical_repairs = [
        create_attempt(iteration=0),
    ]

    context = builder.build(
        state,
        current_source_code="current source",
        historical_repairs=historical_repairs,
    )

    assert len(context.historical_repairs) == 1

    assert (
        context.historical_repairs[0].iteration
        == 0
    )

    assert (
        context.historical_repairs[0].target_file
        == "examples/broken_api_client.py"
    )


def test_context_history_is_immutable() -> None:
    builder = ReasoningContextBuilder()
    state = create_state()

    historical_repairs = [
        create_attempt(iteration=0),
    ]

    context = builder.build(
        state,
        historical_repairs=historical_repairs,
    )

    assert isinstance(
        context.historical_repairs,
        tuple,
    )

    assert len(context.historical_repairs) == 1