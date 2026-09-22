from __future__ import annotations

from datetime import datetime, timezone

from self_healing_agent.agent.decision import (
    DecisionAction,
    RetryDecisionEngine,
)
from self_healing_agent.agent.failure_analyzer import (
    FailureAnalysis,
)
from self_healing_agent.agent.state import AgentState
from self_healing_agent.models.events import ErrorEvent


def create_state() -> AgentState:
    """Create a basic agent state for decision tests."""

    return AgentState(
        error_event=ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            exception_type="AttributeError",
            message="API failure",
            file_path="api_client.py",
            line_number=10,
            function_name="create_customer",
            api_service="stripe",
            api_call="stripe.Customer.create",
            raw_log="API failure",
        )
    )


def test_successful_repair_should_stop() -> None:
    state = create_state()

    state.repair_complete = True

    decision = RetryDecisionEngine().decide(state)

    assert decision.action == DecisionAction.STOP
    assert "completed successfully" in decision.reason


def test_retryable_failure_should_retry() -> None:
    state = create_state()

    state.iteration = 1

    state.failure_analysis = FailureAnalysis(
        failure_type="assertion_error",
        summary="A test assertion failed.",
        details="Expected NEW_API but got WRONG_API.",
        retryable=True,
    )

    decision = RetryDecisionEngine().decide(state)

    assert decision.action == DecisionAction.RETRY
    assert "retryable" in decision.reason


def test_non_retryable_failure_should_escalate() -> None:
    state = create_state()

    state.iteration = 1

    state.failure_analysis = FailureAnalysis(
        failure_type="import_error",
        summary="Import failed.",
        details="No module named stripe.",
        retryable=False,
    )

    decision = RetryDecisionEngine().decide(state)

    assert decision.action == DecisionAction.ESCALATE
    assert "not considered automatically retryable" in (
        decision.reason
    )


def test_max_iterations_should_escalate() -> None:
    state = create_state()

    state.iteration = 5
    state.max_iterations = 5

    state.failure_analysis = FailureAnalysis(
        failure_type="assertion_error",
        summary="Test failed.",
        details="Still failing.",
        retryable=True,
    )

    decision = RetryDecisionEngine().decide(state)

    assert decision.action == DecisionAction.ESCALATE
    assert "Maximum repair iterations" in decision.reason


def test_missing_failure_analysis_should_retry() -> None:
    state = create_state()

    state.iteration = 1

    decision = RetryDecisionEngine().decide(state)

    assert decision.action == DecisionAction.RETRY
    assert "No structured failure analysis" in (
        decision.reason
    )