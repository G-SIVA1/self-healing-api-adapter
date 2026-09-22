from __future__ import annotations

from datetime import datetime, timezone

from self_healing_agent.agent.state import AgentState
from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.rag.chunker import DocumentChunk


def create_error_event() -> ErrorEvent:
    return ErrorEvent(
        timestamp=datetime.now(timezone.utc),
        exception_type="AttributeError",
        message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
        file_path="api_client.py",
        line_number=42,
        function_name="create_customer",
        api_service="stripe",
        api_call="stripe.Customer.create",
        raw_log="Stripe API failure",
    )


def test_agent_state_stores_error_event() -> None:
    event = create_error_event()

    state = AgentState(
        error_event=event,
    )

    assert state.error_event is event
    assert state.documentation == []
    assert state.generated_code is None
    assert state.test_passed is False
    assert state.iteration == 0
    assert state.repair_complete is False


def test_agent_state_adds_documentation() -> None:
    state = AgentState(
        error_event=create_error_event(),
    )

    chunks = [
        DocumentChunk(
            content="Use stripe.customers.create.",
            source="stripe.md",
            chunk_index=0,
        )
    ]

    state.add_documentation(chunks)

    assert len(state.documentation) == 1
    assert (
        state.documentation[0].content
        == "Use stripe.customers.create."
    )


def test_agent_state_records_successful_test() -> None:
    state = AgentState(
        error_event=create_error_event(),
    )

    state.record_test_result(
        passed=True,
        output="1 passed",
    )

    assert state.test_passed is True
    assert state.test_output == "1 passed"
    assert state.repair_complete is True


def test_agent_state_records_failed_test() -> None:
    state = AgentState(
        error_event=create_error_event(),
    )

    state.record_test_result(
        passed=False,
        output="AssertionError",
    )

    assert state.test_passed is False
    assert state.test_output == "AssertionError"
    assert state.repair_complete is False


def test_agent_state_tracks_iterations() -> None:
    state = AgentState(
        error_event=create_error_event(),
        max_iterations=2,
    )

    state.increment_iteration()

    assert state.iteration == 1
    assert state.failure_reason is None

    state.increment_iteration()

    assert state.iteration == 2
    assert (
        state.failure_reason
        == "Maximum repair iterations reached"
    )