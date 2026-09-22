from __future__ import annotations

from datetime import datetime, timezone

from self_healing_agent.agent.context import (
    ReasoningContextBuilder,
)
from self_healing_agent.agent.failure_analyzer import (
    FailureAnalysis,
)
from self_healing_agent.agent.state import AgentState
from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.rag.chunker import DocumentChunk


def create_state() -> AgentState:
    """Create an AgentState for context tests."""

    return AgentState(
        error_event=ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            exception_type="AttributeError",
            message=(
                "module 'stripe' has no attribute 'Customer'"
            ),
            file_path="api_client.py",
            line_number=6,
            function_name="create_customer",
            api_service="stripe",
            api_call="stripe.Customer.create",
            raw_log="API failure",
        )
    )


def test_context_builder_collects_error_information() -> None:
    state = create_state()

    context = ReasoningContextBuilder().build(
        state,
    )

    assert context.exception_type == "AttributeError"

    assert (
        context.error_message
        == "module 'stripe' has no attribute 'Customer'"
    )

    assert context.target_file == "api_client.py"

    assert context.target_line == 6

    assert context.function_name == "create_customer"

    assert context.api_service == "stripe"

    assert context.api_call == (
        "stripe.Customer.create"
    )


def test_context_builder_collects_documentation() -> None:
    state = create_state()

    state.add_documentation(
        [
            DocumentChunk(
                content=(
                    "Replace "
                    "stripe.Customer.create with "
                    "stripe.customers.create."
                ),
                source="stripe.md",
                chunk_index=0,
            ),
            DocumentChunk(
                content=(
                    "The legacy Customer API is "
                    "deprecated."
                ),
                source="stripe.md",
                chunk_index=1,
            ),
        ]
    )

    context = ReasoningContextBuilder().build(
        state,
    )

    assert (
        "stripe.Customer.create"
        in context.documentation
    )

    assert (
        "stripe.customers.create"
        in context.documentation
    )

    assert (
        "The legacy Customer API"
        in context.documentation
    )


def test_context_builder_collects_current_source() -> None:
    state = create_state()

    source_code = """\
import stripe


def create_customer():
    return stripe.Customer.create()
"""

    context = ReasoningContextBuilder().build(
        state,
        current_source_code=source_code,
    )

    assert (
        context.current_source_code
        == source_code
    )


def test_context_builder_collects_previous_attempt() -> None:
    state = create_state()

    state.generated_code = """\
return stripe.customers.create()
"""

    state.test_output = (
        "AssertionError: API call failed"
    )

    state.failure_analysis = FailureAnalysis(
        failure_type="assertion_error",
        summary="A test assertion failed.",
        details="API call failed.",
        retryable=True,
    )

    state.iteration = 2

    context = ReasoningContextBuilder().build(
        state,
    )

    assert (
        context.previous_generated_code
        == state.generated_code
    )

    assert (
        context.test_output
        == "AssertionError: API call failed"
    )

    assert context.failure_analysis is not None

    assert (
        context.failure_analysis.failure_type
        == "assertion_error"
    )

    assert context.iteration == 2

    assert context.max_iterations == 5


def test_context_builder_prefers_current_source() -> None:
    state = create_state()

    state.original_code = "ORIGINAL"

    state.generated_code = "GENERATED"

    current_source = "CURRENT"

    context = ReasoningContextBuilder().build(
        state,
        current_source_code=current_source,
    )

    assert context.current_source_code == "CURRENT"