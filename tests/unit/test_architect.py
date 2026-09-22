from __future__ import annotations

from datetime import datetime, timezone

import pytest

from self_healing_agent.agent.architect import CodeArchitect
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


def create_state() -> AgentState:
    state = AgentState(
        error_event=create_error_event(),
    )

    state.add_documentation(
        [
            DocumentChunk(
                content=(
                    "The legacy API "
                    "stripe.Customer.create "
                    "has been replaced. Use "
                    "stripe.customers.create instead."
                ),
                source="stripe.md",
                chunk_index=0,
            )
        ]
    )

    return state


def test_architect_creates_repair_proposal() -> None:
    state = create_state()

    architect = CodeArchitect()

    proposal = architect.create_proposal(state)

    assert proposal.target_file == "api_client.py"
    assert proposal.target_line == 42

    assert (
        proposal.original_api_call
        == "stripe.Customer.create"
    )

    assert (
        proposal.suggested_api_call
        == "stripe.customers.create"
    )


def test_architect_generates_patch() -> None:
    state = create_state()

    architect = CodeArchitect()

    proposal = architect.create_proposal(state)

    source_code = """\
import stripe


def create_customer(user):
    return stripe.Customer.create(
        email=user.email
    )
"""

    patch = architect.generate_patch(
        proposal,
        source_code,
    )

    assert (
        "stripe.Customer.create"
        not in patch.replacement_code
    )

    assert (
        "stripe.customers.create"
        in patch.replacement_code
    )

    assert (
        "email=user.email"
        in patch.replacement_code
    )

    assert patch.file_path == "api_client.py"


def test_architect_rejects_missing_api_call() -> None:
    state = create_state()

    architect = CodeArchitect()

    proposal = architect.create_proposal(state)

    source_code = """\
def create_customer(user):
    return user.name
"""

    with pytest.raises(ValueError):
        architect.generate_patch(
            proposal,
            source_code,
        )


def test_architect_rejects_ambiguous_patch() -> None:
    state = create_state()

    architect = CodeArchitect()

    proposal = architect.create_proposal(state)

    source_code = """\
def first(user):
    return stripe.Customer.create(
        email=user.email
    )


def second(user):
    return stripe.Customer.create(
        email=user.email
    )
"""

    with pytest.raises(ValueError):
        architect.generate_patch(
            proposal,
            source_code,
        )


def test_architect_rejects_empty_source() -> None:
    state = create_state()

    architect = CodeArchitect()

    proposal = architect.create_proposal(state)

    with pytest.raises(ValueError):
        architect.generate_patch(
            proposal,
            "",
        )