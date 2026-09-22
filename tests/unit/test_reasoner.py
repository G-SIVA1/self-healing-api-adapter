from __future__ import annotations

import pytest

from self_healing_agent.agent.context import (
    ReasoningContext,
)
from self_healing_agent.agent.reasoner import (
    DeterministicRepairReasoner,
)


def create_context() -> ReasoningContext:
    """Create a reasoning context for testing."""

    source_code = """\
import stripe


def create_customer():
    return stripe.Customer.create(
        email="user@example.com"
    )
"""

    return ReasoningContext(
        error_message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
        exception_type="AttributeError",
        target_file="api_client.py",
        target_line=6,
        function_name="create_customer",
        api_service="stripe",
        api_call="stripe.Customer.create",
        current_source_code=source_code,
        documentation=(
            "The legacy API is deprecated. "
            "Replace stripe.Customer.create with "
            "stripe.customers.create."
        ),
        previous_generated_code=None,
        test_output=None,
        failure_analysis=None,
        iteration=0,
        max_iterations=5,
    )


@pytest.mark.asyncio
async def test_reasoner_generates_repair() -> None:
    reasoner = DeterministicRepairReasoner()

    result = await reasoner.reason(
        create_context(),
    )

    assert result.patch.file_path == "api_client.py"

    assert (
        "stripe.Customer.create"
        not in result.patch.replacement_code
    )

    assert (
        "stripe.customers.create"
        in result.patch.replacement_code
    )


@pytest.mark.asyncio
async def test_reasoner_returns_confidence() -> None:
    reasoner = DeterministicRepairReasoner()

    result = await reasoner.reason(
        create_context(),
    )

    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_reasoner_explains_repair() -> None:
    reasoner = DeterministicRepairReasoner()

    result = await reasoner.reason(
        create_context(),
    )

    assert "deprecated API" in result.explanation
    assert "stripe.Customer.create" in result.explanation
    assert "stripe.customers.create" in result.explanation


@pytest.mark.asyncio
async def test_reasoner_requires_source_code() -> None:
    reasoner = DeterministicRepairReasoner()

    context = create_context()

    context_without_source = ReasoningContext(
        error_message=context.error_message,
        exception_type=context.exception_type,
        target_file=context.target_file,
        target_line=context.target_line,
        function_name=context.function_name,
        api_service=context.api_service,
        api_call=context.api_call,
        current_source_code=None,
        documentation=context.documentation,
        previous_generated_code=None,
        test_output=None,
        failure_analysis=None,
        iteration=0,
        max_iterations=5,
    )

    with pytest.raises(
        ValueError,
        match="Current source code",
    ):
        await reasoner.reason(
            context_without_source,
        )


@pytest.mark.asyncio
async def test_reasoner_rejects_missing_api() -> None:
    reasoner = DeterministicRepairReasoner()

    context = create_context()

    context_without_api = ReasoningContext(
        error_message=context.error_message,
        exception_type=context.exception_type,
        target_file=context.target_file,
        target_line=context.target_line,
        function_name=context.function_name,
        api_service=context.api_service,
        api_call=None,
        current_source_code=context.current_source_code,
        documentation=context.documentation,
        previous_generated_code=None,
        test_output=None,
        failure_analysis=None,
        iteration=0,
        max_iterations=5,
    )

    with pytest.raises(
        ValueError,
        match="API call",
    ):
        await reasoner.reason(
            context_without_api,
        )