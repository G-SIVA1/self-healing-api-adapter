from __future__ import annotations

from self_healing_agent.agent.context import (
    ReasoningContext,
)
from self_healing_agent.agent.failure_analyzer import (
    FailureAnalysis,
)
from self_healing_agent.agent.prompt import (
    RepairPromptBuilder,
)


def create_context() -> ReasoningContext:
    """Create a complete reasoning context."""

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
        current_source_code="""\
import stripe


def create_customer():
    return stripe.Customer.create(
        email="user@example.com"
    )
""",
        documentation=(
            "The legacy Customer API is no longer "
            "supported. Replace "
            "stripe.Customer.create with "
            "stripe.customers.create."
        ),
        previous_generated_code=None,
        test_output=None,
        failure_analysis=None,
        iteration=0,
        max_iterations=5,
    )


def test_prompt_contains_runtime_error() -> None:
    prompt = RepairPromptBuilder().build(
        create_context(),
    )

    assert "## Runtime Error" in prompt

    assert "AttributeError" in prompt

    assert (
        "module 'stripe' has no attribute 'Customer'"
        in prompt
    )

    assert "stripe.Customer.create" in prompt


def test_prompt_contains_source_code() -> None:
    prompt = RepairPromptBuilder().build(
        create_context(),
    )

    assert "## Current Source Code" in prompt

    assert "import stripe" in prompt

    assert (
        "stripe.Customer.create"
        in prompt
    )


def test_prompt_contains_documentation() -> None:
    prompt = RepairPromptBuilder().build(
        create_context(),
    )

    assert "## Retrieved Documentation" in prompt

    assert (
        "stripe.customers.create"
        in prompt
    )


def test_prompt_contains_failure_analysis() -> None:
    context = create_context()

    context = ReasoningContext(
        error_message=context.error_message,
        exception_type=context.exception_type,
        target_file=context.target_file,
        target_line=context.target_line,
        function_name=context.function_name,
        api_service=context.api_service,
        api_call=context.api_call,
        current_source_code=context.current_source_code,
        documentation=context.documentation,
        previous_generated_code=(
            "stripe.customers.wrong_create()"
        ),
        test_output=(
            "AssertionError: wrong API"
        ),
        failure_analysis=FailureAnalysis(
            failure_type="assertion_error",
            summary="A test assertion failed.",
            details="The generated API call is invalid.",
            retryable=True,
        ),
        iteration=1,
        max_iterations=5,
    )

    prompt = RepairPromptBuilder().build(
        context,
    )

    assert "## Validation Feedback" in prompt

    assert "assertion_error" in prompt

    assert (
        "The generated API call is invalid."
        in prompt
    )

    assert "Retryable: True" in prompt


def test_prompt_contains_iteration_information() -> None:
    context = create_context()

    context = ReasoningContext(
        error_message=context.error_message,
        exception_type=context.exception_type,
        target_file=context.target_file,
        target_line=context.target_line,
        function_name=context.function_name,
        api_service=context.api_service,
        api_call=context.api_call,
        current_source_code=context.current_source_code,
        documentation=context.documentation,
        previous_generated_code=None,
        test_output=None,
        failure_analysis=None,
        iteration=2,
        max_iterations=5,
    )

    prompt = RepairPromptBuilder().build(
        context,
    )

    assert "Current Iteration: 2" in prompt

    assert "Maximum Iterations: 5" in prompt


def test_prompt_contains_output_contract() -> None:
    prompt = RepairPromptBuilder().build(
        create_context(),
    )

    assert "## Required Output" in prompt

    assert (
        "complete corrected source code"
        in prompt
    )

    assert (
        "Do not modify unrelated functionality."
        in prompt
    )