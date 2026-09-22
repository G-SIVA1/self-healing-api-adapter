from __future__ import annotations

import json

import pytest

from self_healing_agent.agent.context import (
    ReasoningContext,
)
from self_healing_agent.agent.models import (
    RepairReasoning,
)
from self_healing_agent.agent.prompt import (
    RepairPromptBuilder,
)
from self_healing_agent.agent.reasoner import (
    LLMRepairReasoner,
)
from self_healing_agent.agent.response_parser import (
    RepairResponseParser,
)


def create_context() -> ReasoningContext:
    """Create a realistic API repair context."""

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


class FakeLLMClient:
    """Fake asynchronous LLM for deterministic testing."""

    def __init__(self) -> None:
        self.received_prompt: str | None = None

    async def generate(
        self,
        prompt: str,
    ) -> str:
        self.received_prompt = prompt

        replacement_code = """\
import stripe


def create_customer():
    return stripe.customers.create(
        email="user@example.com"
    )
"""

        return json.dumps(
            {
                "explanation": (
                    "Replace the deprecated Stripe "
                    "Customer API with the current "
                    "customers API."
                ),
                "confidence": 0.96,
                "replacement_code": replacement_code,
            }
        )


def create_reasoner(
    client: FakeLLMClient,
) -> LLMRepairReasoner:
    return LLMRepairReasoner(
        client=client,
        prompt_builder=RepairPromptBuilder(),
        response_parser=RepairResponseParser(),
    )


@pytest.mark.asyncio
async def test_llm_reasoner_generates_repair() -> None:
    client = FakeLLMClient()

    reasoner = create_reasoner(client)

    result = await reasoner.reason(
        create_context(),
    )

    assert isinstance(
        result,
        RepairReasoning,
    )

    assert (
        "stripe.customers.create"
        in result.patch.replacement_code
    )

    assert (
        "stripe.Customer.create"
        not in result.patch.replacement_code
    )


@pytest.mark.asyncio
async def test_llm_reasoner_returns_confidence() -> None:
    client = FakeLLMClient()

    reasoner = create_reasoner(client)

    result = await reasoner.reason(
        create_context(),
    )

    assert result.confidence == 0.96


@pytest.mark.asyncio
async def test_llm_reasoner_builds_prompt() -> None:
    client = FakeLLMClient()

    reasoner = create_reasoner(client)

    await reasoner.reason(
        create_context(),
    )

    assert client.received_prompt is not None

    assert (
        "## Runtime Error"
        in client.received_prompt
    )

    assert (
        "## Current Source Code"
        in client.received_prompt
    )

    assert (
        "## Retrieved Documentation"
        in client.received_prompt
    )

    assert (
        "stripe.Customer.create"
        in client.received_prompt
    )

    assert (
        "stripe.customers.create"
        in client.received_prompt
    )


@pytest.mark.asyncio
async def test_llm_reasoner_preserves_target_file() -> None:
    client = FakeLLMClient()

    reasoner = create_reasoner(client)

    result = await reasoner.reason(
        create_context(),
    )

    assert (
        result.patch.file_path
        == "api_client.py"
    )


@pytest.mark.asyncio
async def test_llm_reasoner_rejects_missing_source() -> None:
    client = FakeLLMClient()

    reasoner = create_reasoner(client)

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