from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from self_healing_agent.agent.context import ReasoningContext
from self_healing_agent.agent.llm_client import (
    AsyncLLMClient,
    GeminiTransport,
)
from self_healing_agent.agent.prompt import RepairPromptBuilder


async def main() -> None:
    """Send the repair prompt directly to Gemini and print the raw response."""

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )

    model = os.getenv(
        "LLM_MODEL",
        "gemini-3.6-flash",
    )

    source_code = """\
import stripe


def create_customer(email: str) -> object:
    return stripe.Customer.create(
        email=email,
    )
"""

    documentation = """\
# Stripe API Migration Guide

The legacy Customer API is no longer supported.

### Legacy API

stripe.Customer.create(
    email="user@example.com"
)

### Current API

Use the modern Stripe client interface:

stripe.customers.create(
    email="user@example.com"
)

The legacy stripe.Customer.create method should be
replaced with stripe.customers.create.

If an application reports:

AttributeError: module 'stripe' has no attribute 'Customer'

the application may be using the legacy Customer API.

The migration should preserve the existing function
inputs whenever possible.
"""

    context = ReasoningContext(
        error_message=(
            "module 'stripe' has no attribute "
            "'Customer'"
        ),
        exception_type="AttributeError",
        target_file="examples/broken_api_client.py",
        target_line=5,
        function_name="create_customer",
        api_service="stripe",
        api_call="stripe.Customer.create",
        current_source_code=source_code,
        documentation=documentation,
        previous_generated_code=None,
        test_output=None,
        failure_analysis=None,
        iteration=0,
        max_iterations=3,
    )

    prompt = RepairPromptBuilder().build(context)

    print("\n========================================")
    print("MODEL")
    print("========================================")
    print(model)

    print("\n========================================")
    print("PROMPT")
    print("========================================")
    print(prompt)

    transport = GeminiTransport(
        api_key=api_key,
    )

    client = AsyncLLMClient(
        transport=transport,
        model=model,
        timeout_seconds=30,
        max_retries=2,
        base_delay_seconds=0.5,
    )

    print("\n========================================")
    print("SENDING REQUEST TO GEMINI")
    print("========================================")

    response = await client.generate(prompt)

    print("\n========================================")
    print("RAW GEMINI RESPONSE")
    print("========================================")

    print(response)

    print("\n========================================")
    print("END RAW RESPONSE")
    print("========================================")


if __name__ == "__main__":
    asyncio.run(main())