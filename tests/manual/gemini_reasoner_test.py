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
from self_healing_agent.agent.reasoner import LLMRepairReasoner
from self_healing_agent.agent.response_parser import (
    RepairResponseParser,
)


async def main() -> None:
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

    prompt_builder = RepairPromptBuilder()
    response_parser = RepairResponseParser()

    reasoner = LLMRepairReasoner(
        client=client,
        prompt_builder=prompt_builder,
        response_parser=response_parser,
    )

    context = ReasoningContext(
        error_message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
        exception_type="AttributeError",
        target_file="examples/broken_api_client.py",
        target_line=8,
        function_name="create_customer",
        api_service="stripe",
        api_call="stripe.Customer.create",
        current_source_code="""\
import stripe


def create_customer(email: str) -> object:
    return stripe.Customer.create(
        email=email,
    )
""",
        documentation="""\
The legacy Customer API is no longer supported.

Replace:

stripe.Customer.create(...)

with:

stripe.customers.create(...)

The migration should preserve existing function inputs.
""",
        previous_generated_code=None,
        test_output=None,
        failure_analysis=None,
        iteration=0,
        max_iterations=5,
    )

    reasoning = await reasoner.reason(context)

    print("\n=== GEMINI REPAIR REASONING ===")
    print(f"Explanation: {reasoning.explanation}")
    print(f"Confidence: {reasoning.confidence}")

    print("\n=== GENERATED PATCH ===")
    print(reasoning.patch.replacement_code)

    print("===============================\n")


if __name__ == "__main__":
    asyncio.run(main())