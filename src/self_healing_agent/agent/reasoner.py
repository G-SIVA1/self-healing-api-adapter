from __future__ import annotations

from typing import Protocol

from self_healing_agent.agent.architect import CodePatch
from self_healing_agent.agent.context import ReasoningContext
from self_healing_agent.agent.models import RepairReasoning
from self_healing_agent.agent.prompt import RepairPromptBuilder
from self_healing_agent.agent.response_parser import (
    RepairResponseParser,
)


class RepairReasoner(Protocol):
    """Interface for generating repair reasoning."""

    async def reason(
        self,
        context: ReasoningContext,
    ) -> RepairReasoning:
        ...


class LLMClient(Protocol):
    """Interface for asynchronous LLM clients."""

    async def generate(
        self,
        prompt: str,
    ) -> str:
        ...


class DeterministicRepairReasoner:
    """
    Deterministic repair reasoner.

    This implementation performs the known Stripe Customer
    API migration without requiring an external LLM.
    """

    async def reason(
        self,
        context: ReasoningContext,
    ) -> RepairReasoning:
        source_code = context.current_source_code

        if source_code is None:
            raise ValueError(
                "Current source code is required"
            )

        if not source_code.strip():
            raise ValueError(
                "Current source code cannot be empty"
            )

        if context.api_call is None:
            raise ValueError(
                "API call is required"
            )

        if not context.api_call.strip():
            raise ValueError(
                "API call cannot be empty"
            )

        if not context.target_file:
            raise ValueError(
                "Target file is required"
            )

        legacy_api = "stripe.Customer.create"
        current_api = "stripe.customers.create"

        if legacy_api not in source_code:
            raise ValueError(
                "Deprecated API call was not found "
                "in current source code"
            )

        replacement_code = source_code.replace(
            legacy_api,
            current_api,
            1,
        )

        if replacement_code == source_code:
            raise ValueError(
                "Repair did not modify the source code"
            )

        explanation = (
            "The deprecated API "
            "'stripe.Customer.create' was replaced "
            "with the current "
            "'stripe.customers.create' API."
        )

        patch = CodePatch(
            file_path=context.target_file,
            original_code=source_code,
            replacement_code=replacement_code,
            explanation=explanation,
        )

        return RepairReasoning(
            explanation=explanation,
            confidence=0.95,
            patch=patch,
        )


class LLMRepairReasoner:
    """
    Generate source-code repairs using an LLM.

    The LLM is responsible for producing a complete corrected
    source file. The response parser validates the generated
    repair before it is returned to the agent loop.
    """

    def __init__(
        self,
        client: LLMClient,
        prompt_builder: RepairPromptBuilder,
        response_parser: RepairResponseParser,
    ) -> None:
        self._client = client
        self._prompt_builder = prompt_builder
        self._response_parser = response_parser

    async def reason(
        self,
        context: ReasoningContext,
    ) -> RepairReasoning:
        source_code = context.current_source_code

        if source_code is None:
            raise ValueError(
                "Current source code is required"
            )

        if not source_code.strip():
            raise ValueError(
                "Current source code cannot be empty"
            )

        if not context.target_file:
            raise ValueError(
                "Target file is required"
            )

        if context.api_call is None:
            raise ValueError(
                "API call is required"
            )

        if not context.api_call.strip():
            raise ValueError(
                "API call cannot be empty"
            )

        prompt = self._prompt_builder.build(
            context
        )

        response = await self._client.generate(
            prompt
        )

        return self._response_parser.parse(
            response=response,
            target_file=context.target_file,
            original_code=source_code,
        )