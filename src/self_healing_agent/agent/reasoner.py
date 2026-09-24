from __future__ import annotations

from typing import Protocol

from self_healing_agent.agent.architect import CodePatch
from self_healing_agent.agent.context import ReasoningContext
from self_healing_agent.agent.models import RepairReasoning
from self_healing_agent.agent.prompt import RepairPromptBuilder
from self_healing_agent.agent.response_parser import RepairResponseParser


class RepairReasoner(Protocol):
    async def reason(
        self,
        context: ReasoningContext,
    ) -> RepairReasoning:
        """Generate a repair reasoning result."""


class LLMClient(Protocol):
    async def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate an LLM response."""


class DeterministicRepairReasoner:
    """Deterministic repair reasoner used for local and test scenarios."""

    async def reason(
        self,
        context: ReasoningContext,
    ) -> RepairReasoning:
        if context.current_source_code is None:
            raise ValueError(
                "Current source code is required"
            )

        if not context.current_source_code.strip():
            raise ValueError(
                "Current source code cannot be empty"
            )

        if not context.api_call:
            raise ValueError(
                "API call is required"
            )

        if not context.target_file:
            raise ValueError(
                "Target file is required"
            )

        original_code = context.current_source_code

        legacy_api = "stripe.Customer.create"
        current_api = "stripe.customers.create"

        if legacy_api not in original_code:
            raise ValueError(
                "Deprecated API call was not found "
                "in current source code"
            )

        replacement_code = original_code.replace(
            legacy_api,
            current_api,
            1,
        )

        if replacement_code == original_code:
            raise ValueError(
                "Repair did not modify the source code"
            )

        patch = CodePatch(
            file_path=context.target_file,
            original_code=original_code,
            replacement_code=replacement_code,
            explanation=(
                "Replace the deprecated Stripe Customer API "
                "with the current customers API."
            ),
        )

        return RepairReasoning(
            explanation=(
                "The deprecated API "
                "'stripe.Customer.create' was replaced "
                "with the current "
                "'stripe.customers.create' API."
            ),
            confidence=0.95,
            patch=patch,
            model=None,
        )


class LLMRepairReasoner:
    """Generate repair proposals using an LLM client."""

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
        if context.current_source_code is None:
            raise ValueError(
                "Current source code is required"
            )

        if not context.current_source_code.strip():
            raise ValueError(
                "Current source code cannot be empty"
            )

        if not context.target_file:
            raise ValueError(
                "Target file is required"
            )

        if not context.api_call:
            raise ValueError(
                "API call is required"
            )

        prompt = self._prompt_builder.build(
            context
        )

        response = await self._client.generate(
            prompt
        )

        parsed_reasoning = self._response_parser.parse(
            response=response,
            target_file=context.target_file,
            original_code=context.current_source_code,
        )

        model = getattr(
            self._client,
            "last_model",
            None,
        )

        return RepairReasoning(
            explanation=parsed_reasoning.explanation,
            confidence=parsed_reasoning.confidence,
            patch=parsed_reasoning.patch,
            model=model,
        )