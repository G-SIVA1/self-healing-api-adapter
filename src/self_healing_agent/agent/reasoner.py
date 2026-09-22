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
    async def reason(
        self,
        context: ReasoningContext,
    ) -> RepairReasoning:
        ...


class LLMClient(Protocol):
    """Provider-independent interface for an LLM client."""

    async def generate(
        self,
        prompt: str,
    ) -> str:
        ...


class LLMRepairReasoner:
    """LLM-backed implementation of the repair reasoner."""

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
                "Current source code is required for reasoning"
            )

        if context.target_file is None:
            raise ValueError(
                "Target file is required for reasoning"
            )

        prompt = self._prompt_builder.build(context)

        response = await self._client.generate(prompt)

        return self._response_parser.parse(
            response=response,
            target_file=context.target_file,
            original_code=context.current_source_code,
        )


class DeterministicRepairReasoner:
    """
    Deterministic repair implementation.

    This implements the same asynchronous interface as the
    future LLM-backed reasoner.
    """

    async def reason(
        self,
        context: ReasoningContext,
    ) -> RepairReasoning:
        if context.current_source_code is None:
            raise ValueError(
                "Current source code is required for reasoning"
            )

        original_api = context.api_call

        if original_api is None:
            raise ValueError(
                "API call is required for reasoning"
            )

        suggested_api = self._find_suggested_api(
            context.documentation,
        )

        if suggested_api is None:
            raise ValueError(
                "No supported replacement API was found "
                "in the documentation"
            )

        source_code = context.current_source_code

        if original_api not in source_code:
            raise ValueError(
                "Original API call was not found "
                "in current source code"
            )

        replacement_code = source_code.replace(
            original_api,
            suggested_api,
            1,
        )

        patch = CodePatch(
            file_path=(
                context.target_file
                if context.target_file is not None
                else ""
            ),
            original_code=source_code,
            replacement_code=replacement_code,
            explanation=(
                f"Replace deprecated API "
                f"'{original_api}' with "
                f"'{suggested_api}' according to "
                "the retrieved documentation."
            ),
        )

        return RepairReasoning(
            explanation=patch.explanation,
            confidence=0.95,
            patch=patch,
        )

    @staticmethod
    def _find_suggested_api(
        documentation: str,
    ) -> str | None:
        if "stripe.customers.create" in documentation:
            return "stripe.customers.create"

        return None