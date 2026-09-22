from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.agent.failure_analyzer import (
    FailureAnalysis,
)
from self_healing_agent.agent.state import AgentState


@dataclass(frozen=True, slots=True)
class ReasoningContext:
    """Context supplied to the repair reasoning component."""

    error_message: str
    exception_type: str

    target_file: str | None
    target_line: int | None
    function_name: str | None

    api_service: str | None
    api_call: str | None

    current_source_code: str | None

    documentation: str

    previous_generated_code: str | None

    test_output: str | None

    failure_analysis: FailureAnalysis | None

    iteration: int
    max_iterations: int


class ReasoningContextBuilder:
    """Build structured reasoning context from AgentState."""

    def build(
        self,
        state: AgentState,
        current_source_code: str | None = None,
    ) -> ReasoningContext:
        """
        Build a complete reasoning context.

        The context contains the runtime failure, relevant
        documentation, current source code, previous repair
        information, and validation feedback.
        """

        event = state.error_event

        documentation = self._build_documentation(
            state,
        )

        return ReasoningContext(
            error_message=event.message,
            exception_type=event.exception_type,
            target_file=event.file_path,
            target_line=event.line_number,
            function_name=event.function_name,
            api_service=event.api_service,
            api_call=event.api_call,
            current_source_code=current_source_code
            if current_source_code is not None
            else state.generated_code
            if state.generated_code is not None
            else state.original_code,
            documentation=documentation,
            previous_generated_code=state.generated_code,
            test_output=state.test_output,
            failure_analysis=state.failure_analysis,
            iteration=state.iteration,
            max_iterations=state.max_iterations,
        )

    @staticmethod
    def _build_documentation(
        state: AgentState,
    ) -> str:
        """Combine retrieved documentation chunks."""

        if not state.documentation:
            return ""

        sections: list[str] = []

        for index, chunk in enumerate(
            state.documentation,
            start=1,
        ):
            sections.append(
                f"Documentation Source {index}:\n"
                f"{chunk.content}"
            )

        return "\n\n".join(sections)