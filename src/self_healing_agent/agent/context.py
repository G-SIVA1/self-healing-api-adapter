from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.agent.failure_analyzer import (
    FailureAnalysis,
)
from self_healing_agent.agent.state import AgentState


@dataclass(frozen=True, slots=True)
class ReasoningContext:
    """
    Immutable reasoning context supplied to a repair reasoner.
    """

    error_message: str
    exception_type: str

    target_file: str
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
    """
    Build a reasoning context from the current agent state.
    """

    def build(
        self,
        state: AgentState,
        current_source_code: str | None = None,
    ) -> ReasoningContext:
        """
        Build a complete reasoning context from AgentState.
        """

        error_event = state.error_event

        documentation = self._format_documentation(
            state.documentation,
        )

        return ReasoningContext(
            error_message=error_event.message,
            exception_type=error_event.exception_type,
            target_file=error_event.file_path or "",
            target_line=error_event.line_number,
            function_name=error_event.function_name,
            api_service=error_event.api_service,
            api_call=error_event.api_call,
            current_source_code=current_source_code,
            documentation=documentation,
            previous_generated_code=state.generated_code,
            test_output=state.test_output,
            failure_analysis=state.failure_analysis,
            iteration=state.iteration,
            max_iterations=state.max_iterations,
        )

    @staticmethod
    def _format_documentation(
        documentation: list[object],
    ) -> str:
        """
        Convert retrieved documentation chunks into one
        prompt-ready string.
        """

        if not documentation:
            return "No documentation available."

        formatted_chunks: list[str] = []

        for index, chunk in enumerate(
            documentation,
            start=1,
        ):
            content = getattr(
                chunk,
                "content",
                None,
            )

            if content is None:
                content = str(chunk)

            source = getattr(
                chunk,
                "source",
                None,
            )

            chunk_index = getattr(
                chunk,
                "chunk_index",
                None,
            )

            if source is not None:
                formatted_chunks.append(
                    f"[Document {index}]\n"
                    f"Source: {source}\n"
                    f"Chunk: {chunk_index}\n"
                    f"{content}"
                )
            else:
                formatted_chunks.append(
                    f"[Document {index}]\n"
                    f"{content}"
                )

        return "\n\n".join(formatted_chunks)