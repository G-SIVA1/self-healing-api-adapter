from __future__ import annotations

from dataclasses import dataclass, field

from self_healing_agent.agent.failure_analyzer import (
    FailureAnalysis,
)
from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.rag.chunker import DocumentChunk


@dataclass(slots=True)
class AgentState:
    """Mutable state carried through the self-healing workflow."""

    error_event: ErrorEvent

    documentation: list[DocumentChunk] = field(
        default_factory=list
    )

    original_code: str | None = None

    generated_code: str | None = None

    test_output: str | None = None

    test_passed: bool = False

    iteration: int = 0

    max_iterations: int = 5

    repair_complete: bool = False

    failure_reason: str | None = None

    failure_analysis: FailureAnalysis | None = None

    def add_documentation(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """Add retrieved documentation."""

        self.documentation.extend(chunks)

    def record_test_result(
        self,
        passed: bool,
        output: str,
        failure_analysis: FailureAnalysis | None = None,
    ) -> None:
        """Record sandbox validation results."""

        self.test_passed = passed
        self.test_output = output

        if passed:
            self.repair_complete = True
            self.failure_reason = None
            self.failure_analysis = None
            return

        self.repair_complete = False
        self.failure_reason = output
        self.failure_analysis = failure_analysis

    def increment_iteration(self) -> None:
        """Advance the repair iteration counter."""

        self.iteration += 1

        if (
            self.iteration >= self.max_iterations
            and not self.repair_complete
        ):
            self.failure_reason = (
                "Maximum repair iterations reached"
            )

    def can_continue(self) -> bool:
        """Return whether another repair attempt is allowed."""

        return (
            not self.repair_complete
            and self.iteration < self.max_iterations
        )