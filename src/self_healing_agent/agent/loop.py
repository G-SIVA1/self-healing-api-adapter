from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from self_healing_agent.agent.architect import (
    CodeArchitect,
)
from self_healing_agent.agent.context import (
    ReasoningContextBuilder,
)
from self_healing_agent.agent.decision import (
    DecisionAction,
    RetryDecisionEngine,
)
from self_healing_agent.agent.reasoner import (
    RepairReasoner,
)
from self_healing_agent.agent.state import AgentState
from self_healing_agent.sandbox.repair_executor import (
    RepairExecutionResult,
    RepairExecutor,
)


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Final result of the self-correction loop."""

    state: AgentState
    execution_results: list[RepairExecutionResult]
    final_decision: DecisionAction

    @property
    def succeeded(self) -> bool:
        """Return whether the repair completed successfully."""

        return self.state.repair_complete


class SelfCorrectionLoop:
    """
    Execute the complete self-correction control loop.

    Flow:

        Build Context
             ↓
        Reason About Repair
             ↓
        Apply Patch
             ↓
        Run Tests
             ↓
        Analyze Failure
             ↓
        Make Decision
             ↓
        Retry / Stop / Escalate
    """

    def __init__(
        self,
        architect: CodeArchitect,
        repair_executor: RepairExecutor,
        reasoner: RepairReasoner,
        context_builder: ReasoningContextBuilder,
        decision_engine: RetryDecisionEngine,
    ) -> None:
        self._architect = architect
        self._repair_executor = repair_executor
        self._reasoner = reasoner
        self._context_builder = context_builder
        self._decision_engine = decision_engine

    async def run(
        self,
        state: AgentState,
        workspace: Path,
        validation_command: list[str],
    ) -> AgentLoopResult:
        """
        Run the self-correction loop.

        Every iteration builds fresh reasoning context from
        the current workspace and AgentState.
        """

        execution_results: list[
            RepairExecutionResult
        ] = []

        final_decision = DecisionAction.ESCALATE

        while state.can_continue():
            current_source_code = (
                self._read_current_source(
                    state,
                    workspace,
                )
            )

            context = self._context_builder.build(
                state,
                current_source_code=current_source_code,
            )

            reasoning = await self._reasoner.reason(
                context,
            )

            result = await self._repair_executor.execute(
                patch=reasoning.patch,
                workspace=workspace,
                validation_command=validation_command,
                state=state,
            )

            execution_results.append(result)

            decision = self._decision_engine.decide(
                state,
            )

            final_decision = decision.action

            if decision.action == DecisionAction.STOP:
                break

            if decision.action == DecisionAction.ESCALATE:
                break

        if state.repair_complete:
            final_decision = DecisionAction.STOP
        elif state.iteration >= state.max_iterations:
            final_decision = DecisionAction.ESCALATE

        return AgentLoopResult(
            state=state,
            execution_results=execution_results,
            final_decision=final_decision,
        )

    @staticmethod
    def _read_current_source(
        state: AgentState,
        workspace: Path,
    ) -> str:
        """Read the current target source from the workspace."""

        file_path = state.error_event.file_path

        if file_path is None:
            raise ValueError(
                "AgentState does not contain a target file"
            )

        target_file = (
            workspace / file_path
        ).resolve()

        if not target_file.exists():
            raise FileNotFoundError(
                f"Target file does not exist: {target_file}"
            )

        if not target_file.is_file():
            raise ValueError(
                f"Target path is not a file: {target_file}"
            )

        return target_file.read_text(
            encoding="utf-8",
        )