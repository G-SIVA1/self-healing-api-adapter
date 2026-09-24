from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from self_healing_agent.agent.architect import CodeArchitect
from self_healing_agent.agent.context import ReasoningContextBuilder
from self_healing_agent.agent.decision import (
    DecisionAction,
    RetryDecisionEngine,
)
from self_healing_agent.agent.history import (
    RepairAttempt,
    RepairHistory,
    create_attempt_timestamp,
)
from self_healing_agent.agent.history_retriever import (
    RepairHistoryRetriever,
)
from self_healing_agent.agent.history_store import (
    RepairHistoryStore,
)
from self_healing_agent.agent.reasoner import RepairReasoner
from self_healing_agent.agent.state import AgentState
from self_healing_agent.sandbox.repair_executor import (
    RepairExecutionResult,
    RepairExecutor,
)


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Final result produced by the self-correction loop."""

    state: AgentState
    execution_results: list[RepairExecutionResult]
    final_decision: DecisionAction
    history: RepairHistory

    @property
    def succeeded(self) -> bool:
        """Return whether the repair completed successfully."""

        return self.state.repair_complete


class SelfCorrectionLoop:
    """Coordinate reasoning, patching, validation, and repair history."""

    def __init__(
        self,
        architect: CodeArchitect,
        repair_executor: RepairExecutor,
        reasoner: RepairReasoner,
        context_builder: ReasoningContextBuilder,
        decision_engine: RetryDecisionEngine,
        history: RepairHistory | None = None,
        history_store: RepairHistoryStore | None = None,
        history_retriever: RepairHistoryRetriever | None = None,
    ) -> None:
        self._architect = architect
        self._repair_executor = repair_executor
        self._reasoner = reasoner
        self._context_builder = context_builder
        self._decision_engine = decision_engine

        self._history = (
            history
            if history is not None
            else RepairHistory()
        )

        self._history_store = history_store

        if history_retriever is not None:
            self._history_retriever = history_retriever
        elif history_store is not None:
            self._history_retriever = (
                RepairHistoryRetriever(history_store)
            )
        else:
            self._history_retriever = None

    @property
    def history(self) -> RepairHistory:
        """Return the in-memory repair history."""

        return self._history

    @property
    def history_store(
        self,
    ) -> RepairHistoryStore | None:
        """Return the configured persistent history store."""

        return self._history_store

    @property
    def history_retriever(
        self,
    ) -> RepairHistoryRetriever | None:
        """Return the configured history retriever."""

        return self._history_retriever

    async def run(
        self,
        state: AgentState,
        workspace: Path,
        validation_command: list[str],
    ) -> AgentLoopResult:
        """Execute the self-correction loop."""

        if not workspace.exists():
            raise ValueError(
                "workspace does not exist"
            )

        if not workspace.is_dir():
            raise ValueError(
                "workspace must be a directory"
            )

        execution_results: list[
            RepairExecutionResult
        ] = []

        final_decision = DecisionAction.STOP

        while state.can_continue():
            current_source_code = (
                self._read_current_source(
                    state=state,
                    workspace=workspace,
                )
            )

            historical_repairs = (
                self._retrieve_history(state)
            )

            context = self._context_builder.build(
                state=state,
                current_source_code=current_source_code,
                historical_repairs=historical_repairs,
            )

            reasoning = await self._reasoner.reason(
                context
            )

            execution_result = (
                await self._repair_executor.execute(
                    state=state,
                    patch=reasoning.patch,
                    validation_command=validation_command,
                    workspace=workspace,
                )
            )

            execution_results.append(
                execution_result
            )

            self._record_attempt(
                state=state,
                reasoning=reasoning,
                validation_command=validation_command,
            )

            final_decision = (
                self._decision_engine.decide(
                    state
                )
            )

            if final_decision == DecisionAction.STOP:
                break

            if final_decision == DecisionAction.ESCALATE:
                break

        if state.repair_complete:
            final_decision = DecisionAction.STOP
        elif not state.can_continue():
            final_decision = DecisionAction.ESCALATE

        return AgentLoopResult(
            state=state,
            execution_results=execution_results,
            final_decision=final_decision,
            history=self._history,
        )

    def _retrieve_history(
        self,
        state: AgentState,
    ) -> list[RepairAttempt]:
        """Retrieve relevant successful historical repairs."""

        if self._history_retriever is None:
            return []

        target_file = state.error_event.file_path

        if not target_file:
            return []

        exception_type = (
            state.error_event.exception_type
        )

        historical_repairs = (
            self._history_retriever.find_successful(
                target_file=target_file,
                exception_type=exception_type,
                limit=5,
            )
        )

        return [
            historical_repair.attempt
            for historical_repair in historical_repairs
        ]

    def _record_attempt(
        self,
        state: AgentState,
        reasoning,
        validation_command: list[str],
    ) -> None:
        """Record a completed repair attempt."""

        error_event = state.error_event
        patch = reasoning.patch
        failure_analysis = state.failure_analysis

        attempt = RepairAttempt(
            iteration=state.iteration,
            timestamp=create_attempt_timestamp(),
            target_file=error_event.file_path or "",
            exception_type=error_event.exception_type,
            error_message=error_event.message,
            model=reasoning.model,
            explanation=reasoning.explanation,
            confidence=reasoning.confidence,
            original_code=patch.original_code,
            generated_code=patch.replacement_code,
            validation_command=tuple(
                validation_command
            ),
            test_passed=state.test_passed,
            test_output=state.test_output,
            failure_type=(
                failure_analysis.failure_type
                if failure_analysis is not None
                else None
            ),
            failure_summary=(
                failure_analysis.summary
                if failure_analysis is not None
                else None
            ),
            failure_details=(
                failure_analysis.details
                if failure_analysis is not None
                else None
            ),
            status=(
                "success"
                if state.test_passed
                else "failed"
            ),
        )

        self._history.add(attempt)

        if self._history_store is not None:
            try:
                self._history_store.save(attempt)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to persist repair history"
                ) from exc

    @staticmethod
    def _read_current_source(
        state: AgentState,
        workspace: Path,
    ) -> str | None:
        """Read the current target source code."""

        file_path = state.error_event.file_path

        if not file_path:
            return state.original_code

        path = Path(file_path)

        if not path.is_absolute():
            path = workspace / path

        if not path.exists():
            return state.original_code

        if not path.is_file():
            return state.original_code

        try:
            return path.read_text(
                encoding="utf-8"
            )
        except OSError:
            return state.original_code