from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from self_healing_agent.agent.architect import CodePatch
from self_healing_agent.agent.failure_analyzer import (
    FailureAnalyzer,
)
from self_healing_agent.agent.state import AgentState
from self_healing_agent.sandbox.executor import (
    ExecutionResult,
    SandboxExecutor,
)
from self_healing_agent.sandbox.patcher import (
    SafePatchApplier,
)


@dataclass(frozen=True, slots=True)
class RepairExecutionResult:
    """Result of applying and testing a repair."""

    patch: CodePatch
    execution: ExecutionResult
    patched_file: Path

    @property
    def succeeded(self) -> bool:
        """Return whether the repair passed validation."""

        return self.execution.succeeded


class RepairExecutor:
    """Apply and validate generated repair patches."""

    def __init__(
        self,
        patch_applier: SafePatchApplier,
        sandbox_executor: SandboxExecutor,
        failure_analyzer: FailureAnalyzer,
    ) -> None:
        self._patch_applier = patch_applier
        self._sandbox_executor = sandbox_executor
        self._failure_analyzer = failure_analyzer

    async def execute(
        self,
        patch: CodePatch,
        workspace: Path,
        validation_command: list[str],
        state: AgentState,
    ) -> RepairExecutionResult:
        """
        Apply a patch, execute validation, analyze the result,
        and update AgentState.
        """

        patched_file = await self._patch_applier.apply(
            patch,
            workspace,
        )

        execution = await self._sandbox_executor.execute(
            command=validation_command,
            working_directory=workspace,
        )

        output = self._build_feedback(execution)

        state.generated_code = patch.replacement_code

        failure_analysis = None

        if not execution.succeeded:
            failure_analysis = self._failure_analyzer.analyze(
                output
            )

        state.record_test_result(
            passed=execution.succeeded,
            output=output,
            failure_analysis=failure_analysis,
        )

        state.increment_iteration()

        return RepairExecutionResult(
            patch=patch,
            execution=execution,
            patched_file=patched_file,
        )

    @staticmethod
    def _build_feedback(
        execution: ExecutionResult,
    ) -> str:
        """Combine process output into agent-readable feedback."""

        parts: list[str] = []

        parts.append(
            f"Return code: {execution.return_code}"
        )

        parts.append(
            f"Timed out: {execution.timed_out}"
        )

        if execution.stdout.strip():
            parts.append(
                f"STDOUT:\n{execution.stdout.strip()}"
            )

        if execution.stderr.strip():
            parts.append(
                f"STDERR:\n{execution.stderr.strip()}"
            )

        return "\n\n".join(parts)