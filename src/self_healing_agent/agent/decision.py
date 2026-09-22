from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from self_healing_agent.agent.state import AgentState


class DecisionAction(str, Enum):
    """Possible actions after a repair attempt."""

    RETRY = "retry"
    STOP = "stop"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Decision produced after analyzing a repair attempt."""

    action: DecisionAction
    reason: str


class RetryDecisionEngine:
    """Determine what the self-healing agent should do next."""

    def decide(
        self,
        state: AgentState,
    ) -> RetryDecision:
        """Choose the next action from the current agent state."""

        if state.repair_complete:
            return RetryDecision(
                action=DecisionAction.STOP,
                reason="Repair completed successfully.",
            )

        if state.iteration >= state.max_iterations:
            return RetryDecision(
                action=DecisionAction.ESCALATE,
                reason=(
                    "Maximum repair iterations reached."
                ),
            )

        failure_analysis = state.failure_analysis

        if failure_analysis is None:
            return RetryDecision(
                action=DecisionAction.RETRY,
                reason=(
                    "No structured failure analysis is "
                    "available; another repair attempt "
                    "may provide additional information."
                ),
            )

        if not failure_analysis.retryable:
            return RetryDecision(
                action=DecisionAction.ESCALATE,
                reason=(
                    f"Failure type "
                    f"'{failure_analysis.failure_type}' "
                    "is not considered automatically retryable."
                ),
            )

        return RetryDecision(
            action=DecisionAction.RETRY,
            reason=(
                f"Failure type "
                f"'{failure_analysis.failure_type}' "
                "is retryable."
            ),
        )