from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.agent.architect import CodePatch


@dataclass(frozen=True, slots=True)
class RepairReasoning:
    """Structured reasoning result produced by a repair reasoner."""

    explanation: str
    confidence: float
    patch: CodePatch