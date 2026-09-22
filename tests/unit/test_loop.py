from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from self_healing_agent.agent.architect import (
    CodeArchitect,
    CodePatch,
)
from self_healing_agent.agent.context import (
    ReasoningContext,
)
from self_healing_agent.agent.context import (
    ReasoningContextBuilder,
)
from self_healing_agent.agent.decision import (
    DecisionAction,
    RetryDecisionEngine,
)
from self_healing_agent.agent.failure_analyzer import (
    FailureAnalyzer,
)
from self_healing_agent.agent.loop import (
    SelfCorrectionLoop,
)
from self_healing_agent.agent.reasoner import (
    DeterministicRepairReasoner,
    RepairReasoning,
    RepairReasoner,
)
from self_healing_agent.agent.state import AgentState
from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.rag.chunker import DocumentChunk
from self_healing_agent.sandbox.executor import (
    SandboxExecutor,
)
from self_healing_agent.sandbox.patcher import (
    SafePatchApplier,
)
from self_healing_agent.sandbox.repair_executor import (
    RepairExecutor,
)
from self_healing_agent.sandbox.security import (
    SandboxSecurityPolicy,
)


def create_state(
    target_file: str,
) -> AgentState:
    return AgentState(
        error_event=ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            exception_type="AttributeError",
            message=(
                "module 'stripe' has no attribute 'Customer'"
            ),
            file_path=target_file,
            line_number=6,
            function_name="create_customer",
            api_service="stripe",
            api_call="stripe.Customer.create",
            raw_log=(
                "AttributeError: "
                "module 'stripe' has no attribute 'Customer'"
            ),
        )
    )


def create_loop(
    workspace: Path,
    reasoner: RepairReasoner,
) -> SelfCorrectionLoop:
    security_policy = SandboxSecurityPolicy(
        allowed_working_directory=workspace,
    )

    repair_executor = RepairExecutor(
        patch_applier=SafePatchApplier(),
        sandbox_executor=SandboxExecutor(
            security_policy=security_policy,
            timeout_seconds=10,
        ),
        failure_analyzer=FailureAnalyzer(),
    )

    return SelfCorrectionLoop(
        architect=CodeArchitect(),
        repair_executor=repair_executor,
        reasoner=reasoner,
        context_builder=ReasoningContextBuilder(),
        decision_engine=RetryDecisionEngine(),
    )


class RetryAwareTestReasoner:
    """Produce an incorrect first repair and correct second repair."""

    def __init__(self) -> None:
        self.calls = 0

    async def reason(
        self,
        context: ReasoningContext,
    ) -> RepairReasoning:
        self.calls += 1

        source_code = context.current_source_code

        if source_code is None:
            raise ValueError(
                "Current source code is required"
            )

        if self.calls == 1:
            replacement_code = source_code.replace(
                "stripe.Customer.create",
                "stripe.customers.wrong_create",
                1,
            )

            explanation = (
                "First repair attempt used an incorrect "
                "replacement API."
            )
        else:
            replacement_code = source_code.replace(
                "stripe.customers.wrong_create",
                "stripe.customers.create",
                1,
            )

            explanation = (
                "Second repair attempt corrected the API."
            )

        patch = CodePatch(
            file_path=(
                context.target_file
                if context.target_file is not None
                else ""
            ),
            original_code=source_code,
            replacement_code=replacement_code,
            explanation=explanation,
        )

        return RepairReasoning(
            explanation=explanation,
            confidence=0.90,
            patch=patch,
        )


@pytest.mark.asyncio
async def test_self_correction_loop_succeeds(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "api_client.py"

    original_code = """\
import stripe


def create_customer():
    return stripe.Customer.create(
        email="user@example.com"
    )
"""

    source_file.write_text(
        original_code,
        encoding="utf-8",
    )

    test_file = tmp_path / "test_api_client.py"

    test_file.write_text(
        """\
from pathlib import Path


def test_customer_api():
    source = Path("api_client.py").read_text(
        encoding="utf-8"
    )

    assert "stripe.customers.create" in source
    assert "stripe.Customer.create" not in source
""",
        encoding="utf-8",
    )

    state = create_state(
        "api_client.py",
    )

    state.original_code = original_code

    state.add_documentation(
        [
            DocumentChunk(
                content=(
                    "Replace stripe.Customer.create "
                    "with stripe.customers.create."
                ),
                source="stripe.md",
                chunk_index=0,
            )
        ]
    )

    loop = create_loop(
        tmp_path,
        DeterministicRepairReasoner(),
    )

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=[
            sys.executable,
            "-m",
            "pytest",
            "test_api_client.py",
            "-q",
        ],
    )

    assert result.succeeded is True
    assert result.final_decision == DecisionAction.STOP
    assert state.repair_complete is True
    assert state.iteration == 1
    assert len(result.execution_results) == 1


@pytest.mark.asyncio
async def test_self_correction_loop_reasons_again_after_failure(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "api_client.py"

    original_code = """\
import stripe


def create_customer():
    return stripe.Customer.create(
        email="user@example.com"
    )
"""

    source_file.write_text(
        original_code,
        encoding="utf-8",
    )

    test_file = tmp_path / "test_api_client.py"

    test_file.write_text(
        """\
from pathlib import Path


def test_customer_api():
    source = Path("api_client.py").read_text(
        encoding="utf-8"
    )

    assert "stripe.customers.create" in source
    assert "stripe.customers.wrong_create" not in source
    assert "stripe.Customer.create" not in source
""",
        encoding="utf-8",
    )

    state = create_state(
        "api_client.py",
    )

    state.original_code = original_code

    state.add_documentation(
        [
            DocumentChunk(
                content=(
                    "Replace stripe.Customer.create "
                    "with stripe.customers.create."
                ),
                source="stripe.md",
                chunk_index=0,
            )
        ]
    )

    reasoner = RetryAwareTestReasoner()

    loop = create_loop(
        tmp_path,
        reasoner,
    )

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=[
            sys.executable,
            "-m",
            "pytest",
            "test_api_client.py",
            "-q",
        ],
    )

    assert result.succeeded is True
    assert result.final_decision == DecisionAction.STOP
    assert reasoner.calls == 2
    assert state.iteration == 2
    assert state.repair_complete is True
    assert state.failure_analysis is None
    assert len(result.execution_results) == 2