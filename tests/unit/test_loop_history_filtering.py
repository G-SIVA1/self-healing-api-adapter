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
    ReasoningContextBuilder,
)
from self_healing_agent.agent.decision import (
    RetryDecisionEngine,
)
from self_healing_agent.agent.failure_analyzer import (
    FailureAnalyzer,
)
from self_healing_agent.agent.history import (
    RepairAttempt,
    RepairHistory,
)
from self_healing_agent.agent.history_retriever import (
    RepairHistoryRetriever,
)
from self_healing_agent.agent.history_store import (
    RepairHistoryStore,
)
from self_healing_agent.agent.loop import (
    SelfCorrectionLoop,
)
from self_healing_agent.agent.models import (
    RepairReasoning,
)
from self_healing_agent.agent.reasoner import (
    RepairReasoner,
)
from self_healing_agent.agent.state import (
    AgentState,
)
from self_healing_agent.models.events import (
    ErrorEvent,
)
from self_healing_agent.rag.chunker import (
    DocumentChunk,
)
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
    exception_type: str,
    error_message: str,
) -> AgentState:
    return AgentState(
        error_event=ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            exception_type=exception_type,
            message=error_message,
            file_path=target_file,
            line_number=6,
            function_name="create_customer",
            api_service="stripe",
            api_call="stripe.Customer.create",
            raw_log=error_message,
        )
    )


def create_loop(
    workspace: Path,
    reasoner: RepairReasoner,
    history_store: RepairHistoryStore,
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

    history_retriever = RepairHistoryRetriever(
        history_store,
    )

    return SelfCorrectionLoop(
        architect=CodeArchitect(),
        repair_executor=repair_executor,
        reasoner=reasoner,
        context_builder=ReasoningContextBuilder(),
        decision_engine=RetryDecisionEngine(),
        history=RepairHistory(),
        history_store=history_store,
        history_retriever=history_retriever,
    )


class HistoryFilteringReasoner:
    """Verify unrelated history is not supplied to reasoning."""

    def __init__(self) -> None:
        self.context: ReasoningContext | None = None

    async def reason(
        self,
        context: ReasoningContext,
    ) -> RepairReasoning:
        self.context = context

        if context.historical_repairs:
            raise AssertionError(
                "Unrelated historical repairs were "
                "passed into the reasoning context."
            )

        source_code = context.current_source_code

        if source_code is None:
            raise ValueError(
                "Current source code is required"
            )

        replacement_code = source_code.replace(
            "stripe.Customer.create",
            "stripe.customers.create",
            1,
        )

        patch = CodePatch(
            file_path=context.target_file,
            original_code=source_code,
            replacement_code=replacement_code,
            explanation=(
                "Apply the repair without using "
                "unrelated historical repairs."
            ),
        )

        return RepairReasoning(
            explanation=(
                "No unrelated historical repair was "
                "used for this failure."
            ),
            confidence=0.95,
            patch=patch,
            model="history-filter-test-model",
        )


@pytest.mark.asyncio
async def test_loop_does_not_reuse_unrelated_historical_repairs(
    tmp_path: Path,
) -> None:
    history_database = (
        tmp_path / "repair_history.db"
    )

    history_store = RepairHistoryStore(
        history_database,
    )

    unrelated_attempt = RepairAttempt(
        iteration=1,
        timestamp=datetime.now(timezone.utc),
        target_file="payment_client.py",
        exception_type="ImportError",
        error_message=(
            "payment dependency unavailable"
        ),
        model="previous-model",
        explanation=(
            "Repair an unrelated payment dependency."
        ),
        confidence=0.91,
        original_code=(
            "raise ImportError("
            "'payment dependency unavailable'"
            ")"
        ),
        generated_code=(
            "import payment_sdk"
        ),
        validation_command=(
            "python",
            "-m",
            "pytest",
        ),
        test_passed=True,
        test_output="1 passed",
        failure_type=None,
        failure_summary=None,
        failure_details=None,
        status="success",
    )

    history_store.save(
        unrelated_attempt,
    )

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
        target_file="api_client.py",
        exception_type="AttributeError",
        error_message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
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

    reasoner = HistoryFilteringReasoner()

    loop = create_loop(
        workspace=tmp_path,
        reasoner=reasoner,
        history_store=history_store,
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
    assert result.state.repair_complete is True

    assert reasoner.context is not None

    assert (
        reasoner.context.historical_repairs
        == ()
    )

    assert result.history.count == 1