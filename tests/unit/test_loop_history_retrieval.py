from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from self_healing_agent.agent.architect import CodePatch
from self_healing_agent.agent.context import ReasoningContextBuilder
from self_healing_agent.agent.decision import RetryDecisionEngine
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
from self_healing_agent.agent.loop import SelfCorrectionLoop
from self_healing_agent.agent.models import RepairReasoning
from self_healing_agent.agent.state import AgentState
from self_healing_agent.models.events import ErrorEvent


class HistoryAwareReasoner:
    def __init__(self) -> None:
        self.contexts = []

    async def reason(self, context):
        self.contexts.append(context)

        if context.current_source_code is None:
            raise ValueError(
                "Current source code is required"
            )

        patched_source = context.current_source_code.replace(
            "stripe.Customer.create",
            "stripe.customers.create",
            1,
        )

        patch = CodePatch(
            file_path=context.target_file,
            original_code=context.current_source_code,
            replacement_code=patched_source,
            explanation="Replace deprecated Stripe API.",
        )

        return RepairReasoning(
            explanation="Replace deprecated Stripe API.",
            confidence=0.95,
            patch=patch,
        )


class FakeRepairExecutor:
    async def execute(
        self,
        patch,
        workspace: Path,
        validation_command: list[str],
        state: AgentState,
    ):
        target_file = workspace / patch.file_path

        target_file.write_text(
            patch.replacement_code,
            encoding="utf-8",
        )

        state.record_test_result(
            passed=True,
            output="1 passed",
        )

        state.repair_complete = True

        return object()


def create_state() -> AgentState:
    event = ErrorEvent(
        timestamp=datetime.now(timezone.utc),
        exception_type="AttributeError",
        message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
        file_path="examples/broken_api_client.py",
        line_number=5,
        function_name="create_customer",
        api_service="stripe",
        api_call="stripe.Customer.create",
        raw_log=(
            "AttributeError: module 'stripe' "
            "has no attribute 'Customer'"
        ),
    )

    return AgentState(
        error_event=event,
        max_iterations=3,
    )


def create_workspace(tmp_path: Path) -> None:
    target_file = (
        tmp_path
        / "examples"
        / "broken_api_client.py"
    )

    target_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_file.write_text(
        "import stripe\n\n"
        "def create_customer(email):\n"
        "    return stripe.Customer.create(email=email)\n",
        encoding="utf-8",
    )


def create_historical_attempt() -> RepairAttempt:
    return RepairAttempt(
        iteration=0,
        timestamp=datetime.now(timezone.utc),
        target_file="examples/broken_api_client.py",
        exception_type="AttributeError",
        error_message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
        model="gemini-3.6-flash",
        explanation="Previous successful Stripe migration.",
        confidence=0.95,
        original_code=(
            "stripe.Customer.create(email=email)"
        ),
        generated_code=(
            "stripe.customers.create(email=email)"
        ),
        validation_command=(
            "python",
            "-m",
            "pytest",
            "-q",
        ),
        test_passed=True,
        test_output="1 passed",
        failure_type=None,
        failure_summary=None,
        failure_details=None,
        status="success",
    )


@pytest.mark.asyncio
async def test_loop_passes_previous_successful_repairs_to_reasoner(
    tmp_path: Path,
) -> None:
    create_workspace(tmp_path)

    history_store = RepairHistoryStore(
        tmp_path / "history.db"
    )

    history_store.save(
        create_historical_attempt()
    )

    history = RepairHistory()

    reasoner = HistoryAwareReasoner()

    loop = SelfCorrectionLoop(
        architect=None,
        repair_executor=FakeRepairExecutor(),
        reasoner=reasoner,
        context_builder=ReasoningContextBuilder(),
        decision_engine=RetryDecisionEngine(),
        history=history,
        history_store=history_store,
        history_retriever=RepairHistoryRetriever(
            history_store
        ),
    )

    result = await loop.run(
        state=create_state(),
        workspace=tmp_path,
        validation_command=[
            "python",
            "-m",
            "pytest",
            "-q",
        ],
    )

    assert result.succeeded is True

    assert len(reasoner.contexts) == 1

    context = reasoner.contexts[0]

    assert len(context.historical_repairs) == 1

    historical_repair = (
        context.historical_repairs[0]
    )

    assert (
        historical_repair.target_file
        == "examples/broken_api_client.py"
    )

    assert (
        historical_repair.exception_type
        == "AttributeError"
    )

    assert historical_repair.test_passed is True

    assert (
        historical_repair.generated_code
        == "stripe.customers.create(email=email)"
    )