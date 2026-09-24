from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from self_healing_agent.agent.architect import CodePatch
from self_healing_agent.agent.context import ReasoningContextBuilder
from self_healing_agent.agent.decision import RetryDecisionEngine
from self_healing_agent.agent.history import RepairHistory
from self_healing_agent.agent.loop import SelfCorrectionLoop
from self_healing_agent.agent.models import RepairReasoning
from self_healing_agent.agent.state import AgentState
from self_healing_agent.models.events import ErrorEvent


class FakeReasoner:
    """Deterministic reasoner used for history tests."""

    async def reason(self, context):
        source = context.current_source_code

        if source is None:
            raise ValueError(
                "Current source code is required"
            )

        patched_source = source.replace(
            "stripe.Customer.create",
            "stripe.customers.create",
            1,
        )

        patch = CodePatch(
            file_path=context.target_file,
            original_code=source,
            replacement_code=patched_source,
            explanation="Replace deprecated API.",
        )

        return RepairReasoning(
            explanation="Replace deprecated Stripe API.",
            confidence=0.95,
            patch=patch,
        )


class FakeRepairExecutor:
    """Executor that simulates a successful repair."""

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
    """Create a deterministic AgentState for testing."""

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


def create_loop(
    history: RepairHistory,
) -> SelfCorrectionLoop:
    """Create a self-correction loop for testing."""

    return SelfCorrectionLoop(
        architect=None,
        repair_executor=FakeRepairExecutor(),
        reasoner=FakeReasoner(),
        context_builder=ReasoningContextBuilder(),
        decision_engine=RetryDecisionEngine(),
        history=history,
    )


def create_broken_source(
    tmp_path: Path,
) -> None:
    """Create the broken API client used by the tests."""

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


@pytest.mark.asyncio
async def test_loop_records_successful_repair(
    tmp_path: Path,
) -> None:
    create_broken_source(tmp_path)

    state = create_state()
    history = RepairHistory()

    loop = create_loop(history)

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=[
            "python",
            "-m",
            "pytest",
            "-q",
        ],
    )

    assert result.succeeded is True

    assert result.history.count == 1

    attempt = result.history.latest

    assert attempt is not None

    assert attempt.iteration == 0

    assert (
        attempt.target_file
        == "examples/broken_api_client.py"
    )

    assert (
        attempt.exception_type
        == "AttributeError"
    )

    assert attempt.test_passed is True

    assert attempt.status == "success"

    assert attempt.confidence == 0.95

    assert attempt.original_code is not None

    assert (
        "stripe.Customer.create"
        in attempt.original_code
    )

    assert attempt.generated_code is not None

    assert (
        "stripe.customers.create"
        in attempt.generated_code
    )


@pytest.mark.asyncio
async def test_loop_exposes_history_instance(
    tmp_path: Path,
) -> None:
    create_broken_source(tmp_path)

    state = create_state()
    history = RepairHistory()

    loop = create_loop(history)

    await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=[
            "python",
            "-m",
            "pytest",
            "-q",
        ],
    )

    assert loop.history is history
    assert loop.history.count == 1