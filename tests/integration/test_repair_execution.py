from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from self_healing_agent.agent.architect import CodePatch
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
from self_healing_agent.agent.models import (
    RepairReasoning,
)
from self_healing_agent.agent.prompt import (
    RepairPromptBuilder,
)
from self_healing_agent.agent.reasoner import (
    LLMRepairReasoner,
)
from self_healing_agent.agent.response_parser import (
    RepairResponseParser,
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


class IntegrationFakeLLMClient:
    """Fake LLM used for full-loop integration testing."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(
        self,
        prompt: str,
    ) -> str:
        self.calls += 1
        self.prompts.append(prompt)

        replacement_code = """\
import stripe


def create_customer():
    return stripe.customers.create(
        email="user@example.com"
    )
"""

        return json.dumps(
            {
                "explanation": (
                    "The legacy Stripe Customer API "
                    "must be replaced with the modern "
                    "customers API."
                ),
                "confidence": 0.97,
                "replacement_code": replacement_code,
            }
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


def create_llm_reasoner(
    client: IntegrationFakeLLMClient,
) -> LLMRepairReasoner:
    return LLMRepairReasoner(
        client=client,
        prompt_builder=RepairPromptBuilder(),
        response_parser=RepairResponseParser(),
    )


def create_loop(
    workspace: Path,
    reasoner: LLMRepairReasoner,
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

    from self_healing_agent.agent.architect import (
        CodeArchitect,
    )

    return SelfCorrectionLoop(
        architect=CodeArchitect(),
        repair_executor=repair_executor,
        reasoner=reasoner,
        context_builder=ReasoningContextBuilder(),
        decision_engine=RetryDecisionEngine(),
    )


@pytest.mark.asyncio
async def test_llm_reasoner_repairs_api_end_to_end(
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
                    "The legacy Customer API is no longer "
                    "supported. Replace "
                    "stripe.Customer.create with "
                    "stripe.customers.create."
                ),
                source="stripe_migration.md",
                chunk_index=0,
            )
        ]
    )

    client = IntegrationFakeLLMClient()

    reasoner = create_llm_reasoner(
        client,
    )

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

    assert result.final_decision == (
        DecisionAction.STOP
    )

    assert state.repair_complete is True

    assert state.test_passed is True

    assert state.failure_reason is None

    assert state.failure_analysis is None

    assert state.iteration == 1

    assert len(result.execution_results) == 1

    assert client.calls == 1

    assert len(client.prompts) == 1

    assert (
        "stripe.Customer.create"
        in client.prompts[0]
    )

    assert (
        "stripe.customers.create"
        in client.prompts[0]
    )

    patched_source = source_file.read_text(
        encoding="utf-8",
    )

    assert (
        "stripe.customers.create"
        in patched_source
    )

    assert (
        "stripe.Customer.create"
        not in patched_source
    )