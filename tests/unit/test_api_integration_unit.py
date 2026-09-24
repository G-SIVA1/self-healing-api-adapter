from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from self_healing_agent.agent.architect import (
    CodeArchitect,
    CodePatch,
)
from self_healing_agent.agent.context import (
    ReasoningContextBuilder,
    ReasoningContext,
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


class IntegrationApiLLMClient:
    """Fake LLM representing an API migration decision."""

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
from pathlib import Path


def create_customer() -> str:
    response = Path("api_response.txt").read_text(
        encoding="utf-8"
    )

    return response
"""

        return json.dumps(
            {
                "explanation": (
                    "The API client was using an obsolete "
                    "implementation. The replacement uses "
                    "the currently supported API response."
                ),
                "confidence": 0.98,
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
                "module 'legacy_api' has no attribute "
                "'create_customer'"
            ),
            file_path=target_file,
            line_number=5,
            function_name="create_customer",
            api_service="demo_api",
            api_call="legacy_api.create_customer",
            raw_log=(
                "AttributeError: "
                "module 'legacy_api' has no attribute "
                "'create_customer'"
            ),
        )
    )


def create_reasoner(
    client: IntegrationApiLLMClient,
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

    return SelfCorrectionLoop(
        architect=CodeArchitect(),
        repair_executor=repair_executor,
        reasoner=reasoner,
        context_builder=ReasoningContextBuilder(),
        decision_engine=RetryDecisionEngine(),
    )


@pytest.mark.asyncio
async def test_api_compatibility_repair_end_to_end(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "api_client.py"

    original_code = """\
import legacy_api


def create_customer() -> str:
    return legacy_api.create_customer()
"""

    source_file.write_text(
        original_code,
        encoding="utf-8",
    )

    api_response = tmp_path / "api_response.txt"

    api_response.write_text(
        "customer-created",
        encoding="utf-8",
    )

    test_file = tmp_path / "test_api_client.py"

    test_file.write_text(
        """\
from api_client import create_customer


def test_customer_api():
    result = create_customer()

    assert result == "customer-created"
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
                    "The legacy customer creation API is "
                    "no longer supported. Applications "
                    "must use the current customer response "
                    "interface."
                ),
                source="demo_api_migration.md",
                chunk_index=0,
            )
        ]
    )

    client = IntegrationApiLLMClient()

    reasoner = create_reasoner(
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
    assert state.iteration == 1

    assert client.calls == 1
    assert len(client.prompts) == 1

    assert (
        "legacy_api.create_customer"
        in client.prompts[0]
    )

    assert (
        "demo_api_migration.md"
        in client.prompts[0]
    )

    patched_source = source_file.read_text(
        encoding="utf-8",
    )

    assert (
        "Path(\"api_response.txt\")"
        in patched_source
    )

    assert (
        "legacy_api.create_customer"
        not in patched_source
    )

    assert len(result.execution_results) == 1