from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest

from self_healing_agent.agent.architect import (
    CodeArchitect,
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


class CustomerAPIHandler(BaseHTTPRequestHandler):
    """Small local HTTP API used for deterministic testing."""

    def do_GET(self) -> None:
        if self.path == "/customer":
            body = json.dumps(
                {
                    "customer_id": "customer-001",
                    "name": "Siva",
                }
            ).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)
            return

        body = b'{"error":"not_found"}'

        self.send_response(404)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(
        self,
        format: str,
        *args: object,
    ) -> None:
        return


class HTTPIntegrationFakeLLMClient:
    """Fake LLM producing the HTTP compatibility repair."""

    def __init__(
        self,
        base_url: str,
    ) -> None:
        self.base_url = base_url
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(
        self,
        prompt: str,
    ) -> str:
        self.calls += 1
        self.prompts.append(prompt)

        replacement_code = f"""\
import httpx


def get_customer() -> dict:
    response = httpx.get(
        "{self.base_url}/customer",
        timeout=5.0,
    )

    response.raise_for_status()

    return response.json()
"""

        return json.dumps(
            {
                "explanation": (
                    "The client must use the current "
                    "customer endpoint instead of the "
                    "deprecated endpoint."
                ),
                "confidence": 0.98,
                "replacement_code": replacement_code,
            }
        )


def create_state(
    target_file: str,
    base_url: str,
) -> AgentState:
    return AgentState(
        error_event=ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            exception_type="HTTPStatusError",
            message=(
                "The legacy customer endpoint returned "
                "HTTP 404."
            ),
            file_path=target_file,
            line_number=5,
            function_name="get_customer",
            api_service="local_customer_api",
            api_call="GET /legacy-customer",
            raw_log=(
                "HTTPStatusError: "
                "GET /legacy-customer returned 404"
            ),
        )
    )


def create_reasoner(
    client: HTTPIntegrationFakeLLMClient,
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
async def test_self_healing_repairs_http_api_client(
    tmp_path: Path,
) -> None:
    server = HTTPServer(
        ("127.0.0.1", 0),
        CustomerAPIHandler,
    )

    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    server_thread.start()

    try:
        host, port = server.server_address
        base_url = f"http://{host}:{port}"

        response = httpx.get(
            f"{base_url}/customer",
            timeout=5.0,
        )

        assert response.status_code == 200

        source_file = tmp_path / "api_client.py"

        original_code = """\
import httpx


def get_customer() -> dict:
    response = httpx.get(
        "http://127.0.0.1:9999/legacy-customer",
        timeout=5.0,
    )

    response.raise_for_status()

    return response.json()
"""

        source_file.write_text(
            original_code,
            encoding="utf-8",
        )

        test_file = tmp_path / "test_api_client.py"

        test_file.write_text(
            f"""\
from api_client import get_customer


def test_customer_api():
    customer = get_customer()

    assert customer["customer_id"] == "customer-001"
    assert customer["name"] == "Siva"
""",
            encoding="utf-8",
        )

        state = create_state(
            target_file="api_client.py",
            base_url=base_url,
        )

        state.original_code = original_code

        state.add_documentation(
            [
                DocumentChunk(
                    content=(
                        "The legacy customer endpoint "
                        "GET /legacy-customer is no longer "
                        "supported. Use GET /customer instead. "
                        "The response contains customer_id "
                        "and name."
                    ),
                    source="customer_api_migration.md",
                    chunk_index=0,
                )
            ]
        )

        client = HTTPIntegrationFakeLLMClient(
            base_url=base_url,
        )

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
            "GET /legacy-customer"
            in client.prompts[0]
        )

        assert (
            "GET /customer"
            in client.prompts[0]
        )

        patched_source = source_file.read_text(
            encoding="utf-8",
        )

        assert (
            f"{base_url}/customer"
            in patched_source
        )

        assert (
            "/legacy-customer"
            not in patched_source
        )

        assert len(result.execution_results) == 1

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)