from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from self_healing_agent.agent.architect import CodeArchitect
from self_healing_agent.agent.context import ReasoningContextBuilder
from self_healing_agent.agent.decision import RetryDecisionEngine
from self_healing_agent.agent.failure_analyzer import FailureAnalyzer
from self_healing_agent.agent.llm_client import (
    AsyncLLMClient,
    GeminiTransport,
)
from self_healing_agent.agent.loop import SelfCorrectionLoop
from self_healing_agent.agent.prompt import RepairPromptBuilder
from self_healing_agent.agent.reasoner import LLMRepairReasoner
from self_healing_agent.agent.response_parser import (
    RepairResponseParser,
)
from self_healing_agent.agent.state import AgentState
from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.rag.chunker import DocumentChunk
from self_healing_agent.sandbox.executor import SandboxExecutor
from self_healing_agent.sandbox.patcher import SafePatchApplier
from self_healing_agent.sandbox.repair_executor import RepairExecutor
from self_healing_agent.sandbox.security import SandboxSecurityPolicy


def create_broken_project(
    workspace: Path,
) -> Path:
    """Create an isolated broken API project."""

    source_file = (
        workspace
        / "examples"
        / "broken_api_client.py"
    )

    source_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_file.write_text(
        """\
import stripe


def create_customer(email: str) -> object:
    return stripe.Customer.create(
        email=email,
    )
""",
        encoding="utf-8",
    )

    test_file = workspace / "test_repair.py"

    test_file.write_text(
        """\
from pathlib import Path


def test_customer_api_was_migrated() -> None:
    source_file = (
        Path(__file__).parent
        / "examples"
        / "broken_api_client.py"
    )

    source = source_file.read_text(
        encoding="utf-8",
    )

    assert "stripe.customers.create" in source
    assert "stripe.Customer.create" not in source
""",
        encoding="utf-8",
    )

    return source_file


def create_error_event(
    source_file: Path,
    workspace: Path,
) -> ErrorEvent:
    """Create the runtime error that triggered the repair."""

    relative_file = source_file.relative_to(
        workspace,
    )

    return ErrorEvent(
        timestamp=datetime.now(timezone.utc),
        exception_type="AttributeError",
        message=(
            "module 'stripe' has no attribute "
            "'Customer'"
        ),
        file_path=str(relative_file),
        line_number=5,
        function_name="create_customer",
        api_service="stripe",
        api_call="stripe.Customer.create",
        raw_log=(
            'AttributeError: module "stripe" '
            'has no attribute "Customer"'
        ),
    )


def create_documentation() -> list[DocumentChunk]:
    """Create documentation retrieved by the RAG layer."""

    return [
        DocumentChunk(
            content="""\
# Stripe API Migration Guide

The legacy Customer API is no longer supported.

### Legacy API

stripe.Customer.create(
    email="user@example.com"
)

### Current API

Use the modern Stripe client interface:

stripe.customers.create(
    email="user@example.com"
)

The legacy stripe.Customer.create method should be
replaced with stripe.customers.create.

If an application reports:

AttributeError: module 'stripe' has no attribute 'Customer'

the application may be using the legacy Customer API.

The migration should preserve the existing function
inputs whenever possible.
""",
            source="stripe_migration.md",
            chunk_index=0,
        ),
    ]


async def main() -> None:
    """Run the real Gemini self-correction workflow."""

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )

    model = os.getenv(
        "LLM_MODEL",
        "gemini-3.6-flash",
    )

    with TemporaryDirectory(
        prefix="self_healing_test_",
    ) as temporary_directory:

        workspace = Path(
            temporary_directory,
        )

        print(
            f"Temporary workspace: {workspace}"
        )

        source_file = create_broken_project(
            workspace,
        )

        error_event = create_error_event(
            source_file,
            workspace,
        )

        state = AgentState(
            error_event=error_event,
            max_iterations=3,
        )

        state.add_documentation(
            create_documentation(),
        )

        transport = GeminiTransport(
            api_key=api_key,
        )

        client = AsyncLLMClient(
            transport=transport,
            model=model,
            timeout_seconds=30,
            max_retries=2,
            base_delay_seconds=0.5,
        )

        reasoner = LLMRepairReasoner(
            client=client,
            prompt_builder=RepairPromptBuilder(),
            response_parser=RepairResponseParser(),
        )

        security_policy = SandboxSecurityPolicy(
            allowed_working_directory=workspace,
        )

        sandbox_executor = SandboxExecutor(
            security_policy=security_policy,
            timeout_seconds=30,
        )

        repair_executor = RepairExecutor(
            patch_applier=SafePatchApplier(),
            sandbox_executor=sandbox_executor,
            failure_analyzer=FailureAnalyzer(),
        )

        loop = SelfCorrectionLoop(
            architect=CodeArchitect(),
            repair_executor=repair_executor,
            reasoner=reasoner,
            context_builder=ReasoningContextBuilder(),
            decision_engine=RetryDecisionEngine(),
        )

        print(
            "\nStarting Gemini self-correction loop...\n"
        )

        result = await loop.run(
            state=state,
            workspace=workspace,
            validation_command=[
                "python",
                "-m",
                "pytest",
                "-q",
            ],
        )

        print("\n===================================")
        print("GEMINI SELF-CORRECTION RESULT")
        print("===================================")

        print(
            f"Success: {result.succeeded}"
        )

        print(
            f"Decision: "
            f"{result.final_decision.value}"
        )

        print(
            f"Iterations: "
            f"{result.state.iteration}"
        )

        print(
            f"Repair Complete: "
            f"{result.state.repair_complete}"
        )

        print("\nFinal Source:")
        print("-----------------------------------")

        final_source = source_file.read_text(
            encoding="utf-8",
        )

        print(final_source)

        print("-----------------------------------")

        print("\nExecution Results:")

        for index, execution_result in enumerate(
            result.execution_results,
            start=1,
        ):
            execution = execution_result.execution

            print(
                f"\nIteration {index}"
            )

            print(
                f"Return Code: "
                f"{execution.return_code}"
            )

            print(
                f"Succeeded: "
                f"{execution.succeeded}"
            )

            print(
                f"Timed Out: "
                f"{execution.timed_out}"
            )

            print(
                f"Duration: "
                f"{execution.duration_seconds:.3f}s"
            )

            if execution.stdout:
                print("\nSTDOUT:")
                print(execution.stdout)

            if execution.stderr:
                print("\nSTDERR:")
                print(execution.stderr)

        print("\n===================================")

        if not result.succeeded:
            raise RuntimeError(
                "Gemini self-correction test failed"
            )


if __name__ == "__main__":
    asyncio.run(main())