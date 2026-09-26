from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from self_healing_agent.agent.architect import CodeArchitect
from self_healing_agent.agent.context import ReasoningContextBuilder
from self_healing_agent.agent.decision import RetryDecisionEngine
from self_healing_agent.agent.failure_analyzer import FailureAnalyzer
from self_healing_agent.agent.loop import SelfCorrectionLoop
from self_healing_agent.agent.reasoner import (
    DeterministicRepairReasoner,
)
from self_healing_agent.agent.state import AgentState
from self_healing_agent.git.repair_manager import GitRepairManager
from self_healing_agent.git.repository import GitRepository
from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.sandbox.executor import SandboxExecutor
from self_healing_agent.sandbox.patcher import SafePatchApplier
from self_healing_agent.sandbox.repair_executor import RepairExecutor
from self_healing_agent.sandbox.security import (
    SandboxSecurityPolicy,
)


def run_git_command(
    repository_path: Path,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )


def initialize_git_repository(
    repository_path: Path,
) -> None:
    run_git_command(
        repository_path,
        [
            "init",
            "-b",
            "main",
        ],
    )

    run_git_command(
        repository_path,
        [
            "config",
            "user.name",
            "Test User",
        ],
    )

    run_git_command(
        repository_path,
        [
            "config",
            "user.email",
            "test@example.com",
        ],
    )

    readme_path = repository_path / "README.md"

    readme_path.write_text(
        "Initial repository content\n",
        encoding="utf-8",
    )

    run_git_command(
        repository_path,
        [
            "add",
            "README.md",
        ],
    )

    run_git_command(
        repository_path,
        [
            "commit",
            "-m",
            "Initial commit",
        ],
    )


def configure_local_git_remote(
    repository_path: Path,
) -> Path:
    remote_path = (
        repository_path.parent
        / f"{repository_path.name}-origin.git"
    ).resolve()

    if remote_path.exists():
        subprocess.run(
            [
                "git",
                "init",
                "--bare",
                str(remote_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        subprocess.run(
            [
                "git",
                "init",
                "--bare",
                str(remote_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    bare_check = subprocess.run(
        [
            "git",
            "rev-parse",
            "--is-bare-repository",
        ],
        cwd=remote_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert bare_check.stdout.strip() == "true"

    remote_url = remote_path.as_uri()

    run_git_command(
        repository_path,
        [
            "remote",
            "remove",
            "origin",
        ],
    ) if subprocess.run(
        [
            "git",
            "remote",
            "get-url",
            "origin",
        ],
        cwd=repository_path,
        capture_output=True,
        text=True,
    ).returncode == 0 else None

    run_git_command(
        repository_path,
        [
            "remote",
            "add",
            "origin",
            remote_url,
        ],
    )

    configured_remote = subprocess.run(
        [
            "git",
            "remote",
            "get-url",
            "origin",
        ],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert configured_remote.stdout.strip() == remote_url

    return remote_path


def create_stripe_source(
    repository_path: Path,
) -> str:
    source_file = (
        repository_path
        / "examples"
        / "stripe_client.py"
    )

    source_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_code = (
        "import stripe\n\n"
        "stripe.Customer.create(\n"
        "    email='user@example.com'\n"
        ")\n"
    )

    source_file.write_text(
        original_code,
        encoding="utf-8",
    )

    run_git_command(
        repository_path,
        [
            "add",
            "examples/stripe_client.py",
        ],
    )

    run_git_command(
        repository_path,
        [
            "commit",
            "-m",
            "Add broken Stripe client",
        ],
    )

    return original_code


def create_loop(
    repository_path: Path,
) -> tuple[
    SelfCorrectionLoop,
    GitRepository,
]:
    repository = GitRepository(
        repository_path,
    )

    git_manager = GitRepairManager(
        repository,
    )

    security_policy = SandboxSecurityPolicy(
        allowed_working_directory=repository_path,
    )

    sandbox_executor = SandboxExecutor(
        security_policy=security_policy,
    )

    patch_applier = SafePatchApplier()

    failure_analyzer = FailureAnalyzer()

    repair_executor = RepairExecutor(
        patch_applier=patch_applier,
        sandbox_executor=sandbox_executor,
        failure_analyzer=failure_analyzer,
    )

    architect = CodeArchitect()

    reasoner = DeterministicRepairReasoner()

    context_builder = ReasoningContextBuilder()

    decision_engine = RetryDecisionEngine()

    loop = SelfCorrectionLoop(
        architect=architect,
        repair_executor=repair_executor,
        reasoner=reasoner,
        context_builder=context_builder,
        decision_engine=decision_engine,
        git_repair_manager=git_manager,
    )

    return loop, repository


def create_error_event() -> ErrorEvent:
    return ErrorEvent(
        timestamp=datetime.now(
            timezone.utc,
        ),
        file_path="examples/stripe_client.py",
        line_number=3,
        function_name=None,
        exception_type="AttributeError",
        message=(
            "module 'stripe' has no attribute "
            "'Customer'"
        ),
        raw_log=(
            "AttributeError: module 'stripe' "
            "has no attribute 'Customer'"
        ),
        api_service="Stripe",
        api_call="Customer Create",
    )


def create_state(
    original_code: str,
    max_iterations: int = 1,
) -> AgentState:
    return AgentState(
        error_event=create_error_event(),
        original_code=original_code,
        documentation=[],
        max_iterations=max_iterations,
    )


def successful_validation_command() -> list[str]:
    validation_script = (
        "from pathlib import Path\n"
        "text = Path("
        "'examples/stripe_client.py'"
        ").read_text()\n"
        "assert "
        "'stripe.customers.create' in text\n"
    )

    return [
        "python",
        "-c",
        validation_script,
    ]


def failing_validation_command() -> list[str]:
    validation_script = (
        "from pathlib import Path\n"
        "text = Path("
        "'examples/stripe_client.py'"
        ").read_text()\n"
        "assert "
        "'this_condition_must_fail' in text\n"
    )

    return [
        "python",
        "-c",
        validation_script,
    ]


@pytest.mark.asyncio
async def test_self_correction_loop_commits_verified_repair(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    configure_local_git_remote(
        tmp_path,
    )

    original_code = create_stripe_source(
        tmp_path,
    )

    loop, repository = create_loop(
        tmp_path,
    )

    state = create_state(
        original_code,
    )

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=successful_validation_command(),
        git_service_name="Stripe",
        git_api_call="Customer Create",
    )

    assert result.succeeded

    assert result.state.repair_complete

    assert result.state.test_passed

    assert result.git_commit is not None

    assert (
        result.git_commit.branch.name
        == "repair/stripe-customer-create"
    )

    assert len(
        result.git_commit.commit_sha
    ) == 40

    assert (
        await repository.current_branch()
        == "repair/stripe-customer-create"
    )

    committed_file = subprocess.run(
        [
            "git",
            "show",
            (
                f"{result.git_commit.commit_sha}:"
                "examples/stripe_client.py"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "stripe.customers.create"
        in committed_file.stdout
    )

    assert (
        "stripe.Customer.create"
        not in committed_file.stdout
    )


@pytest.mark.asyncio
async def test_failed_validation_does_not_create_git_commit(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    original_code = create_stripe_source(
        tmp_path,
    )

    loop, repository = create_loop(
        tmp_path,
    )

    state = create_state(
        original_code,
        max_iterations=1,
    )

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=failing_validation_command(),
        git_service_name="Stripe",
        git_api_call="Customer Create",
    )

    assert not result.succeeded

    assert not result.state.repair_complete

    assert not result.state.test_passed

    assert result.git_commit is None

    assert (
        await repository.current_branch()
        == "main"
    )

    assert not await repository.branch_exists(
        "repair/stripe-customer-create"
    )

    log_result = subprocess.run(
        [
            "git",
            "log",
            "--oneline",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    commits = [
        line
        for line in log_result.stdout.splitlines()
        if line.strip()
    ]

    assert len(commits) == 2