from __future__ import annotations

import subprocess
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from self_healing_agent.agent.architect import CodeArchitect
from self_healing_agent.agent.context import ReasoningContextBuilder
from self_healing_agent.agent.decision import RetryDecisionEngine
from self_healing_agent.agent.failure_analyzer import FailureAnalyzer
from self_healing_agent.agent.loop import SelfCorrectionLoop
from self_healing_agent.agent.reasoner import DeterministicRepairReasoner
from self_healing_agent.agent.state import AgentState
from self_healing_agent.git.github_client import GitHubPullRequestClient
from self_healing_agent.git.repair_manager import GitRepairManager
from self_healing_agent.git.repository import GitRepository
from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.sandbox.executor import SandboxExecutor
from self_healing_agent.sandbox.patcher import SafePatchApplier
from self_healing_agent.sandbox.repair_executor import RepairExecutor
from self_healing_agent.sandbox.security import SandboxSecurityPolicy


class MockGitHubHandler(BaseHTTPRequestHandler):
    """Local GitHub API replacement for integration testing."""

    created_pull_requests: list[dict[str, object]] = []

    def log_message(
        self,
        format: str,
        *args: object,
    ) -> None:
        """Disable HTTP server logging."""

    def do_GET(self) -> None:
        """Return no existing pull requests."""

        if self.path.startswith("/repos/"):
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.end_headers()
            self.wfile.write(b"[]")
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        """Create a fake GitHub pull request."""

        if (
            self.path
            != "/repos/G-SIVA1/self-healing-api-adapter/pulls"
        ):
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(
            self.headers.get("Content-Length", "0")
        )

        request_body = self.rfile.read(
            content_length
        )

        MockGitHubHandler.created_pull_requests.append(
            {
                "body": request_body.decode("utf-8"),
                "authorization": self.headers.get(
                    "Authorization",
                    "",
                ),
            }
        )

        response_body = (
            b"{"
            b'"number":42,'
            b'"title":"Fix Stripe Customer Create compatibility",'
            b'"body":"Automated self-healing repair",'
            b'"state":"open",'
            b'"html_url":"http://127.0.0.1/pull/42",'
            b'"head":{"ref":"repair/stripe-customer-create"},'
            b'"base":{"ref":"main"}'
            b"}"
        )

        self.send_response(201)

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(response_body)),
        )

        self.end_headers()

        self.wfile.write(response_body)


def start_mock_github_server() -> tuple[
    HTTPServer,
    threading.Thread,
    str,
]:
    """Start a local HTTP server emulating GitHub."""

    MockGitHubHandler.created_pull_requests = []

    server = HTTPServer(
        ("127.0.0.1", 0),
        MockGitHubHandler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    host, port = server.server_address

    base_url = f"http://{host}:{port}"

    return server, thread, base_url


def initialize_git_repository(
    repository_path: Path,
) -> None:
    """Create a local Git repository."""

    subprocess.run(
        [
            "git",
            "init",
            "-b",
            "main",
        ],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            "git",
            "config",
            "user.name",
            "Test User",
        ],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "test@example.com",
        ],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    readme_path = repository_path / "README.md"

    readme_path.write_text(
        "Initial repository content\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "git",
            "add",
            "README.md",
        ],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Initial commit",
        ],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )


def configure_local_git_remote(
    repository_path: Path,
) -> Path:
    """Create a local bare Git remote."""

    remote_path = (
        repository_path.parent
        / f"{repository_path.name}-remote.git"
    )

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

    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            str(remote_path),
        ],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    return remote_path


def build_agent_loop(
    workspace: Path,
    git_manager: GitRepairManager,
) -> SelfCorrectionLoop:
    """Build the production agent dependency graph."""

    architect = CodeArchitect()

    security_policy = SandboxSecurityPolicy(
        allowed_executables=frozenset(
            {
                "python",
                "python.exe",
                "pytest",
                "pytest.exe",
            }
        ),
        allowed_working_directory=workspace,
    )

    sandbox_executor = SandboxExecutor(
        security_policy=security_policy,
        timeout_seconds=10.0,
    )

    patch_applier = SafePatchApplier()

    failure_analyzer = FailureAnalyzer()

    repair_executor = RepairExecutor(
        patch_applier=patch_applier,
        sandbox_executor=sandbox_executor,
        failure_analyzer=failure_analyzer,
    )

    reasoner = DeterministicRepairReasoner()

    context_builder = ReasoningContextBuilder()

    decision_engine = RetryDecisionEngine()

    return SelfCorrectionLoop(
        architect=architect,
        repair_executor=repair_executor,
        reasoner=reasoner,
        context_builder=context_builder,
        decision_engine=decision_engine,
        git_repair_manager=git_manager,
    )


@pytest.mark.asyncio
async def test_agent_loop_git_github_integration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test verified repair -> Git commit -> GitHub PR."""

    initialize_git_repository(tmp_path)

    configure_local_git_remote(tmp_path)

    repaired_file = (
        tmp_path
        / "examples"
        / "broken_api_client.py"
    )

    repaired_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    repaired_file.write_text(
        "import stripe\n\n"
        "stripe.customers.create(\n"
        "    email='user@example.com'\n"
        ")\n",
        encoding="utf-8",
    )

    server, thread, base_url = (
        start_mock_github_server()
    )

    try:
        monkeypatch.setattr(
            GitHubPullRequestClient,
            "API_BASE_URL",
            base_url,
        )

        github_client = GitHubPullRequestClient(
            token="test-token",
            timeout_seconds=10.0,
        )

        repository = GitRepository(
            tmp_path
        )

        git_manager = GitRepairManager(
            repository=repository,
            pull_request_client=github_client,
        )

        loop = build_agent_loop(
            workspace=tmp_path,
            git_manager=git_manager,
        )

        error_event = ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            exception_type="RuntimeError",
            message=(
                "Stripe Customer.create "
                "is deprecated"
            ),
            file_path=str(repaired_file),
            line_number=3,
            function_name=None,
            api_service="Stripe",
            api_call="Customer Create",
            raw_log=(
                "RuntimeError: "
                "Stripe Customer.create is deprecated"
            ),
        )

        state = AgentState(
            error_event=error_event,
            max_iterations=1,
        )

        state.repair_complete = True
        state.test_passed = True
        state.iteration = 1

        loop._commit_verified_repair = AsyncMock(
            wraps=loop._commit_verified_repair,
        )

        result = await loop.run(
            state=state,
            workspace=tmp_path,
            validation_command=[
                "python",
                "-m",
                "pytest",
            ],
            git_service_name="Stripe",
            git_api_call="Customer Create",
            git_repository=(
                "G-SIVA1/"
                "self-healing-api-adapter"
            ),
            git_target_branch="main",
            git_pull_request_title=(
                "Fix Stripe Customer Create "
                "compatibility"
            ),
            git_pull_request_description=(
                "Automated self-healing repair"
            ),
        )

        assert result.git_commit is not None

        assert result.git_pull_request is not None

        assert (
            result.git_commit.branch.name
            == "repair/stripe-customer-create"
        )

        assert (
            result.git_pull_request.pull_request.number
            == 42
        )

        assert (
            await repository.current_branch()
            == "repair/stripe-customer-create"
        )

        assert len(
            result.git_commit.commit_sha
        ) == 40

        commit_result = subprocess.run(
            [
                "git",
                "show",
                "--format=%s",
                "--no-patch",
                result.git_commit.commit_sha,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        assert (
            commit_result.stdout.strip()
            == "Fix Stripe Customer Create "
            "compatibility"
        )

        repaired_source_result = subprocess.run(
            [
                "git",
                "show",
                (
                    f"{result.git_commit.commit_sha}:"
                    "examples/broken_api_client.py"
                ),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

        repaired_source = (
            repaired_source_result.stdout
        )

        assert (
            "stripe.customers.create"
            in repaired_source
        )

        assert (
            "stripe.Customer.create"
            not in repaired_source
        )

        assert len(
            MockGitHubHandler.created_pull_requests
        ) == 1

        request = (
            MockGitHubHandler
            .created_pull_requests[0]
        )

        assert (
            request["authorization"]
            == "Bearer test-token"
        )

        request_body = str(
            request["body"]
        )

        assert '"head"' in request_body
        assert '"base"' in request_body

        loop._commit_verified_repair.assert_awaited_once()

    finally:
        server.shutdown()
        server.server_close()

        thread.join(
            timeout=5.0
        )
@pytest.mark.asyncio
async def test_github_pull_request_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify repeated PR creation reuses the existing PR."""

    initialize_git_repository(tmp_path)

    configure_local_git_remote(tmp_path)

    server, thread, base_url = (
        start_mock_github_server()
    )

    try:
        monkeypatch.setattr(
            GitHubPullRequestClient,
            "API_BASE_URL",
            base_url,
        )

        github_client = GitHubPullRequestClient(
            token="test-token",
            timeout_seconds=10.0,
        )

        repository = GitRepository(
            tmp_path
        )

        git_manager = GitRepairManager(
            repository=repository,
            pull_request_client=github_client,
        )

        repaired_file = (
            tmp_path
            / "examples"
            / "broken_api_client.py"
        )

        repaired_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        repaired_file.write_text(
            "import stripe\n\n"
            "stripe.customers.create(\n"
            "    email='user@example.com'\n"
            ")\n",
            encoding="utf-8",
        )

        commit = await git_manager.commit_verified_repair(
            service_name="Stripe",
            api_call="Customer Create",
            files=[
                "examples/broken_api_client.py",
            ],
            commit_message=(
                "Fix Stripe Customer Create "
                "compatibility"
            ),
            test_passed=True,
            iteration=1,
        )

        first_result = (
            await git_manager.create_pull_request_for_commit(
                commit=commit,
                repository=(
                    "G-SIVA1/"
                    "self-healing-api-adapter"
                ),
                target_branch="main",
                title=(
                    "Fix Stripe Customer Create "
                    "compatibility"
                ),
                description=(
                    "Automated self-healing repair"
                ),
            )
        )

        assert (
            first_result.pull_request.number
            == 42
        )

        assert len(
            MockGitHubHandler.created_pull_requests
        ) == 1

        original_do_get = (
            MockGitHubHandler.do_GET
        )

        def existing_pull_request_get(
            handler: BaseHTTPRequestHandler,
        ) -> None:
            """Return the already-created PR."""

            if handler.path.startswith("/repos/"):
                response_body = (
                    b"[{"
                    b'"number":42,'
                    b'"title":"Fix Stripe Customer Create compatibility",'
                    b'"body":"Automated self-healing repair",'
                    b'"state":"open",'
                    b'"html_url":"http://127.0.0.1/pull/42",'
                    b'"head":{"ref":"repair/stripe-customer-create"},'
                    b'"base":{"ref":"main"}'
                    b"}]"
                )

                handler.send_response(200)

                handler.send_header(
                    "Content-Type",
                    "application/json",
                )

                handler.send_header(
                    "Content-Length",
                    str(len(response_body)),
                )

                handler.end_headers()

                handler.wfile.write(
                    response_body
                )

                return

            original_do_get()

        MockGitHubHandler.do_GET = (
            existing_pull_request_get
        )

        second_result = (
            await git_manager.create_pull_request_for_commit(
                commit=commit,
                repository=(
                    "G-SIVA1/"
                    "self-healing-api-adapter"
                ),
                target_branch="main",
                title=(
                    "Fix Stripe Customer Create "
                    "compatibility"
                ),
                description=(
                    "Automated self-healing repair"
                ),
            )
        )

        assert (
            second_result.pull_request.number
            == 42
        )

        assert (
            second_result.pull_request
            == first_result.pull_request
        )

        assert len(
            MockGitHubHandler.created_pull_requests
        ) == 1

    finally:
        server.shutdown()
        server.server_close()

        thread.join(
            timeout=5.0
        )