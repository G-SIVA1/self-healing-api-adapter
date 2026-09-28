from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from self_healing_agent.git.github_client import (
    GitHubPullRequestClient,
)
from self_healing_agent.git.pull_request import (
    PullRequestError,
)
from self_healing_agent.git.repository import (
    GitRepository,
)
from self_healing_agent.git.repair_manager import (
    GitRepairManager,
)


class MockGitHubHandler(BaseHTTPRequestHandler):
    """Local HTTP mock for GitHub pull-request operations."""

    received_headers: dict[str, str] = {}
    received_payload: dict[str, Any] = {}
    received_method: str = ""
    response_status: int = 201

    def do_GET(self) -> None:
        """Return no existing pull requests."""

        MockGitHubHandler.received_method = "GET"

        response_body: list[dict[str, Any]] = []

        encoded_response = json.dumps(
            response_body,
        ).encode("utf-8")

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(encoded_response)),
        )

        self.end_headers()

        self.wfile.write(
            encoded_response,
        )

    def do_POST(self) -> None:
        """Capture pull-request creation requests."""

        MockGitHubHandler.received_method = "POST"

        content_length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        request_body = self.rfile.read(
            content_length,
        )

        MockGitHubHandler.received_headers = {
            key: value
            for key, value in self.headers.items()
        }

        MockGitHubHandler.received_payload = json.loads(
            request_body.decode("utf-8"),
        )

        response_body = {
            "number": 42,
            "title": MockGitHubHandler.received_payload[
                "title"
            ],
            "html_url": (
                "http://github.test/"
                "G-SIVA1/self-healing-api-adapter/"
                "pull/42"
            ),
            "head": {
                "ref": MockGitHubHandler.received_payload[
                    "head"
                ],
            },
            "base": {
                "ref": MockGitHubHandler.received_payload[
                    "base"
                ],
            },
        }

        encoded_response = json.dumps(
            response_body,
        ).encode("utf-8")

        self.send_response(
            MockGitHubHandler.response_status,
        )

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(encoded_response)),
        )

        self.end_headers()

        self.wfile.write(
            encoded_response,
        )

    def log_message(
        self,
        format: str,
        *args: object,
    ) -> None:
        """Disable HTTP server logging during tests."""

        return


@pytest.fixture
def mock_github_server() -> Any:
    """Start a local HTTP server that emulates GitHub."""

    MockGitHubHandler.received_headers = {}
    MockGitHubHandler.received_payload = {}
    MockGitHubHandler.received_method = ""
    MockGitHubHandler.response_status = 201

    server = HTTPServer(
        ("127.0.0.1", 0),
        MockGitHubHandler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    try:
        yield server

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_git(
    repository: Path,
    *arguments: str,
) -> str:
    """Run Git inside a test repository."""

    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def configure_git_repository(
    repository: Path,
) -> None:
    """Configure Git identity for tests."""

    run_git(
        repository,
        "config",
        "user.name",
        "Test User",
    )

    run_git(
        repository,
        "config",
        "user.email",
        "test@example.com",
    )


@pytest.fixture
def git_repository() -> Any:
    """Create a temporary Git repository."""

    with TemporaryDirectory() as temporary_directory:
        repository = Path(temporary_directory)

        run_git(
            repository,
            "init",
            "-b",
            "main",
        )

        configure_git_repository(
            repository,
        )

        repaired_file = (
            repository / "repair_target.py"
        )

        repaired_file.write_text(
            "import stripe\n\n"
            "stripe.Customer.create()\n",
            encoding="utf-8",
        )

        run_git(
            repository,
            "add",
            "repair_target.py",
        )

        run_git(
            repository,
            "commit",
            "-m",
            "initial commit",
        )

        yield repository


@pytest.mark.asyncio
async def test_github_client_creates_pull_request_over_http(
    mock_github_server: HTTPServer,
) -> None:
    """Create a PR against the local GitHub mock server."""

    original_base_url = (
        GitHubPullRequestClient.API_BASE_URL
    )

    GitHubPullRequestClient.API_BASE_URL = (
        f"http://127.0.0.1:"
        f"{mock_github_server.server_port}"
    )

    try:
        client = GitHubPullRequestClient(
            token="test-token",
        )

        pull_request = await client.create_pull_request(
            repository=(
                "G-SIVA1/"
                "self-healing-api-adapter"
            ),
            source_branch="repair/test",
            target_branch="main",
            title=(
                "fix: self-heal "
                "stripe compatibility"
            ),
            description="Automated repair.",
        )

        assert pull_request.number == 42

        assert (
            pull_request.title
            == (
                "fix: self-heal "
                "stripe compatibility"
            )
        )

        assert (
            pull_request.url
            == (
                "http://github.test/"
                "G-SIVA1/self-healing-api-adapter/"
                "pull/42"
            )
        )

        assert (
            pull_request.source_branch
            == "repair/test"
        )

        assert (
            pull_request.target_branch
            == "main"
        )

        assert (
            MockGitHubHandler.received_method
            == "POST"
        )

        assert (
            MockGitHubHandler.received_payload[
                "title"
            ]
            == (
                "fix: self-heal "
                "stripe compatibility"
            )
        )

        assert (
            MockGitHubHandler.received_payload[
                "head"
            ]
            == "repair/test"
        )

        assert (
            MockGitHubHandler.received_payload[
                "base"
            ]
            == "main"
        )

        assert (
            MockGitHubHandler.received_payload[
                "body"
            ]
            == "Automated repair."
        )

        assert (
            MockGitHubHandler.received_headers[
                "Authorization"
            ]
            == "Bearer test-token"
        )

        assert (
            MockGitHubHandler.received_headers[
                "X-GitHub-Api-Version"
            ]
            == GitHubPullRequestClient.API_VERSION
        )

    finally:
        GitHubPullRequestClient.API_BASE_URL = (
            original_base_url
        )


@pytest.mark.asyncio
async def test_github_client_does_not_use_real_github(
    mock_github_server: HTTPServer,
) -> None:
    """Verify only the local server is used."""

    original_base_url = (
        GitHubPullRequestClient.API_BASE_URL
    )

    GitHubPullRequestClient.API_BASE_URL = (
        f"http://127.0.0.1:"
        f"{mock_github_server.server_port}"
    )

    try:
        client = GitHubPullRequestClient(
            token="test-token",
        )

        pull_request = await client.create_pull_request(
            repository=(
                "G-SIVA1/"
                "self-healing-api-adapter"
            ),
            source_branch="repair/local-test",
            target_branch="main",
            title="Local integration test",
            description="Local server only.",
        )

        assert pull_request.number == 42

        assert (
            MockGitHubHandler.received_method
            == "POST"
        )

    finally:
        GitHubPullRequestClient.API_BASE_URL = (
            original_base_url
        )


@pytest.mark.asyncio
async def test_github_client_rejects_non_success_response(
    mock_github_server: HTTPServer,
) -> None:
    """Convert non-success responses into PullRequestError."""

    original_base_url = (
        GitHubPullRequestClient.API_BASE_URL
    )

    GitHubPullRequestClient.API_BASE_URL = (
        f"http://127.0.0.1:"
        f"{mock_github_server.server_port}"
    )

    MockGitHubHandler.response_status = 500

    try:
        client = GitHubPullRequestClient(
            token="test-token",
        )

        with pytest.raises(
            PullRequestError,
            match=(
                "GitHub API returned HTTP 500"
            ),
        ):
            await client.create_pull_request(
                repository=(
                    "G-SIVA1/"
                    "self-healing-api-adapter"
                ),
                source_branch="repair/failure-test",
                target_branch="main",
                title="Failure test",
                description="Failure test.",
            )

    finally:
        GitHubPullRequestClient.API_BASE_URL = (
            original_base_url
        )


@pytest.mark.asyncio
async def test_git_repair_manager_commits_pushes_and_creates_pull_request(
    mock_github_server: HTTPServer,
    git_repository: Path,
) -> None:
    """Run the complete Git repair manager flow."""

    original_base_url = (
        GitHubPullRequestClient.API_BASE_URL
    )

    GitHubPullRequestClient.API_BASE_URL = (
        f"http://127.0.0.1:"
        f"{mock_github_server.server_port}"
    )

    # IMPORTANT:
    # The remote must be unique for every test run.
    remote_repository = (
        git_repository.parent
        / f"remote-{git_repository.name}.git"
    )

    run_git(
        git_repository,
        "init",
        "--bare",
        str(remote_repository),
    )

    run_git(
        git_repository,
        "remote",
        "add",
        "origin",
        str(remote_repository),
    )

    client = GitHubPullRequestClient(
        token="test-token",
    )

    repository = GitRepository(
        git_repository,
    )

    manager = GitRepairManager(
        repository=repository,
        pull_request_client=client,
    )

    repaired_file = (
        git_repository / "repair_target.py"
    )

    repaired_file.write_text(
        "import stripe\n\n"
        "stripe.customers.create()\n",
        encoding="utf-8",
    )

    try:
        result = (
            await manager.commit_and_create_pull_request(
                service_name="stripe",
                api_call="Customer.create",
                files=[
                    "repair_target.py",
                ],
                test_passed=True,
                repository=(
                    "G-SIVA1/"
                    "self-healing-api-adapter"
                ),
                target_branch="main",
                commit_message=(
                    "Fix Stripe customer "
                    "API compatibility"
                ),
                pull_request_title=(
                    "Stripe API compatibility repair"
                ),
                pull_request_description=(
                    "Automated repair generated "
                    "by the self-healing agent."
                ),
            )
        )

        assert result.commit.branch.name.startswith(
            "repair/"
        )

        assert result.commit.commit_sha

        assert result.commit.files == (
            "repair_target.py",
        )

        assert (
            result.commit.commit_message
            == (
                "Fix Stripe customer "
                "API compatibility"
            )
        )

        assert (
            result.pull_request.number
            == 42
        )

        assert (
            result.pull_request.title
            == (
                "Stripe API compatibility repair"
            )
        )

        assert (
            result.pull_request.url
            == (
                "http://github.test/"
                "G-SIVA1/self-healing-api-adapter/"
                "pull/42"
            )
        )

        assert (
            result.pull_request.source_branch
            == result.commit.branch.name
        )

        assert (
            result.pull_request.target_branch
            == "main"
        )

        assert (
            MockGitHubHandler.received_method
            == "POST"
        )

        assert (
            MockGitHubHandler.received_headers[
                "Authorization"
            ]
            == "Bearer test-token"
        )

        assert (
            MockGitHubHandler.received_headers[
                "X-GitHub-Api-Version"
            ]
            == GitHubPullRequestClient.API_VERSION
        )

        assert (
            MockGitHubHandler.received_payload[
                "head"
            ]
            == result.commit.branch.name
        )

        assert (
            MockGitHubHandler.received_payload[
                "base"
            ]
            == "main"
        )

        assert (
            MockGitHubHandler.received_payload[
                "title"
            ]
            == (
                "Stripe API compatibility repair"
            )
        )

        assert (
            MockGitHubHandler.received_payload[
                "body"
            ]
            == (
                "Automated repair generated "
                "by the self-healing agent."
            )
        )

        assert (
            repaired_file.read_text(
                encoding="utf-8",
            )
            == (
                "import stripe\n\n"
                "stripe.customers.create()\n"
            )
        )

        current_branch = run_git(
            git_repository,
            "branch",
            "--show-current",
        )

        assert (
            current_branch
            == result.commit.branch.name
        )

        commit_subject = run_git(
            git_repository,
            "log",
            "-1",
            "--pretty=%s",
        )

        assert (
            commit_subject
            == (
                "Fix Stripe customer "
                "API compatibility"
            )
        )

        remote_branches = run_git(
            git_repository,
            "branch",
            "-r",
        )

        assert (
            f"origin/{result.commit.branch.name}"
            in remote_branches
        )

    finally:
        GitHubPullRequestClient.API_BASE_URL = (
            original_base_url
        )