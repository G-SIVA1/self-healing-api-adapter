from __future__ import annotations

import json
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from self_healing_agent.git.github_client import (
    GitHubPullRequestClient,
)
from self_healing_agent.git.repair_manager import (
    GitRepairManager,
)
from self_healing_agent.git.repository import (
    GitRepository,
)


class MockGitHubHandler(BaseHTTPRequestHandler):
    received_headers: dict[str, str] = {}
    received_payload: dict[str, Any] = {}
    response_status: int = 201

    def do_POST(self) -> None:
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
            key.lower(): value
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
                "https://github.com/"
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
        return


@pytest.fixture
def mock_github_server() -> str:
    MockGitHubHandler.received_headers = {}
    MockGitHubHandler.received_payload = {}
    MockGitHubHandler.response_status = 201

    server = HTTPServer(
        ("127.0.0.1", 0),
        MockGitHubHandler,
    )

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    host, port = server.server_address

    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


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

    remote_url = remote_path.as_uri()

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


@pytest.mark.asyncio
async def test_github_client_creates_pull_request_over_http(
    mock_github_server: str,
) -> None:
    client = GitHubPullRequestClient(
        token="test-token",
    )

    original_base_url = client.API_BASE_URL
    client.API_BASE_URL = mock_github_server

    try:
        result = await client.create_pull_request(
            repository=(
                "G-SIVA1/"
                "self-healing-api-adapter"
            ),
            source_branch=(
                "repair/stripe-customer-create"
            ),
            target_branch="main",
            title="Stripe API compatibility repair",
            description=(
                "Automated repair generated by "
                "the self-healing agent."
            ),
        )
    finally:
        client.API_BASE_URL = original_base_url

    assert result.number == 42

    assert result.title == (
        "Stripe API compatibility repair"
    )

    assert result.url == (
        "https://github.com/"
        "G-SIVA1/self-healing-api-adapter/"
        "pull/42"
    )

    assert result.source_branch == (
        "repair/stripe-customer-create"
    )

    assert result.target_branch == "main"

    assert MockGitHubHandler.received_payload == {
        "title": (
            "Stripe API compatibility repair"
        ),
        "head": (
            "repair/stripe-customer-create"
        ),
        "base": "main",
        "body": (
            "Automated repair generated by "
            "the self-healing agent."
        ),
    }

    assert (
        MockGitHubHandler.received_headers[
            "authorization"
        ]
        == "Bearer test-token"
    )

    assert (
        MockGitHubHandler.received_headers[
            "accept"
        ]
        == "application/vnd.github+json"
    )

    assert (
        MockGitHubHandler.received_headers[
            "content-type"
        ]
        == "application/json"
    )

    assert (
        "x-github-api-version"
        in MockGitHubHandler.received_headers
    )


@pytest.mark.asyncio
async def test_github_client_does_not_use_real_github(
    mock_github_server: str,
) -> None:
    client = GitHubPullRequestClient(
        token="fake-local-token",
    )

    original_base_url = client.API_BASE_URL
    client.API_BASE_URL = mock_github_server

    try:
        result = await client.create_pull_request(
            repository="owner/repository",
            source_branch="repair/test",
            target_branch="main",
            title="Test repair",
            description="Local integration test.",
        )
    finally:
        client.API_BASE_URL = original_base_url

    assert result.number == 42

    assert (
        MockGitHubHandler.received_headers[
            "authorization"
        ]
        == "Bearer fake-local-token"
    )

    assert (
        MockGitHubHandler.received_payload[
            "head"
        ]
        == "repair/test"
    )


@pytest.mark.asyncio
async def test_github_client_rejects_non_success_response(
    mock_github_server: str,
) -> None:
    MockGitHubHandler.response_status = 500

    client = GitHubPullRequestClient(
        token="test-token",
    )

    original_base_url = client.API_BASE_URL
    client.API_BASE_URL = mock_github_server

    try:
        with pytest.raises(
            Exception,
            match="GitHub API returned HTTP 500",
        ):
            await client.create_pull_request(
                repository="owner/repository",
                source_branch="repair/test",
                target_branch="main",
                title="Test repair",
                description="Test failure.",
            )
    finally:
        client.API_BASE_URL = original_base_url
        MockGitHubHandler.response_status = 201


@pytest.mark.asyncio
async def test_git_repair_manager_commits_pushes_and_creates_pull_request(
    tmp_path: Path,
    mock_github_server: str,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    configure_local_git_remote(
        tmp_path,
    )

    repaired_file = (
        tmp_path
        / "examples"
        / "stripe_client.py"
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

    repository = GitRepository(
        tmp_path,
    )

    github_client = GitHubPullRequestClient(
        token="local-test-token",
    )

    original_base_url = github_client.API_BASE_URL
    github_client.API_BASE_URL = mock_github_server

    try:
        manager = GitRepairManager(
            repository=repository,
            pull_request_client=github_client,
        )

        result = await manager.commit_and_create_pull_request(
            service_name="Stripe",
            api_call="Customer Create",
            files=[
                "examples/stripe_client.py",
            ],
            test_passed=True,
            repository=(
                "G-SIVA1/"
                "self-healing-api-adapter"
            ),
            target_branch="main",
            commit_message=(
                "Fix Stripe customer API compatibility"
            ),
            pull_request_title=(
                "Stripe API compatibility repair"
            ),
            pull_request_description=(
                "Automated repair generated by "
                "the self-healing agent."
            ),
        )
    finally:
        github_client.API_BASE_URL = original_base_url

    assert result.commit.branch.name == (
        "repair/stripe-customer-create"
    )

    assert len(result.commit.commit_sha) == 40

    assert result.commit.files == (
        "examples/stripe_client.py",
    )

    assert result.pull_request.number == 42

    assert result.pull_request.title == (
        "Stripe API compatibility repair"
    )

    assert result.pull_request.url == (
        "https://github.com/"
        "G-SIVA1/self-healing-api-adapter/"
        "pull/42"
    )

    assert result.pull_request.source_branch == (
        "repair/stripe-customer-create"
    )

    assert result.pull_request.target_branch == "main"

    assert (
        await repository.current_branch()
        == "repair/stripe-customer-create"
    )

    commit_result = subprocess.run(
        [
            "git",
            "show",
            "--format=%s",
            "--no-patch",
            result.commit.commit_sha,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        commit_result.stdout.strip()
        == "Fix Stripe customer API compatibility"
    )

    file_result = subprocess.run(
        [
            "git",
            "show",
            (
                f"{result.commit.commit_sha}:"
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
        in file_result.stdout
    )

    assert (
        "stripe.Customer.create"
        not in file_result.stdout
    )

    assert MockGitHubHandler.received_payload == {
        "title": (
            "Stripe API compatibility repair"
        ),
        "head": (
            "repair/stripe-customer-create"
        ),
        "base": "main",
        "body": (
            "Automated repair generated by "
            "the self-healing agent."
        ),
    }

    assert (
        MockGitHubHandler.received_headers[
            "authorization"
        ]
        == "Bearer local-test-token"
    )

    assert (
        MockGitHubHandler.received_headers[
            "accept"
        ]
        == "application/vnd.github+json"
    )

    assert (
        MockGitHubHandler.received_headers[
            "content-type"
        ]
        == "application/json"
    )

    assert (
        "x-github-api-version"
        in MockGitHubHandler.received_headers
    )