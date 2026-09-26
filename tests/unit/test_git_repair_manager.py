from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from self_healing_agent.git.pull_request import PullRequest
from self_healing_agent.git.repair_manager import (
    GitRepairManager,
    RepairCommit,
)
from self_healing_agent.git.workflow import RepairBranch


@dataclass
class FakeRepository:
    """Test double for GitRepository."""

    changed_files: list[str]
    committed_files: list[str]
    commit_messages: list[str]
    pushed_branches: list[str]

    async def get_changed_files(self) -> list[str]:
        return list(self.changed_files)

    async def commit_changes(
        self,
        *,
        files: list[str],
        message: str,
    ) -> str:
        self.committed_files = list(files)
        self.commit_messages.append(message)

        return "abc123"

    async def push_branch(
        self,
        branch_name: str,
        remote: str = "origin",
    ) -> str:
        self.pushed_branches.append(
            f"{remote}:{branch_name}"
        )

        return branch_name


@dataclass
class FakeWorkflow:
    """Test double for GitRepairWorkflow."""

    branch_name: str = "repair/test-api"

    async def prepare_repair_branch(
        self,
        *,
        service_name: str,
        api_call: str,
        iteration: int = 1,
    ) -> RepairBranch:
        return RepairBranch(
            name=self.branch_name,
            iteration=iteration,
        )


class FakePullRequestClient:
    """Test double for GitHubPullRequestClient."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create_pull_request(
        self,
        *,
        repository: str,
        source_branch: str,
        target_branch: str,
        title: str,
        description: str,
    ) -> PullRequest:
        self.calls.append(
            {
                "repository": repository,
                "source_branch": source_branch,
                "target_branch": target_branch,
                "title": title,
                "description": description,
            }
        )

        return PullRequest(
            number=123,
            title=title,
            url=(
                "https://github.com/"
                "G-SIVA1/"
                "self-healing-api-adapter/"
                "pull/123"
            ),
            source_branch=source_branch,
            target_branch=target_branch,
        )


@pytest.fixture
def repository() -> FakeRepository:
    return FakeRepository(
        changed_files=[
            "src/example.py",
        ],
        committed_files=[],
        commit_messages=[],
        pushed_branches=[],
    )


@pytest.fixture
def workflow() -> FakeWorkflow:
    return FakeWorkflow()


@pytest.fixture
def pull_request_client() -> FakePullRequestClient:
    return FakePullRequestClient()


@pytest.fixture
def manager(
    repository: FakeRepository,
    workflow: FakeWorkflow,
    pull_request_client: FakePullRequestClient,
) -> GitRepairManager:
    return GitRepairManager(
        repository=repository,
        workflow=workflow,
        pull_request_client=pull_request_client,
    )


@pytest.mark.asyncio
async def test_commit_and_create_pull_request(
    manager: GitRepairManager,
    repository: FakeRepository,
    pull_request_client: FakePullRequestClient,
) -> None:
    result = await manager.commit_and_create_pull_request(
        service_name="stripe",
        api_call="stripe.Customer.create",
        files=[
            "src/example.py",
        ],
        test_passed=True,
        repository="G-SIVA1/self-healing-api-adapter",
        target_branch="main",
        pull_request_title="Fix Stripe Customer API",
        pull_request_description="Automated compatibility repair.",
    )

    assert result.commit.commit_sha == "abc123"

    assert (
        result.commit.branch.name
        == "repair/test-api"
    )

    assert result.pull_request.number == 123

    assert (
        result.pull_request.source_branch
        == "repair/test-api"
    )

    assert (
        result.pull_request.target_branch
        == "main"
    )

    assert repository.committed_files == [
        "src/example.py",
    ]

    assert repository.commit_messages == [
        (
            "fix: self-heal "
            "stripe "
            "stripe.Customer.create "
            "compatibility"
        )
    ]

    assert repository.pushed_branches == [
        "origin:repair/test-api",
    ]

    assert len(
        pull_request_client.calls
    ) == 1

    assert pull_request_client.calls[0] == {
        "repository": (
            "G-SIVA1/"
            "self-healing-api-adapter"
        ),
        "source_branch": "repair/test-api",
        "target_branch": "main",
        "title": "Fix Stripe Customer API",
        "description": (
            "Automated compatibility repair."
        ),
    }


@pytest.mark.asyncio
async def test_pull_request_requires_client(
    repository: FakeRepository,
    workflow: FakeWorkflow,
) -> None:
    manager = GitRepairManager(
        repository=repository,
        workflow=workflow,
        pull_request_client=None,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "GitHub pull request client "
            "is not configured"
        ),
    ):
        await manager.commit_and_create_pull_request(
            service_name="stripe",
            api_call="stripe.Customer.create",
            files=[
                "src/example.py",
            ],
            test_passed=True,
            repository="G-SIVA1/self-healing-api-adapter",
            target_branch="main",
        )


@pytest.mark.asyncio
async def test_pull_request_rejects_empty_repository(
    manager: GitRepairManager,
) -> None:
    with pytest.raises(
        ValueError,
        match="repository cannot be empty",
    ):
        await manager.commit_and_create_pull_request(
            service_name="stripe",
            api_call="stripe.Customer.create",
            files=[
                "src/example.py",
            ],
            test_passed=True,
            repository="   ",
            target_branch="main",
        )


@pytest.mark.asyncio
async def test_pull_request_rejects_empty_target_branch(
    manager: GitRepairManager,
) -> None:
    with pytest.raises(
        ValueError,
        match="target_branch cannot be empty",
    ):
        await manager.commit_and_create_pull_request(
            service_name="stripe",
            api_call="stripe.Customer.create",
            files=[
                "src/example.py",
            ],
            test_passed=True,
            repository="G-SIVA1/self-healing-api-adapter",
            target_branch="   ",
        )


@pytest.mark.asyncio
async def test_existing_commit_can_create_pull_request(
    manager: GitRepairManager,
    pull_request_client: FakePullRequestClient,
) -> None:
    commit = RepairCommit(
        branch=RepairBranch(
            name="repair/existing-api",
            iteration=1,
        ),
        commit_sha="existing123",
        files=(
            "src/example.py",
        ),
        commit_message=(
            "fix: existing repair"
        ),
    )

    result = await manager.create_pull_request_for_commit(
        commit=commit,
        repository="G-SIVA1/self-healing-api-adapter",
        target_branch="main",
    )

    assert result.commit == commit

    assert result.pull_request.number == 123

    assert len(
        pull_request_client.calls
    ) == 1

    assert pull_request_client.calls[0][
        "source_branch"
    ] == "repair/existing-api"


@pytest.mark.asyncio
async def test_creating_pull_request_does_not_create_second_commit(
    manager: GitRepairManager,
    repository: FakeRepository,
) -> None:
    commit = await manager.commit_verified_repair(
        service_name="stripe",
        api_call="stripe.Customer.create",
        files=[
            "src/example.py",
        ],
        test_passed=True,
    )

    initial_commit_count = len(
        repository.commit_messages
    )

    initial_push_count = len(
        repository.pushed_branches
    )

    await manager.create_pull_request_for_commit(
        commit=commit,
        repository="G-SIVA1/self-healing-api-adapter",
        target_branch="main",
    )

    assert len(
        repository.commit_messages
    ) == initial_commit_count

    assert len(
        repository.pushed_branches
    ) == initial_push_count


@pytest.mark.asyncio
async def test_unexpected_changed_files_are_rejected(
    repository: FakeRepository,
    workflow: FakeWorkflow,
) -> None:
    repository.changed_files = [
        "src/example.py",
        "README.md",
    ]

    manager = GitRepairManager(
        repository=repository,
        workflow=workflow,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Repair contains unexpected changed files: "
            "README.md"
        ),
    ):
        await manager.commit_verified_repair(
            service_name="stripe",
            api_call="stripe.Customer.create",
            files=[
                "src/example.py",
            ],
            test_passed=True,
        )

    assert repository.committed_files == []

    assert repository.commit_messages == []

    assert repository.pushed_branches == []