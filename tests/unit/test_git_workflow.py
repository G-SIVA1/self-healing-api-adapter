from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from self_healing_agent.git.repository import (
    GitRepository,
)
from self_healing_agent.git.workflow import (
    GitRepairWorkflow,
)


def initialize_git_repository(
    repository_path: Path,
) -> None:
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


@pytest.mark.asyncio
async def test_workflow_creates_repair_branch(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    workflow = GitRepairWorkflow(
        repository,
    )

    repair_branch = (
        await workflow.prepare_repair_branch(
            service_name="Stripe",
            api_call="Customer Create",
        )
    )

    assert (
        repair_branch.name
        == "repair/stripe-customer-create"
    )

    assert repair_branch.iteration == 1

    assert (
        await repository.current_branch()
        == "repair/stripe-customer-create"
    )


@pytest.mark.asyncio
async def test_workflow_uses_iteration_when_branch_exists(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    await repository.create_branch(
        "repair/stripe-customer-create",
    )

    await repository.current_branch()

    await repository._run_git(
        "switch",
        "main",
    )

    workflow = GitRepairWorkflow(
        repository,
    )

    repair_branch = (
        await workflow.prepare_repair_branch(
            service_name="Stripe",
            api_call="Customer Create",
            iteration=2,
        )
    )

    assert (
        repair_branch.name
        == "repair/stripe-customer-create-iteration-2"
    )


@pytest.mark.asyncio
async def test_workflow_uses_suffix_when_iteration_branch_exists(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    await repository.create_branch(
        "repair/stripe-customer-create",
    )

    await repository._run_git(
        "switch",
        "main",
    )

    await repository.create_branch(
        "repair/stripe-customer-create-iteration-2",
    )

    await repository._run_git(
        "switch",
        "main",
    )

    workflow = GitRepairWorkflow(
        repository,
    )

    repair_branch = (
        await workflow.prepare_repair_branch(
            service_name="Stripe",
            api_call="Customer Create",
            iteration=2,
        )
    )

    assert (
        repair_branch.name
        == "repair/stripe-customer-create-iteration-2-2"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_name", "api_call"),
    [
        ("", "create"),
        ("Stripe", ""),
        ("   ", "create"),
        ("Stripe", "   "),
    ],
)
async def test_workflow_rejects_empty_components(
    tmp_path: Path,
    service_name: str,
    api_call: str,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    workflow = GitRepairWorkflow(
        repository,
    )

    with pytest.raises(ValueError):
        await workflow.prepare_repair_branch(
            service_name=service_name,
            api_call=api_call,
        )


@pytest.mark.asyncio
async def test_workflow_rejects_invalid_iteration(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    workflow = GitRepairWorkflow(
        repository,
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        await workflow.prepare_repair_branch(
            service_name="Stripe",
            api_call="Customer Create",
            iteration=0,
        )


@pytest.mark.asyncio
async def test_workflow_rejects_non_git_directory(
    tmp_path: Path,
) -> None:
    repository = GitRepository(
        tmp_path,
    )

    workflow = GitRepairWorkflow(
        repository,
    )

    with pytest.raises(
        Exception,
        match="not a Git repository",
    ):
        await workflow.prepare_repair_branch(
            service_name="Stripe",
            api_call="Customer Create",
        )