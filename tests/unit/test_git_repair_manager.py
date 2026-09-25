from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from self_healing_agent.git.repair_manager import (
    GitRepairManager,
)
from self_healing_agent.git.repository import (
    GitRepository,
)


def initialize_git_repository(
    repository_path: Path,
) -> None:
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
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
        ["git", "add", "README.md"],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.asyncio
async def test_manager_commits_verified_repair(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)

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
        "print('repaired')\n",
        encoding="utf-8",
    )

    repository = GitRepository(tmp_path)
    manager = GitRepairManager(repository)

    result = await manager.commit_verified_repair(
        service_name="Stripe",
        api_call="Customer Create",
        files=[
            "examples/broken_api_client.py",
        ],
        commit_message="Fix Stripe customer API compatibility",
        test_passed=True,
    )

    assert (
        result.branch.name
        == "repair/stripe-customer-create"
    )

    assert result.branch.iteration == 1

    assert len(result.commit_sha) == 40

    assert result.files == (
        "examples/broken_api_client.py",
    )

    assert (
        await repository.current_branch()
        == "repair/stripe-customer-create"
    )


@pytest.mark.asyncio
async def test_manager_rejects_unverified_repair(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)

    repaired_file = tmp_path / "README.md"

    repaired_file.write_text(
        "Unverified repair\n",
        encoding="utf-8",
    )

    repository = GitRepository(tmp_path)
    manager = GitRepairManager(repository)

    with pytest.raises(
        ValueError,
        match="has not passed validation",
    ):
        await manager.commit_verified_repair(
            service_name="Stripe",
            api_call="Customer Create",
            files=["README.md"],
            commit_message="Unsafe repair",
            test_passed=False,
        )

    assert (
        await repository.current_branch()
        == "main"
    )


@pytest.mark.asyncio
async def test_manager_rejects_empty_files(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)

    repository = GitRepository(tmp_path)
    manager = GitRepairManager(repository)

    with pytest.raises(
        ValueError,
        match="At least one repaired file",
    ):
        await manager.commit_verified_repair(
            service_name="Stripe",
            api_call="Customer Create",
            files=[],
            commit_message="Fix API",
            test_passed=True,
        )


@pytest.mark.asyncio
async def test_manager_rejects_empty_file_path(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)

    repository = GitRepository(tmp_path)
    manager = GitRepairManager(repository)

    with pytest.raises(
        ValueError,
        match="file path cannot be empty",
    ):
        await manager.commit_verified_repair(
            service_name="Stripe",
            api_call="Customer Create",
            files=["   "],
            commit_message="Fix API",
            test_passed=True,
        )


@pytest.mark.asyncio
async def test_manager_rejects_duplicate_files(
    tmp_path: Path,
) -> None:
    initialize_git_repository(tmp_path)

    repository = GitRepository(tmp_path)
    manager = GitRepairManager(repository)

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        await manager.commit_verified_repair(
            service_name="Stripe",
            api_call="Customer Create",
            files=[
                "README.md",
                "README.md",
            ],
            commit_message="Fix API",
            test_passed=True,
        )