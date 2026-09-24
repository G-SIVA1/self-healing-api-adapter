from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from self_healing_agent.git.repository import (
    GitRepository,
    GitRepositoryError,
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
async def test_repository_detects_git_repository(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    assert await repository.is_repository() is True


@pytest.mark.asyncio
async def test_repository_detects_current_branch(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    branch = await repository.current_branch()

    assert branch == "main"


@pytest.mark.asyncio
async def test_repository_creates_repair_branch(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    branch_name = (
        "repair/stripe-customer-api"
    )

    created_branch = await repository.create_branch(
        branch_name,
    )

    assert created_branch == branch_name

    assert (
        await repository.current_branch()
        == branch_name
    )

    assert (
        await repository.branch_exists(
            branch_name,
        )
        is True
    )


@pytest.mark.asyncio
async def test_repository_rejects_duplicate_branch(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    branch_name = (
        "repair/stripe-customer-api"
    )

    await repository.create_branch(
        branch_name,
    )

    with pytest.raises(
        GitRepositoryError,
        match="already exists",
    ):
        await repository.create_branch(
            branch_name,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "branch_name",
    [
        "",
        "-invalid",
        "invalid..branch",
        "invalid/",
        "invalid branch",
        "invalid@{branch}",
    ],
)
async def test_repository_rejects_invalid_branch_names(
    tmp_path: Path,
    branch_name: str,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    with pytest.raises(ValueError):
        await repository.create_branch(
            branch_name,
        )


@pytest.mark.asyncio
async def test_repository_rejects_non_git_directory(
    tmp_path: Path,
) -> None:
    repository = GitRepository(
        tmp_path,
    )

    assert await repository.is_repository() is False

    with pytest.raises(
        GitRepositoryError,
        match="not a Git repository",
    ):
        await repository.create_branch(
            "repair/test",
        )