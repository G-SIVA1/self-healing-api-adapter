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


@pytest.mark.asyncio
async def test_repository_commits_specific_file(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    repaired_file = (
        tmp_path / "examples" / "broken_api_client.py"
    )

    repaired_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    repaired_file.write_text(
        "print('repaired')\n",
        encoding="utf-8",
    )

    commit_sha = await repository.commit_changes(
        files=[
            "examples/broken_api_client.py",
        ],
        message="Fix API compatibility",
    )

    assert len(commit_sha) == 40

    assert all(
        character in "0123456789abcdef"
        for character in commit_sha
    )

    result = subprocess.run(
        [
            "git",
            "show",
            "--format=%s",
            "--no-patch",
            "HEAD",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        result.stdout.strip()
        == "Fix API compatibility"
    )

    committed_file = subprocess.run(
        [
            "git",
            "show",
            "--name-only",
            "--format=",
            "HEAD",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "examples/broken_api_client.py"
        in committed_file.stdout.splitlines()
    )


@pytest.mark.asyncio
async def test_repository_commit_returns_head_sha(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    readme_path = tmp_path / "README.md"

    readme_path.write_text(
        "Updated repository content\n",
        encoding="utf-8",
    )

    commit_sha = await repository.commit_changes(
        files=["README.md"],
        message="Update repository content",
    )

    head_result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert commit_sha == head_result.stdout.strip()


@pytest.mark.asyncio
async def test_repository_rejects_empty_commit_files(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="At least one file",
    ):
        await repository.commit_changes(
            files=[],
            message="Fix API compatibility",
        )


@pytest.mark.asyncio
async def test_repository_rejects_empty_commit_message(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    readme_path = tmp_path / "README.md"

    readme_path.write_text(
        "Updated content\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Commit message cannot be empty",
    ):
        await repository.commit_changes(
            files=["README.md"],
            message="   ",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_path",
    [
        "C:/outside/repository.py",
        "C:\\outside\\repository.py",
    ],
)
async def test_repository_rejects_absolute_commit_paths(
    tmp_path: Path,
    file_path: str,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="must be relative",
    ):
        await repository.commit_changes(
            files=[file_path],
            message="Invalid path",
        )


@pytest.mark.asyncio
async def test_repository_rejects_parent_directory_commit_path(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="cannot contain '..'",
    ):
        await repository.commit_changes(
            files=[
                "../outside.py",
            ],
            message="Invalid path",
        )


@pytest.mark.asyncio
async def test_repository_rejects_duplicate_commit_paths(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        await repository.commit_changes(
            files=[
                "README.md",
                "README.md",
            ],
            message="Duplicate paths",
        )


@pytest.mark.asyncio
async def test_repository_rejects_missing_commit_file(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    with pytest.raises(
        GitRepositoryError,
        match="Unable to stage Git files",
    ):
        await repository.commit_changes(
            files=[
                "missing_file.py",
            ],
            message="Fix missing file",
        )


@pytest.mark.asyncio
async def test_repository_rejects_commit_without_changes(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
    )

    repository = GitRepository(
        tmp_path,
    )

    with pytest.raises(
        GitRepositoryError,
        match="Unable to create Git commit",
    ):
        await repository.commit_changes(
            files=[
                "README.md",
            ],
            message="No changes",
        )