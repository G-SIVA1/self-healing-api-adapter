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
    remote_path = (
        repository_path.parent
        / f"{repository_path.name}-remote.git"
    )

    subprocess.run(
        [
            "git",
            "init",
            "--bare",
            remote_path,
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


@pytest.mark.asyncio
async def test_verified_repair_becomes_git_commit(
    tmp_path: Path,
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

    repository = GitRepository(
        tmp_path,
    )

    manager = GitRepairManager(
        repository,
    )

    result = await manager.commit_verified_repair(
        service_name="Stripe",
        api_call="Customer Create",
        files=[
            "examples/broken_api_client.py",
        ],
        commit_message=(
            "Fix Stripe customer API compatibility"
        ),
        test_passed=True,
    )

    assert (
        result.branch.name
        == "repair/stripe-customer-create"
    )

    assert (
        await repository.current_branch()
        == "repair/stripe-customer-create"
    )

    assert len(result.commit_sha) == 40

    commit_result = subprocess.run(
        [
            "git",
            "show",
            "--format=%s",
            "--no-patch",
            result.commit_sha,
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
            f"{result.commit_sha}:"
            "examples/broken_api_client.py",
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


@pytest.mark.asyncio
async def test_unverified_repair_creates_no_git_commit(
    tmp_path: Path,
) -> None:
    initialize_git_repository(
        tmp_path,
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
        "broken repair\n",
        encoding="utf-8",
    )

    repository = GitRepository(
        tmp_path,
    )

    manager = GitRepairManager(
        repository,
    )

    with pytest.raises(
        ValueError,
        match="has not passed validation",
    ):
        await manager.commit_verified_repair(
            service_name="Stripe",
            api_call="Customer Create",
            files=[
                "examples/broken_api_client.py",
            ],
            commit_message="Unsafe repair",
            test_passed=False,
        )

    assert (
        await repository.current_branch()
        == "main"
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

    assert len(commits) == 1