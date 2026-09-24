from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path


class GitRepositoryError(RuntimeError):
    """Raised when a Git repository operation fails."""


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Result returned by a Git command."""

    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        """Return whether the Git command succeeded."""

        return self.return_code == 0


class GitRepository:
    """Manage safe Git operations inside a repository."""

    _BRANCH_PATTERN = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._/-]*$"
    )

    def __init__(
        self,
        repository_path: Path,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._repository_path = repository_path.resolve()
        self._timeout_seconds = timeout_seconds

        if not self._repository_path.exists():
            raise ValueError(
                "Repository path does not exist"
            )

        if not self._repository_path.is_dir():
            raise ValueError(
                "Repository path must be a directory"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero"
            )

    @property
    def repository_path(self) -> Path:
        """Return the configured repository path."""

        return self._repository_path

    async def is_repository(self) -> bool:
        """Return whether the path is a Git repository."""

        result = await self._run_git(
            "rev-parse",
            "--is-inside-work-tree",
        )

        return (
            result.succeeded
            and result.stdout.strip() == "true"
        )

    async def current_branch(self) -> str:
        """Return the currently checked-out branch."""

        result = await self._run_git(
            "branch",
            "--show-current",
        )

        if not result.succeeded:
            raise GitRepositoryError(
                self._format_error(
                    "Unable to determine current Git branch",
                    result,
                )
            )

        branch = result.stdout.strip()

        if not branch:
            raise GitRepositoryError(
                "Repository is in a detached HEAD state"
            )

        return branch

    async def branch_exists(
        self,
        branch_name: str,
    ) -> bool:
        """Return whether a local branch exists."""

        self._validate_branch_name(
            branch_name,
        )

        result = await self._run_git(
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
        )

        if not result.succeeded:
            raise GitRepositoryError(
                self._format_error(
                    "Unable to determine whether "
                    "Git branch exists",
                    result,
                )
            )

        branches = {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        }

        return branch_name in branches

    async def create_branch(
        self,
        branch_name: str,
    ) -> str:
        """Create and check out a new local repair branch."""

        self._validate_branch_name(
            branch_name,
        )

        if not await self.is_repository():
            raise GitRepositoryError(
                "Configured path is not a Git repository"
            )

        if await self.branch_exists(
            branch_name,
        ):
            raise GitRepositoryError(
                f"Git branch already exists: {branch_name}"
            )

        result = await self._run_git(
            "switch",
            "-c",
            branch_name,
        )

        if not result.succeeded:
            raise GitRepositoryError(
                self._format_error(
                    f"Unable to create Git branch "
                    f"'{branch_name}'",
                    result,
                )
            )

        return branch_name

    async def commit_changes(
        self,
        files: list[str],
        message: str,
    ) -> str:
        """Stage specific files and create a Git commit."""

        if not await self.is_repository():
            raise GitRepositoryError(
                "Configured path is not a Git repository"
            )

        if not files:
            raise ValueError(
                "At least one file must be provided"
            )

        normalized_files = self._validate_commit_files(
            files,
        )

        normalized_message = message.strip()

        if not normalized_message:
            raise ValueError(
                "Commit message cannot be empty"
            )

        add_result = await self._run_git(
            "add",
            "--",
            *normalized_files,
        )

        if not add_result.succeeded:
            raise GitRepositoryError(
                self._format_error(
                    "Unable to stage Git files",
                    add_result,
                )
            )

        commit_result = await self._run_git(
            "commit",
            "-m",
            normalized_message,
        )

        if not commit_result.succeeded:
            raise GitRepositoryError(
                self._format_error(
                    "Unable to create Git commit",
                    commit_result,
                )
            )

        commit_sha_result = await self._run_git(
            "rev-parse",
            "HEAD",
        )

        if not commit_sha_result.succeeded:
            raise GitRepositoryError(
                self._format_error(
                    "Unable to determine Git commit SHA",
                    commit_sha_result,
                )
            )

        commit_sha = commit_sha_result.stdout.strip()

        if not commit_sha:
            raise GitRepositoryError(
                "Git commit succeeded but no commit SHA was returned"
            )

        return commit_sha

    def _validate_commit_files(
        self,
        files: list[str],
    ) -> list[str]:
        """Validate file paths supplied for a Git commit."""

        normalized_files: list[str] = []

        for file_path in files:
            normalized_path = file_path.strip()

            if not normalized_path:
                raise ValueError(
                    "Commit file path cannot be empty"
                )

            path = Path(normalized_path)

            if path.is_absolute():
                raise ValueError(
                    "Commit file paths must be relative"
                )

            if any(
                part == ".."
                for part in path.parts
            ):
                raise ValueError(
                    "Commit file paths cannot contain '..'"
                )

            normalized_files.append(
                path.as_posix(),
            )

        if len(set(normalized_files)) != len(
            normalized_files
        ):
            raise ValueError(
                "Commit file paths must be unique"
            )

        return normalized_files

    async def _run_git(
        self,
        *arguments: str,
    ) -> GitCommandResult:
        """Execute a Git command asynchronously."""

        command = (
            "git",
            *arguments,
        )

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self._repository_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = (
                await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._timeout_seconds,
                )
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()

            raise GitRepositoryError(
                "Git command timed out"
            ) from exc

        stdout = stdout_bytes.decode(
            "utf-8",
            errors="replace",
        )

        stderr = stderr_bytes.decode(
            "utf-8",
            errors="replace",
        )

        return_code = (
            process.returncode
            if process.returncode is not None
            else -1
        )

        return GitCommandResult(
            command=command,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )

    @classmethod
    def _validate_branch_name(
        cls,
        branch_name: str,
    ) -> None:
        """Validate a branch name before invoking Git."""

        if not branch_name:
            raise ValueError(
                "Branch name cannot be empty"
            )

        if len(branch_name) > 200:
            raise ValueError(
                "Branch name is too long"
            )

        if branch_name.startswith("-"):
            raise ValueError(
                "Branch name cannot start with '-'"
            )

        if not cls._BRANCH_PATTERN.fullmatch(
            branch_name
        ):
            raise ValueError(
                "Invalid Git branch name"
            )

        if ".." in branch_name:
            raise ValueError(
                "Git branch name cannot contain '..'"
            )

        if branch_name.endswith("."):
            raise ValueError(
                "Git branch name cannot end with '.'"
            )

        if branch_name.endswith("/"):
            raise ValueError(
                "Git branch name cannot end with '/'"
            )

        if "@{" in branch_name:
            raise ValueError(
                "Git branch name cannot contain '@{'"
            )

    @staticmethod
    def _format_error(
        message: str,
        result: GitCommandResult,
    ) -> str:
        """Build a useful Git command error."""

        stderr = result.stderr.strip()

        if stderr:
            return f"{message}: {stderr}"

        return (
            f"{message} "
            f"(exit code {result.return_code})"
        )