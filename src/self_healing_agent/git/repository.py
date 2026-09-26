from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class GitRepositoryError(RuntimeError):
    """Raised when a Git repository operation fails."""


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Result returned by a Git command."""

    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str


class GitRepository:
    """Async wrapper around Git repository operations."""

    def __init__(
        self,
        repository_path: Path,
    ) -> None:
        self.repository_path = repository_path.resolve()

    async def is_repository(self) -> bool:
        """Return True when the configured path is a Git repository."""

        result = await self._run_git(
            "rev-parse",
            "--is-inside-work-tree",
            check=False,
        )

        return (
            result.return_code == 0
            and result.stdout.strip().lower() == "true"
        )

    async def current_branch(self) -> str:
        """Return the currently checked-out branch name."""

        await self._ensure_repository()

        result = await self._run_git(
            "symbolic-ref",
            "--short",
            "HEAD",
        )

        branch_name = result.stdout.strip()

        if not branch_name:
            raise GitRepositoryError(
                "Unable to determine current Git branch"
            )

        return branch_name

    async def branch_exists(
        self,
        branch_name: str,
    ) -> bool:
        """Return True when a local branch exists."""

        self._validate_branch_name(branch_name)

        await self._ensure_repository()

        result = await self._run_git(
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
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
        """Create and switch to a new local branch."""

        self._validate_branch_name(branch_name)

        await self._ensure_repository()

        if await self.branch_exists(branch_name):
            raise GitRepositoryError(
                f"Git branch already exists: {branch_name}"
            )

        result = await self._run_git(
            "switch",
            "-c",
            branch_name,
            check=False,
        )

        if result.return_code != 0:
            raise GitRepositoryError(
                self._format_error(
                    "Unable to create Git branch",
                    result,
                )
            )

        return branch_name

    async def commit_changes(
        self,
        files: Sequence[str],
        message: str,
    ) -> str:
        """Stage selected files and create a Git commit."""

        await self._ensure_repository()

        normalized_files = self._validate_commit_files(
            files,
        )

        if not isinstance(message, str):
            raise ValueError(
                "Commit message must be a string"
            )

        if not message.strip():
            raise ValueError(
                "Commit message cannot be empty"
            )

        stage_result = await self._run_git(
            "add",
            "--",
            *normalized_files,
            check=False,
        )

        if stage_result.return_code != 0:
            raise GitRepositoryError(
                self._format_error(
                    "Unable to stage Git files",
                    stage_result,
                )
            )

        commit_result = await self._run_git(
            "commit",
            "-m",
            message.strip(),
            check=False,
        )

        if commit_result.return_code != 0:
            raise GitRepositoryError(
                self._format_error(
                    "Unable to create Git commit",
                    commit_result,
                )
            )

        head_result = await self._run_git(
            "rev-parse",
            "HEAD",
        )

        commit_sha = head_result.stdout.strip()

        if not commit_sha:
            raise GitRepositoryError(
                "Git commit succeeded but no commit SHA was returned"
            )

        return commit_sha

    async def push_branch(
        self,
        branch_name: str,
        remote: str = "origin",
    ) -> str:
        """Push a local branch to a Git remote."""

        self._validate_branch_name(branch_name)

        if not isinstance(remote, str):
            raise ValueError(
                "Git remote name must be a string"
            )

        normalized_remote = remote.strip()

        if not normalized_remote:
            raise ValueError(
                "Git remote name cannot be empty"
            )

        if normalized_remote.startswith("-"):
            raise ValueError(
                f"Invalid Git remote name: {remote}"
            )

        await self._ensure_repository()

        result = await self._run_git(
            "push",
            "--set-upstream",
            normalized_remote,
            branch_name,
            check=False,
        )

        if result.return_code != 0:
            raise GitRepositoryError(
                self._format_error(
                    f"Unable to push Git branch '{branch_name}'",
                    result,
                )
            )

        return branch_name

    async def get_changed_files(self) -> list[str]:
        """Return working-tree files that differ from HEAD.

        Both tracked modifications and untracked files are included.
        Paths are returned relative to the repository root.
        """

        await self._ensure_repository()

        result = await self._run_git(
            "status",
            "--short",
            "--untracked-files=all",
        )

        changed_files: list[str] = []

        for line in result.stdout.splitlines():
            if not line.strip():
                continue

            if len(line) < 4:
                raise GitRepositoryError(
                    "Unable to parse Git status output"
                )

            status = line[:2]
            path = line[3:]

            if not path:
                raise GitRepositoryError(
                    "Unable to parse Git status output"
                )

            path = self._normalize_git_status_path(path)

            if status[0] == "R" or status[1] == "R":
                rename_parts = path.split(" -> ")

                if len(rename_parts) == 2:
                    path = rename_parts[1]

            changed_files.append(path)

        return sorted(
            set(changed_files),
        )

    async def _ensure_repository(self) -> None:
        """Raise when the configured path is not a Git repository."""

        if not await self.is_repository():
            raise GitRepositoryError(
                f"Path is not a Git repository: "
                f"{self.repository_path}"
            )

    def _validate_commit_files(
        self,
        files: Sequence[str],
    ) -> list[str]:
        """Validate and normalize file paths used for commits."""

        if not files:
            raise ValueError(
                "At least one file must be provided"
            )

        normalized_files: list[str] = []
        seen: set[str] = set()

        for file_path in files:
            if not isinstance(file_path, str):
                raise ValueError(
                    "Git file paths must be strings"
                )

            normalized_path = file_path.strip()

            if not normalized_path:
                raise ValueError(
                    "Git file paths cannot be empty"
                )

            path = Path(normalized_path)

            if path.is_absolute():
                raise ValueError(
                    f"Git file path must be relative: {file_path}"
                )

            if any(
                part == ".."
                for part in path.parts
            ):
                raise ValueError(
                    "Git file path cannot contain '..': "
                    f"{file_path}"
                )

            normalized_path = path.as_posix()

            if normalized_path in seen:
                raise ValueError(
                    "Git file paths must be unique"
                )

            seen.add(normalized_path)
            normalized_files.append(normalized_path)

        return normalized_files

    def _validate_branch_name(
        self,
        branch_name: str,
    ) -> None:
        """Validate a Git branch name."""

        if not isinstance(branch_name, str):
            raise ValueError(
                "Git branch name must be a string"
            )

        if not branch_name.strip():
            raise ValueError(
                "Git branch name cannot be empty"
            )

        branch_name = branch_name.strip()

        if branch_name.startswith("-"):
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if branch_name.endswith("/"):
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if " " in branch_name:
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if ".." in branch_name:
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if "@{" in branch_name:
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if branch_name.endswith("."):
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if branch_name.endswith(".lock"):
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if branch_name.startswith("."):
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if "/." in branch_name:
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if "/-" in branch_name:
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

        if re.search(
            r"[\x00-\x20\x7f\~^:?\*\\\[]",
            branch_name,
        ):
            raise ValueError(
                f"Invalid Git branch name: {branch_name}"
            )

    async def _run_git(
        self,
        *arguments: str,
        check: bool = True,
        timeout: float = 30.0,
    ) -> GitCommandResult:
        """Execute a Git command asynchronously."""

        command = (
            "git",
            *arguments,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.repository_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

        except FileNotFoundError as exc:
            raise GitRepositoryError(
                "Git executable was not found on the system"
            ) from exc

        except asyncio.TimeoutError as exc:
            try:
                process.kill()
            except ProcessLookupError:
                pass

            await process.communicate()

            raise GitRepositoryError(
                f"Git command timed out after {timeout} seconds: "
                f"{' '.join(command)}"
            ) from exc

        stdout = stdout_bytes.decode(
            "utf-8",
            errors="replace",
        )

        stderr = stderr_bytes.decode(
            "utf-8",
            errors="replace",
        )

        result = GitCommandResult(
            command=command,
            return_code=process.returncode or 0,
            stdout=stdout,
            stderr=stderr,
        )

        if check and result.return_code != 0:
            raise GitRepositoryError(
                self._format_error(
                    "Git command failed",
                    result,
                )
            )

        return result

    def _normalize_git_status_path(
        self,
        path: str,
    ) -> str:
        """Normalize a Git status path to a repository-relative POSIX path."""

        normalized_path = path.strip()

        if (
            normalized_path.startswith('"')
            and normalized_path.endswith('"')
            and len(normalized_path) >= 2
        ):
            normalized_path = normalized_path[1:-1]

        normalized_path = normalized_path.replace(
            "\\",
            "/",
        )

        return normalized_path

    def _format_error(
        self,
        message: str,
        result: GitCommandResult,
    ) -> str:
        """Build a useful Git error message."""

        details = (
            result.stderr.strip()
            or result.stdout.strip()
            or "No additional Git output"
        )

        command = " ".join(result.command)

        return (
            f"{message}: {details} "
            f"(command: {command})"
        )