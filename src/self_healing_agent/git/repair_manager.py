from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.git.repository import GitRepository
from self_healing_agent.git.workflow import (
    GitRepairWorkflow,
    RepairBranch,
)


@dataclass(frozen=True, slots=True)
class RepairCommit:
    """Result of committing a verified repair."""

    branch: RepairBranch
    commit_sha: str
    files: tuple[str, ...]


class GitRepairManager:
    """Coordinate verified repair commits."""

    def __init__(
        self,
        repository: GitRepository,
        workflow: GitRepairWorkflow | None = None,
    ) -> None:
        self._repository = repository
        self._workflow = (
            workflow
            if workflow is not None
            else GitRepairWorkflow(repository)
        )

    async def commit_verified_repair(
        self,
        service_name: str,
        api_call: str,
        files: list[str],
        commit_message: str,
        test_passed: bool,
        iteration: int = 1,
    ) -> RepairCommit:
        """Create a branch and commit a verified repair."""

        if not test_passed:
            raise ValueError(
                "Cannot commit a repair that has not passed validation"
            )

        if not files:
            raise ValueError(
                "At least one repaired file must be provided"
            )

        normalized_files = self._normalize_files(
            files,
        )

        branch = (
            await self._workflow.prepare_repair_branch(
                service_name=service_name,
                api_call=api_call,
                iteration=iteration,
            )
        )

        commit_sha = (
            await self._repository.commit_changes(
                files=normalized_files,
                message=commit_message,
            )
        )

        return RepairCommit(
            branch=branch,
            commit_sha=commit_sha,
            files=tuple(normalized_files),
        )

    @staticmethod
    def _normalize_files(
        files: list[str],
    ) -> list[str]:
        """Normalize and validate repaired file paths."""

        normalized_files: list[str] = []

        for file_path in files:
            normalized_path = file_path.strip()

            if not normalized_path:
                raise ValueError(
                    "Repaired file path cannot be empty"
                )

            normalized_files.append(
                normalized_path,
            )

        if len(set(normalized_files)) != len(
            normalized_files
        ):
            raise ValueError(
                "Repaired file paths must be unique"
            )

        return normalized_files