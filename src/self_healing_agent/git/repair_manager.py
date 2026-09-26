from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.git.github_client import (
    GitHubPullRequestClient,
)
from self_healing_agent.git.pull_request import PullRequest
from self_healing_agent.git.repository import GitRepository
from self_healing_agent.git.workflow import (
    GitRepairWorkflow,
    RepairBranch,
)


@dataclass(frozen=True, slots=True)
class RepairCommit:
    """Information about a committed repair."""

    branch: RepairBranch
    commit_sha: str
    files: tuple[str, ...]
    commit_message: str


@dataclass(frozen=True, slots=True)
class RepairPullRequest:
    """Information about a created repair pull request."""

    commit: RepairCommit
    pull_request: PullRequest


class GitRepairManager:
    """Coordinate branch creation, commit, push, and pull request creation."""

    def __init__(
        self,
        repository: GitRepository,
        workflow: GitRepairWorkflow | None = None,
        pull_request_client: GitHubPullRequestClient | None = None,
    ) -> None:
        self._repository = repository
        self._workflow = workflow or GitRepairWorkflow(
            repository,
        )
        self._pull_request_client = pull_request_client

    @property
    def pull_request_client(
        self,
    ) -> GitHubPullRequestClient | None:
        return self._pull_request_client

    async def commit_verified_repair(
        self,
        *,
        service_name: str,
        api_call: str,
        files: list[str],
        test_passed: bool,
        iteration: int = 1,
        commit_message: str | None = None,
    ) -> RepairCommit:
        if not test_passed:
            raise ValueError(
                "Repair has not passed validation"
            )

        normalized_files = self._normalize_files(
            files,
        )

        changed_files = await self._repository.get_changed_files()

        normalized_changed_files = {
            self._normalize_path(path)
            for path in changed_files
        }

        expected_files = set(
            normalized_files,
        )

        missing_files = sorted(
            expected_files
            - normalized_changed_files,
        )

        if missing_files:
            raise ValueError(
                "Repair files were not changed: "
                + ", ".join(missing_files)
            )

        unexpected_files = sorted(
            normalized_changed_files
            - expected_files,
        )

        if unexpected_files:
            raise ValueError(
                "Repair contains unexpected changed files: "
                + ", ".join(unexpected_files)
            )

        normalized_service = self._normalize_metadata(
            service_name,
            "service_name",
        )

        normalized_api_call = self._normalize_metadata(
            api_call,
            "api_call",
        )

        message = (
            commit_message.strip()
            if commit_message is not None
            else self._generate_commit_message(
                normalized_service,
                normalized_api_call,
            )
        )

        if not message:
            raise ValueError(
                "Commit message cannot be empty"
            )

        branch = await self._workflow.prepare_repair_branch(
            service_name=normalized_service,
            api_call=normalized_api_call,
            iteration=iteration,
        )

        commit_sha = await self._repository.commit_changes(
            files=normalized_files,
            message=message,
        )

        await self._repository.push_branch(
            branch.name,
        )

        return RepairCommit(
            branch=branch,
            commit_sha=commit_sha,
            files=tuple(normalized_files),
            commit_message=message,
        )

    async def create_pull_request_for_commit(
        self,
        *,
        commit: RepairCommit,
        repository: str,
        target_branch: str,
        title: str | None = None,
        description: str = "",
    ) -> RepairPullRequest:
        if self._pull_request_client is None:
            raise RuntimeError(
                "GitHub pull request client is not configured"
            )

        if not isinstance(
            commit,
            RepairCommit,
        ):
            raise TypeError(
                "commit must be a RepairCommit"
            )

        normalized_repository = self._normalize_metadata(
            repository,
            "repository",
        )

        normalized_target_branch = self._normalize_metadata(
            target_branch,
            "target_branch",
        )

        normalized_title = (
            title.strip()
            if title is not None
            else commit.commit_message
        )

        if not normalized_title:
            raise ValueError(
                "Pull request title cannot be empty"
            )

        if not isinstance(
            description,
            str,
        ):
            raise TypeError(
                "Pull request description must be a string"
            )

        pull_request = (
            await self._pull_request_client.create_pull_request(
                repository=normalized_repository,
                source_branch=commit.branch.name,
                target_branch=normalized_target_branch,
                title=normalized_title,
                description=description,
            )
        )

        return RepairPullRequest(
            commit=commit,
            pull_request=pull_request,
        )

    async def commit_and_create_pull_request(
        self,
        *,
        service_name: str,
        api_call: str,
        files: list[str],
        test_passed: bool,
        repository: str,
        target_branch: str,
        iteration: int = 1,
        commit_message: str | None = None,
        pull_request_title: str | None = None,
        pull_request_description: str = "",
    ) -> RepairPullRequest:
        commit = await self.commit_verified_repair(
            service_name=service_name,
            api_call=api_call,
            files=files,
            test_passed=test_passed,
            iteration=iteration,
            commit_message=commit_message,
        )

        return await self.create_pull_request_for_commit(
            commit=commit,
            repository=repository,
            target_branch=target_branch,
            title=pull_request_title,
            description=pull_request_description,
        )

    @staticmethod
    def _normalize_files(
        files: list[str],
    ) -> list[str]:
        if not files:
            raise ValueError(
                "At least one repair file is required"
            )

        normalized: list[str] = []

        for file_path in files:
            if not isinstance(
                file_path,
                str,
            ):
                raise TypeError(
                    "Repair file paths must be strings"
                )

            path = file_path.strip()

            if not path:
                raise ValueError(
                    "Repair file path cannot be empty"
                )

            normalized.append(
                GitRepairManager._normalize_path(
                    path,
                )
            )

        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "Repair file paths must be unique"
            )

        return normalized

    @staticmethod
    def _normalize_path(
        path: str,
    ) -> str:
        normalized = path.strip().replace(
            "\\",
            "/",
        )

        while normalized.startswith("./"):
            normalized = normalized[2:]

        return normalized

    @staticmethod
    def _normalize_metadata(
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                f"{field_name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty"
            )

        return normalized

    @staticmethod
    def _generate_commit_message(
        service_name: str,
        api_call: str,
    ) -> str:
        return (
            "fix: self-heal "
            f"{service_name} "
            f"{api_call.strip()} compatibility"
        )