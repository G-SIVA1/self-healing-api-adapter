from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.git.github_client import GitHubPullRequestClient
from self_healing_agent.git.pull_request import PullRequest
from self_healing_agent.git.repository import GitRepository
from self_healing_agent.git.workflow import GitRepairWorkflow, RepairBranch


@dataclass(frozen=True, slots=True)
class RepairCommit:
    branch: RepairBranch
    commit_sha: str
    files: tuple[str, ...]
    commit_message: str


@dataclass(frozen=True, slots=True)
class RepairPullRequest:
    commit: RepairCommit
    pull_request: PullRequest


class GitRepairManager:
    def __init__(
        self,
        repository: GitRepository,
        workflow: GitRepairWorkflow | None = None,
        pull_request_client: GitHubPullRequestClient | None = None,
    ) -> None:
        self.repository = repository
        self.workflow = workflow or GitRepairWorkflow(repository)
        self._pull_request_client = pull_request_client

    @property
    def pull_request_client(self) -> GitHubPullRequestClient:
        if self._pull_request_client is None:
            raise RuntimeError(
                "GitHub pull request client is not configured"
            )

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

        normalized_files = self._normalize_files(files)

        changed_files = await self.repository.get_changed_files()

        normalized_changed_files = {
            self._normalize_path(path)
            for path in changed_files
        }

        expected_files = set(normalized_files)

        missing_files = sorted(
            expected_files - normalized_changed_files
        )

        if missing_files:
            raise ValueError(
                "Repair files were not changed: "
                + ", ".join(missing_files)
            )

        unexpected_files = sorted(
            normalized_changed_files - expected_files
        )

        if unexpected_files:
            raise ValueError(
                "Repair contains unexpected changed files: "
                + ", ".join(unexpected_files)
            )

        normalized_service_name = self._normalize_metadata(
            service_name,
            "service_name",
        )

        normalized_api_call = self._normalize_metadata(
            api_call,
            "api_call",
        )

        normalized_message = (
            commit_message.strip()
            if isinstance(commit_message, str)
            else (
                "fix: self-heal "
                f"{normalized_service_name} "
                f"{normalized_api_call} "
                "compatibility"
            )
        )

        if not normalized_message:
            raise ValueError(
                "Commit message cannot be empty"
            )

        branch = await self.workflow.prepare_repair_branch(
            service_name=normalized_service_name,
            api_call=normalized_api_call,
            iteration=iteration,
        )

        commit_sha = await self.repository.commit_changes(
            files=normalized_files,
            message=normalized_message,
        )

        try:
            await self.repository.push_branch(
                branch.name
            )
        except Exception as push_error:
            try:
                await self.repository.revert_commit(
                    commit_sha
                )
            except Exception as rollback_error:
                raise RuntimeError(
                    "Repair commit "
                    f"{commit_sha} could not be pushed "
                    "and rollback failed: "
                    f"{rollback_error}"
                ) from rollback_error

            raise push_error

        return RepairCommit(
            branch=branch,
            commit_sha=commit_sha,
            files=tuple(normalized_files),
            commit_message=normalized_message,
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
            if isinstance(title, str)
            else (
                "Automated repair: "
                f"{commit.commit_message}"
            )
        )

        if not normalized_title:
            raise ValueError(
                "Pull request title cannot be empty"
            )

        normalized_description = (
            description.strip()
            if isinstance(description, str)
            else ""
        )

        existing_pull_request = (
            await self.pull_request_client.find_existing_pull_request(
                repository=normalized_repository,
                source_branch=commit.branch.name,
                target_branch=normalized_target_branch,
            )
        )

        if existing_pull_request is not None:
            return RepairPullRequest(
                commit=commit,
                pull_request=existing_pull_request,
            )

        pull_request = (
            await self.pull_request_client.create_pull_request(
                repository=normalized_repository,
                source_branch=commit.branch.name,
                target_branch=normalized_target_branch,
                title=normalized_title,
                description=normalized_description,
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
                "At least one repair file must be provided"
            )

        normalized_files: list[str] = []
        seen: set[str] = set()

        for file_path in files:
            if not isinstance(file_path, str):
                raise ValueError(
                    "Repair file paths must be strings"
                )

            normalized_path = (
                GitRepairManager._normalize_path(
                    file_path
                )
            )

            if not normalized_path:
                raise ValueError(
                    "Repair file paths cannot be empty"
                )

            if normalized_path in seen:
                raise ValueError(
                    "Repair file paths must be unique"
                )

            seen.add(normalized_path)
            normalized_files.append(
                normalized_path
            )

        return normalized_files

    @staticmethod
    def _normalize_path(
        path: str,
    ) -> str:
        return (
            path.strip()
            .replace("\\", "/")
            .lstrip("./")
        )

    @staticmethod
    def _normalize_metadata(
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty"
            )

        return normalized
