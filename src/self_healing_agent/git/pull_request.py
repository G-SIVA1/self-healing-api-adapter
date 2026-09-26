from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PullRequest:
    """Represents a GitHub pull request created for a repair."""

    number: int
    title: str
    url: str
    source_branch: str
    target_branch: str


class PullRequestError(RuntimeError):
    """Raised when pull request operations fail."""


class PullRequestValidator:
    """Validate pull request metadata before creation."""

    @staticmethod
    def validate_branch_name(
        branch_name: str,
        field_name: str,
    ) -> str:
        """Validate and normalize a branch name."""

        if not isinstance(branch_name, str):
            raise ValueError(
                f"{field_name} must be a string"
            )

        normalized_branch = branch_name.strip()

        if not normalized_branch:
            raise ValueError(
                f"{field_name} cannot be empty"
            )

        return normalized_branch

    @staticmethod
    def validate_title(
        title: str,
    ) -> str:
        """Validate and normalize a pull request title."""

        if not isinstance(title, str):
            raise ValueError(
                "Pull request title must be a string"
            )

        normalized_title = title.strip()

        if not normalized_title:
            raise ValueError(
                "Pull request title cannot be empty"
            )

        return normalized_title

    @staticmethod
    def validate_repository(
        repository: str,
    ) -> str:
        """Validate a GitHub repository identifier."""

        if not isinstance(repository, str):
            raise ValueError(
                "GitHub repository must be a string"
            )

        normalized_repository = repository.strip()

        if not normalized_repository:
            raise ValueError(
                "GitHub repository cannot be empty"
            )

        if normalized_repository.count("/") != 1:
            raise ValueError(
                "GitHub repository must use the format "
                "'owner/repository'"
            )

        owner, repository_name = (
            normalized_repository.split("/", 1)
        )

        if not owner or not repository_name:
            raise ValueError(
                "GitHub repository must use the format "
                "'owner/repository'"
            )

        return normalized_repository


class PullRequestClient(Protocol):
    """Protocol for asynchronous GitHub pull request clients."""

    async def create_pull_request(
        self,
        repository: str,
        source_branch: str,
        target_branch: str,
        title: str,
        description: str,
    ) -> PullRequest:
        """Create and return a pull request."""
        ...