from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.git.repository import (
    GitRepository,
    GitRepositoryError,
)


@dataclass(frozen=True, slots=True)
class RepairBranch:
    """Information about a Git branch created for a repair."""

    name: str
    iteration: int


class GitRepairWorkflow:
    """Coordinate Git branch preparation for API repairs."""

    def __init__(
        self,
        repository: GitRepository,
    ) -> None:
        self._repository = repository

    async def prepare_repair_branch(
        self,
        service_name: str,
        api_call: str,
        iteration: int = 1,
    ) -> RepairBranch:
        """Create a dedicated repair branch."""

        normalized_service = self._normalize_component(
            service_name,
            "service_name",
        )

        normalized_api_call = self._normalize_component(
            api_call,
            "api_call",
        )

        if iteration < 1:
            raise ValueError(
                "iteration must be greater than zero"
            )

        if not await self._repository.is_repository():
            raise GitRepositoryError(
                "Configured path is not a Git repository"
            )

        base_branch = (
            f"repair/"
            f"{normalized_service}-"
            f"{normalized_api_call}"
        )

        branch_name = await self._resolve_branch_name(
            base_branch=base_branch,
            iteration=iteration,
        )

        created_branch = (
            await self._repository.create_branch(
                branch_name,
            )
        )

        return RepairBranch(
            name=created_branch,
            iteration=iteration,
        )

    async def _resolve_branch_name(
        self,
        base_branch: str,
        iteration: int,
    ) -> str:
        """Return an available deterministic repair branch."""

        if not await self._repository.branch_exists(
            base_branch,
        ):
            return base_branch

        candidate = (
            f"{base_branch}-iteration-{iteration}"
        )

        if not await self._repository.branch_exists(
            candidate,
        ):
            return candidate

        suffix = 2

        while True:
            candidate = (
                f"{base_branch}-iteration-"
                f"{iteration}-{suffix}"
            )

            if not await self._repository.branch_exists(
                candidate,
            ):
                return candidate

            suffix += 1

    @staticmethod
    def _normalize_component(
        value: str,
        field_name: str,
    ) -> str:
        """Normalize a branch-name component safely."""

        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty"
            )

        normalized = normalized.replace(
            " ",
            "-",
        )

        allowed_characters = (
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            "-_."
        )

        normalized = "".join(
            character
            if character in allowed_characters
            else "-"
            for character in normalized
        )

        while "--" in normalized:
            normalized = normalized.replace(
                "--",
                "-",
            )

        normalized = normalized.strip(
            "-_."
        )

        if not normalized:
            raise ValueError(
                f"{field_name} does not contain "
                "valid branch-name characters"
            )

        return normalized