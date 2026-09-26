from __future__ import annotations

from typing import Any

import httpx

from self_healing_agent.config.settings import (
    Settings,
    SettingsError,
)
from self_healing_agent.git.pull_request import (
    PullRequest,
    PullRequestError,
    PullRequestValidator,
)


class GitHubPullRequestClient:
    """GitHub REST API client for creating pull requests."""

    API_BASE_URL = "https://api.github.com"
    API_VERSION = "2026-03-10"

    def __init__(
        self,
        token: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        normalized_token = token.strip()

        if not normalized_token:
            raise PullRequestError(
                "GitHub token cannot be empty"
            )

        if timeout_seconds <= 0:
            raise PullRequestError(
                "GitHub timeout must be greater than zero"
            )

        self._token = normalized_token
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
    ) -> GitHubPullRequestClient:
        """Create a GitHub client from application settings."""

        if not isinstance(settings, Settings):
            raise PullRequestError(
                "settings must be a Settings instance"
            )

        if not settings.github_token:
            raise SettingsError(
                "GITHUB_TOKEN is required for GitHub operations"
            )

        return cls(
            token=settings.github_token,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    async def create_pull_request(
        self,
        repository: str,
        source_branch: str,
        target_branch: str,
        title: str,
        description: str,
    ) -> PullRequest:
        """Create a pull request on GitHub."""

        normalized_repository = (
            PullRequestValidator.validate_repository(
                repository,
            )
        )

        normalized_source_branch = (
            PullRequestValidator.validate_branch_name(
                source_branch,
                "source_branch",
            )
        )

        normalized_target_branch = (
            PullRequestValidator.validate_branch_name(
                target_branch,
                "target_branch",
            )
        )

        normalized_title = (
            PullRequestValidator.validate_title(title)
        )

        if not isinstance(description, str):
            raise PullRequestError(
                "description must be a string"
            )

        normalized_description = description.strip()

        owner, repo = normalized_repository.split("/", 1)

        url = (
            f"{self.API_BASE_URL}"
            f"/repos/{owner}/{repo}/pulls"
        )

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": self.API_VERSION,
        }

        payload = {
            "title": normalized_title,
            "head": normalized_source_branch,
            "base": normalized_target_branch,
            "body": normalized_description,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
            ) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise PullRequestError(
                f"GitHub API request failed: {exc}"
            ) from exc

        if response.status_code != 201:
            self._raise_api_error(response)

        try:
            response_data: Any = response.json()
        except ValueError as exc:
            raise PullRequestError(
                "GitHub API returned invalid JSON"
            ) from exc

        return self._parse_pull_request(response_data)

    @staticmethod
    def _raise_api_error(
        response: httpx.Response,
    ) -> None:
        """Convert a GitHub HTTP error into a domain error."""

        try:
            data: Any = response.json()
        except ValueError:
            data = {}

        message = data.get("message")

        if not isinstance(message, str) or not message.strip():
            message = response.text.strip()

        if not message:
            message = "Unknown GitHub API error"

        raise PullRequestError(
            f"GitHub API returned HTTP "
            f"{response.status_code}: {message}"
        )

    @staticmethod
    def _parse_pull_request(
        data: Any,
    ) -> PullRequest:
        """Convert GitHub API response into a PullRequest."""

        if not isinstance(data, dict):
            raise PullRequestError(
                "GitHub API response must be a JSON object"
            )

        number = data.get("number")
        title = data.get("title")
        url = data.get("html_url")

        head = data.get("head")
        base = data.get("base")

        if not isinstance(number, int):
            raise PullRequestError(
                "GitHub API response contains invalid PR number"
            )

        if not isinstance(title, str) or not title.strip():
            raise PullRequestError(
                "GitHub API response contains invalid PR title"
            )

        if not isinstance(url, str) or not url.strip():
            raise PullRequestError(
                "GitHub API response contains invalid PR URL"
            )

        if not isinstance(head, dict):
            raise PullRequestError(
                "GitHub API response contains invalid head"
            )

        if not isinstance(base, dict):
            raise PullRequestError(
                "GitHub API response contains invalid base"
            )

        source_branch = head.get("ref")
        target_branch = base.get("ref")

        if (
            not isinstance(source_branch, str)
            or not source_branch.strip()
        ):
            raise PullRequestError(
                "GitHub API response contains invalid source branch"
            )

        if (
            not isinstance(target_branch, str)
            or not target_branch.strip()
        ):
            raise PullRequestError(
                "GitHub API response contains invalid target branch"
            )

        return PullRequest(
            number=number,
            title=title.strip(),
            url=url.strip(),
            source_branch=source_branch.strip(),
            target_branch=target_branch.strip(),
        )