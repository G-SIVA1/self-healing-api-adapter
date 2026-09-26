from __future__ import annotations

import pytest

from self_healing_agent.git.pull_request import (
    PullRequest,
    PullRequestClient,
    PullRequestError,
    PullRequestValidator,
)


class FakePullRequestClient:
    """Test implementation of the PullRequestClient protocol."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def create_pull_request(
        self,
        repository: str,
        source_branch: str,
        target_branch: str,
        title: str,
        description: str,
    ) -> PullRequest:
        self.calls.append(
            {
                "repository": repository,
                "source_branch": source_branch,
                "target_branch": target_branch,
                "title": title,
                "description": description,
            }
        )

        return PullRequest(
            number=42,
            title=title,
            url=(
                "https://github.com/"
                f"{repository}/pull/42"
            ),
            source_branch=source_branch,
            target_branch=target_branch,
        )


def test_pull_request_model_stores_metadata() -> None:
    pull_request = PullRequest(
        number=42,
        title="fix: self-heal Stripe Customer Create compatibility",
        url="https://github.com/example/project/pull/42",
        source_branch="repair/stripe-customer-create",
        target_branch="main",
    )

    assert pull_request.number == 42

    assert (
        pull_request.title
        == "fix: self-heal Stripe Customer Create compatibility"
    )

    assert (
        pull_request.url
        == "https://github.com/example/project/pull/42"
    )

    assert (
        pull_request.source_branch
        == "repair/stripe-customer-create"
    )

    assert pull_request.target_branch == "main"


def test_pull_request_model_is_immutable() -> None:
    pull_request = PullRequest(
        number=42,
        title="Fix API compatibility",
        url="https://github.com/example/project/pull/42",
        source_branch="repair/test",
        target_branch="main",
    )

    with pytest.raises(
        AttributeError,
    ):
        pull_request.number = 43


@pytest.mark.parametrize(
    "branch_name",
    [
        "main",
        "repair/stripe-customer-create",
        "feature/api-fix",
        "repair/test-1",
    ],
)
def test_validator_accepts_valid_branch_names(
    branch_name: str,
) -> None:
    result = PullRequestValidator.validate_branch_name(
        branch_name,
        "source branch",
    )

    assert result == branch_name


def test_validator_strips_branch_name() -> None:
    result = PullRequestValidator.validate_branch_name(
        "  repair/test  ",
        "source branch",
    )

    assert result == "repair/test"


@pytest.mark.parametrize(
    "branch_name",
    [
        "",
        "   ",
    ],
)
def test_validator_rejects_empty_branch_names(
    branch_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="source branch cannot be empty",
    ):
        PullRequestValidator.validate_branch_name(
            branch_name,
            "source branch",
        )


def test_validator_rejects_non_string_branch_name() -> None:
    with pytest.raises(
        ValueError,
        match="source branch must be a string",
    ):
        PullRequestValidator.validate_branch_name(
            123,  # type: ignore[arg-type]
            "source branch",
        )


def test_validator_accepts_title() -> None:
    result = PullRequestValidator.validate_title(
        "Fix Stripe API compatibility",
    )

    assert result == "Fix Stripe API compatibility"


def test_validator_strips_title() -> None:
    result = PullRequestValidator.validate_title(
        "  Fix Stripe API compatibility  ",
    )

    assert result == "Fix Stripe API compatibility"


@pytest.mark.parametrize(
    "title",
    [
        "",
        "   ",
    ],
)
def test_validator_rejects_empty_title(
    title: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Pull request title cannot be empty",
    ):
        PullRequestValidator.validate_title(title)


def test_validator_rejects_non_string_title() -> None:
    with pytest.raises(
        ValueError,
        match="Pull request title must be a string",
    ):
        PullRequestValidator.validate_title(
            123,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "repository",
    [
        "G-SIVA1/self-healing-api-adapter",
        "openai/example",
        "owner/repository",
    ],
)
def test_validator_accepts_valid_repository(
    repository: str,
) -> None:
    result = PullRequestValidator.validate_repository(
        repository,
    )

    assert result == repository


def test_validator_strips_repository() -> None:
    result = PullRequestValidator.validate_repository(
        "  G-SIVA1/self-healing-api-adapter  ",
    )

    assert result == "G-SIVA1/self-healing-api-adapter"


@pytest.mark.parametrize(
    "repository",
    [
        "",
        "   ",
        "repository",
        "owner/",
        "/repository",
        "owner/repository/extra",
    ],
)
def test_validator_rejects_invalid_repository(
    repository: str,
) -> None:
    with pytest.raises(
        ValueError,
    ):
        PullRequestValidator.validate_repository(
            repository,
        )


def test_validator_rejects_non_string_repository() -> None:
    with pytest.raises(
        ValueError,
        match="GitHub repository must be a string",
    ):
        PullRequestValidator.validate_repository(
            123,  # type: ignore[arg-type]
        )


def test_pull_request_error_is_runtime_error() -> None:
    error = PullRequestError(
        "GitHub operation failed",
    )

    assert isinstance(
        error,
        RuntimeError,
    )

    assert str(error) == "GitHub operation failed"


@pytest.mark.asyncio
async def test_pull_request_client_creates_pull_request() -> None:
    client: PullRequestClient = (
        FakePullRequestClient()
    )

    result = await client.create_pull_request(
        repository="G-SIVA1/self-healing-api-adapter",
        source_branch="repair/stripe-customer-create",
        target_branch="main",
        title="Fix Stripe Customer Create compatibility",
        description="Automatically generated repair.",
    )

    assert result.number == 42

    assert (
        result.title
        == "Fix Stripe Customer Create compatibility"
    )

    assert (
        result.url
        == (
            "https://github.com/"
            "G-SIVA1/self-healing-api-adapter"
            "/pull/42"
        )
    )

    assert (
        result.source_branch
        == "repair/stripe-customer-create"
    )

    assert result.target_branch == "main"


@pytest.mark.asyncio
async def test_pull_request_client_receives_all_arguments() -> None:
    fake_client = FakePullRequestClient()

    client: PullRequestClient = fake_client

    await client.create_pull_request(
        repository="owner/repository",
        source_branch="repair/test",
        target_branch="main",
        title="Fix API compatibility",
        description="Self-healing repair.",
    )

    assert fake_client.calls == [
        {
            "repository": "owner/repository",
            "source_branch": "repair/test",
            "target_branch": "main",
            "title": "Fix API compatibility",
            "description": "Self-healing repair.",
        }
    ]