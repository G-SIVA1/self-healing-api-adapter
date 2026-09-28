from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from self_healing_agent.agent.loop import SelfCorrectionLoop
from self_healing_agent.agent.state import AgentState
from self_healing_agent.git.pull_request import PullRequest
from self_healing_agent.git.repair_manager import (
    GitRepairManager,
    RepairCommit,
    RepairPullRequest,
)


def _build_loop(
    git_repair_manager: GitRepairManager | None = None,
) -> SelfCorrectionLoop:
    """Build a SelfCorrectionLoop with isolated test dependencies."""

    return SelfCorrectionLoop(
        architect=Mock(),
        repair_executor=Mock(),
        reasoner=Mock(),
        context_builder=Mock(),
        decision_engine=Mock(),
        git_repair_manager=git_repair_manager,
    )


def _build_state(
    *,
    repair_complete: bool,
    test_passed: bool,
) -> AgentState:
    """Build a minimal AgentState for Git integration tests."""

    state = Mock(spec=AgentState)

    state.repair_complete = repair_complete
    state.test_passed = test_passed
    state.iteration = 1

    state.error_event = Mock()
    state.error_event.file_path = "repair_target.py"

    return state


def _build_commit() -> RepairCommit:
    """Build a deterministic repair commit for tests."""

    return RepairCommit(
        branch=Mock(),
        commit_sha="abcdef1234567890",
        files=("repair_target.py",),
        commit_message="Fix stripe Customer.create compatibility",
    )


def _build_pull_request() -> PullRequest:
    """Build a deterministic pull request for tests."""

    return PullRequest(
        number=42,
        url="https://github.com/G-SIVA1/self-healing-api-adapter/pull/42",
        title="Stripe API compatibility repair",
        source_branch="repair/stripe-customer-create",
        target_branch="main",
    )


@pytest.mark.asyncio
async def test_successful_repair_commits_to_git(
    tmp_path: Path,
) -> None:
    """A verified successful repair is passed to the Git repair manager."""

    git_manager = Mock(spec=GitRepairManager)

    expected_commit = _build_commit()

    git_manager.commit_verified_repair = AsyncMock(
        return_value=expected_commit,
    )

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    result = await loop._commit_verified_repair(
        state=state,
        workspace=tmp_path,
        service_name="stripe",
        api_call="Customer.create",
    )

    assert result is expected_commit

    git_manager.commit_verified_repair.assert_awaited_once_with(
        service_name="stripe",
        api_call="Customer.create",
        files=["repair_target.py"],
        commit_message="Fix stripe Customer.create compatibility",
        test_passed=True,
        iteration=1,
    )


@pytest.mark.asyncio
async def test_failed_repair_does_not_commit_to_git(
    tmp_path: Path,
) -> None:
    """The loop must not call Git when the repair is not complete."""

    git_manager = Mock(spec=GitRepairManager)
    git_manager.commit_verified_repair = AsyncMock()

    loop = _build_loop(git_repair_manager=git_manager)

    commit_method = AsyncMock()
    loop._commit_verified_repair = commit_method

    state = _build_state(
        repair_complete=False,
        test_passed=False,
    )

    state.can_continue = Mock(return_value=False)

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=["python", "-m", "pytest"],
    )

    assert result.git_commit is None
    assert result.git_pull_request is None

    commit_method.assert_not_awaited()
    git_manager.commit_verified_repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_git_integration_is_skipped_without_manager(
    tmp_path: Path,
) -> None:
    """Git integration is skipped when no Git manager is configured."""

    loop = _build_loop(git_repair_manager=None)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    result = await loop._commit_verified_repair(
        state=state,
        workspace=tmp_path,
        service_name="stripe",
        api_call="Customer.create",
    )

    assert result is None


@pytest.mark.asyncio
async def test_git_integration_requires_service_name(
    tmp_path: Path,
) -> None:
    """Git integration is skipped when the service name is missing."""

    git_manager = Mock(spec=GitRepairManager)
    git_manager.commit_verified_repair = AsyncMock()

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    result = await loop._commit_verified_repair(
        state=state,
        workspace=tmp_path,
        service_name=None,
        api_call="Customer.create",
    )

    assert result is None

    git_manager.commit_verified_repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_git_integration_requires_api_call(
    tmp_path: Path,
) -> None:
    """Git integration is skipped when the API call is missing."""

    git_manager = Mock(spec=GitRepairManager)
    git_manager.commit_verified_repair = AsyncMock()

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    result = await loop._commit_verified_repair(
        state=state,
        workspace=tmp_path,
        service_name="stripe",
        api_call=None,
    )

    assert result is None

    git_manager.commit_verified_repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_git_integration_requires_target_file(
    tmp_path: Path,
) -> None:
    """Git integration rejects a successful repair without a target file."""

    git_manager = Mock(spec=GitRepairManager)
    git_manager.commit_verified_repair = AsyncMock()

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    state.error_event.file_path = None

    with pytest.raises(
        RuntimeError,
        match="Cannot commit verified repair without a target file",
    ):
        await loop._commit_verified_repair(
            state=state,
            workspace=tmp_path,
            service_name="stripe",
            api_call="Customer.create",
        )

    git_manager.commit_verified_repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_absolute_target_file_is_converted_to_git_relative_path(
    tmp_path: Path,
) -> None:
    """An absolute repair path is converted to a workspace-relative path."""

    target_file = tmp_path / "src" / "repair_target.py"
    target_file.parent.mkdir(parents=True)
    target_file.write_text(
        "print('repair')",
        encoding="utf-8",
    )

    git_manager = Mock(spec=GitRepairManager)

    expected_commit = RepairCommit(
        branch=Mock(),
        commit_sha="abcdef1234567890",
        files=("src/repair_target.py",),
        commit_message="Fix stripe Customer.create compatibility",
    )

    git_manager.commit_verified_repair = AsyncMock(
        return_value=expected_commit,
    )

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    state.error_event.file_path = str(target_file)

    result = await loop._commit_verified_repair(
        state=state,
        workspace=tmp_path,
        service_name="stripe",
        api_call="Customer.create",
    )

    assert result is expected_commit

    git_manager.commit_verified_repair.assert_awaited_once_with(
        service_name="stripe",
        api_call="Customer.create",
        files=["src/repair_target.py"],
        commit_message="Fix stripe Customer.create compatibility",
        test_passed=True,
        iteration=1,
    )


@pytest.mark.asyncio
async def test_target_file_outside_workspace_is_rejected(
    tmp_path: Path,
) -> None:
    """A repair target outside the workspace cannot be committed."""

    git_manager = Mock(spec=GitRepairManager)
    git_manager.commit_verified_repair = AsyncMock()

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    outside_file = tmp_path.parent / "outside.py"

    state.error_event.file_path = str(outside_file)

    with pytest.raises(
        RuntimeError,
        match="Repair target file is outside the workspace",
    ):
        await loop._commit_verified_repair(
            state=state,
            workspace=tmp_path,
            service_name="stripe",
            api_call="Customer.create",
        )

    git_manager.commit_verified_repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_commits_successful_repair_to_git(
    tmp_path: Path,
) -> None:
    """The real loop commits the repair after successful validation."""

    git_manager = Mock(spec=GitRepairManager)

    expected_commit = _build_commit()

    git_manager.commit_verified_repair = AsyncMock(
        return_value=expected_commit,
    )

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    state.can_continue = Mock(return_value=False)

    commit_method = AsyncMock(
        return_value=expected_commit,
    )

    loop._commit_verified_repair = commit_method

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=["python", "-m", "pytest"],
        git_service_name="stripe",
        git_api_call="Customer.create",
    )

    assert result.git_commit is expected_commit
    assert result.git_pull_request is None

    commit_method.assert_awaited_once_with(
        state=state,
        workspace=tmp_path,
        service_name="stripe",
        api_call="Customer.create",
    )


@pytest.mark.asyncio
async def test_run_does_not_commit_when_repair_is_incomplete(
    tmp_path: Path,
) -> None:
    """The real loop never commits an incomplete repair."""

    git_manager = Mock(spec=GitRepairManager)
    git_manager.commit_verified_repair = AsyncMock()

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=False,
        test_passed=False,
    )

    state.can_continue = Mock(return_value=False)

    commit_method = AsyncMock()
    loop._commit_verified_repair = commit_method

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=["python", "-m", "pytest"],
        git_service_name="stripe",
        git_api_call="Customer.create",
    )

    assert result.git_commit is None
    assert result.git_pull_request is None

    commit_method.assert_not_awaited()
    git_manager.commit_verified_repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_creates_pull_request_after_successful_git_commit(
    tmp_path: Path,
) -> None:
    """The real loop creates a pull request after a successful Git commit."""

    git_manager = Mock(spec=GitRepairManager)

    expected_commit = _build_commit()

    expected_pull_request = RepairPullRequest(
        commit=expected_commit,
        pull_request=_build_pull_request(),
    )

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    state.can_continue = Mock(return_value=False)

    commit_method = AsyncMock(
        return_value=expected_commit,
    )

    pull_request_method = AsyncMock(
        return_value=expected_pull_request,
    )

    loop._commit_verified_repair = commit_method
    loop._create_pull_request = pull_request_method

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=["python", "-m", "pytest"],
        git_service_name="stripe",
        git_api_call="Customer.create",
        git_repository="G-SIVA1/self-healing-api-adapter",
        git_target_branch="main",
        git_pull_request_title="Stripe API compatibility repair",
        git_pull_request_description=(
            "Automated repair generated by the self-healing agent."
        ),
    )

    assert result.git_commit is expected_commit
    assert result.git_pull_request is expected_pull_request

    commit_method.assert_awaited_once_with(
        state=state,
        workspace=tmp_path,
        service_name="stripe",
        api_call="Customer.create",
    )

    pull_request_method.assert_awaited_once_with(
        commit=expected_commit,
        repository="G-SIVA1/self-healing-api-adapter",
        target_branch="main",
        title="Stripe API compatibility repair",
        description=(
            "Automated repair generated by the self-healing agent."
        ),
    )


@pytest.mark.asyncio
async def test_run_does_not_create_pull_request_without_git_commit(
    tmp_path: Path,
) -> None:
    """The loop never creates a PR when no Git commit was produced."""

    git_manager = Mock(spec=GitRepairManager)

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    state.can_continue = Mock(return_value=False)

    commit_method = AsyncMock(
        return_value=None,
    )

    pull_request_method = AsyncMock()

    loop._commit_verified_repair = commit_method
    loop._create_pull_request = pull_request_method

    result = await loop.run(
        state=state,
        workspace=tmp_path,
        validation_command=["python", "-m", "pytest"],
        git_service_name="stripe",
        git_api_call="Customer.create",
        git_repository="G-SIVA1/self-healing-api-adapter",
        git_target_branch="main",
        git_pull_request_title="Stripe API compatibility repair",
    )

    assert result.git_commit is None
    assert result.git_pull_request is None

    commit_method.assert_awaited_once()

    pull_request_method.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_preserves_successful_commit_when_pull_request_creation_fails(
    tmp_path: Path,
) -> None:
    """A PR failure does not cause the successful Git commit to be retried."""

    git_manager = Mock(spec=GitRepairManager)

    expected_commit = _build_commit()

    loop = _build_loop(git_repair_manager=git_manager)

    state = _build_state(
        repair_complete=True,
        test_passed=True,
    )

    state.can_continue = Mock(return_value=False)

    commit_method = AsyncMock(
        return_value=expected_commit,
    )

    pull_request_method = AsyncMock(
        side_effect=RuntimeError("GitHub PR creation failed"),
    )

    loop._commit_verified_repair = commit_method
    loop._create_pull_request = pull_request_method

    with pytest.raises(
        RuntimeError,
        match="GitHub PR creation failed",
    ):
        await loop.run(
            state=state,
            workspace=tmp_path,
            validation_command=["python", "-m", "pytest"],
            git_service_name="stripe",
            git_api_call="Customer.create",
            git_repository="G-SIVA1/self-healing-api-adapter",
            git_target_branch="main",
            git_pull_request_title="Stripe API compatibility repair",
        )

    assert commit_method.await_count == 1
    assert commit_method.await_args.kwargs == {
        "state": state,
        "workspace": tmp_path,
        "service_name": "stripe",
        "api_call": "Customer.create",
    }

    assert pull_request_method.await_count == 1