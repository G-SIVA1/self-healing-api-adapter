from __future__ import annotations

import sys
from pathlib import Path

import pytest

from self_healing_agent.sandbox.executor import (
    SandboxExecutor,
)
from self_healing_agent.sandbox.security import (
    SandboxSecurityPolicy,
    SecurityPolicyError,
)


def create_executor(
    sandbox_directory: Path,
    timeout: float = 5.0,
) -> SandboxExecutor:
    policy = SandboxSecurityPolicy(
        allowed_working_directory=sandbox_directory,
    )

    return SandboxExecutor(
        security_policy=policy,
        timeout_seconds=timeout,
    )


@pytest.mark.asyncio
async def test_executor_runs_successful_command(
    tmp_path: Path,
) -> None:
    script = tmp_path / "success.py"

    script.write_text(
        "print('hello sandbox')",
        encoding="utf-8",
    )

    executor = create_executor(tmp_path)

    result = await executor.execute(
        command=[
            sys.executable,
            str(script),
        ],
        working_directory=tmp_path,
    )

    assert result.succeeded is True
    assert result.return_code == 0
    assert "hello sandbox" in result.stdout
    assert result.stderr == ""
    assert result.timed_out is False
    assert result.duration_seconds >= 0


@pytest.mark.asyncio
async def test_executor_captures_stderr(
    tmp_path: Path,
) -> None:
    script = tmp_path / "failure.py"

    script.write_text(
        """
import sys

print("something went wrong", file=sys.stderr)
sys.exit(1)
""",
        encoding="utf-8",
    )

    executor = create_executor(tmp_path)

    result = await executor.execute(
        command=[
            sys.executable,
            str(script),
        ],
        working_directory=tmp_path,
    )

    assert result.succeeded is False
    assert result.return_code == 1
    assert "something went wrong" in result.stderr
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_executor_handles_timeout(
    tmp_path: Path,
) -> None:
    script = tmp_path / "timeout.py"

    script.write_text(
        """
import time

time.sleep(5)
""",
        encoding="utf-8",
    )

    executor = create_executor(
        tmp_path,
        timeout=0.1,
    )

    result = await executor.execute(
        command=[
            sys.executable,
            str(script),
        ],
        working_directory=tmp_path,
    )

    assert result.succeeded is False
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_executor_rejects_empty_command(
    tmp_path: Path,
) -> None:
    executor = create_executor(tmp_path)

    with pytest.raises(SecurityPolicyError):
        await executor.execute(
            command=[],
            working_directory=tmp_path,
        )


@pytest.mark.asyncio
async def test_executor_rejects_unknown_executable(
    tmp_path: Path,
) -> None:
    executor = create_executor(tmp_path)

    with pytest.raises(SecurityPolicyError):
        await executor.execute(
            command=[
                "powershell",
                "dangerous-command",
            ],
            working_directory=tmp_path,
        )


@pytest.mark.asyncio
async def test_executor_rejects_outside_directory(
    tmp_path: Path,
) -> None:
    sandbox_directory = tmp_path / "sandbox"
    outside_directory = tmp_path / "outside"

    sandbox_directory.mkdir()
    outside_directory.mkdir()

    executor = create_executor(
        sandbox_directory,
    )

    with pytest.raises(SecurityPolicyError):
        await executor.execute(
            command=[
                sys.executable,
                "--version",
            ],
            working_directory=outside_directory,
        )