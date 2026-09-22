from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from self_healing_agent.sandbox.security import (
    SandboxSecurityPolicy,
    SecurityPolicyError,
)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result produced by a sandbox execution."""

    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool

    @property
    def succeeded(self) -> bool:
        """Return True when execution completed successfully."""

        return (
            self.return_code == 0
            and not self.timed_out
        )


class SandboxExecutionError(Exception):
    """Raised when the sandbox cannot start execution."""


class SandboxExecutor:
    """
    Execute commands through a security policy.

    Every command and working directory is validated before
    a subprocess is started.
    """

    def __init__(
        self,
        security_policy: SandboxSecurityPolicy,
        timeout_seconds: float = 10.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero"
            )

        self._security_policy = security_policy
        self._timeout_seconds = timeout_seconds

    async def execute(
        self,
        command: list[str],
        working_directory: Path,
    ) -> ExecutionResult:
        """
        Validate and execute a command asynchronously.

        Raises:
            SecurityPolicyError:
                If the command or working directory violates
                the security policy.

            SandboxExecutionError:
                If the process cannot be started.
        """

        self._security_policy.validate_command(command)

        self._security_policy.validate_working_directory(
            working_directory
        )

        start_time = time.perf_counter()

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(working_directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise SandboxExecutionError(
                f"Unable to start command: {command!r}"
            ) from exc

        timed_out = False

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True

            process.kill()

            stdout_bytes, stderr_bytes = await process.communicate()

        duration_seconds = time.perf_counter() - start_time

        return ExecutionResult(
            return_code=process.returncode or 0,
            stdout=stdout_bytes.decode(
                "utf-8",
                errors="replace",
            ),
            stderr=stderr_bytes.decode(
                "utf-8",
                errors="replace",
            ),
            duration_seconds=duration_seconds,
            timed_out=timed_out,
        )