from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SecurityPolicyError(Exception):
    """Raised when a command violates the sandbox security policy."""


@dataclass(frozen=True, slots=True)
class SandboxSecurityPolicy:
    """
    Defines which commands the sandbox is allowed to execute.

    The policy is intentionally restrictive. Only explicitly
    permitted executables can be launched.
    """

    allowed_executables: frozenset[str] = frozenset(
        {
            "python",
            "python.exe",
            "pytest",
            "pytest.exe",
        }
    )

    allowed_working_directory: Path | None = None

    def validate_command(
        self,
        command: list[str],
    ) -> None:
        """Validate a command before execution."""

        if not command:
            raise SecurityPolicyError(
                "Command cannot be empty"
            )

        executable = Path(command[0]).name.lower()

        if executable not in self.allowed_executables:
            raise SecurityPolicyError(
                f"Executable is not allowed: {executable}"
            )

        self._validate_arguments(command)

    def validate_working_directory(
        self,
        working_directory: Path,
    ) -> None:
        """Validate the execution directory."""

        if not working_directory.exists():
            raise SecurityPolicyError(
                "Working directory does not exist"
            )

        if not working_directory.is_dir():
            raise SecurityPolicyError(
                "Working directory is not a directory"
            )

        if self.allowed_working_directory is None:
            return

        try:
            working_directory.resolve().relative_to(
                self.allowed_working_directory.resolve()
            )
        except ValueError as exc:
            raise SecurityPolicyError(
                "Working directory is outside the allowed sandbox"
            ) from exc

    @staticmethod
    def _validate_arguments(
        command: list[str],
    ) -> None:
        """Reject arguments containing dangerous shell operations."""

        dangerous_tokens = {
            "&&",
            "||",
            ";",
            "|",
            ">",
            ">>",
            "<",
            "$(",
            "`",
        }

        for argument in command[1:]:
            for token in dangerous_tokens:
                if token in argument:
                    raise SecurityPolicyError(
                        f"Dangerous command token detected: {token}"
                    )