from __future__ import annotations

import sys
from pathlib import Path

import pytest

from self_healing_agent.sandbox.security import (
    SandboxSecurityPolicy,
    SecurityPolicyError,
)


def test_security_policy_allows_python(
    tmp_path: Path,
) -> None:
    policy = SandboxSecurityPolicy(
        allowed_working_directory=tmp_path,
    )

    policy.validate_command(
        [
            sys.executable,
            "test.py",
        ]
    )

    policy.validate_working_directory(tmp_path)


def test_security_policy_rejects_unknown_executable(
    tmp_path: Path,
) -> None:
    policy = SandboxSecurityPolicy(
        allowed_working_directory=tmp_path,
    )

    with pytest.raises(SecurityPolicyError):
        policy.validate_command(
            [
                "powershell",
                "something-dangerous",
            ]
        )


def test_security_policy_rejects_shell_operator(
    tmp_path: Path,
) -> None:
    policy = SandboxSecurityPolicy(
        allowed_working_directory=tmp_path,
    )

    with pytest.raises(SecurityPolicyError):
        policy.validate_command(
            [
                sys.executable,
                "test.py",
                "&&",
                "dangerous-command",
            ]
        )


def test_security_policy_rejects_outside_directory(
    tmp_path: Path,
) -> None:
    allowed_directory = tmp_path / "sandbox"
    outside_directory = tmp_path / "outside"

    allowed_directory.mkdir()
    outside_directory.mkdir()

    policy = SandboxSecurityPolicy(
        allowed_working_directory=allowed_directory,
    )

    with pytest.raises(SecurityPolicyError):
        policy.validate_working_directory(
            outside_directory,
        )


def test_security_policy_rejects_empty_command(
    tmp_path: Path,
) -> None:
    policy = SandboxSecurityPolicy(
        allowed_working_directory=tmp_path,
    )

    with pytest.raises(SecurityPolicyError):
        policy.validate_command([])


def test_security_policy_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    policy = SandboxSecurityPolicy(
        allowed_working_directory=tmp_path,
    )

    with pytest.raises(SecurityPolicyError):
        policy.validate_working_directory(
            tmp_path / "does-not-exist"
        )