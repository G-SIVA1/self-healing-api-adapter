from __future__ import annotations

from pathlib import Path

import pytest

from self_healing_agent.agent.architect import CodePatch
from self_healing_agent.sandbox.patcher import (
    PatchApplicationError,
    SafePatchApplier,
)


@pytest.mark.asyncio
async def test_patcher_applies_valid_patch(
    tmp_path: Path,
) -> None:
    target = tmp_path / "api_client.py"

    original_code = """\
import stripe


def create_customer(user):
    return stripe.Customer.create(
        email=user.email
    )
"""

    replacement_code = """\
import stripe


def create_customer(user):
    return stripe.customers.create(
        email=user.email
    )
"""

    target.write_text(
        original_code,
        encoding="utf-8",
    )

    patch = CodePatch(
        file_path="api_client.py",
        original_code=original_code,
        replacement_code=replacement_code,
        explanation="Migrate the legacy Stripe API.",
    )

    applier = SafePatchApplier()

    result = await applier.apply(
        patch,
        tmp_path,
    )

    assert result == target
    assert target.read_text(encoding="utf-8") == replacement_code


@pytest.mark.asyncio
async def test_patcher_rejects_changed_source(
    tmp_path: Path,
) -> None:
    target = tmp_path / "api_client.py"

    original_code = "stripe.Customer.create()"
    current_code = "stripe.Customer.create(email='x')"

    target.write_text(
        current_code,
        encoding="utf-8",
    )

    patch = CodePatch(
        file_path="api_client.py",
        original_code=original_code,
        replacement_code="stripe.customers.create()",
        explanation="API migration.",
    )

    applier = SafePatchApplier()

    with pytest.raises(PatchApplicationError):
        await applier.apply(
            patch,
            tmp_path,
        )


@pytest.mark.asyncio
async def test_patcher_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    target = tmp_path / "api_client.py"

    target.write_text(
        "print('safe')",
        encoding="utf-8",
    )

    patch = CodePatch(
        file_path="../api_client.py",
        original_code="print('safe')",
        replacement_code="print('modified')",
        explanation="Test patch.",
    )

    applier = SafePatchApplier()

    with pytest.raises(PatchApplicationError):
        await applier.apply(
            patch,
            tmp_path,
        )


@pytest.mark.asyncio
async def test_patcher_rejects_missing_target(
    tmp_path: Path,
) -> None:
    patch = CodePatch(
        file_path="missing.py",
        original_code="old",
        replacement_code="new",
        explanation="Test patch.",
    )

    applier = SafePatchApplier()

    with pytest.raises(PatchApplicationError):
        await applier.apply(
            patch,
            tmp_path,
        )


@pytest.mark.asyncio
async def test_patcher_rejects_non_matching_original_code(
    tmp_path: Path,
) -> None:
    target = tmp_path / "api_client.py"

    target.write_text(
        "actual code",
        encoding="utf-8",
    )

    patch = CodePatch(
        file_path="api_client.py",
        original_code="different code",
        replacement_code="new code",
        explanation="Test patch.",
    )

    applier = SafePatchApplier()

    with pytest.raises(PatchApplicationError):
        await applier.apply(
            patch,
            tmp_path,
        )