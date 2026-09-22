from __future__ import annotations

from pathlib import Path

import pytest

from self_healing_agent.rag.document_loader import (
    DocumentLoader,
    DocumentLoaderError,
)


@pytest.mark.asyncio
async def test_document_loader_reads_markdown_file(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "stripe_migration.md"

    document_path.write_text(
        "# Stripe Migration\n\nUse the modern API.",
        encoding="utf-8",
    )

    loader = DocumentLoader()

    document = await loader.load(document_path)

    assert document.source == str(document_path)
    assert document.content == (
        "# Stripe Migration\n\nUse the modern API."
    )


@pytest.mark.asyncio
async def test_document_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "missing.md"

    loader = DocumentLoader()

    with pytest.raises(DocumentLoaderError):
        await loader.load(document_path)


@pytest.mark.asyncio
async def test_document_loader_rejects_empty_file(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "empty.md"

    document_path.write_text("", encoding="utf-8")

    loader = DocumentLoader()

    with pytest.raises(DocumentLoaderError):
        await loader.load(document_path)