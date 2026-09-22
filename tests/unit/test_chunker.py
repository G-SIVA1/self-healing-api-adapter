from __future__ import annotations

from self_healing_agent.rag.chunker import DocumentChunker
from self_healing_agent.rag.document_loader import Document


def test_chunker_splits_markdown_sections() -> None:
    document = Document(
        source="stripe.md",
        content="""\
# Stripe API

Stripe API documentation.

## Customer Creation

Use the modern customer API.

## Migration

Replace the legacy Customer API.
""",
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk(document)

    assert len(chunks) == 3

    assert chunks[0].content.startswith("# Stripe API")
    assert chunks[1].content.startswith("## Customer Creation")
    assert chunks[2].content.startswith("## Migration")

    assert chunks[0].source == "stripe.md"
    assert chunks[1].source == "stripe.md"
    assert chunks[2].source == "stripe.md"

    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2


def test_chunker_splits_large_sections() -> None:
    document = Document(
        source="large.md",
        content="""\
# Large Section

This is the first paragraph.

This is the second paragraph.

This is the third paragraph.
""",
    )

    chunker = DocumentChunker(max_characters=50)

    chunks = chunker.chunk(document)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.content) <= 50