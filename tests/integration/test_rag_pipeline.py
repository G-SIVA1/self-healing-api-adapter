from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.rag.chunker import DocumentChunker
from self_healing_agent.rag.document_loader import DocumentLoader
from self_healing_agent.rag.retriever import DocumentationRetriever
from self_healing_agent.rag.vector_store import ChromaVectorStore


@pytest.mark.asyncio
async def test_complete_rag_pipeline(
    tmp_path: Path,
) -> None:
    # ---------------------------------------------------------
    # 1. Create documentation
    # ---------------------------------------------------------

    document_path = tmp_path / "stripe_migration.md"

    document_path.write_text(
        """\
# Stripe API Migration Guide

## Customer Creation

The legacy Customer API is no longer supported.

The old API was:

stripe.Customer.create(email="user@example.com")

The current API is:

stripe.customers.create(email="user@example.com")

Applications should migrate from the legacy Customer API
to the modern customers interface.

## Authentication

Stripe API requests require valid API credentials.
""",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 2. Load documentation
    # ---------------------------------------------------------

    loader = DocumentLoader()

    document = await loader.load(document_path)

    assert "stripe.Customer.create" in document.content

    # ---------------------------------------------------------
    # 3. Chunk documentation
    # ---------------------------------------------------------

    chunker = DocumentChunker(
        max_characters=1000,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2

    # ---------------------------------------------------------
    # 4. Store chunks in ChromaDB
    # ---------------------------------------------------------

    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="integration_rag",
    )

    vector_store.add_chunks(chunks)

    # ---------------------------------------------------------
    # 5. Create runtime error event
    # ---------------------------------------------------------

    event = ErrorEvent(
        timestamp=datetime.now(timezone.utc),
        exception_type="AttributeError",
        message=(
            "module 'stripe' has no attribute 'Customer'"
        ),
        file_path="api_client.py",
        line_number=42,
        function_name="create_customer",
        api_service="stripe",
        api_call="stripe.Customer.create",
        raw_log=(
            "AttributeError: module 'stripe' "
            "has no attribute 'Customer'"
        ),
    )

    # ---------------------------------------------------------
    # 6. Retrieve relevant documentation
    # ---------------------------------------------------------

    retriever = DocumentationRetriever(
        vector_store,
    )

    result = retriever.retrieve(
        event,
        top_k=2,
    )

    # ---------------------------------------------------------
    # 7. Verify semantic retrieval
    # ---------------------------------------------------------

    assert len(result.chunks) > 0

    retrieved_content = " ".join(
        chunk.content
        for chunk in result.chunks
    )

    assert "stripe.customers.create" in retrieved_content