from __future__ import annotations

from pathlib import Path

from self_healing_agent.rag.chunker import DocumentChunk
from self_healing_agent.rag.vector_store import ChromaVectorStore


def test_vector_store_adds_and_searches_chunks(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_documents",
    )

    chunks = [
        DocumentChunk(
            content=(
                "The legacy stripe.Customer.create API "
                "has been replaced."
            ),
            source="stripe.md",
            chunk_index=0,
        ),
        DocumentChunk(
            content=(
                "Use stripe.customers.create for "
                "creating customers."
            ),
            source="stripe.md",
            chunk_index=1,
        ),
        DocumentChunk(
            content=(
                "Authentication uses API keys."
            ),
            source="stripe.md",
            chunk_index=2,
        ),
    ]

    store.add_chunks(chunks)

    results = store.search(
        "How do I replace the old Customer creation API?",
        top_k=2,
    )

    assert len(results) == 2

    combined_content = " ".join(
        chunk.content
        for chunk in results
    )

    assert "stripe.customers.create" in combined_content


def test_vector_store_returns_empty_for_no_documents(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="empty_documents",
    )

    results = store.search(
        "How do I create a customer?",
    )

    assert results == []