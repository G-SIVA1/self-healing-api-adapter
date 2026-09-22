from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.rag.chunker import DocumentChunk
from self_healing_agent.rag.retriever import DocumentationRetriever
from self_healing_agent.rag.vector_store import ChromaVectorStore


def test_retriever_finds_relevant_documentation(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="retriever_test",
    )

    store.add_chunks(
        [
            DocumentChunk(
                content=(
                    "The legacy stripe.Customer.create API "
                    "has been replaced. Use "
                    "stripe.customers.create instead."
                ),
                source="stripe.md",
                chunk_index=0,
            ),
            DocumentChunk(
                content=(
                    "Stripe authentication uses API keys."
                ),
                source="stripe.md",
                chunk_index=1,
            ),
        ]
    )

    retriever = DocumentationRetriever(store)

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
        raw_log="legacy Stripe failure",
    )

    result = retriever.retrieve(event, top_k=1)

    assert len(result.chunks) == 1

    assert (
        "stripe.customers.create"
        in result.chunks[0].content
    )

    assert "stripe.Customer.create" in result.query
    assert "AttributeError" in result.query


def test_retriever_returns_empty_when_no_documents_exist(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="empty_retriever_test",
    )

    retriever = DocumentationRetriever(store)

    event = ErrorEvent(
        timestamp=datetime.now(timezone.utc),
        exception_type="AttributeError",
        message="module 'stripe' has no attribute 'Customer'",
        file_path=None,
        line_number=None,
        function_name=None,
        api_service="stripe",
        api_call="stripe.Customer.create",
        raw_log="failure",
    )

    result = retriever.retrieve(event)

    assert result.chunks == []