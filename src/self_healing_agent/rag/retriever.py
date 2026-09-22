from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.rag.chunker import DocumentChunk
from self_healing_agent.rag.vector_store import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Documentation retrieved for an API failure."""

    query: str
    chunks: list[DocumentChunk]


class DocumentationRetriever:
    """Retrieve relevant API documentation for runtime failures."""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self._vector_store = vector_store

    def retrieve(
        self,
        event: ErrorEvent,
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        Retrieve documentation relevant to an ErrorEvent.
        """

        query = self._build_query(event)

        chunks = self._vector_store.search(
            query=query,
            top_k=top_k,
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
        )

    @staticmethod
    def _build_query(event: ErrorEvent) -> str:
        """Build a semantic search query from an ErrorEvent."""

        parts: list[str] = []

        if event.api_service:
            parts.append(
                f"API service: {event.api_service}"
            )

        if event.api_call:
            parts.append(
                f"API call: {event.api_call}"
            )

        parts.append(
            f"Exception: {event.exception_type}"
        )

        parts.append(
            f"Error message: {event.message}"
        )

        if event.function_name:
            parts.append(
                f"Function: {event.function_name}"
            )

        return "\n".join(parts)