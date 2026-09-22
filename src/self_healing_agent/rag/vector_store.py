from __future__ import annotations

from pathlib import Path

import chromadb

from self_healing_agent.rag.chunker import DocumentChunk


class VectorStoreError(Exception):
    """Raised when the vector store encounters an error."""


class ChromaVectorStore:
    """Persistent ChromaDB-backed vector store."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "api_documentation",
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        self._persist_directory = persist_directory
        self._persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            self._client = chromadb.PersistentClient(
                path=str(self._persist_directory),
            )

            self._collection = self._client.get_or_create_collection(
                name=collection_name,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Unable to initialize ChromaDB"
            ) from exc

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """Add documentation chunks to ChromaDB."""

        if not chunks:
            return

        ids = [
            f"{chunk.source}:{chunk.chunk_index}"
            for chunk in chunks
        ]

        documents = [
            chunk.content
            for chunk in chunks
        ]

        metadatas = [
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Unable to add documentation chunks to ChromaDB"
            ) from exc

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[DocumentChunk]:
        """
        Search documentation using semantic similarity.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of chunks to return.
        """

        if not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        try:
            result = self._collection.query(
                query_texts=[query],
                n_results=top_k,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Unable to search ChromaDB"
            ) from exc

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        chunks: list[DocumentChunk] = []

        for document, metadata in zip(
            documents,
            metadatas,
            strict=True,
        ):
            if not isinstance(document, str):
                continue

            source = str(metadata.get("source", ""))
            chunk_index = int(metadata.get("chunk_index", 0))

            chunks.append(
                DocumentChunk(
                    content=document,
                    source=source,
                    chunk_index=chunk_index,
                )
            )

        return chunks