from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.rag.document_loader import Document


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A searchable section of a documentation document."""

    content: str
    source: str
    chunk_index: int


class DocumentChunker:
    """Split documentation into meaningful chunks."""

    def __init__(self, max_characters: int = 1000) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be greater than zero")

        self._max_characters = max_characters

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """
        Split a document into searchable chunks.

        Sections are separated using Markdown headings.
        Large sections are further split by paragraphs.
        """

        sections = self._split_sections(document.content)

        chunks: list[DocumentChunk] = []

        for section in sections:
            if len(section) <= self._max_characters:
                chunks.append(
                    DocumentChunk(
                        content=section,
                        source=document.source,
                        chunk_index=len(chunks),
                    )
                )
                continue

            paragraphs = [
                paragraph.strip()
                for paragraph in section.split("\n\n")
                if paragraph.strip()
            ]

            current = ""

            for paragraph in paragraphs:
                if not current:
                    current = paragraph
                    continue

                candidate = f"{current}\n\n{paragraph}"

                if len(candidate) <= self._max_characters:
                    current = candidate
                else:
                    chunks.append(
                        DocumentChunk(
                            content=current,
                            source=document.source,
                            chunk_index=len(chunks),
                        )
                    )

                    current = paragraph

            if current:
                chunks.append(
                    DocumentChunk(
                        content=current,
                        source=document.source,
                        chunk_index=len(chunks),
                    )
                )

        return chunks

    @staticmethod
    def _split_sections(content: str) -> list[str]:
        """Split Markdown content whenever a heading begins."""

        lines = content.splitlines()

        sections: list[str] = []
        current_lines: list[str] = []

        for line in lines:
            if line.startswith("#") and current_lines:
                section = "\n".join(current_lines).strip()

                if section:
                    sections.append(section)

                current_lines = []

            current_lines.append(line)

        if current_lines:
            section = "\n".join(current_lines).strip()

            if section:
                sections.append(section)

        return sections