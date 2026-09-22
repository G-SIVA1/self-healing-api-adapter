from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """Represents a loaded documentation document."""

    source: str
    content: str


class DocumentLoaderError(Exception):
    """Raised when a documentation document cannot be loaded."""


class DocumentLoader:
    """Load documentation files from the local filesystem."""

    async def load(self, file_path: Path) -> Document:
        """
        Asynchronously load a documentation file.

        Args:
            file_path: Path to the documentation file.

        Returns:
            A Document containing the source path and file contents.

        Raises:
            DocumentLoaderError: If the file cannot be read.
        """

        if not file_path.exists():
            raise DocumentLoaderError(
                f"Documentation file does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise DocumentLoaderError(
                f"Documentation path is not a file: {file_path}"
            )

        try:
            content = await asyncio.to_thread(
                file_path.read_text,
                encoding="utf-8",
            )
        except OSError as exc:
            raise DocumentLoaderError(
                f"Unable to read documentation file: {file_path}"
            ) from exc

        if not content.strip():
            raise DocumentLoaderError(
                f"Documentation file is empty: {file_path}"
            )

        return Document(
            source=str(file_path),
            content=content,
        )