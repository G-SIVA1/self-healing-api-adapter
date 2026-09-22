from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.monitoring.parser import RuntimeErrorParser


ErrorHandler = Callable[[ErrorEvent], Awaitable[None]]


class LogMonitorError(Exception):
    """Raised when the log monitor encounters an unrecoverable error."""


class AsyncLogMonitor:
    """
    Asynchronously monitors an application log file.

    New log content is parsed into ErrorEvent objects and passed
    to the configured asynchronous error handler.
    """

    def __init__(
        self,
        log_path: Path,
        parser: RuntimeErrorParser,
        error_handler: ErrorHandler,
        poll_interval: float = 0.5,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")

        self._log_path = log_path
        self._parser = parser
        self._error_handler = error_handler
        self._poll_interval = poll_interval

        self._running = False
        self._position = 0

    async def start(self) -> None:
        """Start monitoring the configured log file."""

        self._running = True

        try:
            await self._monitor_loop()
        except asyncio.CancelledError:
            self._running = False
            raise
        except Exception as exc:
            self._running = False
            raise LogMonitorError(
                f"Log monitoring failed for {self._log_path}"
            ) from exc

    async def stop(self) -> None:
        """Stop monitoring the log file."""

        self._running = False

    async def _monitor_loop(self) -> None:
        while self._running:
            if self._log_path.exists():
                await self._process_new_content()

            await asyncio.sleep(self._poll_interval)

    async def _process_new_content(self) -> None:
        """Read and process only newly appended log content."""

        try:
            content = await asyncio.to_thread(
                self._log_path.read_text,
                encoding="utf-8",
            )
        except OSError as exc:
            raise LogMonitorError(
                f"Unable to read log file: {self._log_path}"
            ) from exc

        if self._position > len(content):
            self._position = 0

        new_content = content[self._position :]
        self._position = len(content)

        if not new_content.strip():
            return

        log_blocks = self._split_log_blocks(new_content)

        for block in log_blocks:
            event = self._parser.parse(block)

            if event is None:
                continue

            await self._error_handler(event)

    @staticmethod
    def _split_log_blocks(content: str) -> list[str]:
        """Split log content into non-empty logical blocks."""

        return [
            block.strip()
            for block in content.split("\n\n")
            if block.strip()
        ]