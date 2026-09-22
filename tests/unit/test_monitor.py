from __future__ import annotations

import asyncio
from pathlib import Path

from self_healing_agent.models.events import ErrorEvent
from self_healing_agent.monitoring.monitor import AsyncLogMonitor
from self_healing_agent.monitoring.parser import RuntimeErrorParser


def test_monitor_processes_new_log_content(tmp_path: Path) -> None:
    log_file = tmp_path / "application.log"

    log_file.write_text(
        """\
2026-09-20 18:30:15 ERROR api_client.py:42
Traceback (most recent call last):
  File "api_client.py", line 42, in create_customer
    stripe.Customer.create(email=user.email)
AttributeError: module 'stripe' has no attribute 'Customer'

""",
        encoding="utf-8",
    )

    received_events: list[ErrorEvent] = []

    async def handle_error(event: ErrorEvent) -> None:
        received_events.append(event)

    monitor = AsyncLogMonitor(
        log_path=log_file,
        parser=RuntimeErrorParser(),
        error_handler=handle_error,
    )

    asyncio.run(monitor._process_new_content())

    assert len(received_events) == 1

    event = received_events[0]

    assert event.exception_type == "AttributeError"
    assert event.api_service == "stripe"
    assert event.api_call == "stripe.Customer.create"
    assert event.file_path == "api_client.py"
    assert event.line_number == 42


def test_monitor_ignores_non_error_content(tmp_path: Path) -> None:
    log_file = tmp_path / "application.log"

    log_file.write_text(
        "Application started successfully.\n",
        encoding="utf-8",
    )

    received_events: list[ErrorEvent] = []

    async def handle_error(event: ErrorEvent) -> None:
        received_events.append(event)

    monitor = AsyncLogMonitor(
        log_path=log_file,
        parser=RuntimeErrorParser(),
        error_handler=handle_error,
    )

    asyncio.run(monitor._process_new_content())

    assert received_events == []