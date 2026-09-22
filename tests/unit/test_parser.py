from __future__ import annotations

from self_healing_agent.monitoring.parser import RuntimeErrorParser


def test_parser_extracts_runtime_error() -> None:
    raw_log = """\
2026-09-20 18:30:15 ERROR api_client.py:42
Traceback (most recent call last):
  File "api_client.py", line 42, in create_customer
    stripe.Customer.create(email=user.email)
AttributeError: module 'stripe' has no attribute 'Customer'
"""

    parser = RuntimeErrorParser()

    event = parser.parse(raw_log)

    assert event is not None

    assert event.exception_type == "AttributeError"
    assert event.message == "module 'stripe' has no attribute 'Customer'"

    assert event.file_path == "api_client.py"
    assert event.line_number == 42
    assert event.function_name == "create_customer"

    assert event.api_service == "stripe"
    assert event.api_call == "stripe.Customer.create"


def test_parser_returns_none_for_empty_log() -> None:
    parser = RuntimeErrorParser()

    event = parser.parse("")

    assert event is None