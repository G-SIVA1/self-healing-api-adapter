from __future__ import annotations

from datetime import datetime, timezone

from self_healing_agent.models.events import ErrorEvent


def test_error_event_creation() -> None:
    event = ErrorEvent(
        timestamp=datetime.now(timezone.utc),
        exception_type="AttributeError",
        message="module 'stripe' has no attribute 'Customer'",
        file_path="api_client.py",
        line_number=42,
        function_name="create_customer",
        api_service="stripe",
        api_call="stripe.Customer.create",
        raw_log=(
            "AttributeError: "
            "module 'stripe' has no attribute 'Customer'"
        ),
    )

    assert event.exception_type == "AttributeError"
    assert event.api_service == "stripe"
    assert event.api_call == "stripe.Customer.create"
    assert event.file_path == "api_client.py"
    assert event.line_number == 42
    assert event.function_name == "create_customer"