from __future__ import annotations

from self_healing_agent.agent.failure_analyzer import (
    FailureAnalyzer,
)


def test_analyzer_detects_assertion_error() -> None:
    analyzer = FailureAnalyzer()

    output = """
FAILED test_api_client.py
AssertionError: expected NEW_API but got WRONG_API
"""

    result = analyzer.analyze(output)

    assert result.failure_type == "assertion_error"
    assert result.summary == "A test assertion failed."
    assert result.details == (
        "expected NEW_API but got WRONG_API"
    )
    assert result.retryable is True


def test_analyzer_detects_syntax_error() -> None:
    analyzer = FailureAnalyzer()

    output = """
SyntaxError: invalid syntax
"""

    result = analyzer.analyze(output)

    assert result.failure_type == "syntax_error"
    assert result.retryable is True


def test_analyzer_detects_import_error() -> None:
    analyzer = FailureAnalyzer()

    output = """
ModuleNotFoundError: No module named 'stripe'
"""

    result = analyzer.analyze(output)

    assert result.failure_type == "import_error"
    assert result.retryable is False


def test_analyzer_handles_empty_output() -> None:
    analyzer = FailureAnalyzer()

    result = analyzer.analyze("")

    assert result.failure_type == "unknown"
    assert result.retryable is True


def test_analyzer_handles_unknown_failure() -> None:
    analyzer = FailureAnalyzer()

    output = """
Process terminated unexpectedly.
"""

    result = analyzer.analyze(output)

    assert result.failure_type == "unknown"
    assert result.retryable is True