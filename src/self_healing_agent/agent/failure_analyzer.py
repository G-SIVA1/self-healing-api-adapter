from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FailureAnalysis:
    """Structured analysis of a failed repair validation."""

    failure_type: str
    summary: str
    details: str
    retryable: bool


class FailureAnalyzer:
    """Analyze sandbox validation output."""

    _ASSERTION_PATTERN = re.compile(
        r"AssertionError(?::\s*(?P<message>.*))?",
        re.IGNORECASE,
    )

    _SYNTAX_PATTERN = re.compile(
        r"SyntaxError(?::\s*(?P<message>.*))?",
        re.IGNORECASE,
    )

    _IMPORT_PATTERN = re.compile(
        r"(?:ModuleNotFoundError|ImportError)"
        r"(?::\s*(?P<message>.*))?",
        re.IGNORECASE,
    )

    def analyze(
        self,
        output: str,
    ) -> FailureAnalysis:
        """
        Analyze test or sandbox output.

        The analyzer classifies common Python failure types
        and determines whether another repair attempt may
        reasonably be useful.
        """

        if not output.strip():
            return FailureAnalysis(
                failure_type="unknown",
                summary="Validation failed without diagnostic output.",
                details="No stdout or stderr was produced.",
                retryable=True,
            )

        assertion_match = self._ASSERTION_PATTERN.search(
            output
        )

        if assertion_match:
            message = (
                assertion_match.group("message")
                or "Test assertion failed."
            )

            return FailureAnalysis(
                failure_type="assertion_error",
                summary="A test assertion failed.",
                details=message.strip(),
                retryable=True,
            )

        syntax_match = self._SYNTAX_PATTERN.search(
            output
        )

        if syntax_match:
            message = (
                syntax_match.group("message")
                or "Python syntax is invalid."
            )

            return FailureAnalysis(
                failure_type="syntax_error",
                summary="The generated code contains invalid syntax.",
                details=message.strip(),
                retryable=True,
            )

        import_match = self._IMPORT_PATTERN.search(
            output
        )

        if import_match:
            message = (
                import_match.group("message")
                or "A required Python module could not be imported."
            )

            return FailureAnalysis(
                failure_type="import_error",
                summary="The repaired code has an import problem.",
                details=message.strip(),
                retryable=False,
            )

        if "FAILED" in output:
            return FailureAnalysis(
                failure_type="test_failure",
                summary="One or more validation tests failed.",
                details=output.strip(),
                retryable=True,
            )

        return FailureAnalysis(
            failure_type="unknown",
            summary="Validation failed for an unrecognized reason.",
            details=output.strip(),
            retryable=True,
        )