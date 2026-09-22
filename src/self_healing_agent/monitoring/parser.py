from __future__ import annotations

import re
from datetime import datetime, timezone

from self_healing_agent.models.events import ErrorEvent


class RuntimeErrorParser:
    """Parse Python runtime errors into structured ErrorEvent objects."""

    _TRACEBACK_PATTERN = re.compile(
        r'File "(?P<file>[^"]+)", '
        r"line (?P<line>\d+), "
        r"in (?P<function>[^\n]+)"
    )

    _EXCEPTION_PATTERN = re.compile(
        r"(?P<exception>[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))"
        r":\s*(?P<message>.+)"
    )

    _API_CALL_PATTERN = re.compile(
        r"\b(?P<service>stripe|openai|anthropic|github)"
        r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
    )

    def parse(self, raw_log: str) -> ErrorEvent | None:
        """Parse a raw Python traceback into an ErrorEvent."""

        if not raw_log.strip():
            return None

        exception_match = self._EXCEPTION_PATTERN.search(raw_log)

        if exception_match is None:
            return None

        traceback_match = self._TRACEBACK_PATTERN.search(raw_log)

        file_path: str | None = None
        line_number: int | None = None
        function_name: str | None = None

        if traceback_match is not None:
            file_path = traceback_match.group("file")
            line_number = int(traceback_match.group("line"))
            function_name = traceback_match.group("function").strip()

        api_service: str | None = None
        api_call: str | None = None

        for line in raw_log.splitlines():
            api_match = self._API_CALL_PATTERN.search(line)

            if api_match is not None:
                api_service = api_match.group("service")
                api_call = api_match.group(0)
                break

        return ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            exception_type=exception_match.group("exception"),
            message=exception_match.group("message").strip(),
            file_path=file_path,
            line_number=line_number,
            function_name=function_name,
            api_service=api_service,
            api_call=api_call,
            raw_log=raw_log,
        )