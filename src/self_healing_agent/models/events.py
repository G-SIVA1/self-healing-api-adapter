from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """
    Represents a runtime failure detected by the log monitor.
    """

    timestamp: datetime
    exception_type: str
    message: str

    file_path: Optional[str]
    line_number: Optional[int]
    function_name: Optional[str]

    api_service: Optional[str]
    api_call: Optional[str]

    raw_log: str