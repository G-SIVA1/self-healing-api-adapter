from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from self_healing_agent.agent.architect import CodePatch
from self_healing_agent.agent.models import RepairReasoning


class RepairResponseError(Exception):
    """Raised when an LLM repair response is invalid."""


@dataclass(frozen=True, slots=True)
class RepairResponseParser:
    """Parse and validate structured LLM repair responses."""

    def parse(
        self,
        response: str,
        target_file: str,
        original_code: str,
    ) -> RepairReasoning:
        if not response.strip():
            raise RepairResponseError(
                "LLM response cannot be empty"
            )

        try:
            payload: Any = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RepairResponseError(
                "LLM response is not valid JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise RepairResponseError(
                "LLM response must be a JSON object"
            )

        explanation = self._require_string(
            payload,
            "explanation",
        )

        confidence = self._require_confidence(
            payload,
        )

        replacement_code = self._require_string(
            payload,
            "replacement_code",
        )

        if replacement_code == original_code:
            raise RepairResponseError(
                "LLM repair does not change the source code"
            )

        patch = CodePatch(
            file_path=target_file,
            original_code=original_code,
            replacement_code=replacement_code,
            explanation=explanation,
        )

        return RepairReasoning(
            explanation=explanation,
            confidence=confidence,
            patch=patch,
        )

    @staticmethod
    def _require_string(
        payload: dict[str, Any],
        field_name: str,
    ) -> str:
        value = payload.get(field_name)

        if not isinstance(value, str):
            raise RepairResponseError(
                f"'{field_name}' must be a string"
            )

        if not value.strip():
            raise RepairResponseError(
                f"'{field_name}' cannot be empty"
            )

        return value

    @staticmethod
    def _require_confidence(
        payload: dict[str, Any],
    ) -> float:
        value = payload.get("confidence")

        if isinstance(value, bool):
            raise RepairResponseError(
                "'confidence' must be a number"
            )

        if not isinstance(value, (int, float)):
            raise RepairResponseError(
                "'confidence' must be a number"
            )

        confidence = float(value)

        if not 0.0 <= confidence <= 1.0:
            raise RepairResponseError(
                "'confidence' must be between 0 and 1"
            )

        return confidence