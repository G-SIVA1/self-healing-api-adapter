from __future__ import annotations

from self_healing_agent.agent.context import (
    ReasoningContext,
)


class RepairPromptBuilder:
    """Build structured prompts for the repair reasoning model."""

    def build(
        self,
        context: ReasoningContext,
    ) -> str:
        sections: list[str] = []

        sections.append(
            self._build_system_instruction()
        )
        sections.append(
            self._build_error_section(context)
        )
        sections.append(
            self._build_source_section(context)
        )
        sections.append(
            self._build_documentation_section(context)
        )
        sections.append(
            self._build_previous_attempt_section(context)
        )
        sections.append(
            self._build_failure_section(context)
        )
        sections.append(
            self._build_iteration_section(context)
        )
        sections.append(
            self._build_output_instruction()
        )

        return "\n\n".join(
            section
            for section in sections
            if section.strip()
        )

    @staticmethod
    def _build_system_instruction() -> str:
        return """\
You are an API compatibility repair agent.

Your task is to analyze a runtime API failure and
produce a minimal, safe source-code repair.

Rules:
1. Use the supplied documentation as the primary source
   for API compatibility decisions.
2. Preserve existing application behavior whenever possible.
3. Make the smallest possible code change.
4. Do not modify unrelated functionality.
5. Do not invent APIs that are not supported by the
   supplied documentation.
6. Consider previous failed repair attempts.
7. Use test feedback to improve the next repair.
8. Return a concrete source-code repair.
9. Your response MUST be a JSON object.
10. Do not use Markdown.
11. Do not use code fences.
12. Do not include any text before or after the JSON object.
"""

    @staticmethod
    def _build_error_section(
        context: ReasoningContext,
    ) -> str:
        lines = [
            "## Runtime Error",
            f"Exception: {context.exception_type}",
            f"Message: {context.error_message}",
        ]

        if context.api_service:
            lines.append(
                f"API Service: {context.api_service}"
            )

        if context.api_call:
            lines.append(
                f"API Call: {context.api_call}"
            )

        if context.target_file:
            lines.append(
                f"Target File: {context.target_file}"
            )

        if context.target_line is not None:
            lines.append(
                f"Target Line: {context.target_line}"
            )

        if context.function_name:
            lines.append(
                f"Function: {context.function_name}"
            )

        return "\n".join(lines)

    @staticmethod
    def _build_source_section(
        context: ReasoningContext,
    ) -> str:
        source = context.current_source_code

        if not source:
            source = "(No source code available.)"

        return (
            "## Current Source Code\n"
            "```python\n"
            f"{source}\n"
            "```"
        )

    @staticmethod
    def _build_documentation_section(
        context: ReasoningContext,
    ) -> str:
        documentation = context.documentation

        if not documentation:
            documentation = (
                "(No documentation was retrieved.)"
            )

        return (
            "## Retrieved Documentation\n"
            f"{documentation}"
        )

    @staticmethod
    def _build_previous_attempt_section(
        context: ReasoningContext,
    ) -> str:
        if not context.previous_generated_code:
            return (
                "## Previous Repair Attempt\n"
                "No previous repair attempt exists."
            )

        return (
            "## Previous Repair Attempt\n"
            "```python\n"
            f"{context.previous_generated_code}\n"
            "```"
        )

    @staticmethod
    def _build_failure_section(
        context: ReasoningContext,
    ) -> str:
        if context.failure_analysis is None:
            if not context.test_output:
                return (
                    "## Validation Feedback\n"
                    "No previous validation failure exists."
                )

            return (
                "## Validation Feedback\n"
                f"{context.test_output}"
            )

        analysis = context.failure_analysis

        return (
            "## Validation Feedback\n"
            f"Failure Type: {analysis.failure_type}\n"
            f"Summary: {analysis.summary}\n"
            f"Details: {analysis.details}\n"
            f"Retryable: {analysis.retryable}"
        )

    @staticmethod
    def _build_iteration_section(
        context: ReasoningContext,
    ) -> str:
        return (
            "## Repair Iteration\n"
            f"Current Iteration: {context.iteration}\n"
            f"Maximum Iterations: {context.max_iterations}"
        )

    @staticmethod
    def _build_output_instruction() -> str:
        return """\
## Required Output

Return EXACTLY one JSON object with EXACTLY these fields:

{
  "explanation": "A concise explanation of the API compatibility issue and repair.",
  "confidence": 0.95,
  "replacement_code": "The complete corrected source code."
}

Field requirements:

- "explanation" must be a non-empty string.
- "confidence" must be a number between 0.0 and 1.0.
- "replacement_code" must be a non-empty string containing
  the complete corrected source code.
- Do not add any other fields.
- Do not return Markdown.
- Do not return code fences.
- Do not return commentary outside the JSON object.
"""