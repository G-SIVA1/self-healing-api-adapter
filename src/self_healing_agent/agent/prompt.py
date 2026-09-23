from __future__ import annotations

from self_healing_agent.agent.context import ReasoningContext


class RepairPromptBuilder:
    """
    Build the structured prompt used by the LLM repair reasoner.
    """

    def build(
        self,
        context: ReasoningContext,
    ) -> str:
        source_code = (
            context.current_source_code
            if context.current_source_code is not None
            else "No current source code available."
        )

        previous_code = (
            context.previous_generated_code
            if context.previous_generated_code
            else "No previous repair attempt."
        )

        test_output = (
            context.test_output
            if context.test_output
            else "No validation output available."
        )

        failure_analysis = (
            self._format_failure_analysis(context)
        )

        function_name = (
            context.function_name
            if context.function_name
            else "Unknown"
        )

        api_service = (
            context.api_service
            if context.api_service
            else "Unknown"
        )

        api_call = (
            context.api_call
            if context.api_call
            else "Unknown"
        )

        target_line = (
            str(context.target_line)
            if context.target_line is not None
            else "Unknown"
        )

        return f"""
You are an API compatibility repair agent.

Your task is to analyze a runtime API failure and
produce a minimal, safe source-code repair.

## Rules

1. Use the supplied documentation as the primary source
   for API compatibility decisions.

2. Use the runtime error to identify the broken API call.

3. Preserve existing application behavior whenever possible.

4. Make the smallest possible code change.

5. Do not modify unrelated functionality.

6. Do not invent APIs that are not supported by the
   supplied documentation.

7. Consider previous failed repair attempts.

8. Use validation feedback to improve the next repair.

9. If the supplied documentation identifies a legacy API
   and a replacement API, the replacement MUST be applied
   to the source code.

10. The replacement_code MUST be different from the
    original source code whenever the documentation
    provides a valid repair.

11. Never return the original source code unchanged when
    a valid documented repair exists.

12. Return the complete corrected source file.

13. Your response MUST be a JSON object.

14. Do not use Markdown inside the JSON response.

15. Do not use code fences inside the JSON response.

16. Do not include any text before or after the JSON object.


## Runtime Error

Exception Type:
{context.exception_type}

Error Message:
{context.error_message}

Target File:
{context.target_file}

Target Line:
{target_line}

Function:
{function_name}

API Service:
{api_service}

API Call:
{api_call}


## Current Source Code

{source_code}


## Retrieved Documentation

{context.documentation}


## Previous Generated Code

{previous_code}


## Validation Feedback

Test Output:
{test_output}

Failure Analysis:
{failure_analysis}


## Iteration

Current Iteration: {context.iteration}

Maximum Iterations: {context.max_iterations}


## Required Output

Return exactly one JSON object:

{{
  "explanation": "Explain the concrete API compatibility repair.",
  "confidence": 0.95,
  "replacement_code": "complete corrected source code."
}}

The replacement_code MUST contain the complete corrected
source file and MUST apply the documented API migration.
""".strip()

    @staticmethod
    def _format_failure_analysis(
        context: ReasoningContext,
    ) -> str:
        analysis = context.failure_analysis

        if analysis is None:
            return "No previous failure analysis available."

        return (
            f"Failure Type: {analysis.failure_type}\n"
            f"Summary: {analysis.summary}\n"
            f"Details: {analysis.details}\n"
            f"Retryable: {analysis.retryable}"
        )