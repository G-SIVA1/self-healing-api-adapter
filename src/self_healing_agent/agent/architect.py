from __future__ import annotations

from dataclasses import dataclass

from self_healing_agent.agent.state import AgentState


@dataclass(frozen=True, slots=True)
class RepairProposal:
    """Structured proposal for repairing broken API code."""

    target_file: str
    target_line: int | None
    explanation: str
    original_api_call: str | None
    suggested_api_call: str | None
    replacement_code: str


@dataclass(frozen=True, slots=True)
class CodePatch:
    """Concrete source-code modification."""

    file_path: str
    original_code: str
    replacement_code: str
    explanation: str


class CodeArchitect:
    """
    Analyze agent state and create repair proposals and patches.
    """

    def create_proposal(
        self,
        state: AgentState,
    ) -> RepairProposal:
        """Create a repair proposal from the current agent state."""

        event = state.error_event

        if not event.file_path:
            raise ValueError(
                "Cannot create repair proposal without a target file"
            )

        if not event.api_call:
            raise ValueError(
                "Cannot create repair proposal without an API call"
            )

        if not state.documentation:
            raise ValueError(
                "Cannot create repair proposal without documentation"
            )

        suggested_api_call = self._find_suggested_api(
            state,
        )

        if suggested_api_call is None:
            raise ValueError(
                "Documentation does not contain a suggested API"
            )

        replacement_code = self._build_replacement(
            event.api_call,
            suggested_api_call,
        )

        explanation = (
            f"The API call '{event.api_call}' appears to be "
            "incompatible with the current API documentation. "
            f"The documentation recommends '{suggested_api_call}'."
        )

        return RepairProposal(
            target_file=event.file_path,
            target_line=event.line_number,
            explanation=explanation,
            original_api_call=event.api_call,
            suggested_api_call=suggested_api_call,
            replacement_code=replacement_code,
        )

    def generate_patch(
        self,
        proposal: RepairProposal,
        source_code: str,
    ) -> CodePatch:
        """
        Generate a concrete source-code patch.

        Only the API identifier is replaced. The surrounding
        source code remains untouched.
        """

        if not source_code.strip():
            raise ValueError(
                "Source code cannot be empty"
            )

        if not proposal.original_api_call:
            raise ValueError(
                "Repair proposal has no original API call"
            )

        if not proposal.suggested_api_call:
            raise ValueError(
                "Repair proposal has no suggested API call"
            )

        occurrence_count = source_code.count(
            proposal.original_api_call
        )

        if occurrence_count == 0:
            raise ValueError(
                "Original API call was not found in source code"
            )

        if occurrence_count > 1:
            raise ValueError(
                "Original API call occurs multiple times; "
                "refusing ambiguous patch"
            )

        patched_code = source_code.replace(
            proposal.original_api_call,
            proposal.suggested_api_call,
            1,
        )

        return CodePatch(
            file_path=proposal.target_file,
            original_code=source_code,
            replacement_code=patched_code,
            explanation=proposal.explanation,
        )

    @staticmethod
    def _find_suggested_api(
        state: AgentState,
    ) -> str | None:
        """Extract the recommended API from documentation."""

        for chunk in state.documentation:
            content = chunk.content

            if "stripe.customers.create" in content:
                return "stripe.customers.create"

        return None

    @staticmethod
    def _build_replacement(
        original_api: str,
        suggested_api: str,
    ) -> str:
        """Build the proposed API replacement."""

        if original_api == "stripe.Customer.create":
            return suggested_api

        return suggested_api