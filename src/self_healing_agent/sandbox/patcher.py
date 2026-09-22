from __future__ import annotations

import asyncio
from pathlib import Path

from self_healing_agent.agent.architect import CodePatch


class PatchApplicationError(Exception):
    """Raised when a code patch cannot be safely applied."""


class SafePatchApplier:
    """
    Apply a CodePatch inside a controlled workspace.

    The original source project is never modified by this class.
    """

    async def apply(
        self,
        patch: CodePatch,
        workspace: Path,
    ) -> Path:
        """
        Apply a patch to the target file inside a workspace.

        Args:
            patch: Concrete code patch.
            workspace: Temporary repair workspace.

        Returns:
            Path to the patched file.

        Raises:
            PatchApplicationError:
                If the target file cannot be safely patched.
        """

        if not workspace.exists():
            raise PatchApplicationError(
                f"Workspace does not exist: {workspace}"
            )

        if not workspace.is_dir():
            raise PatchApplicationError(
                f"Workspace is not a directory: {workspace}"
            )

        if not patch.file_path.strip():
            raise PatchApplicationError(
                "Patch target file cannot be empty"
            )

        target_path = self._resolve_target(
            workspace,
            patch.file_path,
        )

        try:
            current_code = await asyncio.to_thread(
                target_path.read_text,
                encoding="utf-8",
            )
        except OSError as exc:
            raise PatchApplicationError(
                f"Unable to read target file: {target_path}"
            ) from exc

        if current_code != patch.original_code:
            raise PatchApplicationError(
                "Target file has changed since the patch was generated"
            )

        if current_code == patch.replacement_code:
            raise PatchApplicationError(
                "Patch would not change the source code"
            )

        try:
            await asyncio.to_thread(
                target_path.write_text,
                patch.replacement_code,
                encoding="utf-8",
            )
        except OSError as exc:
            raise PatchApplicationError(
                f"Unable to write patched file: {target_path}"
            ) from exc

        return target_path

    @staticmethod
    def _resolve_target(
        workspace: Path,
        file_path: str,
    ) -> Path:
        """
        Resolve a patch target and prevent path traversal.
        """

        workspace_resolved = workspace.resolve()

        target_path = (
            workspace_resolved / Path(file_path)
        ).resolve()

        try:
            target_path.relative_to(workspace_resolved)
        except ValueError as exc:
            raise PatchApplicationError(
                "Patch target is outside the workspace"
            ) from exc

        if not target_path.exists():
            raise PatchApplicationError(
                f"Patch target does not exist: {target_path}"
            )

        if not target_path.is_file():
            raise PatchApplicationError(
                f"Patch target is not a file: {target_path}"
            )

        return target_path