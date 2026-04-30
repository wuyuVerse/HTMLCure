"""CodeCleanupStrategy — improve code quality without changing behavior for Tier A pages."""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import (
    CODE_CLEANUP, _SCORING_RUBRIC,
    format_prev_iterations, format_preservation_list, format_output_instructions,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


class CodeCleanupStrategy(RepairStrategy):
    name = "code_cleanup"
    mode = "patch"

    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
    ) -> str:
        preserve_str = format_preservation_list(diag)
        prev = format_prev_iterations(prev_iterations or [])

        return CODE_CLEANUP.format(
            query=query,
            score=diag.score,
            code_quality=diag.code_quality,
            rendering=diag.rendering,
            visual_design=diag.visual_design,
            functionality=diag.functionality,
            interaction=diag.interaction,
            prev_iterations=prev + "\n" if prev else "",
            summary=diag.summary or "(no summary available)",
            preservation_list=preserve_str,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )
