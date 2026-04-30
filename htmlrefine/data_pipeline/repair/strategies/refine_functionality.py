"""RefineFunctionalityStrategy — complete missing features / edge cases for Tier A pages."""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import (
    REFINE_FUNCTIONALITY, _SCORING_RUBRIC,
    format_prev_iterations, format_preservation_list,
    format_requirement_checklist, format_output_instructions,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


class RefineFunctionalityStrategy(RepairStrategy):
    name = "refine_functionality"
    mode = "patch"

    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
    ) -> str:
        # Build missing/weak areas list
        if diag.missing_features:
            missing = "\n".join(f"- {f}" for f in diag.missing_features)
        else:
            weak = [f"{name} ({cur}/{mx})" for name, cur, mx in diag.weak_dims[:3]]
            missing = f"- Improve weak areas: {', '.join(weak)}"

        req_cl       = format_requirement_checklist(diag)
        preserve_str = format_preservation_list(diag)
        prev         = format_prev_iterations(prev_iterations or [])

        return REFINE_FUNCTIONALITY.format(
            query=query,
            score=diag.score,
            functionality=diag.functionality,
            interaction=diag.interaction,
            visual_design=diag.visual_design,
            rendering=diag.rendering,
            code_quality=diag.code_quality,
            prev_iterations=prev + "\n" if prev else "",
            summary=diag.summary or "(no summary available)",
            requirement_checklist=req_cl + "\n" if req_cl else "",
            missing=missing,
            preservation_list=preserve_str,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )
