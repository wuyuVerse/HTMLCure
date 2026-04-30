"""VisualEnhanceStrategy — improve visual design while preserving functionality."""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import VISUAL_ENHANCE, _SCORING_RUBRIC, format_prev_iterations, format_output_instructions
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


class VisualEnhanceStrategy(RepairStrategy):
    name = "visual_enhance"
    mode = "patch"

    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
    ) -> str:
        issues: list[str] = []
        if diag.visual_design < 10:
            issues.append("Very plain or unstyled appearance")
        if diag.rendering < 15:
            issues.append("Rendering issues affecting visual quality")
        for bug in diag.bugs:
            if any(kw in bug.lower() for kw in ("visual", "style", "css", "color", "layout", "design")):
                issues.append(bug)
        if not issues:
            issues.append(
                f"Visual design score is low ({diag.visual_design}/20); "
                "improve polish, consistency, and presentation"
            )
        highlights = "\n".join(f"- {h}" for h in diag.highlights) or "- (none noted)"
        prev = format_prev_iterations(prev_iterations or [])
        return VISUAL_ENHANCE.format(
            query=query,
            visual_design=diag.visual_design,
            rendering=diag.rendering,
            functionality=diag.functionality,
            prev_iterations=prev + "\n" if prev else "",
            visual_issues="\n".join(f"- {i}" for i in issues),
            summary=diag.summary or "(no summary available)",
            highlights=highlights,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )
