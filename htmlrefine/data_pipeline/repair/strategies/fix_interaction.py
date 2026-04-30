"""FixInteractionStrategy — fix broken general interaction (not just keyboard)."""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import (
    FIX_INTERACTION, _SCORING_RUBRIC,
    format_prev_iterations, format_probe_evidence, format_preservation_list,
    format_output_instructions,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


class FixInteractionStrategy(RepairStrategy):
    name = "fix_interaction"
    mode = "patch"

    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
    ) -> str:
        probe_ev     = format_probe_evidence(diag)
        preserve_str = format_preservation_list(diag)
        prev         = format_prev_iterations(prev_iterations or [])

        # Build interaction issues from reliable_issues, VLM bugs, and console errors
        issues = []
        if diag.reliable_issues:
            issues.extend(diag.reliable_issues[:3])
        if diag.bugs:
            remaining = [b for b in diag.bugs if b not in issues]
            issues.extend(remaining[:3])
        if not issues:
            issues.append(
                f"Interaction score is {diag.interaction}/25 — elements are "
                "unresponsive or event handling is broken"
            )

        return FIX_INTERACTION.format(
            query=query,
            rendering=diag.rendering,
            visual_design=diag.visual_design,
            functionality=diag.functionality,
            interaction=diag.interaction,
            code_quality=diag.code_quality,
            score=diag.score,
            prev_iterations=prev + "\n" if prev else "",
            probe_evidence=probe_ev + "\n" if probe_ev else "",
            interaction_issues="\n".join(f"- {i}" for i in issues),
            summary=diag.summary or "(no summary available)",
            preservation_list=preserve_str,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )
