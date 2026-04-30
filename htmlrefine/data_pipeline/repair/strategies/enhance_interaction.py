"""EnhanceInteractionStrategy — UX polish for Tier A pages."""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import (
    ENHANCE_INTERACTION, _SCORING_RUBRIC,
    format_prev_iterations, format_preservation_list, format_output_instructions,
    format_observer_evidence, format_probe_evidence,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


class EnhanceInteractionStrategy(RepairStrategy):
    name = "enhance_interaction"
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
        observer_ev = format_observer_evidence(diag)
        probe_ev = format_probe_evidence(diag)

        return ENHANCE_INTERACTION.format(
            query=query,
            score=diag.score,
            interaction=diag.interaction,
            functionality=diag.functionality,
            visual_design=diag.visual_design,
            rendering=diag.rendering,
            code_quality=diag.code_quality,
            prev_iterations=prev + "\n" if prev else "",
            observer_evidence=observer_ev + "\n" if observer_ev else "",
            probe_evidence=probe_ev + "\n" if probe_ev else "",
            summary=diag.summary or "(no summary available)",
            preservation_list=preserve_str,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )
