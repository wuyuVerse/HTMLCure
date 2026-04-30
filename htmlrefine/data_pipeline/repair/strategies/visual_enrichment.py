"""VisualEnrichmentStrategy — VLM-driven visual enrichment for Tier A pages (score >= 80)."""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import (
    VISUAL_ENRICHMENT, _SCORING_RUBRIC,
    format_prev_iterations, format_preservation_list, format_output_instructions,
    format_observer_evidence, format_visual_diagnosis,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


class VisualEnrichmentStrategy(RepairStrategy):
    name = "visual_enrichment"
    mode = "rewrite"

    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
        visual_diagnosis_text: str = "",
    ) -> str:
        preserve_str = format_preservation_list(diag)
        prev = format_prev_iterations(prev_iterations or [])
        observer_ev = format_observer_evidence(diag)

        return VISUAL_ENRICHMENT.format(
            query=query,
            score=diag.score,
            visual_design=diag.visual_design,
            rendering=diag.rendering,
            functionality=diag.functionality,
            interaction=diag.interaction,
            code_quality=diag.code_quality,
            visual_diagnosis=visual_diagnosis_text,
            prev_iterations=prev + "\n" if prev else "",
            observer_evidence=observer_ev + "\n" if observer_ev else "",
            preservation_list=preserve_str,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )
