"""FixPlayabilityStrategy — fix broken keyboard/input bindings (probe-confirmed)."""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import (
    FIX_PLAYABILITY, _SCORING_RUBRIC,
    format_prev_iterations, format_probe_evidence, format_preservation_list,
    format_output_instructions,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


class FixPlayabilityStrategy(RepairStrategy):
    name = "fix_playability"
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

        return FIX_PLAYABILITY.format(
            query=query,
            rendering=diag.rendering,
            visual_design=diag.visual_design,
            functionality=diag.functionality,
            interaction=diag.interaction,
            code_quality=diag.code_quality,
            score=diag.score,
            prev_iterations=prev + "\n" if prev else "",
            probe_evidence=probe_ev + "\n" if probe_ev else "",
            keyboard_visual_change="no",
            summary=diag.summary or "(no summary available)",
            preservation_list=preserve_str,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )
