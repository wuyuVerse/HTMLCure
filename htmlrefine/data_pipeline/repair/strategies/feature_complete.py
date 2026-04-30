"""FeatureCompleteStrategy — add missing features to a partially working page."""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import (
    FEATURE_COMPLETE, _SCORING_RUBRIC,
    format_prev_iterations, format_probe_evidence, format_preservation_list,
    format_observer_evidence, format_requirement_checklist, format_output_instructions,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


class FeatureCompleteStrategy(RepairStrategy):
    name = "feature_complete"
    mode = "patch"

    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
    ) -> str:
        if diag.missing_features:
            missing = "\n".join(f"- {f}" for f in diag.missing_features)
        else:
            weak = [f"{name} ({cur}/{mx})" for name, cur, mx in diag.weak_dims[:3]]
            missing = f"- Improve weak areas: {', '.join(weak)}"

        probe_ev     = format_probe_evidence(diag)
        obs_ev       = format_observer_evidence(diag)
        req_cl       = format_requirement_checklist(diag)
        preserve_str = format_preservation_list(diag)
        prev         = format_prev_iterations(prev_iterations or [])

        # Game-specific constraints: don't add new features, only fix existing bugs.
        # Data: feature_complete on games avg Δ=-9.3 because LLM adds new game modes,
        # enemy types, levels which break existing logic.
        game_constraint = ""
        if diag.is_game:
            game_constraint = (
                "\n\n### Game-specific constraints\n"
                "- Do NOT add new game modes, new levels, new enemy types, or new power-ups\n"
                "- Only fix existing features that are broken: collision, state transitions, scoring\n"
                "- Do NOT modify physics parameters (gravity, velocity, friction) unless clearly wrong\n"
                "- Preserve ALL existing requestAnimationFrame loop structure\n"
                "- Do NOT restructure the game state machine — fix individual transitions"
            )

        prompt = FEATURE_COMPLETE.format(
            query=query,
            functionality=diag.functionality,
            interaction=diag.interaction,
            visual_design=diag.visual_design,
            rendering=diag.rendering,
            code_quality=diag.code_quality,
            prev_iterations=prev + "\n" if prev else "",
            probe_evidence=probe_ev + "\n" if probe_ev else "",
            observer_evidence=obs_ev + "\n" if obs_ev else "",
            requirement_checklist=req_cl + "\n" if req_cl else "",
            missing=missing,
            summary=diag.summary or "(no summary available)",
            preservation_list=preserve_str,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )
        return prompt + game_constraint
