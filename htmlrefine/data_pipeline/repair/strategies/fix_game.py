"""FixGameStrategy — probe-driven, layer-specific game repair.

Diagnoses which game subsystem is broken (input, game_loop, canvas, overlay,
gameplay) based on structural probe data, then selects the matching single-focus
prompt. This avoids holistic_rewrite's oscillation problem for games stuck at 40-79.
"""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import (
    FIX_GAME_INPUT, FIX_GAME_LOOP, FIX_GAME_CANVAS, FIX_GAME_OVERLAY,
    FIX_GAME_GAMEPLAY,
    _SCORING_RUBRIC,
    format_prev_iterations, format_probe_evidence, format_preservation_list,
    format_output_instructions,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


def _diagnose_layer(diag: Diagnosis) -> str:
    """Determine which game subsystem is broken based on structural probe data.

    Returns one of: "input", "game_loop", "canvas", "overlay", "gameplay".
    Priority order matches typical failure frequency in game_html data.
    """
    # Overlay blocking viewport → fix overlay first (game may be fine underneath)
    if diag.structural_visible_overlays:
        return "overlay"

    # Canvas game with no rAF calls → game loop never started
    if diag.canvas_game and diag.structural_raf_calls_2s == 0:
        return "game_loop"

    # Keyboard confirmed broken → input wiring issue
    if diag.keyboard_broken:
        return "input"

    # Canvas game with low rendering → canvas probably empty
    if diag.canvas_game and diag.rendering < 14:
        return "canvas"

    # Fallback: gameplay layer — game renders and accepts input but has logic bugs
    # (collision detection, state machines, scoring, level transitions).
    # Data: 58 game_html stuck < 60 with functional rendering + input.
    return "gameplay"


# Map layer → prompt template
_LAYER_PROMPTS = {
    "input":     FIX_GAME_INPUT,
    "game_loop": FIX_GAME_LOOP,
    "canvas":    FIX_GAME_CANVAS,
    "overlay":   FIX_GAME_OVERLAY,
    "gameplay":  FIX_GAME_GAMEPLAY,
}


class FixGameStrategy(RepairStrategy):
    name = "fix_game"
    mode = "patch"

    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
    ) -> str:
        layer = _diagnose_layer(diag)
        template = _LAYER_PROMPTS[layer]

        probe_ev = format_probe_evidence(diag)
        preserve_str = format_preservation_list(diag)
        prev = format_prev_iterations(prev_iterations or [])

        # Base format kwargs shared by all layer prompts
        kwargs = dict(
            query=query,
            rendering=diag.rendering,
            visual_design=diag.visual_design,
            functionality=diag.functionality,
            interaction=diag.interaction,
            code_quality=diag.code_quality,
            score=diag.score,
            prev_iterations=prev + "\n" if prev else "",
            probe_evidence=probe_ev + "\n" if probe_ev else "",
            preservation_list=preserve_str,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )

        # Overlay prompt needs overlay_details
        if layer == "overlay":
            overlay_lines = []
            for ov in diag.structural_visible_overlays[:3]:
                overlay_lines.append(
                    f"- {ov.get('selector', '?')} covering {ov.get('coverage', '?')}% "
                    f"of viewport (z-index={ov.get('z_index', '?')}): "
                    f"'{ov.get('text_preview', '')[:60]}'"
                )
            kwargs["overlay_details"] = "\n".join(overlay_lines) if overlay_lines else "(detected by structural probe)"

        return template.format(**kwargs)
