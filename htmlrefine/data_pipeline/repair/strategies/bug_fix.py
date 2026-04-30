"""BugFixStrategy — surgical fix for confirmed (objective) bugs only."""

from __future__ import annotations

from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.core.evidence import KEYBOARD_FIX_HINT
from htmlrefine.data_pipeline.repair.prompts import (
    BUG_FIX, _SCORING_RUBRIC,
    format_prev_iterations, format_probe_evidence, format_preservation_list,
    format_observer_evidence, format_output_instructions,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


class BugFixStrategy(RepairStrategy):
    name = "bug_fix"
    mode = "patch"

    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
    ) -> str:
        # Use reliable_issues (objectively confirmed) NOT vl_bugs (inferred)
        if diag.reliable_issues:
            reliable_str = "\n".join(f"- {b}" for b in diag.reliable_issues)
        elif diag.bugs:
            # Fallback: use VLM bugs with disclaimer
            reliable_str = (
                "Note: these are VLM-inferred (not directly observed). Fix cautiously:\n"
                + "\n".join(f"- {b}" for b in diag.bugs[:5])
            )
        else:
            reliable_str = "- No specific bugs listed; fix all console errors."

        console = "\n".join(f"- {e}" for e in diag.console_errors) or "- None"

        # Keyboard fix hint (only when confirmed broken by probe)
        keyboard_hint = ""
        if diag.keyboard_broken:
            keyboard_hint = "## Keyboard Fix (probe-confirmed broken)\n" + KEYBOARD_FIX_HINT + "\n"

        probe_ev     = format_probe_evidence(diag)
        obs_ev       = format_observer_evidence(diag)
        preserve_str = format_preservation_list(diag)
        prev         = format_prev_iterations(prev_iterations or [])

        return BUG_FIX.format(
            query=query,
            rendering=diag.rendering,
            functionality=diag.functionality,
            interaction=diag.interaction,
            code_quality=diag.code_quality,
            prev_iterations=prev + "\n" if prev else "",
            probe_evidence=probe_ev + "\n" if probe_ev else "",
            observer_evidence=obs_ev + "\n" if obs_ev else "",
            reliable_issues=reliable_str,
            console_errors=console,
            keyboard_hint=keyboard_hint,
            summary=diag.summary or "(no summary available)",
            preservation_list=preserve_str,
            rubric=_SCORING_RUBRIC,
            html=html,
            output_instructions=format_output_instructions(self.mode),
        )
