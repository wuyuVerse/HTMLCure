"""
Evidence computation for render_test phase.

Aggregates derived evidence fields from annotated frames and probe data.
Called once after all probes have run and inter-frame diffs are computed.

Two public functions:
  - compute_evidence(): pure data computation, sets data[...] fields
  - annotate_gameplay_smoothness(): mutates frame descriptions (side effect)
    Must be called before keyframe selection so descriptions are up to date.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from htmleval.phases.render_test.frame_types import AnnotatedFrame, FPSResult


# ── Single-pass frame classifier ─────────────────────────────────

@dataclass
class _FrameIndex:
    """Pre-classified frame index for O(1) lookups by layer/label."""
    observation: List[AnnotatedFrame] = field(default_factory=list)
    click: List[AnnotatedFrame] = field(default_factory=list)
    hover: List[AnnotatedFrame] = field(default_factory=list)
    keyboard: List[AnnotatedFrame] = field(default_factory=list)
    form: List[AnnotatedFrame] = field(default_factory=list)
    drag: List[AnnotatedFrame] = field(default_factory=list)
    responsive: List[AnnotatedFrame] = field(default_factory=list)
    deep: List[AnnotatedFrame] = field(default_factory=list)
    form_submitted: List[AnnotatedFrame] = field(default_factory=list)

    @classmethod
    def build(cls, all_frames: List[AnnotatedFrame]) -> _FrameIndex:
        idx = cls()
        for f in all_frames:
            # Layer-based (mutually exclusive)
            if f.layer == "observation":
                idx.observation.append(f)
            elif f.layer == "deep":
                idx.deep.append(f)
            elif f.layer == "responsive":
                idx.responsive.append(f)
            # Label-based — use `if` (not elif) because labels
            # can overlap (e.g. "form_submitted" is both form + submitted)
            if f.label.startswith("after_click"):
                idx.click.append(f)
            if f.label.startswith("hover"):
                idx.hover.append(f)
            if f.label.startswith("keyboard"):
                idx.keyboard.append(f)
            if f.label.startswith("form_"):
                idx.form.append(f)
            if f.label == "after_drag":
                idx.drag.append(f)
            if f.label == "form_submitted":
                idx.form_submitted.append(f)
        return idx


# ── Public API ────────────────────────────────────────────────────

def compute_evidence(
    all_frames: List[AnnotatedFrame],
    data: Dict[str, Any],
    probe_evidence: Dict[str, Any],
) -> None:
    """Compute all derived evidence fields from frames and probe data.

    Pure data computation — sets fields on *data* dict.
    Does NOT modify frame objects.

    Args:
        all_frames: All captured frames from every probe.
        data: The mutable data dict that flows to the evaluator.
        probe_evidence: Merged evidence dict from all ProbeResults
                        (e.g. fps_result, kb_data).
    """
    # ── Extract probe-specific evidence ───────────────────────────
    fps_result: Optional[FPSResult] = probe_evidence.get("fps_result")
    kb_data = probe_evidence.get("kb_data", {"keyboard_probed": False, "keys_responded": []})
    page_height = data.get("scroll_height")
    viewport_height = data.get("viewport_height")

    # ── Single-pass frame classification ──────────────────────────
    idx = _FrameIndex.build(all_frames)

    # ── Animation detection from observation layer ────────────────
    if len(idx.observation) >= 2:
        changes = sum(1 for f in idx.observation[1:] if f.diff_from_prev > 0.015)
        data["animation_detected"] = changes >= 2  # require sustained change, not single-frame flicker
        data["frame_change_rate"] = round(changes / max(len(idx.observation) - 1, 1), 2)
    else:
        data["animation_detected"] = False
        data["frame_change_rate"] = 0.0

    # ── FPS sampling data from Layer 1 ────────────────────────────
    if fps_result:
        data["animation_fps_estimate"] = fps_result.visual_change_count
        data["fps_quality"] = fps_result.quality

    # ── Scroll detection ──────────────────────────────────────────
    if page_height is not None and viewport_height is not None:
        data["has_below_fold"] = page_height > viewport_height * 1.2
    else:
        data["has_below_fold"] = False

    # ── Button responsiveness (from frame diffs) ──────────────────
    data["button_responsive"] = any(f.diff_from_prev > 0.01 for f in idx.click)
    data["hover_effects_detected"] = sum(
        1 for f in idx.hover if f.diff_from_prev > 0.015
    )

    # ── Keyboard responsiveness ───────────────────────────────────
    data["keyboard_visual_change"] = any(f.diff_from_prev > 0.015 for f in idx.keyboard)
    data["keyboard_responsive"] = (
        bool(kb_data.get("keys_responded"))
        or data["keyboard_visual_change"]
    )
    data["keyboard_probed"] = kb_data.get("keyboard_probed", False)
    data["keys_responded"] = kb_data.get("keys_responded", [])

    # ── Form probing results ──────────────────────────────────────
    data["form_probed"] = len(idx.form) > 0
    data["form_submitted"] = len(idx.form_submitted) > 0
    data["form_submit_changed"] = any(
        f.diff_from_prev > 0.01 for f in idx.form_submitted
    )

    # ── Drag probing results ──────────────────────────────────────
    data["drag_probed"] = len(idx.drag) > 0
    data["drag_responsive"] = any(f.diff_from_prev > 0.01 for f in idx.drag)

    # ── Responsive viewport count ─────────────────────────────────
    data["responsive_viewports_tested"] = len(idx.responsive)

    # ── Deep gameplay evidence ─────────────────────────────────────
    if "gameplay_state_changed" in probe_evidence:
        data["gameplay_state_changed"] = probe_evidence["gameplay_state_changed"]
        data["gameplay_vars_diff"] = probe_evidence.get("gameplay_vars_diff", {})
        data["gameplay_mode"] = probe_evidence.get("gameplay_mode", "unknown")

    # ── Game completion evidence (from deep gameplay probe) ──────────
    gc = probe_evidence.get("game_completion")
    if gc and isinstance(gc, dict):
        data["game_completion_detected"] = gc.get("completed", False)
        data["game_won"] = gc.get("won", False)
        data["game_completion_state"] = gc.get("state", "unknown")

    # ── Structural validation evidence (Layer 0) ───────────────────
    sv = probe_evidence.get("structural_validation")
    if sv and isinstance(sv, dict):
        data["structural_event_listener_count"] = sv.get("event_listener_count", 0)
        data["structural_raf_calls_2s"] = sv.get("raf_calls_2s", 0)
        data["structural_visible_overlays"] = sv.get("visible_overlays", [])
        data["structural_unbound_buttons"] = sv.get("unbound_buttons", 0)
        data["structural_total_buttons"] = sv.get("total_buttons", 0)


def annotate_gameplay_smoothness(deep_frames: List[AnnotatedFrame]) -> None:
    """Annotate deep gameplay frames with smoothness verdict.

    Mutates frame descriptions in-place. Must be called BEFORE keyframe
    selection so the VLM prompt receives the updated descriptions.

    Args:
        deep_frames: Only frames with layer=="deep" (pre-filtered by caller).
    """
    if not deep_frames:
        return
    change_count = sum(1 for f in deep_frames[1:] if f.diff_from_prev > 0.015)
    if change_count >= 4:
        deep_frames[-1].description += " — continuous visual change (smooth)"
    elif change_count >= 2:
        deep_frames[-1].description += " — some visual change (partial)"
    else:
        deep_frames[-1].description += " — little/no visual change (may be unresponsive)"


# Labels whose frames represent user actions where change detection matters
# gameplay_ is excluded — it gets dedicated annotation via annotate_gameplay_smoothness
_INTERACTION_PREFIXES = (
    "after_click", "hover", "keyboard", "form_", "canvas_click",
    "after_drag", "select_", "range_", "checkbox_", "link_click",
)

_SIGNIFICANT_CHANGE = 0.03   # diff above this = definite visual response
_MINOR_CHANGE = 0.015        # diff above this but below significant = subtle change


def annotate_interaction_outcomes(all_frames: List[AnnotatedFrame]) -> None:
    """Enrich interaction frame descriptions with change detection verdicts.

    After diff scores are computed, this adds human-readable observations like
    "— visual change detected (responsive)" or "— NO visual change (may be broken)"
    so the VLM doesn't have to interpret raw Δ numbers.

    Must be called AFTER _compute_frame_diffs and BEFORE keyframe selection.
    """
    for f in all_frames:
        if f.diff_from_prev == 0.0:
            continue  # first frame or no diff data
        if not any(f.label.startswith(p) for p in _INTERACTION_PREFIXES):
            continue  # observation/responsive frames don't need this

        if f.diff_from_prev >= _SIGNIFICANT_CHANGE:
            f.description += " — visual change detected (responsive)"
        elif f.diff_from_prev >= _MINOR_CHANGE:
            f.description += " — subtle visual change"
        else:
            f.description += " — NO visual change (may be broken)"
