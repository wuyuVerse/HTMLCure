"""
Deterministic composite scoring for benchmark records with test_cases.

When test_cases exist: deterministic signals drive most dimensions,
VLM only controls visual_design.

When test_cases are absent: this module is never called — VLM scores stand as-is.
"""

from __future__ import annotations

from typing import Any, Dict

SCORE_VERSION = "drop_cov_tc_transfer_rend_linear_6000tc_v3_20260430"

INTERACTIVE_WEIGHTS = {
    "rendering": 10,
    "visual_design": 20,
    "functionality": 55,
    "interaction": 10,
    "code_quality": 5,
}

NON_INTERACTIVE_WEIGHTS = {
    "rendering": 10,
    "visual_design": 20,
    "functionality": 65,
    "interaction": 0,
    "code_quality": 5,
}


def _functionality_score(pass_rate: float, max_points: int) -> int:
    """Convert weighted TC pass rate into direct TC points."""
    t = max(0.0, min(1.0, float(pass_rate or 0.0)))
    return max(0, min(max_points, round(t * max_points)))


def _scale_score(value: float, old_max: int, new_max: int) -> int:
    return max(0, min(new_max, round(float(value or 0.0) / old_max * new_max)))


def composite_score(
    test_runner_data: Dict[str, Any],
    static_data: Dict[str, Any],
    render_data: Dict[str, Any],
    vlm_scores: Dict[str, Any],
    has_interaction: bool = True,
) -> Dict[str, Any]:
    """Produce composite scores: deterministic + VLM visual_design.

    Args:
        test_runner_data: PhaseResult.data from TestRunnerPhase
        static_data:      PhaseResult.data from StaticAnalysisPhase
        render_data:      PhaseResult.data from RenderTestPhase
        vlm_scores:       raw eval result from VisionEvalPhase (scorer agent output)
        has_interaction:   whether the prompt requires user interaction

    Returns:
        Full score dict compatible with existing final_score format.
    """
    # ── rendering (0-10): deterministic ──────────────────────────
    rendering = 10
    # Static-only interactive apps usually cannot satisfy requested behavior.
    # Non-interactive CSS-only prompts remain eligible for full rendering.
    has_script = static_data.get("has_script", False)
    has_canvas = static_data.get("has_canvas", False)
    has_svg = static_data.get("has_svg", False)
    if has_interaction and not has_script and not has_canvas and not has_svg:
        rendering = min(rendering, 4)
    console_entries = render_data.get("console_errors", [])
    console_errs = [
        e for e in console_entries
        if isinstance(e, dict) and e.get("type") == "error"
    ]
    console_warns = [
        e for e in console_entries
        if isinstance(e, dict) and e.get("type") == "warning"
    ]
    rendering -= len(console_errs) * 3
    rendering -= min(len(console_warns), 5)
    rendering -= render_data.get("page_err_count", 0) * 5
    if len(console_errs) + len(console_warns) > 0:
        rendering -= 1
    if not render_data.get("rendered", True):
        rendering = 0
    fps_q = render_data.get("fps_quality", "not measured")
    if fps_q == "frozen":
        rendering -= 5
    elif fps_q == "choppy":
        rendering -= 4
    if not static_data.get("has_lang_attr", False):
        rendering -= 1
    unbound = render_data.get("structural_unbound_buttons", 0)
    total_btns = render_data.get("structural_total_buttons", 0)
    if total_btns > 0 and unbound / total_btns > 0.3:
        rendering -= 2
    if render_data.get("structural_overlays", 0) > 0:
        rendering -= 2
    rendering = max(0, rendering)

    weights = INTERACTIVE_WEIGHTS if has_interaction else NON_INTERACTIVE_WEIGHTS

    # ── weighted rendering contribution ───────────────────────────
    rendering = _scale_score(rendering, 10, weights["rendering"])

    # ── functionality: direct score-bearing TC pass rate ──────────
    pass_rate = test_runner_data.get("test_pass_rate", 0.0)
    if test_runner_data.get("tests_total", 0) > 0:
        functionality = _functionality_score(pass_rate, weights["functionality"])
    else:
        vlm_func = vlm_scores.get("functionality", {})
        raw_func = vlm_func.get("score", 0) if isinstance(vlm_func, dict) else 0
        functionality = _scale_score(raw_func, 25, weights["functionality"])

    # ── interaction: conditional on has_interaction ────────────────
    if has_interaction:
        interaction = _scale_score(_probe_interaction_score(render_data), 20, weights["interaction"])
        if (
            pass_rate >= 0.995
            and interaction > 0
            and not render_data.get("interactions_timed_out", 0)
        ):
            interaction = weights["interaction"]
        else:
            interaction = max(0, min(weights["interaction"], interaction))
    else:
        interaction = None  # skipped — not counted in total

    # ── visual_design (0-25): VLM score + guardrails ───────────────
    vd = vlm_scores.get("visual_design", {})
    visual_design = vd.get("score", 8) if isinstance(vd, dict) else 8  # fast-mode default; full-mode uses VLM score (up to 25)
    visual_design = min(visual_design, 25)
    if render_data.get("responsive_has_horizontal_overflow"):
        visual_design -= 4
    if not static_data.get("has_meta_viewport", True):
        visual_design -= 3
    if render_data.get("responsive_mobile_broken"):
        visual_design -= 3
    if not static_data.get("has_style", True) and not static_data.get("has_css_transitions", False):
        visual_design -= 3
    visual_design = max(0, visual_design)
    visual_design = _scale_score(visual_design, 25, weights["visual_design"])

    # ── code_quality (0-15): deterministic ────────────────────────
    code_quality = _scale_score(_static_code_quality_score(static_data), 15, weights["code_quality"])

    # ── direct 100-point contribution sum ─────────────────────────
    raw_sum = rendering + visual_design + functionality + code_quality
    if has_interaction:
        raw_sum += interaction
    max_possible = 100
    total = round(raw_sum)
    total = max(0, min(100, total))

    result = {
        "rendering": {
            "score": rendering,
            "reason": f"deterministic: {len(console_errs)} errors, {len(console_warns)} warnings, fps={fps_q}",
        },
        "visual_design": {"score": visual_design, "reason": vd.get("reason", "") if isinstance(vd, dict) else ""},
        "functionality": {
            "score": functionality,
            "reason": f"test_pass_rate={pass_rate:.0%}",
        },
        "interaction": {
            "score": interaction if interaction is not None else 0,
            "reason": "graduated probe scoring" if has_interaction else "skipped (non-interactive prompt)",
        },
        "code_quality": {
            "score": code_quality,
            "reason": "static analysis",
        },
        "total_score": total,
        "has_interaction": has_interaction,
        "max_possible": max_possible,
        "raw_sum": raw_sum,
        "bugs": vlm_scores.get("bugs", []),
        "missing_features": vlm_scores.get("missing_features", []),
        "highlights": vlm_scores.get("highlights", []),
        "improvement_hints": vlm_scores.get("improvement_hints", []),
        "summary": vlm_scores.get("summary", ""),
        "observer_report": vlm_scores.get("observer_report"),
        "task_auditor_report": vlm_scores.get("task_auditor_report"),
    }
    return result


def _probe_interaction_score(render_data: Dict[str, Any]) -> int:
    """Compute interaction contribution (0-20) from render_test probe data.

    Uses graduated scoring across all available probe signals.
    Stricter thresholds for higher discrimination.
    """
    score = 0

    # Button responsiveness: graduated by response rate (0-5)
    btn_rate = render_data.get("button_response_rate", 0)
    if btn_rate > 0:
        score += round(btn_rate * 5)
    elif render_data.get("button_responsive"):
        score += 2

    # Keyboard responsiveness: graduated by keys responded (0-4)
    kb_keys = len(render_data.get("keys_responded", []))
    if kb_keys >= 5:
        score += 4
    elif kb_keys >= 3:
        score += 3
    elif kb_keys >= 1:
        score += 1
    elif render_data.get("keyboard_responsive"):
        score += 1

    # Hover effects (0-2)
    score += 2 if render_data.get("hover_effects_detected") else 0

    # Drag responsiveness (0-2)
    score += 2 if render_data.get("drag_responsive") else 0

    # Form submit (0-2)
    score += 2 if render_data.get("form_submit_changed") else 0

    # Animation detected (0 — removed, not discriminating)
    # Most pages have some animation, gives no signal

    # Gameplay state changed (0-3)
    if render_data.get("gameplay_state_changed"):
        score += 3

    # Below-fold scrollable content (0-1)
    if render_data.get("has_below_fold"):
        score += 1

    # Interaction latency bonus (0-2, stricter thresholds)
    avg_latency = render_data.get("avg_latency_ms", None)
    if avg_latency is not None:
        if avg_latency < 50:
            score += 2
        elif avg_latency < 150:
            score += 1

    timed_out = int(render_data.get("interactions_timed_out", 0) or 0)
    if timed_out:
        score -= min(4, timed_out)

    return max(0, min(20, score))


def _static_code_quality_score(static_data: Dict[str, Any]) -> int:
    """Compute code quality score (0-15) from static analysis data.

    Penalizes heavy CDN usage, missing semantic tags, excessive inline styles,
    missing title, missing meta viewport, and lack of modern CSS practices.
    """
    score = 15

    # CDN dependency penalty (restored — CDN usage is acceptable)
    ext_count = static_data.get("ext_count", 0)
    if ext_count >= 8:
        score -= 3
    elif ext_count >= 5:
        score -= 1

    # Missing DOCTYPE penalty
    if not static_data.get("has_doctype", True):
        score -= 2

    # Missing <title> penalty
    if not static_data.get("has_title", True):
        score -= 2                              # stricter: was -1

    # Missing meta viewport penalty
    if not static_data.get("has_meta_viewport", True):
        score -= 2

    # Semantic HTML penalty (stricter: require 4+ tags)
    semantic_count = sum(
        1 for t in ("header", "main", "section", "nav", "footer", "article", "aside")
        if static_data.get(f"has_{t}", False)
    )
    if semantic_count == 0:
        score -= 4                              # stricter: was -3
    elif semantic_count < 2:
        score -= 3
    elif semantic_count < 4:
        score -= 1                              # new: penalize <4 semantic tags

    # Excessive inline styles penalty (stricter)
    inline_count = static_data.get("inline_style_count", 0)
    if inline_count > 30:
        score -= 3                              # stricter: was -2 at >50
    elif inline_count > 15:
        score -= 1                              # stricter: was -1 at >20

    # No responsive media queries penalty (discriminates — kimi uses fewer)
    # Removed: has_media_queries check (unfair — kimi generates good HTML without @media)

    # No CSS transitions check (removed — kimi uses JS animations, qwen uses CSS)
    # These were unfairly penalizing kimi

    return max(0, score)
