"""
Multi-agent evaluation pipeline: EvalAgent ABC + concrete agents.

Architecture:
    AgentContext flows through a list of EvalAgent instances.
    Each agent builds multimodal content, calls the VLM, and parses the response.
    Results accumulate in ctx.stage_outputs[agent.name].

Pipeline:  Observer (VLM+vision) → TaskAuditor (text-only) → Scorer (text-only)

To add a new agent:
    1. Subclass EvalAgent, implement build_content() and parse_response().
    2. Append an instance to EVAL_AGENTS at the desired position.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from htmleval.phases.vision_eval.llm import (
    encode_image_b64,
    format_dom_inventory,
)
from htmleval.phases.vision_eval.prompts import (
    ANALYST_PROMPT,
    OBSERVER_PROMPT,
    SCORER_PROMPT,
    TASK_AUDITOR_PROMPT,
)

logger = logging.getLogger("htmleval")

SCORE_DIMENSIONS = ("rendering", "visual_design", "functionality", "interaction", "code_quality")

_NON_SCORING_CONSTRAINT_PATTERNS = (
    r"\bno external resources?\b",
    r"\bexternal resources?\b",
    r"\bexternal dependenc(?:y|ies)\b",
    r"\bcdn\b",
    r"\bsingle[- ]html\b",
    r"\bsingle[- ]file\b",
    r"\bself-contained\b",
    r"\braw html only\b",
    r"\bplain html\b",
    r"\bhtml only\b",
)
_BENIGN_CONSOLE_SUBSTRINGS = (
    "notallowederror",
    "failed to execute 'writetext' on 'clipboard'",
    "failed to execute 'readtext' on 'clipboard'",
    "navigator.clipboard",
    "failed to load resource",
    "net::err_",
    "blocked by cors policy",
    "cross-origin request blocked",
    "access to fetch at",
)


# ---------------------------------------------------------------------------
# Shared context flowing through the agent pipeline
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Accumulated context flowing through the agent pipeline."""

    query: str
    static: Dict[str, Any]
    render: Dict[str, Any]
    agent: Dict[str, Any]
    agent_ran: bool
    screenshots: List[str]
    max_screenshots: int
    stage_outputs: Dict[str, Any] = field(default_factory=dict)

    # Pre-computed from static/render — set by evaluator before pipeline runs
    input_types: List[str] = field(default_factory=list)
    kb_probed: bool = False
    kb_responded: List[str] = field(default_factory=list)
    kb_vis_change: bool = False
    kb_responsive: bool = False
    frame_annotations: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent ABC
# ---------------------------------------------------------------------------

class EvalAgent(ABC):
    """Base class for evaluation agents. Subclass to add a new pipeline stage."""

    name: str
    max_tokens: int = 65536  # Must be large for reasoning models (think+output share this budget)

    @abstractmethod
    def build_content(self, ctx: AgentContext) -> List[Dict[str, Any]]:
        """Build multimodal content (text + images) for the LLM call."""

    @abstractmethod
    def parse_response(self, raw_json: Dict[str, Any], ctx: AgentContext) -> Dict[str, Any]:
        """Validate / post-process parsed LLM JSON. Return cleaned result."""


# ---------------------------------------------------------------------------
# Helpers shared across agents
# ---------------------------------------------------------------------------

def _format_frame_annotations(frame_annots: List[Dict[str, Any]]) -> str:
    if not frame_annots:
        return "  (no annotated frames available)"
    lines = []
    for i, fa in enumerate(frame_annots):
        lines.append(
            f"  Frame {i+1}/{len(frame_annots)} [{fa.get('label','')}] "
            f"(t={fa.get('timestamp',0):.1f}s, Δ={fa.get('diff_from_prev',0):.3f}): "
            f"{fa.get('description','')}"
        )
    return "\n".join(lines)


def _is_benign_console_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    text = str(entry.get("text", "") or "").lower()
    return any(token in text for token in _BENIGN_CONSOLE_SUBSTRINGS)


def _common_prompt_vars(ctx: AgentContext) -> Dict[str, Any]:
    """Build the prompt-variable dict shared across agents."""
    static = ctx.static
    render = ctx.render
    agent = ctx.agent

    raw_console_errors = render.get("console_errors", [])
    console_errors = [
        entry for entry in raw_console_errors
        if isinstance(entry, dict) and not _is_benign_console_entry(entry)
    ]
    page_errors = render.get("page_errors", [])
    ext_resources = static.get("external_resources", [])
    agent_actions = agent.get("actions", [])

    discovered_keys = agent.get("discovered_keys", [])
    game_vars = agent.get("game_vars_initial", {})
    if not game_vars:
        game_vars = render.get("game_vars", {})

    avg_lat = render.get("avg_interaction_latency_ms")
    max_lat = render.get("max_interaction_latency_ms")
    int_timeouts = render.get("interactions_timed_out", 0)

    resp_count = render.get("responsive_viewports_tested", 0)
    responsive_str = ""
    if resp_count > 0:
        responsive_str = f", mobile (375x667), tablet (768x1024) — {resp_count} extra viewport(s)"

    btn_rate = render.get("button_response_rate", -1)

    return {
        "query": ctx.query,
        "html_size": static.get("html_size", 0),
        "has_canvas": "yes" if static.get("has_canvas") else "no",
        "has_script": "yes" if static.get("has_script") else "no",
        "has_style": "yes" if static.get("has_style") else "no",
        "has_svg": "yes" if static.get("has_svg") else "no",
        "has_raf": "yes" if static.get("has_requestanimationframe") else "no",
        "input_types": ", ".join(ctx.input_types) or "none detected",
        "ext_count": len(ext_resources),
        "ext_list": ", ".join(ext_resources[:5]) or "none",
        "static_issues": "; ".join(static.get("issues", [])) or "none",
        "rendered": "yes" if render.get("rendered") else "no",
        "page_title": render.get("page_title") or "none",
        "console_count": len(console_errors),
        "console_list": json.dumps(console_errors[:5], ensure_ascii=False),
        "benign_console_count": max(0, len(raw_console_errors) - len(console_errors)),
        "page_err_count": len(page_errors),
        "page_err_list": json.dumps(page_errors[:5], ensure_ascii=False),
        "agent_ran": "yes" if ctx.agent_ran else "no (skipped)",
        "agent_steps": agent.get("steps_taken", 0),
        "agent_actions": len(agent_actions),
        "agent_errors": json.dumps(agent.get("errors", [])[:5], ensure_ascii=False),
        "agent_summary": agent.get("agent_summary", "(not run)")[:3000],
        "discovered_keys": ", ".join(discovered_keys) if discovered_keys else "none detected",
        "game_vars": (
            ", ".join(f"{k}={v}" for k, v in list(game_vars.items())[:8])
            if game_vars else "none detected"
        ),
        "keyboard_probed": "yes" if ctx.kb_probed else "no",
        "keys_responded": ", ".join(ctx.kb_responded) if ctx.kb_responded else "none",
        "keyboard_visual_change": "yes" if ctx.kb_vis_change else "no",
        "keyboard_responsive": "yes" if ctx.kb_responsive else "no",
        "frame_annotations": _format_frame_annotations(ctx.frame_annotations),
        "animation_detected": "yes" if render.get("animation_detected") else "no",
        "frame_change_rate": render.get("frame_change_rate", 0),
        "has_below_fold": "yes" if render.get("has_below_fold") else "no",
        "button_responsive": "yes" if render.get("button_responsive") else "no",
        "hover_effects_detected": render.get("hover_effects_detected", 0),
        "avg_latency": f"{avg_lat}ms" if avg_lat is not None else "not measured",
        "max_latency": f"{max_lat}ms" if max_lat is not None else "not measured",
        "interactions_timed_out": int_timeouts,
        "responsive_viewports": responsive_str,
        "form_elements": ", ".join(
            name for name, key in [
                ("form", "has_form"), ("text_input", "has_text_input"),
                ("textarea", "has_textarea"), ("select", "has_select"),
                ("range", "has_range"),
            ] if static.get(key)
        ) or "none",
        "form_probed": "yes" if render.get("form_probed") else "no",
        "form_submitted": "yes" if render.get("form_submitted") else "no",
        "form_submit_changed": "yes" if render.get("form_submit_changed") else "no",
        "buttons_total": render.get("buttons_total", "n/a"),
        "buttons_tested": render.get("buttons_tested", "n/a"),
        "buttons_responsive": render.get("buttons_responsive_census", "n/a"),
        "button_response_rate_str": (
            f"{btn_rate:.0%}" if btn_rate >= 0 else "not tested"
        ),
        "drag_detected": "yes" if "mouse_drag" in ctx.input_types else "no",
        "drag_probed": "yes" if render.get("drag_probed") else "no",
        "drag_responsive": "yes" if render.get("drag_responsive") else "no",
        "gameplay_state_changed": (
            "yes" if render.get("gameplay_state_changed") is True else (
                "no" if render.get("gameplay_state_changed") is False else "not tested"
            )
        ),
        "gameplay_mode": render.get("gameplay_mode", "not tested"),
        "canvas_type": render.get("canvas_type", "n/a"),
        "canvas_has_content": "yes" if render.get("canvas_has_content") else (
            "no" if render.get("canvas_type") else "n/a"),
        "canvas_fill_ratio": render.get("canvas_fill_ratio", "n/a"),
        "fps_estimate": render.get("animation_fps_estimate", "n/a"),
        "fps_quality": render.get("fps_quality", "not measured"),
        "audio_elements": render.get("audio_elements", 0) + (
            1 if static.get("has_audio") else 0),
        "has_audio_context": "yes" if render.get("has_audio_context") or
            static.get("has_audio") else "no",
        "dom_inventory_str": format_dom_inventory(render.get("dom_inventory")),
        "visible_text": render.get("visible_text", "(not available)")[:2000],
        # Structural validation data
        "structural_event_listeners": render.get("structural_event_listener_count", "n/a"),
        "structural_raf_calls": render.get("structural_raf_calls_2s", "n/a"),
        "structural_overlays": len(render.get("structural_visible_overlays", [])),
        "structural_overlay_details": json.dumps(
            render.get("structural_visible_overlays", [])[:3], ensure_ascii=False
        ) if render.get("structural_visible_overlays") else "none",
        "structural_unbound_buttons": render.get("structural_unbound_buttons", "n/a"),
        "structural_total_buttons_struct": render.get("structural_total_buttons", "n/a"),
    }


def _encode_screenshots_content(
    paths: List[str],
    frame_annots: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Synchronously encode screenshots into multimodal content blocks.

    Called within asyncio.to_thread from the agent pipeline.
    Returns a list of {type: text/image_url} dicts ready for the LLM.
    """
    content: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths):
        try:
            b64 = encode_image_b64(path)
        except Exception as e:
            logger.warning(f"Failed to encode screenshot {path}: {e}")
            continue
        if idx < len(frame_annots):
            fa = frame_annots[idx]
            diff = fa.get("diff_from_prev", 0)
            # Surface visual change magnitude so Observer has objective evidence
            diff_tag = ""
            if idx > 0 and diff is not None:
                if diff < 0.003:
                    diff_tag = " ⚠ IDENTICAL to previous"
                elif diff < 0.01:
                    diff_tag = f" (subtle change, Δ={diff:.3f})"
                else:
                    diff_tag = f" (visual change, Δ={diff:.3f})"
            content.append({
                "type": "text",
                "text": (
                    f"[Frame {idx+1}: {fa.get('label','')} — "
                    f"{fa.get('description','')}{diff_tag}]"
                ),
            })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return content


def _is_non_scoring_constraint_text(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    return any(re.search(pattern, low) for pattern in _NON_SCORING_CONSTRAINT_PATTERNS)


def _sanitize_requirement_rows(requirements: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    cleaned: List[Dict[str, Any]] = []
    for req in requirements:
        requirement = str(req.get("requirement", "") or "")
        evidence = str(req.get("evidence", "") or "")
        if _is_non_scoring_constraint_text(requirement) or _is_non_scoring_constraint_text(evidence):
            continue
        cleaned.append({
            "requirement": requirement,
            "status": req.get("status", "missing"),
            "evidence": evidence,
        })

    summary = {
        "total": len(cleaned),
        "done": sum(1 for r in cleaned if r["status"] == "done"),
        "broken": sum(1 for r in cleaned if r["status"] == "broken"),
        "missing": sum(1 for r in cleaned if r["status"] == "missing"),
    }
    return cleaned, summary


def _sanitize_observation_rows(rows: List[Any]) -> List[str]:
    cleaned: List[str] = []
    for row in rows:
        text = str(row or "")
        if _is_non_scoring_constraint_text(text):
            continue
        cleaned.append(text)
    return cleaned


# ---------------------------------------------------------------------------
# Analyst — combined Observer + TaskAuditor in one VLM call (default)
# ---------------------------------------------------------------------------

class AnalystAgent(EvalAgent):
    """Combined perception + requirement audit. Sees all screenshots + probe data.

    Outputs both observation fields (page_type, working, broken, etc.)
    and requirement audit (requirements[], summary). Single VLM call replaces
    the separate Observer + TaskAuditor calls.
    """

    name = "analyst"
    max_tokens = 16384

    def build_content(self, ctx: AgentContext) -> List[Dict[str, Any]]:
        pvars = _common_prompt_vars(ctx)
        prompt_text = ANALYST_PROMPT.format(**pvars)

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        # All screenshots — Analyst is the VLM agent with visual perception
        paths = ctx.screenshots[:ctx.max_screenshots]
        content.extend(
            _encode_screenshots_content(paths, ctx.frame_annotations)
        )
        return content

    def parse_response(self, raw_json: Dict[str, Any], ctx: AgentContext) -> Dict[str, Any]:
        # Parse observation fields (same as ObserverAgent)
        result: Dict[str, Any] = {
            "page_type": raw_json.get("page_type", "unknown"),
            "visual_state": raw_json.get("visual_state", ""),
            "visual_elements": raw_json.get("visual_elements", []),
            "template_like_signals": raw_json.get("template_like_signals", []),
            "distinctive_design_signals": raw_json.get("distinctive_design_signals", []),
            "design_specificity": raw_json.get("design_specificity", ""),
            "working": _sanitize_observation_rows(raw_json.get("working", [])),
            "broken": _sanitize_observation_rows(raw_json.get("broken", [])),
            "interaction_quality": raw_json.get("interaction_quality", ""),
            "layout_notes": raw_json.get("layout_notes", ""),
        }

        # Parse requirement audit fields (same as TaskAuditorAgent)
        requirements = raw_json.get("requirements", [])
        cleaned, summary = _sanitize_requirement_rows(requirements)
        result["requirements"] = cleaned
        result["summary"] = summary

        return result


# ---------------------------------------------------------------------------
# Agent 1: Observer — perception (VLM + screenshots, no scoring) [legacy]
# ---------------------------------------------------------------------------

class ObserverAgent(EvalAgent):
    """Sees all screenshots + probe data. Outputs factual report. Does NOT score."""

    name = "observer"
    max_tokens = 16384

    def build_content(self, ctx: AgentContext) -> List[Dict[str, Any]]:
        pvars = _common_prompt_vars(ctx)
        prompt_text = OBSERVER_PROMPT.format(**pvars)

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        # All screenshots — Observer is the only agent with visual perception
        paths = ctx.screenshots[:ctx.max_screenshots]
        content.extend(
            _encode_screenshots_content(paths, ctx.frame_annotations)
        )
        return content

    def parse_response(self, raw_json: Dict[str, Any], ctx: AgentContext) -> Dict[str, Any]:
        return {
            "page_type": raw_json.get("page_type", "unknown"),
            "visual_state": raw_json.get("visual_state", ""),
            "visual_elements": raw_json.get("visual_elements", []),
            "template_like_signals": raw_json.get("template_like_signals", []),
            "distinctive_design_signals": raw_json.get("distinctive_design_signals", []),
            "design_specificity": raw_json.get("design_specificity", ""),
            "working": _sanitize_observation_rows(raw_json.get("working", [])),
            "broken": _sanitize_observation_rows(raw_json.get("broken", [])),
            "interaction_quality": raw_json.get("interaction_quality", ""),
            "layout_notes": raw_json.get("layout_notes", ""),
        }


# ---------------------------------------------------------------------------
# Agent 2: TaskAuditor — requirement checklist (text-only, no screenshots)
# ---------------------------------------------------------------------------

class TaskAuditorAgent(EvalAgent):
    """Compares Observer report against task requirements. Outputs structured checklist."""

    name = "task_auditor"
    max_tokens = 8192

    def build_content(self, ctx: AgentContext) -> List[Dict[str, Any]]:
        observer_report = ctx.stage_outputs.get("observer", {})
        pvars = _common_prompt_vars(ctx)
        pvars["observer_report_json"] = json.dumps(
            observer_report, indent=2, ensure_ascii=False,
        )
        prompt_text = TASK_AUDITOR_PROMPT.format(**pvars)
        # Text-only — no screenshots
        return [{"type": "text", "text": prompt_text}]

    def parse_response(self, raw_json: Dict[str, Any], ctx: AgentContext) -> Dict[str, Any]:
        requirements = raw_json.get("requirements", [])
        cleaned, summary = _sanitize_requirement_rows(requirements)

        return {
            "requirements": cleaned,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Agent 3: Scorer — judgment (text-only, NO screenshots)
# ---------------------------------------------------------------------------

class ScorerAgent(EvalAgent):
    """Scores based on Observer report + TaskAuditor checklist + objective metrics.

    Has NO visual input — true perception-judgment separation.
    """

    name = "scorer"
    max_tokens = 8192

    def build_content(self, ctx: AgentContext) -> List[Dict[str, Any]]:
        # Support both new (analyst) and legacy (observer + task_auditor) pipelines
        analyst_report = ctx.stage_outputs.get("analyst")
        if analyst_report:
            # New 2-agent pipeline: analyst contains both observation + requirements
            observer_report = analyst_report
            task_auditor_report = {
                "requirements": analyst_report.get("requirements", []),
                "summary": analyst_report.get("summary", {}),
            }
        else:
            # Legacy 3-agent pipeline
            observer_report = ctx.stage_outputs.get("observer", {})
            task_auditor_report = ctx.stage_outputs.get("task_auditor", {})

        pvars = _common_prompt_vars(ctx)
        pvars["observer_report_json"] = json.dumps(
            observer_report, indent=2, ensure_ascii=False,
        )
        pvars["task_auditor_report_json"] = json.dumps(
            task_auditor_report, indent=2, ensure_ascii=False,
        )

        prompt_text = SCORER_PROMPT.format(**pvars)

        # Text-only — NO screenshots. Scorer judges purely from evidence.
        return [{"type": "text", "text": prompt_text}]

    def parse_response(self, raw_json: Dict[str, Any], ctx: AgentContext) -> Dict[str, Any]:
        ev = raw_json

        # ── Post-parse clamping (defense in depth) ──────────────

        # Clamp 1: Keyboard broken → interaction ≤ 8, functionality ≤ 15
        if (
            ctx.kb_probed
            and "keyboard" in ctx.input_types
            and not ctx.kb_responsive
        ):
            if isinstance(ev.get("interaction"), dict):
                prev = ev["interaction"].get("score", 0)
                capped = min(prev, 8)
                ev["interaction"]["score"] = capped
                if capped != prev:
                    ev["interaction"]["reason"] = (
                        ev["interaction"].get("reason", "") +
                        " [CLAMPED: keyboard probed but unresponsive]"
                    ).strip()
            if isinstance(ev.get("functionality"), dict):
                prev = ev["functionality"].get("score", 0)
                capped = min(prev, 15)
                ev["functionality"]["score"] = capped
                if capped != prev:
                    ev["functionality"]["reason"] = (
                        ev["functionality"].get("reason", "") +
                        " [CLAMPED: keyboard-dependent requirement is unresponsive]"
                    ).strip()
            logger.info(
                "[scorer] clamped: keyboard probed but unresponsive → interaction≤8, functionality≤15"
            )

        # Clamp 2: Canvas blank → rendering ≤ 2
        static = ctx.static
        render = ctx.render
        if (
            static.get("has_canvas")
            and render.get("canvas_type")
            and not render.get("canvas_has_content", True)
            and not render.get("canvas_tainted", False)
        ):
            r_dim = ev.get("rendering")
            if isinstance(r_dim, dict) and r_dim.get("score", 0) > 2:
                r_dim["score"] = min(r_dim["score"], 2)
                r_dim["reason"] = (
                    r_dim.get("reason", "") + " [CLAMPED: canvas exists but is blank]"
                )
                logger.info("[scorer] clamped rendering≤2: blank canvas")

        # Clamp 3 removed: button response rate should affect interaction evidence,
        # but not directly cap visual design.
        btn_rate = render.get("button_response_rate", -1)
        if 0 <= btn_rate < 0.3 and render.get("buttons_tested", 0) >= 2:
            i_dim = ev.get("interaction")
            if isinstance(i_dim, dict) and i_dim.get("score", 0) > 8:
                i_dim["score"] = min(i_dim["score"], 8)
                i_dim["reason"] = (
                    i_dim.get("reason", "") +
                    f" [CLAMPED: {btn_rate:.0%} button response rate]"
                )
            logger.info(
                f"[scorer] clamped interaction≤8: "
                f"button_response_rate={btn_rate:.0%}"
            )

        # ── Recompute total ─────────────────────────────────────
        total = sum(
            ev[k].get("score", 0) for k in SCORE_DIMENSIONS
            if isinstance(ev.get(k), dict)
        )
        ev["total_score"] = total
        ev["agent_phase_run"] = ctx.agent_ran

        return ev


# ---------------------------------------------------------------------------
# Default agent pipeline — extend by inserting/appending to this list
# ---------------------------------------------------------------------------

# Default: 2-agent pipeline (Analyst + Scorer) — saves one VLM call
EVAL_AGENTS: List[EvalAgent] = [
    AnalystAgent(),   # Stage 1: combined perception + audit (VLM + screenshots)
    ScorerAgent(),    # Stage 2: scoring (text-only, reads analyst output)
]

# Legacy 3-agent pipeline (kept for reference / fallback)
EVAL_AGENTS_LEGACY: List[EvalAgent] = [
    ObserverAgent(),      # Stage 1: perception (VLM + screenshots)
    TaskAuditorAgent(),   # Stage 2: requirement audit (text-only)
    ScorerAgent(),        # Stage 3: scoring (text-only, no screenshots)
]
