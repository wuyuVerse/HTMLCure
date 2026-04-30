"""
Phase 1: StaticAnalysisPhase — lightweight structural analysis of the HTML.

No browser required.  Detects structural signals (canvas, script, animations,
external resources, input types) for use by downstream phases and the evaluator.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext, PhaseResult
from htmleval.core.phase import Phase

logger = logging.getLogger("htmleval")


def analyse_html(html_code: str) -> Dict[str, Any]:
    """Run static analysis on raw HTML — no browser needed."""
    result: Dict[str, Any] = {
        "html_size": len(html_code),
        "has_doctype": html_code.strip().lower().startswith("<!doctype"),
        "has_canvas": bool(re.search(r'<canvas', html_code, re.I)),
        "has_script": bool(re.search(r'<script', html_code, re.I)),
        "has_style":  bool(re.search(r'<style',  html_code, re.I)),
        "has_svg":    bool(re.search(r'<svg',    html_code, re.I)),
        "has_requestanimationframe": bool(
            re.search(r'requestanimationframe', html_code, re.I)
        ),
        "external_resources": [],
        "issues": [],
        "input_types": [],
    }

    # External resource URLs (src= and href= attributes)
    urls = re.findall(r'(?:src|href)=["\']((https?://[^"\']+))["\']', html_code)
    result["external_resources"] = list(set(u[0] for u in urls))

    # Structural issues
    if not result["has_doctype"]:
        result["issues"].append("Missing DOCTYPE declaration")
    if result["html_size"] < 200:
        result["issues"].append(f"HTML very short ({result['html_size']} chars)")

    # Detect input interaction types
    code_lower = html_code.lower()
    input_types: set[str] = set()

    if any(p in code_lower for p in [
        'keydown', 'keyup', 'keypress',
        'arrowup', 'arrowdown', 'arrowleft', 'arrowright',
        'e.key', 'e.code', 'e.which', 'keycode',
        "key === 'w'", "key === 'a'", "key === 's'", "key === 'd'",
    ]):
        input_types.add('keyboard')

    has_mousedown  = any(p in code_lower for p in ['mousedown', 'pointerdown'])
    has_mousemove  = any(p in code_lower for p in ['mousemove', 'pointermove'])
    has_touchstart = 'touchstart' in code_lower
    has_touchmove  = 'touchmove'  in code_lower

    if (has_mousedown and has_mousemove) or (has_touchstart and has_touchmove):
        input_types.add('mouse_drag')

    if any(p in code_lower for p in ['onclick', 'click', 'mousedown', 'pointerdown', 'mouseup']):
        input_types.add('mouse_click')

    if has_touchstart:
        input_types.add('touch')

    result["input_types"] = sorted(input_types)

    # Detect form elements (for targeted Layer 2 probing)
    result["has_form"] = bool(re.search(r'<form', html_code, re.I))
    result["has_text_input"] = bool(re.search(
        r'<input[^>]*type=["\']?(text|email|password|search|number|tel|url)',
        html_code, re.I,
    )) or bool(re.search(r'<input(?![^>]*type=)[^>]*>', html_code, re.I))  # input without type = text
    result["has_textarea"] = bool(re.search(r'<textarea', html_code, re.I))
    result["has_select"] = bool(re.search(r'<select', html_code, re.I))
    result["has_range"] = bool(re.search(r'<input[^>]*type=["\']?range', html_code, re.I))
    result["has_checkbox"] = bool(re.search(
        r'<input[^>]*type=["\']?(checkbox|radio)', html_code, re.I,
    ))
    result["has_audio"] = bool(re.search(r'<audio|new\s+Audio|AudioContext|webkitAudioContext', html_code, re.I))
    result["has_video"] = bool(re.search(r'<video', html_code, re.I))
    result["has_threejs"] = bool(re.search(r'THREE\.\w|from\s+[\'"]three[\'"]|import.*THREE', html_code))
    result["has_webgl"] = bool(re.search(r"getContext\s*\(\s*['\"]webgl", html_code, re.IGNORECASE))

    # Semantic HTML tags and heading hierarchy. These are lightweight
    # structural signals used later for implementation-quality scoring.
    semantic_tags = ("header", "main", "section", "nav", "footer", "article", "aside")
    for tag in semantic_tags:
        count = len(re.findall(rf'<{tag}\b', html_code, re.I))
        result[f"{tag}_count"] = count
        result[f"has_{tag}"] = count > 0
    result["semantic_layout_count"] = sum(
        1 for tag in semantic_tags if result.get(f"has_{tag}", False)
    )
    result["h1_count"] = len(re.findall(r'<h1\b', html_code, re.I))
    result["h2_count"] = len(re.findall(r'<h2\b', html_code, re.I))

    # Title and meta viewport
    result["has_title"] = bool(re.search(r'<title[^>]*>.+</title>', html_code, re.I | re.DOTALL))
    result["has_meta_viewport"] = bool(re.search(r'<meta[^>]*viewport', html_code, re.I))

    # lang attribute on <html>
    result["has_lang_attr"] = bool(re.search(r'<html[^>]*\blang\s*=', html_code, re.I))

    # Modern CSS detection
    result["has_media_queries"] = bool(re.search(r'@media\s', html_code))
    result["has_css_transitions"] = bool(re.search(r'transition\s*:', html_code, re.I))
    result["has_css_animations"] = bool(re.search(r'@keyframes\s|animation\s*:', html_code, re.I))
    result["has_css_variables"] = bool(re.search(r'var\(--', html_code))
    result["has_flexbox"] = bool(re.search(r'display\s*:\s*flex', html_code, re.I))
    result["has_grid"] = bool(re.search(r'display\s*:\s*grid', html_code, re.I))

    # Interactivity signals. Count several common event wiring styles so
    # single-file HTML apps are not penalized for avoiding addEventListener.
    result["has_addeventlistener"] = bool(re.search(r'addEventListener', html_code))
    addeventlistener_count = len(re.findall(r'addEventListener', html_code))
    result["addeventlistener_count"] = addeventlistener_count
    inline_event_attr_count = len(re.findall(
        r'\bon(?:click|change|input|submit|keydown|keyup|keypress|mousedown|mouseup|mousemove|mouseenter|mouseleave|touchstart|touchmove|pointerdown|pointerup|dragstart|dragover|drop)\s*=',
        html_code,
        re.I,
    ))
    js_event_assignment_count = len(re.findall(
        r'\.\s*on(?:click|change|input|submit|keydown|keyup|keypress|mousedown|mouseup|mousemove|mouseenter|mouseleave|touchstart|touchmove|pointerdown|pointerup|dragstart|dragover|drop)\s*=',
        html_code,
        re.I,
    ))
    result["event_handler_count"] = (
        addeventlistener_count + inline_event_attr_count + js_event_assignment_count
    )
    result["has_gradient"] = bool(re.search(r'gradient\s*\(', html_code, re.I))
    result["has_opacity_usage"] = bool(re.search(r'opacity\s*:', html_code, re.I))
    result["has_hover_css"] = bool(re.search(r':hover\s*\{', html_code))

    # Derived counts for scoring
    result["ext_count"] = len(result["external_resources"])
    result["inline_style_count"] = len(re.findall(r'\bstyle\s*=\s*["\']', html_code))

    return result


class StaticAnalysisPhase(Phase):
    """Phase 1 — structural analysis of extracted HTML.

    PhaseResult.data fields:
        html_size                   int   — character count
        has_doctype/canvas/script/
          style/svg/requestanimationframe bool
        external_resources          list  — absolute URLs in src/href
        issues                      list  — human-readable warnings
        input_types                 list  — detected interaction modes
    """

    @property
    def name(self) -> str:
        return "static_analysis"

    async def execute(self, ctx: EvalContext) -> PhaseResult:
        if ctx.html_code is None:
            return PhaseResult(
                phase_name=self.name,
                success=False,
                errors=["No HTML code available (extract phase may have failed)"],
            )

        analysis = analyse_html(ctx.html_code)

        logger.info(
            "[static_analysis] %s: size=%d canvas=%s raf=%s inputs=%s issues=%d",
            ctx.game_id,
            analysis["html_size"],
            analysis["has_canvas"],
            analysis["has_requestanimationframe"],
            analysis["input_types"],
            len(analysis["issues"]),
        )

        return PhaseResult(phase_name=self.name, success=True, data=analysis)

    def should_stop_pipeline(self, result: PhaseResult, ctx: EvalContext) -> bool:
        return False
