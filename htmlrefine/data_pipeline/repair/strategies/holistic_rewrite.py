"""HolisticRewriteStrategy — complete rewrite preserving visual fidelity.

For score >= 40: the original HTML IS passed so the LLM can preserve
SVG paths, gradients, animations, and other visual details that cannot
be reconstructed from a text description alone.

For score < 40: original HTML is omitted (broken code anchors the LLM).
"""

from __future__ import annotations

import re
from typing import List, Optional

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.prompts import (
    HOLISTIC_REWRITE, _SCORING_RUBRIC, _GAME_SKELETON,
    format_prev_iterations, format_observer_evidence, format_requirement_checklist,
)
from htmlrefine.data_pipeline.repair.strategies.base import RepairStrategy


def _extract_visual_features(html: str) -> str:
    """Summarize visual AND gameplay features that MUST be preserved and enhanced."""
    features = []
    lower = html.lower()

    # --- Visual assets ---
    svg_paths = len(re.findall(r'<path\b', lower))
    if svg_paths > 0:
        features.append(f"{svg_paths} SVG <path> elements (keep ALL path data, add more detail)")

    gradients = len(re.findall(r'gradient\s*\(', lower))
    if gradients > 0:
        features.append(f"{gradients} CSS/SVG gradients (keep and enrich)")

    keyframes = re.findall(r'@keyframes\s+(\w+)', lower)
    if keyframes:
        features.append(f"{len(keyframes)} @keyframes animations: {', '.join(keyframes[:8])} (keep all, add more)")

    filters_count = lower.count('filter:') + lower.count('filter(')
    if filters_count > 0:
        features.append(f"{filters_count} CSS filter effects (keep and enhance)")

    if 'clip-path' in lower:
        features.append("clip-path shapes (keep exact values)")

    if 'perspective' in lower or 'transform3d' in lower or 'rotatex' in lower or 'rotatey' in lower:
        features.append("3D CSS transforms (keep perspective/rotate effects)")

    if '<canvas' in lower:
        features.append("HTML5 Canvas (keep ALL drawing/animation code)")

    # --- Gameplay mechanics ---
    physics_kws = [kw for kw in ['velocity', 'gravity', 'acceleration', 'friction', 'bounce', 'momentum'] if kw in lower]
    if physics_kws:
        features.append(f"Physics engine ({', '.join(physics_kws)}) — MUST keep realistic physics, do NOT simplify")

    collision_kws = [kw for kw in ['collision', 'intersect', 'hitbox', 'boundingbox', 'collide'] if kw in lower]
    if collision_kws:
        features.append(f"Collision detection ({', '.join(collision_kws)}) — MUST keep accurate collision")

    # --- Multi-state / multi-screen ---
    state_kws = [kw for kw in ['gamestate', 'game_state', 'currentstate', 'showscreen', 'currentscreen'] if kw in lower]
    if state_kws:
        features.append(f"State management ({', '.join(state_kws)}) — keep ALL game states and transitions")

    level_kws = [kw for kw in ['currentlevel', 'nextlevel', 'loadlevel', 'leveldata', 'levels['] if kw in lower]
    if level_kws:
        features.append(f"Level system ({', '.join(level_kws)}) — keep ALL levels with progression")

    screen_count = len(re.findall(r'(?:start|menu|game.?over|pause|settings|tutorial|loading)\s*(?:screen|menu|overlay|modal)', lower))
    if screen_count > 0:
        features.append(f"{screen_count} distinct screens/overlays (menu, game-over, etc.) — keep ALL transitions")

    # --- External resources ---
    img_tags = re.findall(r'<img[^>]+src=["\']([^"\']+)', html)
    if img_tags:
        features.append(f"{len(img_tags)} <img> tags with external URLs — KEEP ALL, add more if appropriate")

    font_imports = re.findall(r'fonts\.googleapis\.com|@import.*font|@font-face', lower)
    if font_imports:
        features.append(f"{len(font_imports)} external font imports — KEEP ALL font imports and font-family references")

    emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', html))
    if emoji_count > 2:
        features.append(f"{emoji_count} emoji/unicode characters — keep ALL")

    form_elements = len(re.findall(r'<(?:input|select|textarea)\b', lower))
    if form_elements > 2:
        features.append(f"{form_elements} form elements (input/select/textarea) — keep ALL")

    if not features:
        return ""

    lines = ["## Original Implementation Features (keep ALL and ENHANCE)"]
    for f in features:
        lines.append(f"- {f}")
    lines.append(
        "\n⚠ Your implementation MUST be at least as complex as the original. "
        "A simpler version that 'works' but loses visual-detail/resources/functionality "
        "is a FAILURE. The goal is to make it BETTER and MORE polished, not simpler."
    )
    return "\n".join(lines)


class HolisticRewriteStrategy(RepairStrategy):
    name = "holistic_rewrite"

    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
    ) -> str:
        issues: list[str] = []

        if not diag.render_ok:
            issues.append("Page failed to render (blank screen or crash)")

        if diag.reliable_issues:
            issues.extend(diag.reliable_issues[:4])

        if diag.bugs and len(issues) < 5:
            remaining = [b for b in diag.bugs if b not in diag.reliable_issues]
            issues.extend(remaining[:3])

        if diag.missing_features and len(issues) < 6:
            issues.extend(diag.missing_features[:3])

        if not issues:
            issues.append(
                f"Overall quality is too low (score {diag.score}/100); "
                "needs significant improvement"
            )

        prev = format_prev_iterations(prev_iterations or [])
        obs_ev = format_observer_evidence(diag)
        req_cl = format_requirement_checklist(diag)

        # Score >= 40: pass original HTML so LLM can preserve visual details
        # Score < 40: omit (broken code anchors the LLM, fresh start is better)
        if diag.score >= 40 and html:
            vis_features = _extract_visual_features(html)
            existing_html_block = (
                "## Existing HTML (use as reference — preserve all visual details)\n"
                "```html\n" + html + "\n```"
            )
        else:
            vis_features = ""
            existing_html_block = ""

        # Inject game skeleton for game pages to prevent broken input/loop patterns
        if diag.is_game:
            game_logic_guard = (
                "\n\n## Game Logic Protection\n"
                "If the original HTML contains any of the following, your version MUST preserve "
                "the exact algorithm (you may rename variables and restyle, but NOT change the logic):\n"
                "- Collision detection functions (collision/intersect/overlap/hitTest)\n"
                "- Physics simulation (velocity/gravity/friction/bounce/acceleration)\n"
                "- Level/map data arrays and progression logic\n"
                "- State machine transitions (gameState/currentScreen/phase)\n"
                "- Scoring/life/timer calculation formulas\n\n"
                "Do NOT simplify or rewrite game mechanics. A visually different but "
                "mechanically identical game is acceptable. A simpler game is NOT."
            )
            if vis_features:
                vis_features = vis_features + "\n\n" + _GAME_SKELETON + game_logic_guard
            else:
                vis_features = _GAME_SKELETON + game_logic_guard

        return HOLISTIC_REWRITE.format(
            query=query,
            score=diag.score,
            issues="\n".join(f"- {i}" for i in issues),
            summary=diag.summary or "(previous attempt had significant issues)",
            observer_evidence=obs_ev + "\n" if obs_ev else "",
            requirement_checklist=req_cl + "\n" if req_cl else "",
            prev_iterations=prev + "\n" if prev else "",
            visual_preservation=vis_features + "\n" if vis_features else "",
            rubric=_SCORING_RUBRIC,
            existing_html=existing_html_block + "\n" if existing_html_block else "",
        )
