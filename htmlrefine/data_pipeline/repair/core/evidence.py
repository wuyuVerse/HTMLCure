"""
Evidence — structured evidence collected before repair.

Evidence quality determines which repair strategies are safe:
  "high"   → agent ran + observed bugs/features = use surgical bug_fix
  "medium" → console errors OR keyscan data = use targeted fix for known issues
  "low"    → VLM-inferred only (77.7% of skip-agent records) = holistic_rewrite only

Key insight from data:
  - bug_fix with inferred evidence: 22% success, avg Δ=-5.4 (actively harmful)
  - feature_complete with inferred evidence: 34% success, avg Δ=-3.5 (usually harmful)
  - holistic_rewrite with any evidence: 76% success, avg Δ=+14.5 (consistently works)
  → Only use surgical strategies when evidence is objective/observed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("htmlrefine.repair")


# ---------------------------------------------------------------------------
# Evidence dataclass
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """
    Structured evidence about an HTML page's quality problems.

    Tiered by reliability:
      objective  → ground truth (console errors, keyscan probe, render result)
      observed   → agent saw it (agent bugs, agent summary, steps)
      inferred   → VLM guessed from screenshots (bugs, missing features, scores)
    """

    # ── Objective evidence (always reliable) ──────────────────────────────
    render_ok: bool = False
    console_errors: List[str] = field(default_factory=list)

    # Keyscan probe results (None = not run; [] = run but nothing worked)
    keyboard_game: bool = False          # static analysis detected keyboard input
    canvas_game: bool = False            # has canvas + requestAnimationFrame
    discovered_keys: Optional[List[str]] = None   # None = probe not run
    game_vars_initial: dict = field(default_factory=dict)

    # Render-test keyboard probe (lightweight, ~1.5s)
    keyboard_probed: bool = False        # whether render_test ran the keyboard probe
    keys_responded: List[str] = field(default_factory=list)  # keys that got a response
    keyboard_visual_change: bool = False # screenshot changed after key input

    # ── Observed evidence (agent ran) ──────────────────────────────────────
    agent_ran: bool = False
    agent_summary: str = ""
    agent_bugs: List[str] = field(default_factory=list)    # from agent report
    agent_steps: int = 0

    # ── Inferred evidence (VLM, always available but unreliable alone) ─────
    vl_bugs: List[str] = field(default_factory=list)
    vl_missing: List[str] = field(default_factory=list)
    vl_highlights: List[str] = field(default_factory=list)
    vl_hints: List[str] = field(default_factory=list)
    vl_scores: dict = field(default_factory=dict)  # raw final_score dict
    vl_summary: str = ""

    # ── Multi-agent reports (from 3-agent eval pipeline) ─────────────────
    observer_report: dict = field(default_factory=dict)       # {page_type, visual_state, working, broken, ...}
    task_auditor_report: dict = field(default_factory=dict)   # {requirements: [{requirement, status, evidence}], summary}

    # ── Dynamic experience evidence (from render_test Layer 1-3) ─────
    dynamic_experience_ran: bool = False
    button_responsive: bool = False
    animation_detected: bool = False
    has_below_fold: bool = False
    hover_effects_count: int = 0
    frame_change_rate: float = 0.0
    frame_annotations: List[dict] = field(default_factory=list)  # [{label, description, ...}]

    # ── Interaction latency (from probe measurement) ──────────────
    avg_interaction_latency_ms: Optional[int] = None   # None = not measured
    max_interaction_latency_ms: Optional[int] = None
    interactions_timed_out: int = 0

    # ── Responsive viewport evidence (from Layer 4) ───────────────
    responsive_viewports_tested: int = 0

    # ── Structural validation evidence (from Layer 0) ──────────────
    structural_event_listener_count: int = 0
    structural_raf_calls_2s: int = 0
    structural_visible_overlays: List[dict] = field(default_factory=list)
    structural_unbound_buttons: int = 0
    structural_total_buttons: int = 0

    # ── Game completion evidence (from render_test deep gameplay) ──────
    game_completion_detected: bool = False
    game_won: bool = False
    game_completion_state: str = "unknown"

    # ── Derived ───────────────────────────────────────────────────────────

    @property
    def quality(self) -> str:
        """
        Evidence quality tier.

        high   → agent ran with real observations
        medium → dynamic experience OR objective signals OR observer broken/working evidence
        low    → only VLM inference (no agent, no errors, no probe, no observer detail)
        """
        if self.agent_ran and self.agent_steps > 0:
            return "high"
        if self.dynamic_experience_ran:
            return "medium"
        if self.console_errors or self.discovered_keys is not None or self.keyboard_probed:
            return "medium"
        # Observer broken/working lists cite specific evidence (delta, latency, etc.)
        # This is more reliable than raw VLM inference — upgrade to medium.
        obs = self.observer_report
        if obs and (obs.get("broken") or obs.get("working")):
            return "medium"
        return "low"

    @property
    def keyboard_broken(self) -> bool:
        """Keyboard probe ran AND found zero responsive keys."""
        # Render-test lightweight probe
        if self.keyboard_probed and self.keyboard_game and not self.keys_responded and not self.keyboard_visual_change:
            return True
        # Legacy keyscan probe
        return (
            self.keyboard_game
            and self.discovered_keys is not None
            and len(self.discovered_keys) == 0
        )

    @property
    def keyboard_verified(self) -> bool:
        """Keyboard probe ran AND found at least one responsive key."""
        # Render-test lightweight probe
        if self.keyboard_probed and self.keyboard_game and (self.keys_responded or self.keyboard_visual_change):
            return True
        # Legacy keyscan probe
        return (
            self.keyboard_game
            and self.discovered_keys is not None
            and len(self.discovered_keys) > 0
        )

    @property
    def reliable_issues(self) -> List[str]:
        """
        Only issues we're objectively confident about.
        Safe to use for targeted repair prompts.

        Priority (highest → lowest reliability):
          1. Console errors (ground truth)
          2. Keyboard probe (ground truth)
          3. Dynamic experience (frame-level observation)
          4. Observer broken list (VLM saw it + cited evidence)
          5. TaskAuditor broken items (cross-referenced with Observer)
          6. Agent observations (agent ran and saw it)
        """
        issues = []

        # Console errors: always objective
        for e in self.console_errors[:3]:
            issues.append(f"JS console error: {e}")

        # Keyboard broken: proven by probe
        if self.keyboard_broken:
            issues.append(
                "Keyboard interaction broken (automated probe sent key events, "
                "page produced zero visual response). Most likely: canvas not "
                "focused, event listener on wrong element, or game loop not started."
            )

        # Dynamic experience evidence (frame-level)
        if self.dynamic_experience_ran:
            if not self.button_responsive:
                issues.append(
                    "Button clicks produce no visible change (frame comparison "
                    "before/after click shows identical screenshots)."
                )
            if self.animation_detected is False and self.frame_change_rate == 0:
                issues.append(
                    "No animation detected despite code using requestAnimationFrame. "
                    "Game loop may not be starting."
                )

        # Structural validation evidence (runtime probe)
        if self.structural_visible_overlays:
            for ov in self.structural_visible_overlays[:2]:
                issues.append(
                    f"Visible overlay blocking {ov.get('coverage', '?')}% of viewport "
                    f"({ov.get('selector', '?')}, z-index={ov.get('z_index', '?')}): "
                    f"'{ov.get('text_preview', '')[:40]}'. "
                    f"Likely a game-over/modal screen showing on page load."
                )
        if self.structural_unbound_buttons > 0 and self.structural_total_buttons > 0:
            issues.append(
                f"{self.structural_unbound_buttons}/{self.structural_total_buttons} "
                f"buttons have no event handler (no onclick or addEventListener). "
                f"These buttons are non-functional."
            )

        # Game completion not detected (for active games only).
        # Only report when probe confirmed the game is actually running
        # (has animation or button responses). 15-second probe rarely reaches
        # game-over, so skip this for games that didn't show activity.
        if (self.canvas_game or self.keyboard_game) and not self.game_completion_detected:
            if self.dynamic_experience_ran and (
                self.animation_detected or self.button_responsive
            ):
                issues.append(
                    "Game has no detectable completion state (no win/lose/game-over screen). "
                    "Consider adding a game-over overlay or score display when the game ends."
                )

        # Observer broken list: evidence-backed facts from VLM perception
        observer_broken = self.observer_report.get("broken", [])
        for item in observer_broken[:5]:
            if item not in issues:
                issues.append(item)

        # TaskAuditor broken items: cross-referenced with Observer
        for req in self.task_auditor_report.get("requirements", []):
            if req.get("status") == "broken" and len(issues) < 10:
                desc = req.get("requirement", "")
                evidence = req.get("evidence", "")
                entry = f"{desc} — {evidence}" if evidence else desc
                if entry not in issues:
                    issues.append(entry)

        # Agent observations: real, not inferred
        if self.agent_ran:
            for bug in self.agent_bugs[:3]:
                if bug not in issues:
                    issues.append(bug)

        return issues

    @property
    def preservation_list(self) -> List[str]:
        """
        Features confirmed working — MUST NOT be broken by repair.

        Priority: Observer working list + visual elements (evidence-backed) > VLM highlights > probe keys.
        """
        items = []
        # Observer working list: each item cites evidence (e.g., "delta=0.18")
        observer_working = self.observer_report.get("working", [])
        items.extend(observer_working)
        # Visual elements inventory: decorative details that must be preserved
        visual_elements = self.observer_report.get("visual_elements", [])
        for ve in visual_elements:
            if ve not in items:
                items.append(f"[visual] {ve}")
        # Fall back to VLM highlights if no observer data
        if not items:
            items = list(self.vl_highlights)
        # Add probe-verified keyboard (objective)
        if self.keyboard_verified and self.discovered_keys:
            items.append(
                f"Keyboard interaction works (probe-verified responsive keys: "
                f"{', '.join(self.discovered_keys)})"
            )
        return items

    @property
    def keyboard_fix_hint(self) -> str:
        """Targeted hint for fixing broken keyboard interaction."""
        return KEYBOARD_FIX_HINT


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

# Module-level constant so callers don't need to instantiate Evidence just for this text
KEYBOARD_FIX_HINT = (
    "The keyboard probe confirmed: key events ARE being dispatched to the page, "
    "but nothing visually responds. Check in order:\n"
    "1. canvas.setAttribute('tabIndex', '0') — canvas must be focusable\n"
    "2. canvas.focus() on page load or game-start button click\n"
    "3. Move keydown/keyup listeners from `document` to `canvas` (or keep on document "
    "but ensure the game loop is running when keys are pressed)\n"
    "4. Verify requestAnimationFrame loop starts immediately, not only after a click\n"
    "Fix ONLY the input handling. Do not rewrite game logic."
)


async def collect_evidence(
    ctx,                    # EvalContext
    browser_args: list,
    run_keyscan: bool = True,
    keyscan_timeout: float = 15.0,
) -> Evidence:
    """
    Build an Evidence object from a fully-evaluated EvalContext.

    Side-effect: if keyboard game + agent not run → runs keyscan probe (~8s).
    """
    # ── Static analysis ────────────────────────────────────────────────────
    static_phase = ctx.get_phase("static_analysis")
    static = static_phase.data if static_phase else {}

    input_types = static.get("input_types", [])
    keyboard_game = "keyboard" in input_types
    canvas_game = (
        static.get("has_canvas", False)
        and (static.get("has_requestanimationframe", False) or static.get("has_threejs", False))
    )
    # Three.js pages are interactive games even without explicit keyboard listeners
    threejs_game = static.get("has_threejs", False) and static.get("has_canvas", False)
    if threejs_game and not keyboard_game:
        keyboard_game = True

    # ── Render test ────────────────────────────────────────────────────────
    render_phase = ctx.get_phase("render_test")
    render = render_phase.data if render_phase else {}
    render_ok = bool(render.get("rendered"))
    console_errors = render.get("console_errors", [])[:5]

    # Render-test keyboard probe results
    kb_probed     = bool(render.get("keyboard_probed"))
    kb_responded  = render.get("keys_responded", []) or []
    kb_vis_change = bool(render.get("keyboard_visual_change"))

    # In repair context (no render_test phase), infer render_ok from VLM score.
    # rendering >= 12/20 means the page almost certainly renders visually.
    if not render_phase and ctx.final_score:
        fs_r = ctx.final_score.get("rendering", {})
        r_score = fs_r.get("score", 0) if isinstance(fs_r, dict) else int(fs_r or 0)
        if r_score >= 12:
            render_ok = True

    # ── Agent test ─────────────────────────────────────────────────────────
    agent_phase = ctx.get_phase("agent_test")
    agent_ran = agent_phase is not None and agent_phase.data.get("agent_completed", False)
    agent_data = agent_phase.data if agent_phase else {}

    # Keys from Phase 3 pre-scan (if agent ran)
    discovered_keys: Optional[List[str]] = None
    if agent_ran:
        dk = agent_data.get("discovered_keys")
        if dk is not None:
            discovered_keys = dk

    # ── keyscan probe (if keyboard game + agent not run + render probe didn't run) ─
    if (
        run_keyscan
        and keyboard_game
        and not agent_ran
        and not kb_probed
        and ctx.game_url_http
        and discovered_keys is None
    ):
        logger.info("[evidence] running keyscan probe for %s", ctx.game_id)
        try:
            from htmleval.phases.agent_test.keyscan import discover_keys
            scan = await asyncio.wait_for(
                discover_keys(ctx.game_url_http, list(browser_args)),
                timeout=keyscan_timeout,
            )
            discovered_keys = scan.get("working", [])
            logger.info("[evidence] keyscan: working=%s", discovered_keys)
        except asyncio.TimeoutError:
            logger.warning("[evidence] keyscan timed out")
            discovered_keys = None  # unknown, not "broken"
        except Exception as e:
            logger.warning("[evidence] keyscan error: %s", e)
            discovered_keys = None

    # ── VLM final_score fields ─────────────────────────────────────────────
    fs = ctx.final_score or {}
    vl_bugs     = fs.get("bugs", []) or []
    vl_missing  = fs.get("missing_features", []) or []
    vl_highlights = fs.get("highlights", []) or []
    vl_hints    = fs.get("improvement_hints", []) or []
    vl_summary  = fs.get("summary", "") or ""

    # ── Multi-agent reports (from 3-agent eval pipeline) ───────────────
    observer_report      = fs.get("observer_report", {}) or {}
    task_auditor_report  = fs.get("task_auditor_report", {}) or {}

    # Agent-observed bugs: from agent summary (heuristic extraction)
    agent_bugs: List[str] = []
    if agent_ran:
        # Use vl_bugs as they incorporate agent observations when agent ran
        agent_bugs = vl_bugs[:5]

    # ── Dynamic experience evidence (from render_test Layer 1-3) ──────
    dyn_ran    = bool(render.get("rendered"))
    btn_resp   = bool(render.get("button_responsive"))
    anim_det   = bool(render.get("animation_detected"))
    below_fold = bool(render.get("has_below_fold"))
    hover_fx   = int(render.get("hover_effects_detected", 0))
    fcr        = float(render.get("frame_change_rate", 0))
    f_annots   = render.get("frame_annotations", [])

    # ── Interaction latency + responsive ──────────────────────────────
    avg_lat    = render.get("avg_interaction_latency_ms")
    max_lat    = render.get("max_interaction_latency_ms")
    int_tout   = render.get("interactions_timed_out", 0)
    resp_vp    = render.get("responsive_viewports_tested", 0)

    # ── Structural validation (from Layer 0) ──────────────────────────
    sv_listeners   = int(render.get("structural_event_listener_count", 0))
    sv_raf         = int(render.get("structural_raf_calls_2s", 0))
    sv_overlays    = render.get("structural_visible_overlays", [])
    sv_unbound_btn = int(render.get("structural_unbound_buttons", 0))
    sv_total_btn   = int(render.get("structural_total_buttons", 0))

    # ── Game completion evidence (from deep gameplay probe) ──────────
    gc_detected = bool(render.get("game_completion_detected"))
    gc_won      = bool(render.get("game_won"))
    gc_state    = str(render.get("game_completion_state", "unknown"))

    return Evidence(
        render_ok=render_ok,
        console_errors=list(console_errors),
        keyboard_game=keyboard_game,
        canvas_game=canvas_game,
        discovered_keys=discovered_keys,
        game_vars_initial=agent_data.get("game_vars_initial", {}),
        keyboard_probed=kb_probed,
        keys_responded=list(kb_responded),
        keyboard_visual_change=kb_vis_change,
        agent_ran=agent_ran,
        agent_summary=agent_data.get("agent_summary", ""),
        agent_bugs=agent_bugs,
        agent_steps=agent_data.get("steps_taken", 0),
        vl_bugs=list(vl_bugs),
        vl_missing=list(vl_missing),
        vl_highlights=list(vl_highlights),
        vl_hints=list(vl_hints),
        vl_scores=dict(fs),
        vl_summary=vl_summary,
        observer_report=observer_report,
        task_auditor_report=task_auditor_report,
        dynamic_experience_ran=dyn_ran,
        button_responsive=btn_resp,
        animation_detected=anim_det,
        has_below_fold=below_fold,
        hover_effects_count=hover_fx,
        frame_change_rate=fcr,
        frame_annotations=f_annots,
        avg_interaction_latency_ms=avg_lat,
        max_interaction_latency_ms=max_lat,
        interactions_timed_out=int_tout,
        responsive_viewports_tested=resp_vp,
        structural_event_listener_count=sv_listeners,
        structural_raf_calls_2s=sv_raf,
        structural_visible_overlays=sv_overlays,
        structural_unbound_buttons=sv_unbound_btn,
        structural_total_buttons=sv_total_btn,
        game_completion_detected=gc_detected,
        game_won=gc_won,
        game_completion_state=gc_state,
    )
