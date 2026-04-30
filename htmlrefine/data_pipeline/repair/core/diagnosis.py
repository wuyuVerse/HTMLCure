"""
Diagnosis extractor — parses EvalContext + Evidence into a structured repair diagnosis.

Evidence quality gating (KEY DESIGN PRINCIPLE):
  Data shows surgical strategies fail on inferred-only evidence:
    - bug_fix with inferred evidence:     22% success, avg Δ = -5.4  (harmful)
    - feature_complete with inferred:     34% success, avg Δ = -3.5  (usually harmful)
    - holistic_rewrite with any evidence: 76% success, avg Δ = +14.5 (consistently works)

  Strategy selection is therefore GATED by evidence quality:
    - "low"    (VLM only, 77.7% of records)  → holistic_rewrite ONLY
    - "medium" (console errors or keyscan)    → targeted fix for proven issues only
    - "high"   (agent ran with observations)  → full score-based selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from htmleval.core.context import EvalContext

if TYPE_CHECKING:
    from htmlrefine.data_pipeline.repair.core.evidence import Evidence


# ---------------------------------------------------------------------------
# Diagnosis dataclass
# ---------------------------------------------------------------------------

@dataclass
class Diagnosis:
    """Structured analysis of what's wrong with an HTML page."""

    # Scores
    score: int
    rendering: int
    visual_design: int
    functionality: int
    interaction: int
    code_quality: int

    # Tier
    tier: str                        # "A" / "B" / "C"

    # Structured findings from LLM
    bugs: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)
    summary: str = ""

    # Weak dimensions (name, current, max) sorted by deficit
    weak_dims: List[tuple] = field(default_factory=list)

    # Context flags
    render_ok: bool = True
    has_agent_data: bool = False
    console_errors: List[str] = field(default_factory=list)

    # ── Evidence quality fields (NEW) ──────────────────────────────────────
    evidence_quality: str = "low"   # "high" | "medium" | "low"
    keyboard_broken: bool = False   # keyscan confirmed: key events sent but nothing responds
    keyboard_verified: bool = False # keyscan confirmed: keyboard actually works
    discovered_keys: Optional[List[str]] = None   # probe-verified responsive keys
    game_vars_initial: dict = field(default_factory=dict)

    # ── Game detection fields ─────────────────────────────────────────────
    is_game: bool = False
    canvas_game: bool = False
    structural_raf_calls_2s: int = 0
    structural_visible_overlays: List[dict] = field(default_factory=list)
    structural_event_listener_count: int = 0

    # Only issues we're objectively confident about (from reliable_issues property)
    reliable_issues: List[str] = field(default_factory=list)

    # Features confirmed working — must not be broken by repair
    preservation_list: List[str] = field(default_factory=list)

    # Multi-agent reports (from 3-agent eval pipeline)
    observer_report: dict = field(default_factory=dict)
    task_auditor_report: dict = field(default_factory=dict)

    # Chosen strategy
    strategy: str = "holistic_rewrite"

    def __str__(self) -> str:
        lines = [
            f"Score: {self.score}/100  Tier: {self.tier}  "
            f"Evidence: {self.evidence_quality}  Strategy: {self.strategy}",
            f"  rendering={self.rendering}/20  visual={self.visual_design}/20  "
            f"func={self.functionality}/25  interaction={self.interaction}/25  "
            f"code={self.code_quality}/10",
        ]
        if self.reliable_issues:
            lines.append(f"  Reliable issues: {self.reliable_issues[:3]}")
        elif self.bugs:
            lines.append(f"  VLM bugs (inferred): {self.bugs[:3]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

_DIM_MAX = {
    "rendering": 20,
    "visual_design": 20,
    "functionality": 25,
    "interaction": 25,
    "code_quality": 10,
}

TIER_A = 80
TIER_B = 40


def _tier(score: int) -> str:
    if score >= TIER_A:
        return "A"
    if score >= TIER_B:
        return "B"
    return "C"


def _extract_scores(ctx: EvalContext):
    """Pull per-dimension scores from ctx.final_score."""
    fs = ctx.final_score or {}
    def _get(k):
        v = fs.get(k, {})
        return v.get("score", 0) if isinstance(v, dict) else int(v or 0)
    return {k: _get(k) for k in _DIM_MAX}


def _weak_dims(scores: dict) -> List[tuple]:
    """Return dims sorted by (current/max) ascending — weakest first."""
    items = [(k, scores[k], _DIM_MAX[k]) for k in _DIM_MAX]
    items.sort(key=lambda x: x[1] / x[2])
    return items


def _select_strategy(
    diag: "Diagnosis",
    prev_strategy: str = "",
    consecutive_holistic: int = 0,
    total_fix_playability: int = 0,
    blacklist: Optional[set] = None,
) -> str:
    """
    Choose repair/refinement strategy with blacklist post-filter.

    Wraps _select_strategy_core: if the chosen strategy is blacklisted
    (catastrophic regression delta < -15), falls through to alternatives.
    """
    chosen = _select_strategy_core(
        diag, prev_strategy=prev_strategy,
        consecutive_holistic=consecutive_holistic,
        total_fix_playability=total_fix_playability,
    )
    if not blacklist or chosen not in blacklist:
        return chosen

    # Blacklisted → try alternatives in priority order
    _alternatives = [
        "visual_enrichment", "feature_complete", "bug_fix",
        "holistic_rewrite", "fix_game", "fix_interaction",
    ]
    for alt in _alternatives:
        if alt != chosen and alt not in blacklist:
            return alt
    # Everything blacklisted → holistic_rewrite as last resort
    return "holistic_rewrite"


def _select_strategy_core(
    diag: "Diagnosis",
    prev_strategy: str = "",
    consecutive_holistic: int = 0,
    total_fix_playability: int = 0,
) -> str:
    """
    Choose repair/refinement strategy from diagnosis state.

    Strategy selection priority:
      - P0: keyboard broken → fix_playability (capped at 2 uses)
      - P1: very low interaction → fix_interaction
      - Score 60-79: prefer surgical strategies (feature_complete, bug_fix)
        over holistic_rewrite which tends to regress at these scores
      - Score >= 80: dimension-targeted refinement
      - Score < 60: evidence-gated (holistic for low evidence, diversify after 2)

    Anti-stagnation rules:
      - consecutive_holistic >= 2 → force diversification (visual_enrichment at 70+)
      - total_fix_playability >= 2 → stop trying fix_playability
      - After fix_playability succeeds → use feature_complete (protect keyboard)
      - holistic_rewrite at score 60+ → use feature_complete instead
    """
    # ── P0: keyboard confirmed broken → fix_playability ──────────────────
    # Cap at 2 total uses — data shows 410 records stuck in playability loop
    if diag.keyboard_broken:
        if total_fix_playability < 2 and prev_strategy != "fix_playability":
            return "fix_playability"
        # fix_playability exhausted or just tried — use feature_complete
        return "feature_complete"

    # ── P0.5: game stuck → fix_game (probe-driven layer fix) ───────────
    # Data: 58 game_html records stuck < 60 because feature_complete cycles
    # 5-7 times with all-negative deltas. fix_game has 4 diagnostic layers
    # (overlay/game_loop/input/canvas) + new "gameplay" layer for logic bugs.
    # Widened range: 40-79 (was 50-65) to cover more stuck games.
    # Default: fix_game with "gameplay" layer (NOT feature_complete).
    if diag.is_game and 40 <= diag.score < 80:
        if diag.canvas_game and diag.structural_raf_calls_2s == 0:
            return "fix_game"
        if diag.structural_visible_overlays:
            return "fix_game"
        # Game with reliable issues mentioning completion/collision/physics → fix_game
        if any(kw in issue.lower() for issue in diag.reliable_issues
               for kw in ("completion", "game over", "collision", "boundary", "physics")):
            return "fix_game"
        # Default for games: fix_game (gameplay layer) — not feature_complete which
        # adds new features that break existing game logic.
        if prev_strategy != "fix_game":
            return "fix_game"
        # Just tried fix_game → rotate to feature_complete for variety
        return "feature_complete"

    # ── Post-fix_playability guard ────────────────────────────────────────
    _just_fixed_playability = (
        prev_strategy == "fix_playability" and not diag.keyboard_broken
    )

    # ── P1: interaction extremely low → fix_interaction ──────────────────
    if (
        diag.interaction < 15
        and diag.score >= 60
        and diag.evidence_quality in ("medium", "high")
    ):
        if prev_strategy != "fix_interaction":
            return "fix_interaction"
        return "feature_complete"

    # ── Tier A (score >= 80): dimension-targeted refinement ──────────────
    if diag.score >= 80:
        return _dimension_targeted(diag, prev_strategy)

    # ── Anti-stagnation: diversify after 2+ consecutive holistic_rewrite ──
    # Note: 80+ is handled above by _dimension_targeted() — this block is for < 80.
    if consecutive_holistic >= 2 and diag.score >= 40:
        func_ratio = diag.functionality / 25
        vis_ratio = diag.visual_design / 20
        if func_ratio < 0.72:
            return "feature_complete"
        elif vis_ratio < 0.75:
            return "visual_enrichment" if diag.score >= 70 else "feature_complete"
        else:
            return "feature_complete"

    # ── Post-fix_playability: build on working code, don't rewrite ────────
    if _just_fixed_playability and diag.score >= 40:
        return "feature_complete"

    # ── Score 60-79: prefer surgical strategies over holistic_rewrite ─────
    # Data shows holistic_rewrite at 60-79 regresses 493/800 records.
    # These pages have decent structure — rewriting from scratch loses it.
    if diag.score >= 60:
        func_ratio = diag.functionality / 25
        vis_ratio = diag.visual_design / 20
        int_ratio = diag.interaction / 25

        if diag.reliable_issues:
            return "bug_fix" if prev_strategy != "bug_fix" else "feature_complete"
        if func_ratio < 0.72:
            return "feature_complete" if prev_strategy != "feature_complete" else "bug_fix"
        if int_ratio < 0.60:
            return "fix_interaction" if prev_strategy != "fix_interaction" else "feature_complete"
        if vis_ratio < 0.75:
            # visual_enhance is the worst strategy at all tiers (avg Δ=-11.9).
            # Route to visual_enrichment (VLM-guided) at 70+ or feature_complete below.
            if diag.score >= 70:
                return "visual_enrichment" if prev_strategy != "visual_enrichment" else "feature_complete"
            return "feature_complete" if prev_strategy != "feature_complete" else "holistic_rewrite"
        return "feature_complete" if prev_strategy != "feature_complete" else "bug_fix"

    # ── Tier B/C (score < 60): evidence-gated repair ────────────────────
    if diag.evidence_quality == "low":
        return "holistic_rewrite"

    if diag.evidence_quality == "medium":
        if diag.console_errors:
            return "bug_fix"
        return "holistic_rewrite"

    # ── evidence_quality == "high" (agent ran + real observations) ─────────
    if not diag.render_ok or diag.score < 50:
        return "holistic_rewrite"

    func_ratio = diag.functionality / 25
    vis_ratio  = diag.visual_design / 20

    natural = None
    if diag.reliable_issues and func_ratio >= 0.5:
        natural = "bug_fix"
    elif func_ratio < 0.72:
        natural = "feature_complete"
    elif vis_ratio < 0.65 and func_ratio >= 0.72:
        natural = "feature_complete"  # was visual_enhance, but it's destructive at all tiers
    else:
        natural = "feature_complete"

    if prev_strategy and prev_strategy == natural:
        _fallback = {
            "bug_fix":          "feature_complete",
            "feature_complete": "bug_fix",
            "holistic_rewrite": "feature_complete",
        }
        return _fallback.get(natural, natural)

    return natural


# ---------------------------------------------------------------------------
# Refinement strategy selection (Tier A: 80-94)
# ---------------------------------------------------------------------------

# Dimension thresholds: below these → target that dimension.
# Set ~85% of max so refinement activates before near-max scores,
# giving strategies room to push toward 95+.
_REFINE_THRESHOLDS = {
    "visual_design": 17,    # out of 20
    "interaction":   21,    # out of 25
    "functionality": 22,    # out of 25
    "code_quality":  8,     # out of 10
}

# Stricter thresholds for 90+ scores: only intervene on truly weak dimensions.
# Data: at 94 func=22 still triggered refine_functionality → LLM invented changes → Δ=-72.
_REFINE_THRESHOLDS_HIGH = {
    "visual_design": 16,    # out of 20  — only ≤15 is worth touching
    "interaction":   20,    # out of 25  — only ≤19 is worth touching
    "functionality": 21,    # out of 25  — only ≤20 is worth touching
    "code_quality":  7,     # out of 10  — only ≤6 is worth touching
}

# Map dimension → refinement strategy
_REFINE_DIM_TO_STRATEGY = {
    "visual_design": "polish_visual",
    "interaction":   "enhance_interaction",
    "functionality": "refine_functionality",
    "code_quality":  "code_cleanup",
}


def _dimension_targeted(diag: "Diagnosis", prev_strategy: str = "") -> str:
    """
    Choose refinement strategy for Tier A pages based on weakest dimension.

    Score-adaptive thresholds:
      - score < 90: standard thresholds (_REFINE_THRESHOLDS)
      - score >= 90: stricter thresholds (_REFINE_THRESHOLDS_HIGH)

    Special rule for 90+ scores:
      If prev_strategy is already visual_enrichment (we tried it once and it
      didn't improve), stop retrying — return polish_visual for a different
      angle, or converge. Data shows visual_enrichment avg Δ=-7.2 on 90+ records.
    """
    # Detect whether this page actually has meaningful user interaction.
    page_is_interactive = _page_has_interaction(diag)

    # Score-adaptive thresholds
    thresholds = _REFINE_THRESHOLDS_HIGH if diag.score >= 90 else _REFINE_THRESHOLDS

    # Compute deficit ratios for refineable dimensions
    dim_scores = {
        "visual_design": diag.visual_design,
        "interaction":   diag.interaction,
        "functionality": diag.functionality,
        "code_quality":  diag.code_quality,
    }

    # Filter to dimensions at or below their refinement threshold
    candidates = []
    for dim, current in dim_scores.items():
        threshold = thresholds[dim]
        if current <= threshold:
            # Skip interaction refinement for non-interactive pages
            if dim == "interaction" and not page_is_interactive:
                continue
            deficit_ratio = (_DIM_MAX[dim] - current) / _DIM_MAX[dim]
            strategy_name = _REFINE_DIM_TO_STRATEGY[dim]
            # code_cleanup: almost never improves total score (data: 452 records
            # used it as first strategy, nearly all regressed). Only use when
            # code_quality is truly terrible (deficit > 30%, i.e. score <= 6/10).
            if strategy_name == "code_cleanup":
                if deficit_ratio <= 0.30:
                    continue
            # Skip truly marginal deficits for risky strategies
            elif strategy_name in ("refine_functionality", "enhance_interaction"):
                if deficit_ratio <= 0.10:
                    continue
            candidates.append((deficit_ratio, dim))

    # Sort by deficit ratio descending (worst dimension first)
    candidates.sort(reverse=True)

    if candidates:
        _, best_dim = candidates[0]
        natural = _REFINE_DIM_TO_STRATEGY[best_dim]

        # Rotate if same as previous strategy
        if prev_strategy and prev_strategy == natural and len(candidates) > 1:
            _, next_dim = candidates[1]
            return _REFINE_DIM_TO_STRATEGY[next_dim]

        return natural

    # All dimensions near-max: visual_enrichment has VLM diagnosis + zero catastrophe rate at 80+
    # But data shows: on 90+ scores, visual_enrichment avg Δ=-7.2 (still harmful).
    # After one failed visual_enrichment attempt, rotate to polish_visual for variety.
    if diag.score >= 90 and prev_strategy == "visual_enrichment":
        return "polish_visual"
    return "visual_enrichment"


def _page_has_interaction(diag: "Diagnosis") -> bool:
    """Detect whether a page has meaningful user interaction elements.

    Returns False for purely visual/decorative pages (SVG art, CSS animations,
    static illustrations) where adding keyboard handlers would be harmful.

    Uses observer_report and task_auditor_report to determine page nature.
    """
    obs = diag.observer_report
    audit = diag.task_auditor_report

    # If observer identified interactive elements or working interactions
    if obs:
        page_type = obs.get("page_type", "").lower()
        # Explicitly non-interactive types
        if any(kw in page_type for kw in ("illustration", "animation", "art", "display", "decoration", "static")):
            return False
        # Explicitly interactive types
        if any(kw in page_type for kw in ("game", "app", "tool", "dashboard", "form", "editor", "calculator")):
            return True
        # Check if observer found working/broken interaction elements
        working = obs.get("working", [])
        broken = obs.get("broken", [])
        interaction_kws = ("click", "button", "input", "drag", "keyboard", "key", "submit", "toggle", "slider")
        for item in working + broken:
            if any(kw in item.lower() for kw in interaction_kws):
                return True

    # If task auditor found interaction-related requirements
    if audit:
        for req in audit.get("requirements", []):
            desc = req.get("requirement", "").lower()
            if any(kw in desc for kw in ("click", "button", "input", "keyboard", "drag", "interact", "control")):
                return True

    # Check keyboard_verified from probe — if keyboard works, page is interactive
    if diag.keyboard_verified:
        return True

    # Default: assume interactive (safer than skipping needed fixes)
    # But if interaction score is already high (≥20), likely not much to fix
    if diag.interaction >= 20:
        return True

    # Heuristic: if functionality is high but interaction is low on a Tier A page,
    # the page likely works but just isn't "interactive" in the traditional sense
    # (e.g., an animation that scores well on func but low on interaction UX polish)
    if diag.functionality >= 22 and diag.interaction < 18:
        # Check query keywords for non-interactive content
        summary = (diag.summary or "").lower()
        non_interactive_kws = ("svg", "animation", "illustration", "art", "display",
                               "visualization", "ornament", "decoration", "scene",
                               "landscape", "portrait", "pattern", "drawing")
        if any(kw in summary for kw in non_interactive_kws):
            return False

    return True


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_diagnosis(
    ctx: EvalContext,
    tier_a: int = TIER_A,
    tier_b: int = TIER_B,
    prev_strategy: str = "",
    evidence: Optional["Evidence"] = None,
    consecutive_holistic: int = 0,
    total_fix_playability: int = 0,
    strategy_blacklist: Optional[set] = None,
) -> Diagnosis:
    """
    Build a Diagnosis from a fully evaluated EvalContext.

    Args:
        ctx:           EvalContext after pipeline.evaluate() or _inject_scores().
        tier_a:        Score threshold for Tier A (default 80).
        tier_b:        Score threshold for Tier B (default 40).
        prev_strategy: Last repair strategy used (for diversification).
        evidence:      Pre-collected Evidence object (if available).
                       If None, falls back to phase data or VLM-only.

    Returns:
        Diagnosis with unified strategy selection (score-based routing).
    """
    scores = _extract_scores(ctx)
    total  = ctx.total_score
    fs     = ctx.final_score or {}

    # LLM-generated findings
    bugs     = fs.get("bugs", []) if isinstance(fs.get("bugs"), list) else []
    missing  = fs.get("missing_features", []) if isinstance(fs.get("missing_features"), list) else []
    summary  = fs.get("summary", "") if isinstance(fs.get("summary"), str) else ""
    highlights = fs.get("highlights", []) if isinstance(fs.get("highlights"), list) else []

    # Render state — check phase first, then infer from VLM rendering score
    render_res  = ctx.get_phase("render_test")
    if render_res:
        render_ok   = bool(render_res.data.get("rendered"))
        console_err = (render_res.data.get("console_errors", []))[:5]
    else:
        # In repair context (no pipeline run): infer from VLM rendering score
        render_score_v = scores.get("rendering", 0)
        render_ok   = render_score_v >= 12  # ≥12/20 means page likely renders
        console_err = []

    # Agent state
    agent_res      = ctx.get_phase("agent_test")
    has_agent_data = agent_res is not None

    # ── Evidence integration ───────────────────────────────────────────────
    observer_report = {}
    task_auditor_report = {}
    # Game detection fields (populated from evidence or static analysis)
    is_game = False
    canvas_game_flag = False
    sv_raf_calls = 0
    sv_overlays: List[dict] = []
    sv_listeners = 0

    if evidence is not None:
        evidence_quality  = evidence.quality
        keyboard_broken   = evidence.keyboard_broken
        keyboard_verified = evidence.keyboard_verified
        discovered_keys   = evidence.discovered_keys
        game_vars_initial = evidence.game_vars_initial
        reliable_issues   = list(evidence.reliable_issues)
        preservation      = list(evidence.preservation_list)
        observer_report   = evidence.observer_report
        task_auditor_report = evidence.task_auditor_report
        # Game detection from evidence
        canvas_game_flag = evidence.canvas_game
        sv_raf_calls = evidence.structural_raf_calls_2s
        sv_overlays = list(evidence.structural_visible_overlays)
        sv_listeners = evidence.structural_event_listener_count
        is_game = evidence.keyboard_game or canvas_game_flag
        # Use console errors from evidence if phase data not available
        if not console_err:
            console_err = list(evidence.console_errors)
        # Evidence may have a better render_ok determination
        if evidence.render_ok:
            render_ok = True
        # Enrich missing_features from TaskAuditor (more structured)
        auditor_missing = [
            req.get("requirement", "")
            for req in task_auditor_report.get("requirements", [])
            if req.get("status") == "missing" and req.get("requirement")
        ]
        if auditor_missing:
            # Prefer auditor's list (requirement-level granularity)
            missing = auditor_missing
    else:
        # Fall back to phase data or low-quality inference
        evidence_quality  = "high" if has_agent_data else (
            "medium" if console_err else "low"
        )
        keyboard_broken   = False
        keyboard_verified = False
        discovered_keys   = None
        game_vars_initial = {}
        # Game detection from static analysis phase
        if static_phase := ctx.get_phase("static_analysis"):
            sa = static_phase.data
            input_types = sa.get("input_types", [])
            canvas_game_flag = (
                sa.get("has_canvas", False)
                and (sa.get("has_requestanimationframe", False) or sa.get("has_threejs", False))
            )
            is_game = "keyboard" in input_types or canvas_game_flag
        # Structural data from render_test phase
        if render_res:
            sv_raf_calls = int(render_res.data.get("structural_raf_calls_2s", 0))
            sv_overlays = render_res.data.get("structural_visible_overlays", [])
            sv_listeners = int(render_res.data.get("structural_event_listener_count", 0))
        # Reliable issues from phase data
        reliable_issues = []
        if console_err:
            reliable_issues = [f"Console error: {e}" for e in console_err[:3]]
        if has_agent_data and bugs:
            reliable_issues.extend(bugs[:3])
        # Preservation from VLM highlights (less reliable, but best we have)
        preservation = list(highlights)

    # If no bugs found, supplement from console errors (only objective ones)
    if not bugs and console_err:
        bugs = [f"Console error: {e}" for e in console_err[:3]]

    diag = Diagnosis(
        score=total,
        rendering=scores["rendering"],
        visual_design=scores["visual_design"],
        functionality=scores["functionality"],
        interaction=scores["interaction"],
        code_quality=scores["code_quality"],
        tier=_tier(total),
        bugs=bugs,
        missing_features=missing,
        highlights=highlights,
        summary=summary,
        weak_dims=_weak_dims(scores),
        render_ok=render_ok,
        has_agent_data=has_agent_data,
        console_errors=console_err,
        # Evidence fields
        evidence_quality=evidence_quality,
        keyboard_broken=keyboard_broken,
        keyboard_verified=keyboard_verified,
        discovered_keys=discovered_keys,
        game_vars_initial=game_vars_initial,
        reliable_issues=reliable_issues,
        preservation_list=preservation,
        observer_report=observer_report,
        task_auditor_report=task_auditor_report,
        # Game detection fields
        is_game=is_game,
        canvas_game=canvas_game_flag,
        structural_raf_calls_2s=sv_raf_calls,
        structural_visible_overlays=sv_overlays,
        structural_event_listener_count=sv_listeners,
    )
    diag.strategy = _select_strategy(
        diag, prev_strategy=prev_strategy,
        consecutive_holistic=consecutive_holistic,
        total_fix_playability=total_fix_playability,
        blacklist=strategy_blacklist,
    )
    return diag
