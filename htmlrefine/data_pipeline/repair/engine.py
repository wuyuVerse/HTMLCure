"""
RepairEngine — progressive iterative HTML improvement engine.

Handles both repair (Tier B/C, score < 80) and refinement (Tier A, score 80-94)
in a single continuous loop. Strategies adapt automatically as score crosses
tier boundaries — a B-tier record that reaches 80+ seamlessly transitions to
A-tier dimension-targeted refinement.

Design principles (data-driven):
  1. Evidence quality gating (Tier B/C): only use surgical strategies (bug_fix,
     feature_complete) when objective evidence confirms the issues.
       - bug_fix on inferred evidence:  22% success, avg Δ = -5.4  (harmful)
       - holistic_rewrite (any quality): 76% success, avg Δ = +14.5 (consistently works)

  2. Dimension targeting (Tier A): VLM scores are reliable at 80+, so target the
     weakest dimension directly (polish_visual, enhance_interaction, etc.).

  3. Evidence refresh: after each iteration's quick-eval, re-collect evidence from
     the updated ctx. This keeps strategy selection current as the HTML evolves.

  4. Layered composite anti-regression score: penalizes dimension regressions
     with weights that vary by original score tier.

  5. Rejection sampling: generate n_candidates in parallel for score < 70,
     keep the best by composite score.

  6. Convergence-based stopping: no premature quality gate truncation. Loop
     continues until 2 consecutive low-delta iterations (natural convergence),
     near-perfect score (≥97), or max_iterations reached.

Flow per record:
  1. collect_evidence(ctx)               → Evidence (quality: high/medium/low)
  2. extract_diagnosis(ctx, evidence)    → Diagnosis (strategy auto-routes by score)
  3. Generate n_candidates repairs       → parallel LLM calls
  4. Quick-eval all candidates           → pick best by composite score
  5. Refresh evidence from updated ctx   → next iteration sees fresh signals
  6. Convergence check (2 consecutive low-delta → stop)
  7. Repeat up to max_iterations
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext
from htmleval.core.pipeline import PipelineEngine
from htmlrefine.core.config import RepairConfig
from htmlrefine.data_pipeline.repair.feedback.contrastive import (
    ContrastiveReport, generate_contrastive_feedback, format_contrastive_feedback,
)
from htmlrefine.data_pipeline.repair.feedback.visual_diagnosis import (
    diagnose_visual_issues, verify_visual_change, VisualDiagnosis,
)
from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis, extract_diagnosis, _DIM_MAX
from htmlrefine.data_pipeline.repair.core.evidence import Evidence, collect_evidence
from htmlrefine.data_pipeline.repair.prompts import format_visual_context, format_visual_diagnosis
from htmlrefine.data_pipeline.repair.strategies import ALL_STRATEGIES

logger = logging.getLogger("htmlrefine.repair")


# ---------------------------------------------------------------------------
# Markdown wrapper stripping
# ---------------------------------------------------------------------------

_DOCTYPE_OR_HTML = re.compile(r"(<!DOCTYPE\b|<html[\s>])", re.IGNORECASE)
_FENCE_HTML = re.compile(r"```html?\s*\n", re.IGNORECASE)


def _strip_markdown_wrapper(html: str) -> str:
    """Strip markdown description / ```html fences wrapping raw HTML.

    Handles:
      1. Already clean HTML (starts with <!DOCTYPE or <html>) → return as-is
      2. Has ```html fence → strip fence + prefix text + trailing ```
      3. No fence but embedded <!DOCTYPE or <html> → strip prefix text
      4. None of the above → return as-is
    """
    stripped = html.lstrip()
    # Case 1: already clean
    if _DOCTYPE_OR_HTML.match(stripped):
        return html

    # Case 2: has ```html fence
    m = _FENCE_HTML.search(html)
    if m:
        inner = html[m.end():]
        # Strip trailing ``` (may have whitespace after)
        tail = inner.rstrip()
        if tail.endswith("```"):
            inner = tail[:-3].rstrip()
        return inner

    # Case 3: no fence but has embedded HTML start
    m = _DOCTYPE_OR_HTML.search(html)
    if m:
        return html[m.start():]

    # Case 4: no recognizable HTML
    return html


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    iteration: int
    strategy: str
    score_before: int
    score_after: int
    delta: int
    html_before: str
    html_after: str
    elapsed_s: float
    success: bool          # False if LLM failed or HTML invalid
    n_candidates: int = 1  # how many candidates were generated
    error: str = ""
    composite_before: int = 0   # composite anti-regression score before
    composite_after: int = 0    # composite anti-regression score after
    dim_deltas: Dict[str, int] = field(default_factory=dict)  # per-dimension score changes
    contrastive_summary: str = ""  # formatted contrastive visual feedback text


@dataclass
class RepairResult:
    record_id: str
    query: str
    original_score: int
    final_score: int
    improvement: int
    best_html: str
    original_html: str
    iterations: List[IterationResult] = field(default_factory=list)
    converged: bool = False    # stopped early (improvement < threshold or quality gate)
    elapsed_s: float = 0.0
    sft_eligible: bool = False  # True if quality gate passed
    evidence_quality: str = "low"  # for analytics


# ---------------------------------------------------------------------------
# Composite anti-regression score
# ---------------------------------------------------------------------------

def _get_dim_scores(ctx: EvalContext) -> Dict[str, int]:
    """Extract per-dimension scores from ctx.final_score."""
    fs = ctx.final_score or {}
    result = {}
    for k in _DIM_MAX:
        v = fs.get(k, {})
        result[k] = v.get("score", 0) if isinstance(v, dict) else int(v or 0)
    return result


def _composite_score(ctx: EvalContext, original_scores: Dict[str, int]) -> int:
    """
    Layered composite score that penalizes dimension regressions.

    Penalty structure depends on original score tier:
      - original < 85: func×3, interaction×3, render×2, visual×2
      - original ≥ 85: func×2, interaction×2, render×1, visual×1, code×1

    Prevents "robbing Peter to pay Paul" repairs where one dimension improves
    at the expense of others.
    """
    new_scores = _get_dim_scores(ctx)
    total = ctx.total_score
    original_total = sum(original_scores.values())

    penalty = 0
    if original_total >= 85:
        for dim, mult in [
            ("functionality", 2),
            ("interaction", 2),
            ("rendering", 1),
            ("visual_design", 1),
            ("code_quality", 1),
        ]:
            regression = max(0, original_scores.get(dim, 0) - new_scores.get(dim, 0))
            if regression <= 2:  # ≤2pt = VLM scoring noise, don't penalize
                regression = 0
            penalty += regression * mult
    else:
        for dim, mult in [
            ("functionality", 3),
            ("interaction", 3),
            ("rendering", 2),
            ("visual_design", 2),
        ]:
            regression = max(0, original_scores.get(dim, 0) - new_scores.get(dim, 0))
            if regression <= 2:  # ≤2pt = VLM scoring noise, don't penalize
                regression = 0
            penalty += regression * mult

    # Cap penalty: when raw score genuinely improved by 3+ points, don't let
    # regression penalty erase the entire gain.  This prevents cases like
    # "score 77→82 (+5) but visual_design -6 → composite 70 < 77 → rejected".
    raw_gain = total - original_total
    if raw_gain >= 3:
        max_penalty = raw_gain // 2  # at most half the gain
        penalty = min(penalty, max_penalty)

    return total - penalty


# ---------------------------------------------------------------------------
# Trajectory context
# ---------------------------------------------------------------------------

def _build_prev_iter_summaries(iterations: List[IterationResult]) -> List[dict]:
    """Convert IterationResult list → compact dicts for trajectory-aware prompts."""
    result = []
    for it in iterations:
        entry: dict = {
            "iteration":    it.iteration,
            "strategy":     it.strategy,
            "score_before": it.score_before,
            "score_after":  it.score_after,
            "delta":        it.delta,
        }
        if it.delta > 5:
            entry["what_improved"] = f"Score rose by {it.delta} points"
        elif it.delta > 0:
            entry["what_improved"] = f"Minor improvement (+{it.delta})"
        else:
            entry["what_improved"] = "No improvement"

        if it.delta < 5 and it.success:
            entry["what_remains"] = (
                "Most issues persist — this iteration made minimal progress; "
                "try a different approach"
            )

        # Per-dimension changes so the next iteration knows what improved/regressed
        if it.dim_deltas:
            changed = {k: v for k, v in it.dim_deltas.items() if v != 0}
            if changed:
                entry["dim_changes"] = changed

        # Contrastive visual feedback from VLM (if available)
        if it.contrastive_summary:
            entry["contrastive"] = it.contrastive_summary

        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RepairEngine:
    """
    Progressive iterative improvement engine for all score tiers.

    Key features:
    - Score-adaptive: auto-routes repair (< 80) vs refine (>= 80) strategies
    - Seamless tier transitions: B→A promotion continues with A-tier refinement
    - Evidence quality gating: prevents harmful surgical repairs on weak evidence
    - Evidence refresh: re-collects evidence each iteration for up-to-date signals
    - Dimension targeting: directly improves weakest dimension for Tier A
    - Convergence-based stopping: 2 consecutive low-delta iterations, no premature gate
    - Layered composite anti-regression: penalty weights vary by original tier
    - Rejection sampling: n_candidates repairs, best by composite score
    - Vision-in-repair: screenshots passed when available
    """

    def __init__(self, config: RepairConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url or None,
            api_key=config.api_key or "empty",
        )
        self.strategies = {
            name: cls()
            for name, cls in ALL_STRATEGIES.items()
            if name in config.strategies
        }

    async def repair(
        self,
        ctx: EvalContext,
        eval_pipeline: PipelineEngine,
        eval_config: EvalConfig,
        max_iterations: Optional[int] = None,
    ) -> RepairResult:
        """
        Run the progressive iterative improvement loop on an already-evaluated EvalContext.

        Automatically adapts strategy as score evolves:
          - score < 80 (repair): evidence-gated strategy, rejection sampling for <70
          - score >= 80 (refine): dimension-targeted strategy, single candidate
          - B→A transition: when repair pushes score past 80, seamlessly switches
            to dimension-targeted refinement in subsequent iterations.

        Stopping conditions (convergence-only, no premature gate truncation):
          - 2 consecutive iterations with delta < improvement_threshold
          - Near-perfect score (≥97)
          - max_iterations reached

        Args:
            max_iterations: Per-record override for iteration limit.
                            If None, uses self.config.max_iterations.

        Returns:
            RepairResult with best HTML, score trajectory, sft_eligible flag.
        """
        t0 = time.time()
        record_id = ctx.game_id
        query     = ctx.query

        original_html   = ctx.html_code or ctx.response or ""
        original_html   = _strip_markdown_wrapper(original_html)
        # Write back cleaned HTML so downstream code sees it consistently
        if ctx.html_code:
            ctx.html_code = original_html
        else:
            ctx.response = original_html
        original_score  = ctx.total_score
        original_scores = _get_dim_scores(ctx)
        best_html       = original_html
        best_score      = original_score

        # Unified composite function (layered by original score internally)
        _comp_fn = _composite_score
        best_composite  = _comp_fn(ctx, original_scores)
        best_ctx        = ctx           # track context of best composite
        current_ctx     = ctx
        iterations: List[IterationResult] = []

        # Effective iteration limit (per-record override or config default)
        effective_max_iters = max_iterations if max_iterations is not None else self.config.max_iterations

        # Tier label for logging (dynamic — updates as score crosses thresholds)
        tier_str = "refine" if original_score >= 80 else "repair"

        # ── Step 1: Collect initial evidence ──────────────────────────────
        # In batch repair context: keyscan won't run (no game_url_http),
        # so evidence comes from phase data + injected VLM scores.
        # Evidence quality gates strategy selection:
        #   low  (VLM only)   → holistic_rewrite always (76% success)
        #   medium (proven)   → targeted fix for confirmed issues
        #   high  (agent ran) → full score-based selection
        evidence = await collect_evidence(ctx, browser_args=[])
        visual_ctx_str = format_visual_context(evidence)
        logger.info(
            f"[{tier_str}:{record_id}] start  score={original_score}  "
            f"evidence={evidence.quality}  max_iters={effective_max_iters}  "
            f"dynamic={'yes' if evidence.dynamic_experience_ran else 'no'}  "
            f"{'keyboard_broken  ' if evidence.keyboard_broken else ''}"
            f"{'keyboard_ok  ' if evidence.keyboard_verified else ''}"
        )

        # Force skip_agent for ALL repair evaluations (too slow, not needed).
        # Set once here — NOT inside _quick_eval — to avoid asyncio race condition:
        # multiple concurrent _quick_eval calls can't safely toggle a shared bool
        # around an 'await' boundary.
        _saved_skip_agent = eval_config.processing.skip_agent_phase
        eval_config.processing.skip_agent_phase = True

        prev_strategy = ""
        consecutive_low_delta = 0  # convergence: stop after 2 consecutive low-delta iters
        contrastive_report: Optional[ContrastiveReport] = None
        contrastive_text: str = ""
        _contrastive_task = None  # background task for non-blocking contrastive
        holistic_rewrite_count = 0  # total holistic_rewrite uses per record
        consecutive_holistic = 0   # consecutive holistic_rewrite counter for anti-stagnation
        total_fix_playability = 0  # total fix_playability uses (capped at 2)
        strategy_blacklist: set = set()  # strategies that caused catastrophic regressions (Δ < -8)
        iters_since_best: int = 0      # post-peak patience: stop after 2 iters without improving best
        force_visual_diagnosis: bool = False  # set on plateau to trigger VLM diagnosis

        try:
            # Expand short/vague queries with VLM-generated requirements
            repair_query = await self._expand_short_query(query, ctx)
            for i in range(1, effective_max_iters + 1):
                iter_t0 = time.time()
                diag    = extract_diagnosis(
                    current_ctx,
                    prev_strategy=prev_strategy,
                    evidence=evidence,
                    consecutive_holistic=consecutive_holistic,
                    total_fix_playability=total_fix_playability,
                    strategy_blacklist=strategy_blacklist,
                )
                strategy = self.strategies.get(diag.strategy)

                if strategy is None:
                    # Fallback to holistic_rewrite when selected strategy not enabled
                    strategy = self.strategies.get("holistic_rewrite")
                    if strategy is None:
                        logger.warning(
                            f"[{tier_str}:{record_id}] strategy '{diag.strategy}' "
                            f"not enabled, no fallback available"
                        )
                        break
                    logger.info(
                        f"[{tier_str}:{record_id}] strategy '{diag.strategy}' "
                        f"not enabled, falling back to holistic_rewrite"
                    )
                    diag.strategy = "holistic_rewrite"

                # Cap holistic_rewrite at 2 uses for small-enough files;
                # switch to feature_complete or bug_fix for variety.
                # Skip at 80+: let _dimension_targeted() control strategy selection
                # (data: feature_complete at 90+ avg Δ=-10.8, 480 catastrophes).
                if diag.strategy == "holistic_rewrite" and holistic_rewrite_count >= 2 \
                   and diag.score < 80 \
                   and len(
                    (current_ctx.html_code or current_ctx.response or "")
                ) <= 35000:
                    fallback_name = "feature_complete" if self.strategies.get("feature_complete") else "bug_fix"
                    fallback_strat = self.strategies.get(fallback_name)
                    if fallback_strat:
                        logger.info(
                            f"[{tier_str}:{record_id}] holistic_rewrite used {holistic_rewrite_count}x, "
                            f"switching to {fallback_name}"
                        )
                        strategy = fallback_strat
                        diag.strategy = fallback_name

                # Screenshots + frame annotations for multimodal repair
                screenshots: Optional[list] = None
                frame_annotations: Optional[list] = None
                if self.config.vision_in_repair and current_ctx.all_screenshots:
                    screenshots = current_ctx.all_screenshots
                    render_phase = current_ctx.get_phase("render_test")
                    if render_phase:
                        frame_annotations = render_phase.data.get("frame_annotations", [])

                prev_iter_summaries = _build_prev_iter_summaries(iterations)

                logger.info(
                    f"[{tier_str}:{record_id}] iter {i}/{effective_max_iters}  "
                    f"strategy={diag.strategy}  score={diag.score}  "
                    f"evidence={diag.evidence_quality}  "
                    f"screenshots={len(screenshots) if screenshots else 0}"
                )

                html_before      = current_ctx.html_code or current_ctx.response or ""
                composite_before = _comp_fn(current_ctx, original_scores)

                # Large files (>35k chars): patch strategies fail 96-100% of the time,
                # auto-route to a rewrite-mode strategy.
                # At 80+: use visual_enrichment (VLM-guided, zero catastrophe rate)
                # Below 80: use holistic_rewrite (aggressive but effective at low scores)
                if len(html_before) > 35000 and strategy.mode == "patch":
                    if diag.score >= 80:
                        fallback = self.strategies.get("visual_enrichment") or self.strategies.get("holistic_rewrite")
                        fallback_name = "visual_enrichment" if self.strategies.get("visual_enrichment") else "holistic_rewrite"
                    else:
                        fallback = self.strategies.get("holistic_rewrite")
                        fallback_name = "holistic_rewrite"
                    if fallback:
                        logger.info(
                            f"[{tier_str}:{record_id}] large file ({len(html_before)} chars), "
                            f"switching {diag.strategy} (patch) to {fallback_name}"
                        )
                        strategy = fallback
                        diag.strategy = fallback_name

                # Rejection sampling: more candidates when score is low (harder cases)
                # Score >= 90: single candidate (refinement is precise)
                # Score 70-89: 2 candidates (moderate diversity for Tier A/B)
                # Score < 70: n_candidates (3 for hardest cases)
                if diag.score < 70:
                    n = self.config.n_candidates
                elif diag.score < 90:
                    n = 2
                else:
                    n = 1
                # VLM visual diagnosis: triggered on plateau or for high-score records
                visual_diagnosis_text = ""
                should_diagnose = (
                    force_visual_diagnosis
                    or (diag.score >= 75 and self.config.vision_in_repair and current_ctx.all_screenshots)
                )
                if should_diagnose and self.config.vision_in_repair and current_ctx.all_screenshots:
                    try:
                        vis_diag = await diagnose_visual_issues(
                            current_ctx, query, self.client, self.config.model,
                        )
                        if vis_diag and vis_diag.issues:
                            visual_diagnosis_text = format_visual_diagnosis(vis_diag)
                    except Exception as e:
                        logger.warning(f"[{tier_str}:{record_id}] VLM diagnosis failed: {e}")
                # Always reset — even if vision wasn't available, the plateau was acknowledged
                if force_visual_diagnosis:
                    force_visual_diagnosis = False

                # Combine VLM diagnosis + contrastive feedback + visual context
                combined_visual_ctx = ""
                if visual_diagnosis_text:
                    combined_visual_ctx += visual_diagnosis_text + "\n\n"
                if contrastive_text:
                    combined_visual_ctx += contrastive_text + "\n\n"
                if visual_ctx_str:
                    combined_visual_ctx += visual_ctx_str
                candidate_htmls = await self._generate_candidates(
                    n, strategy, html_before, repair_query, diag, screenshots,
                    frame_annotations=frame_annotations,
                    prev_iterations=prev_iter_summaries,
                    visual_context=combined_visual_ctx,
                )

                if not candidate_htmls:
                    # Patch failure fallback: retry once with holistic_rewrite
                    if strategy.mode == "patch":
                        rewrite_strategy = self.strategies.get("holistic_rewrite")
                        if rewrite_strategy:
                            logger.info(
                                f"[{tier_str}:{record_id}] iter {i} — patch failed, "
                                f"retrying with holistic_rewrite"
                            )
                            candidate_htmls = await self._generate_candidates(
                                1, rewrite_strategy, html_before, repair_query, diag, screenshots,
                                frame_annotations=frame_annotations,
                                prev_iterations=prev_iter_summaries,
                                visual_context=combined_visual_ctx,
                            )
                            if candidate_htmls:
                                diag.strategy = "holistic_rewrite"
                    if not candidate_htmls:
                        itr = IterationResult(
                            iteration=i, strategy=diag.strategy,
                            score_before=diag.score, score_after=diag.score, delta=0,
                            html_before=html_before, html_after="",
                            elapsed_s=time.time() - iter_t0, success=False,
                            n_candidates=n, error="LLM returned no valid HTML",
                            composite_before=composite_before, composite_after=composite_before,
                        )
                        iterations.append(itr)
                        logger.warning(f"[{tier_str}:{record_id}] iter {i} — no valid candidates, stopping")
                        break

                # Quick-eval all candidates; pick best by composite anti-regression score
                best_ctx_candidate, best_candidate_html = await self._eval_best_candidate(
                    candidate_htmls, query, record_id, i,
                    eval_pipeline, original_scores, _comp_fn,
                )
                new_score     = best_ctx_candidate.total_score
                new_composite = _comp_fn(best_ctx_candidate, original_scores)
                delta         = new_score - diag.score

                logger.info(
                    f"[{tier_str}:{record_id}] iter {i}  "
                    f"{diag.score} → {new_score}  Δ={delta:+d}  "
                    f"composite: {composite_before} → {new_composite}  "
                    f"candidates={len(candidate_htmls)}  elapsed={time.time()-iter_t0:.1f}s"
                )

                # Per-dimension deltas for trajectory context
                before_dims = _get_dim_scores(current_ctx)
                after_dims = _get_dim_scores(best_ctx_candidate)
                dim_deltas = {k: after_dims[k] - before_dims[k] for k in _DIM_MAX}

                itr = IterationResult(
                    iteration=i, strategy=diag.strategy,
                    score_before=diag.score, score_after=new_score, delta=delta,
                    html_before=html_before, html_after=best_candidate_html,
                    elapsed_s=time.time() - iter_t0, success=True,
                    n_candidates=len(candidate_htmls),
                    composite_before=composite_before, composite_after=new_composite,
                    dim_deltas=dim_deltas,
                )
                iterations.append(itr)

                # Anti-regression: keep globally best by COMPOSITE score
                if new_composite > best_composite:
                    best_composite = new_composite
                    best_score     = new_score
                    best_html      = best_candidate_html
                    best_ctx       = best_ctx_candidate  # track best context for restart
                    iters_since_best = 0
                else:
                    # Regression — will restart from best HTML next iteration
                    iters_since_best += 1
                    logger.info(
                        f"[{tier_str}:{record_id}] regression (composite {new_composite} <= {best_composite}), "
                        f"restarting from best (score={best_score})"
                    )

                # ── Contrastive visual feedback (non-blocking) ─────────
                # Compare before/after screenshots via VLM to identify what
                # improved, regressed, or remains broken. The formatted text
                # is injected into the NEXT iteration's repair prompt.
                # Runs as background task to avoid blocking the next iteration.
                if _contrastive_task is not None:
                    # Harvest previous iteration's contrastive result
                    try:
                        contrastive_report = await asyncio.wait_for(_contrastive_task, timeout=5.0)
                        contrastive_text = format_contrastive_feedback(contrastive_report) if contrastive_report else ""
                    except Exception:
                        contrastive_text = ""
                        contrastive_report = None
                    _contrastive_task = None

                    # Contrastive regression → blacklist the strategy that caused it.
                    # This prevents cycling: e.g., "form broken" after feature_complete
                    # should stop the engine from picking feature_complete again.
                    if contrastive_report and contrastive_report.regressed:
                        strategy_blacklist.add(prev_strategy)
                        logger.info(
                            f"[{tier_str}:{record_id}] contrastive found "
                            f"{len(contrastive_report.regressed)} regressions after "
                            f"'{prev_strategy}', blacklisting it"
                        )

                if self.config.contrastive_feedback and self.config.vision_in_repair:
                    _prev_ctx = current_ctx  # capture for closure
                    _itr = itr

                    async def _contrastive_bg(prev_ctx, after_ctx, _query, _i, _itr_ref):
                        try:
                            report = await generate_contrastive_feedback(
                                before_ctx=prev_ctx,
                                after_ctx=after_ctx,
                                query=_query, iteration=_i,
                                client=self.client, model=self.config.model,
                            )
                            if report:
                                _itr_ref.contrastive_summary = format_contrastive_feedback(report)
                                logger.info(
                                    f"[{tier_str}:{record_id}] contrastive iter {_i}: "
                                    f"improved={len(report.improved)} "
                                    f"regressed={len(report.regressed)} "
                                    f"unchanged={len(report.unchanged_issues)}"
                                )
                            return report
                        except Exception as e:
                            logger.warning(f"[{tier_str}:{record_id}] contrastive failed: {e}")
                            return None

                    _contrastive_task = asyncio.create_task(
                        _contrastive_bg(_prev_ctx, best_ctx_candidate, query, i, _itr)
                    )

                prev_strategy = diag.strategy
                # Blacklist strategies that cause catastrophic regressions
                if delta < -8:
                    strategy_blacklist.add(diag.strategy)
                    logger.info(
                        f"[{tier_str}:{record_id}] blacklisted '{diag.strategy}' "
                        f"(Δ={delta:+d} < -8)"
                    )
                if diag.strategy == "holistic_rewrite":
                    holistic_rewrite_count += 1
                    consecutive_holistic += 1
                else:
                    consecutive_holistic = 0
                if diag.strategy == "fix_playability":
                    total_fix_playability += 1
                # Always restart from best — prevents cascading degradation
                # when an iteration regresses (e.g., 61→57→48)
                current_ctx   = best_ctx

                # ── Refresh evidence from updated ctx ─────────────────────
                # After quick-eval, current_ctx has fresh phase data (render_test,
                # static_analysis, vision_eval). Re-collecting evidence ensures
                # strategy selection in the next iteration uses up-to-date signals.
                # This is critical for B→A transitions: after holistic_rewrite,
                # the new HTML may have different render_test results, keyboard
                # probe data, dynamic experience, etc.
                evidence = await collect_evidence(current_ctx, browser_args=[])
                visual_ctx_str = format_visual_context(evidence)

                # Update tier label as score may have crossed thresholds
                tier_str = "refine" if new_score >= 80 else "repair"

                # ── Convergence-based stopping ────────────────────────────
                # No premature quality gate — iterate until natural convergence.
                # This allows B→A transitions and continued refinement.

                # Playability override: keep going if keyboard broken + interaction low
                new_interaction = _get_dim_scores(best_ctx).get("interaction", 0)
                if evidence.keyboard_broken and new_interaction < 20:
                    logger.info(
                        f"[{tier_str}:{record_id}] keyboard broken, "
                        f"interaction={new_interaction} < 20 — forcing continue"
                    )
                    consecutive_low_delta = 0
                    continue

                # Post-peak plateau: if 2 iterations without improving best,
                # force VLM diagnosis + strategy switch instead of hard stop
                if iters_since_best >= 2:
                    force_visual_diagnosis = True
                    iters_since_best = 0  # reset
                    consecutive_low_delta = 0  # reset so convergence doesn't kill us before VLM runs
                    logger.info(
                        f"[{tier_str}:{record_id}] plateau detected "
                        f"(2 iters without improving best={best_score}), "
                        f"forcing VLM diagnosis + strategy switch"
                    )

                # Near-perfect: minimal room for improvement
                if new_score >= 97:
                    logger.info(
                        f"[{tier_str}:{record_id}] near-perfect "
                        f"({new_score} ≥ 97), stopping"
                    )
                    iterations[-1].error = "near_perfect"
                    break

                # Convergence: consecutive NON-IMPROVING iterations.
                # Score-adaptive patience:
                #   score < 80: 3 consecutive (data: 800 records stopped too early at 2)
                #   score 80-89: 2 consecutive
                #   score 90+: 2 consecutive (but lower delta threshold)
                effective_threshold = self.config.improvement_threshold  # default 2.0
                if new_score >= 90:
                    effective_threshold = 0.5  # 90+ every point is valuable
                elif new_score >= 85:
                    effective_threshold = 1.0  # 85-89 slightly relaxed

                # Patience: how many consecutive low-delta before stopping
                convergence_patience = 3 if new_score < 80 else 2

                if delta < effective_threshold:
                    consecutive_low_delta += 1
                    if consecutive_low_delta >= convergence_patience:
                        logger.info(
                            f"[{tier_str}:{record_id}] converged "
                            f"({convergence_patience} consecutive Δ < {effective_threshold})"
                        )
                        iterations[-1].error = "converged"
                        break
                    logger.info(
                        f"[{tier_str}:{record_id}] low delta "
                        f"(Δ={delta:+d}), {consecutive_low_delta}/{convergence_patience} — continuing"
                    )
                else:
                    consecutive_low_delta = 0

            # ── Visual enrichment (post-convergence, score >= 70) ─────────
            if (
                best_score >= 70
                and self.config.visual_enrichment
                and self.config.vision_in_repair
            ):
                ve_result = await self._run_visual_enrichment(
                    current_ctx=current_ctx,
                    best_html=best_html,
                    best_score=best_score,
                    best_composite=best_composite,
                    query=repair_query,
                    record_id=record_id,
                    eval_pipeline=eval_pipeline,
                    original_scores=original_scores,
                    evidence=evidence,
                    iterations=iterations,
                )
                if ve_result:
                    best_html, best_score, best_composite, current_ctx = ve_result

        finally:
            # Cancel any pending contrastive background task
            if _contrastive_task is not None and not _contrastive_task.done():
                _contrastive_task.cancel()
            # Restore skip_agent_phase to its value before repair started.
            eval_config.processing.skip_agent_phase = _saved_skip_agent

        improvement  = best_score - original_score
        sft_eligible = (
            best_score >= self.config.quality_gate_score
            and improvement >= self.config.min_improvement_for_sft
        )
        stop_reason = iterations[-1].error if iterations else "no_iterations"


        result = RepairResult(
            record_id=record_id,
            query=query,
            original_score=original_score,
            final_score=best_score,
            improvement=improvement,
            best_html=best_html,
            original_html=original_html,
            iterations=iterations,
            converged=stop_reason in ("converged", "near_perfect"),
            elapsed_s=time.time() - t0,
            sft_eligible=sft_eligible,
            evidence_quality=evidence.quality,
        )
        logger.info(
            f"[{tier_str}:{record_id}] done  "
            f"{original_score} → {best_score}  Δ={improvement:+d}  "
            f"iters={len(iterations)}  stop={stop_reason}  "
            f"sft={'✓' if sft_eligible else '✗'}  elapsed={result.elapsed_s:.1f}s"
        )
        return result

    # -----------------------------------------------------------------------
    # Rejection sampling helpers
    # -----------------------------------------------------------------------

    async def _generate_candidates(
        self,
        n: int,
        strategy,
        html: str,
        query: str,
        diag: Diagnosis,
        screenshots: Optional[list],
        frame_annotations: Optional[list] = None,
        prev_iterations: Optional[list] = None,
        visual_context: str = "",
    ) -> List[str]:
        """Generate n repair candidates in parallel; return valid ones."""
        tasks = [
            strategy.repair(
                html=html, query=query, diag=diag,
                client=self.client, model=self.config.model,
                screenshots=screenshots,
                frame_annotations=frame_annotations,
                prev_iterations=prev_iterations,
                visual_context=visual_context,
            )
            for _ in range(n)
        ]
        results = await asyncio.gather(*tasks)
        valid = [r for r in results if r]
        if n > 1:
            logger.info(f"  rejection sampling: {len(valid)}/{n} valid candidates")
        return valid

    async def _eval_best_candidate(
        self,
        candidates: List[str],
        query: str,
        record_id: str,
        iteration: int,
        pipeline: PipelineEngine,
        original_scores: Dict[str, int],
        composite_fn=_composite_score,
    ) -> tuple[EvalContext, str]:
        """
        Quick-eval all candidates concurrently; return (best_ctx, best_html).

        Uses composite anti-regression score to select best candidate
        rather than raw total_score.
        """
        tasks = [
            self._quick_eval(html, query, record_id, iteration, idx, pipeline)
            for idx, html in enumerate(candidates)
        ]
        ctxs = await asyncio.gather(*tasks)

        # Select by composite score (penalizes dimension regressions)
        composites = [composite_fn(c, original_scores) for c in ctxs]
        best_idx   = max(range(len(composites)), key=lambda i: composites[i])

        if len(candidates) > 1:
            totals = [c.total_score for c in ctxs]
            logger.info(
                f"  candidate totals: {totals}  composites: {composites}"
                f"  → best={totals[best_idx]} (composite={composites[best_idx]})"
            )
        return ctxs[best_idx], candidates[best_idx]

    async def _quick_eval(
        self,
        html: str,
        query: str,
        record_id: str,
        iteration: int,
        candidate_idx: int,
        pipeline: PipelineEngine,
    ) -> EvalContext:
        """
        Run a fast re-evaluation on one repaired HTML candidate.

        Wraps pipeline.evaluate() with a 300s timeout to prevent individual
        evaluations from hanging indefinitely (e.g. browser pool starvation).
        """
        suffix = f"_c{candidate_idx}" if candidate_idx > 0 else ""
        new_ctx = EvalContext(
            query=query,
            response=html,
            game_id=f"{record_id}_repair_{iteration}{suffix}",
            variant="repair",
        )
        new_ctx = await asyncio.wait_for(pipeline.evaluate(new_ctx), timeout=600)
        return new_ctx

    # -----------------------------------------------------------------------
    # Visual enrichment (post-convergence VLM-driven visual improvement)
    # -----------------------------------------------------------------------

    async def _run_visual_enrichment(
        self,
        current_ctx: EvalContext,
        best_html: str,
        best_score: int,
        best_composite: int,
        query: str,
        record_id: str,
        eval_pipeline: PipelineEngine,
        original_scores: Dict[str, int],
        evidence: Evidence,
        iterations: List[IterationResult],
    ) -> Optional[tuple[str, int, int, EvalContext]]:
        """
        VLM-driven visual enrichment loop. Runs after main repair converges at score >= 80.

        Flow per iteration:
          1. VLM diagnoses visual issues from screenshot
          2. If no issues → stop (visually good enough)
          3. VisualEnrichmentStrategy generates CSS/HTML-only improvement
          4. Quick-eval the candidate
          5. VLM verifies: improved AND no regression → accept
          6. Otherwise → rollback and stop

        Returns:
            (best_html, best_score, best_composite, current_ctx) if improved, else None.
        """
        strategy = self.strategies.get("visual_enrichment")
        if not strategy:
            return None

        max_vi = self.config.visual_enrichment_max_iters
        logger.info(
            f"[visual_enrichment:{record_id}] starting — "
            f"score={best_score}  max_iters={max_vi}"
        )

        ve_best_html = best_html
        ve_best_score = best_score
        ve_best_composite = best_composite
        ve_ctx = current_ctx
        improved_any = False

        for vi in range(1, max_vi + 1):
            iter_t0 = time.time()

            # Step 1: VLM diagnoses visual issues
            logger.info(f"[visual_enrichment:{record_id}] iter {vi}/{max_vi} — diagnosing...")
            diagnosis = await diagnose_visual_issues(
                ve_ctx, query, self.client, self.config.model,
            )
            if not diagnosis.issues:
                logger.info(
                    f"[visual_enrichment:{record_id}] iter {vi} — "
                    f"no visual issues found, stopping"
                )
                break

            # Step 2: Build diagnosis text for the strategy prompt
            diag_text = format_visual_diagnosis(diagnosis)

            # Step 3: Generate visual enrichment candidate
            diag = extract_diagnosis(ve_ctx, evidence=evidence)
            # Override strategy to visual_enrichment
            diag.strategy = "visual_enrichment"

            # The VisualEnrichmentStrategy.build_prompt accepts visual_diagnosis_text
            # but the base repair() calls build_prompt without it.
            # We pass it via visual_context which gets prepended to the prompt.
            screenshots = ve_ctx.all_screenshots if self.config.vision_in_repair else None
            frame_annotations = None
            if screenshots:
                render_phase = ve_ctx.get_phase("render_test")
                if render_phase:
                    frame_annotations = render_phase.data.get("frame_annotations", [])

            prev_summaries = _build_prev_iter_summaries(iterations)
            candidate_htmls = await self._generate_candidates(
                1, strategy, ve_best_html, query, diag, screenshots,
                frame_annotations=frame_annotations,
                prev_iterations=prev_summaries,
                visual_context=diag_text,
            )

            if not candidate_htmls:
                logger.warning(
                    f"[visual_enrichment:{record_id}] iter {vi} — "
                    f"no valid candidate, stopping"
                )
                break

            # Step 4: Quick-eval
            new_ctx = await self._quick_eval(
                candidate_htmls[0], query, record_id,
                len(iterations) + vi, 0, eval_pipeline,
            )
            new_score = new_ctx.total_score
            new_composite = _composite_score(new_ctx, original_scores)

            logger.info(
                f"[visual_enrichment:{record_id}] iter {vi} — "
                f"score: {ve_best_score} → {new_score}  "
                f"composite: {ve_best_composite} → {new_composite}  "
                f"elapsed={time.time() - iter_t0:.1f}s"
            )

            # Step 5: VLM verification
            verification = await verify_visual_change(
                ve_ctx, new_ctx, query, self.client, self.config.model,
            )

            if verification.improved and not verification.functional_regression:
                # Step 6a: Accept if composite doesn't regress
                if new_composite >= ve_best_composite:
                    ve_best_html = candidate_htmls[0]
                    ve_best_score = new_score
                    ve_best_composite = new_composite
                    ve_ctx = new_ctx
                    improved_any = True

                    itr = IterationResult(
                        iteration=len(iterations) + vi,
                        strategy="visual_enrichment",
                        score_before=best_score if vi == 1 else ve_best_score,
                        score_after=new_score,
                        delta=new_score - (best_score if vi == 1 else ve_best_score),
                        html_before=best_html if vi == 1 else ve_best_html,
                        html_after=candidate_htmls[0],
                        elapsed_s=time.time() - iter_t0,
                        success=True,
                    )
                    iterations.append(itr)

                    logger.info(
                        f"[visual_enrichment:{record_id}] iter {vi} — "
                        f"ACCEPTED (VLM: improved, no regression)"
                    )
                else:
                    logger.info(
                        f"[visual_enrichment:{record_id}] iter {vi} — "
                        f"REJECTED (composite regression: {ve_best_composite} → {new_composite})"
                    )
                    break
            else:
                # Step 6b: VLM says worse or regression → stop
                logger.info(
                    f"[visual_enrichment:{record_id}] iter {vi} — "
                    f"REJECTED (VLM: improved={verification.improved}, "
                    f"regression={verification.functional_regression})"
                )
                break

        if improved_any:
            logger.info(
                f"[visual_enrichment:{record_id}] done — "
                f"{best_score} → {ve_best_score}  Δ={ve_best_score - best_score:+d}"
            )
            return (ve_best_html, ve_best_score, ve_best_composite, ve_ctx)

        logger.info(f"[visual_enrichment:{record_id}] done — no improvement")
        return None

    # -----------------------------------------------------------------------
    # Short query expansion
    # -----------------------------------------------------------------------

    async def _expand_short_query(
        self,
        query: str,
        ctx: EvalContext,
    ) -> str:
        """Expand short/vague queries into detailed requirements using VLM.

        If query > 150 chars or no screenshots available, returns as-is.
        Calls VLM with screenshots + prompt to generate 8-12 specific requirements.
        """
        if len(query) > 150:
            return query
        if not self.config.vision_in_repair or not ctx.all_screenshots:
            return query

        try:
            screenshots = ctx.all_screenshots[:3]  # limit to save tokens
            import base64
            content: list = []
            for sc in screenshots:
                if isinstance(sc, (str, Path)):
                    with open(sc, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
                elif isinstance(sc, dict) and "url" in sc:
                    content.append({"type": "image_url", "image_url": sc})

            content.append({
                "type": "text",
                "text": (
                    f"The user asked to build: \"{query}\"\n\n"
                    "Looking at the current implementation screenshots above, "
                    "generate 8-12 specific, concrete requirements that this project should meet. "
                    "Include requirements for:\n"
                    "- Visual design (colors, layout, typography)\n"
                    "- Functionality (what should work, game mechanics if applicable)\n"
                    "- Interaction (controls, buttons, feedback)\n"
                    "- Completeness (win/lose states, scoring, menus)\n\n"
                    "Format as a numbered list. Be specific and actionable."
                ),
            })

            resp = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=800,
                temperature=0.3,
            )
            spec = resp.choices[0].message.content.strip()
            if spec:
                logger.info(f"[query_expansion] expanded {len(query)}-char query → +{len(spec)} chars")
                return f"{query}\n\n## Detailed Requirements\n{spec}"
        except Exception as e:
            logger.warning(f"[query_expansion] failed: {e}")

        return query
