"""
Contrastive Visual Feedback — VLM-driven before/after comparison for iterative repair.

After each repair iteration's quick-eval, this module:
  1. Pairs before/after screenshots by semantic label (stable↔stable, after_click↔after_click)
  2. Asks a VLM to compare each pair and classify changes as improved/regressed/unchanged
  3. Produces a structured ContrastiveReport
  4. Formats it into text injected into the next iteration's repair prompt

This is the core innovation: giving the repair LLM explicit visual feedback about what
its previous edit fixed, what it broke, and what still needs attention — preventing
oscillation and guiding convergent improvement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from htmleval.core.context import EvalContext

logger = logging.getLogger("htmlrefine.repair")

# Reuse helpers from base.py (same package)
from htmlrefine.data_pipeline.repair.strategies import base as _repair_base
from htmlrefine.data_pipeline.repair.strategies.base import _encode_screenshot

# ---------------------------------------------------------------------------
# Think-block stripping (same regex as base.py, duplicated to avoid circular)
# ---------------------------------------------------------------------------
import re
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# ContrastiveReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveReport:
    """Structured output from VLM contrastive comparison."""
    iteration: int
    score_before: int
    score_after: int
    dim_deltas: Dict[str, int]      # {"rendering": +3, "interaction": -2, ...}
    improved: List[str]             # VLM-identified visual improvements
    regressed: List[str]            # VLM-identified visual regressions
    unchanged_issues: List[str]     # Issues present in both before and after
    priority_fix: str               # What to fix next (VLM recommendation)
    pairs_compared: int
    elapsed_ms: float
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Frame pairing — match before/after screenshots by semantic label
# ---------------------------------------------------------------------------

# Labels ordered by diagnostic value for contrastive comparison
PAIR_PRIORITY = [
    "stable", "after_click", "keyboard", "idle_animation",
    "scroll_bottom", "canvas_click", "hover", "gameplay",
]


def _pair_frames(
    before_screenshots: List[str],
    before_annotations: List[dict],
    after_screenshots: List[str],
    after_annotations: List[dict],
    max_pairs: int = 3,
) -> List[Tuple[str, str, dict, dict]]:
    """
    Pair before/after frames by semantic label for contrastive comparison.

    Returns list of (before_path, after_path, before_annot, after_annot) tuples.
    Matches by label prefix (e.g. "after_click_btn_0" matches "after_click_btn_0").
    Falls back to positional pairing when labels don't match.
    """
    # Build {label: (path, annotation)} dicts
    before_map: Dict[str, Tuple[str, dict]] = {}
    for i, fa in enumerate(before_annotations):
        if i < len(before_screenshots):
            label = fa.get("label", f"frame_{i}")
            before_map[label] = (before_screenshots[i], fa)

    after_map: Dict[str, Tuple[str, dict]] = {}
    for i, fa in enumerate(after_annotations):
        if i < len(after_screenshots):
            label = fa.get("label", f"frame_{i}")
            after_map[label] = (after_screenshots[i], fa)

    pairs: List[Tuple[str, str, dict, dict]] = []
    used_before: set = set()
    used_after: set = set()

    # Pass 1: exact label match by priority
    for priority_prefix in PAIR_PRIORITY:
        if len(pairs) >= max_pairs:
            break
        for b_label, (b_path, b_fa) in before_map.items():
            if b_label in used_before or not b_label.startswith(priority_prefix):
                continue
            # Find matching after label
            for a_label, (a_path, a_fa) in after_map.items():
                if a_label in used_after:
                    continue
                if a_label == b_label or (
                    a_label.startswith(priority_prefix) and b_label.startswith(priority_prefix)
                ):
                    pairs.append((b_path, a_path, b_fa, a_fa))
                    used_before.add(b_label)
                    used_after.add(a_label)
                    break
            if len(pairs) >= max_pairs:
                break

    # Pass 2: positional fallback for remaining slots
    if len(pairs) < max_pairs:
        remaining_before = [
            (i, before_screenshots[i], before_annotations[i] if i < len(before_annotations) else {})
            for i in range(min(len(before_screenshots), len(before_annotations)))
            if before_annotations[i].get("label", f"frame_{i}") not in used_before
        ]
        remaining_after = [
            (i, after_screenshots[i], after_annotations[i] if i < len(after_annotations) else {})
            for i in range(min(len(after_screenshots), len(after_annotations)))
            if after_annotations[i].get("label", f"frame_{i}") not in used_after
        ]
        for (_, b_path, b_fa), (_, a_path, a_fa) in zip(remaining_before, remaining_after):
            if len(pairs) >= max_pairs:
                break
            pairs.append((b_path, a_path, b_fa, a_fa))

    return pairs


# ---------------------------------------------------------------------------
# Dimension delta computation
# ---------------------------------------------------------------------------

_DIM_KEYS = ["rendering", "visual_design", "functionality", "interaction", "code_quality"]


def _compute_dim_deltas(before_ctx: EvalContext, after_ctx: EvalContext) -> Dict[str, int]:
    """Compute per-dimension score deltas between before and after contexts."""
    deltas = {}
    fs_before = before_ctx.final_score or {}
    fs_after = after_ctx.final_score or {}
    for k in _DIM_KEYS:
        b = fs_before.get(k, {})
        a = fs_after.get(k, {})
        b_score = b.get("score", 0) if isinstance(b, dict) else int(b or 0)
        a_score = a.get("score", 0) if isinstance(a, dict) else int(a or 0)
        deltas[k] = a_score - b_score
    return deltas


# ---------------------------------------------------------------------------
# Contrastive VLM prompt
# ---------------------------------------------------------------------------

CONTRASTIVE_PROMPT = """\
You are a visual quality analyst comparing two versions of an HTML page.
The page was modified between BEFORE and AFTER. Your job: identify what \
improved, what regressed, and what problems remain unchanged.

## Task
{query}

## Score change: {score_before} → {score_after} ({delta:+d})
{dim_deltas_str}

## Frame Pairs
{pair_count} paired screenshots below. For each pair, BEFORE is the old version \
and AFTER is the new version at the same interaction state.

Compare the visual differences carefully. For each aspect, classify as:
- IMPROVED: was broken/missing/ugly before → fixed/present/better after
- REGRESSED: was working/good before → broken/worse/missing after
- UNCHANGED: same problem visible in both versions

Focus on functional and visual differences, not minor pixel shifts.

Reply ONLY with this JSON (no markdown fences, no explanation):
{{"improved": ["description1", ...], "regressed": ["description1", ...], \
"unchanged_issues": ["description1", ...], "priority_fix": "single most important thing to fix next"}}\
"""


# ---------------------------------------------------------------------------
# Core async function
# ---------------------------------------------------------------------------

async def generate_contrastive_feedback(
    before_ctx: EvalContext,
    after_ctx: EvalContext,
    query: str,
    iteration: int,
    client: AsyncOpenAI,
    model: str,
    max_pairs: int = 3,
) -> Optional[ContrastiveReport]:
    """
    Generate contrastive visual feedback by comparing before/after screenshots via VLM.

    Returns None on any failure (graceful degradation — repair loop continues without it).
    """
    t0 = time.time()

    # Need screenshots from both contexts
    if not before_ctx.all_screenshots or not after_ctx.all_screenshots:
        logger.debug(f"contrastive: skipping — no screenshots (before={len(before_ctx.all_screenshots)}, after={len(after_ctx.all_screenshots)})")
        return None

    # Get frame annotations from render_test phase
    before_render = before_ctx.get_phase("render_test")
    after_render = after_ctx.get_phase("render_test")
    before_annots = before_render.data.get("frame_annotations", []) if before_render else []
    after_annots = after_render.data.get("frame_annotations", []) if after_render else []

    # Pair frames by semantic label
    pairs = _pair_frames(
        before_ctx.all_screenshots, before_annots,
        after_ctx.all_screenshots, after_annots,
        max_pairs=max_pairs,
    )
    if not pairs:
        logger.debug("contrastive: no frame pairs found")
        return None

    # Compute dimension deltas
    dim_deltas = _compute_dim_deltas(before_ctx, after_ctx)
    score_before = before_ctx.total_score
    score_after = after_ctx.total_score
    delta = score_after - score_before

    dim_deltas_str = "  ".join(
        f"{k}={dim_deltas[k]:+d}" for k in _DIM_KEYS if dim_deltas.get(k, 0) != 0
    )
    if not dim_deltas_str:
        dim_deltas_str = "(no dimension changes)"

    # Build multimodal content: text prompt + interleaved image pairs
    prompt_text = CONTRASTIVE_PROMPT.format(
        query=query,
        score_before=score_before,
        score_after=score_after,
        delta=delta,
        dim_deltas_str=dim_deltas_str,
        pair_count=len(pairs),
    )

    content: list = [{"type": "text", "text": prompt_text}]

    # Encode all images concurrently (smaller size for contrastive — 6 images)
    all_paths = []
    for b_path, a_path, _, _ in pairs:
        all_paths.extend([b_path, a_path])

    encoded = await asyncio.gather(
        *[asyncio.to_thread(_encode_screenshot, p, max_dim=768) for p in all_paths],
        return_exceptions=True,
    )

    # Interleave pairs with captions
    enc_idx = 0
    valid_pairs = 0
    for pair_num, (b_path, a_path, b_fa, a_fa) in enumerate(pairs):
        b_enc = encoded[enc_idx]
        a_enc = encoded[enc_idx + 1]
        enc_idx += 2

        if isinstance(b_enc, Exception) or isinstance(a_enc, Exception):
            logger.warning(f"contrastive: failed to encode pair {pair_num}: "
                          f"before={b_enc if isinstance(b_enc, Exception) else 'ok'}, "
                          f"after={a_enc if isinstance(a_enc, Exception) else 'ok'}")
            continue

        label = b_fa.get("label", a_fa.get("label", f"frame_{pair_num}"))
        desc = b_fa.get("description", a_fa.get("description", ""))

        content.append({"type": "text", "text": f"\n--- Pair {pair_num + 1}: {label} — {desc} ---"})
        content.append({"type": "text", "text": "BEFORE:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b_enc}"},
        })
        content.append({"type": "text", "text": "AFTER:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{a_enc}"},
        })
        valid_pairs += 1

    if valid_pairs == 0:
        logger.debug("contrastive: no valid image pairs after encoding")
        return None

    # Call VLM with retries
    raw_response = ""
    for attempt in range(1, 3):
        try:
            if _repair_base._LLM_SEM:
                async with _repair_base._LLM_SEM:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": content}],
                            max_tokens=32768,
                            temperature=0,
                        ),
                        timeout=300,
                    )
            else:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=32768,
                        temperature=0,
                    ),
                    timeout=300,
                )

            msg = resp.choices[0].message
            raw_response = msg.content or getattr(msg, "reasoning_content", None) or ""
            raw_response = raw_response.strip()
            break

        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"contrastive VLM attempt {attempt}/2 failed: {type(e).__name__}: {e}")
            if attempt < 2:
                await asyncio.sleep(2)
    else:
        return None

    if not raw_response:
        return None

    # Parse JSON (strip think blocks, markdown fences)
    clean = _THINK_BLOCK.sub("", raw_response).strip()
    # Strip markdown JSON fences if present
    clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
    clean = re.sub(r"\n?```\s*$", "", clean)
    clean = clean.strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                logger.warning(f"contrastive: failed to parse JSON from VLM response (len={len(raw_response)})")
                return None
        else:
            logger.warning(f"contrastive: no JSON found in VLM response (len={len(raw_response)})")
            return None

    elapsed_ms = (time.time() - t0) * 1000

    return ContrastiveReport(
        iteration=iteration,
        score_before=score_before,
        score_after=score_after,
        dim_deltas=dim_deltas,
        improved=data.get("improved", [])[:5],
        regressed=data.get("regressed", [])[:5],
        unchanged_issues=data.get("unchanged_issues", [])[:5],
        priority_fix=data.get("priority_fix", ""),
        pairs_compared=valid_pairs,
        elapsed_ms=elapsed_ms,
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# Format for injection into repair prompt
# ---------------------------------------------------------------------------

def format_contrastive_feedback(report: Optional[ContrastiveReport]) -> str:
    """
    Format a ContrastiveReport into text for injection into the repair prompt.

    Returns empty string if report is None.
    """
    if not report:
        return ""

    delta = report.score_after - report.score_before
    lines = [
        f"## Contrastive Visual Feedback (iteration {report.iteration}: "
        f"{report.score_before} → {report.score_after}, {delta:+d})"
    ]

    if report.improved:
        lines.append("\nWhat improved after last repair:")
        for item in report.improved:
            lines.append(f"  + {item}")

    if report.regressed:
        lines.append("\nWhat REGRESSED (MUST fix — do not repeat this mistake):")
        for item in report.regressed:
            lines.append(f"  ! {item}")

    if report.unchanged_issues:
        lines.append("\nIssues still present (focus here):")
        for item in report.unchanged_issues:
            lines.append(f"  - {item}")

    if report.priority_fix:
        lines.append(f"\nPriority for this iteration: {report.priority_fix}")

    return "\n".join(lines)
