"""
VLM-driven visual diagnosis and verification for visual enrichment.

Two main functions:
  - diagnose_visual_issues(): VLM examines screenshot + query → specific visual issues
  - verify_visual_change(): VLM compares before/after → improved? regression?

Follows the same VLM call pattern as contrastive.py (semaphore, retries, JSON parsing).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from openai import AsyncOpenAI

from htmleval.core.context import EvalContext
from htmlrefine.data_pipeline.repair.strategies import base as _repair_base
from htmlrefine.data_pipeline.repair.strategies.base import _encode_screenshot

logger = logging.getLogger("htmlrefine.repair")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VisualDiagnosis:
    """Output from VLM visual diagnosis."""
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    css_focus_areas: List[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class VisualVerification:
    """Output from VLM before/after visual comparison."""
    improved: bool = False
    functional_regression: bool = False
    improvements: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    raw_response: str = ""


# ---------------------------------------------------------------------------
# VLM prompts
# ---------------------------------------------------------------------------

_DIAGNOSIS_PROMPT = """\
You are a visual quality expert. Examine this HTML page screenshot and identify \
specific visual issues preventing it from reaching professional quality.

## Task Description
{query}

## Current Score
Total: {score}/100 | visual_design: {visual_design}/20 | rendering: {rendering}/20

## Your Job
Identify concrete, actionable visual improvements. Focus on:
- Color palette: bland/clashing? Missing cohesion?
- Typography: generic system fonts? No hierarchy?
- Spacing/layout: cramped or excessive whitespace? Misaligned?
- Visual depth: flat appearance? Missing shadows, gradients, borders?
- Polish: missing hover states, transitions, micro-animations?
- Responsive: does it look good at this viewport?

Do NOT suggest functionality changes — CSS/HTML visual improvements ONLY.

Reply ONLY with this JSON (no markdown fences):
{{"issues": ["specific visual problem 1", ...], \
"suggestions": ["concrete CSS fix 1", ...], \
"css_focus_areas": ["typography", "color_palette", "spacing", "depth", "animation", "responsive"]}}\
"""

_VERIFICATION_PROMPT = """\
You are a visual quality analyst comparing BEFORE and AFTER versions of an HTML page.
The page was modified to improve visual quality. Determine:
1. Did the visual quality actually improve?
2. Was any functionality broken by the change?

## Task Description
{query}

## Score Change
Before: {score_before}/100 | After: {score_after}/100

Look carefully at both screenshots. Compare:
- Visual polish (better or worse?)
- Layout integrity (anything shifted/broken?)
- Content completeness (anything missing?)
- Interactive elements (buttons, controls still visible and properly styled?)

Reply ONLY with this JSON (no markdown fences):
{{"improved": true/false, \
"functional_regression": true/false, \
"improvements": ["what got better"], \
"regressions": ["what got worse or broke"]}}\
"""


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> Optional[dict]:
    """Parse JSON from VLM response, stripping think blocks and fences."""
    clean = _THINK_BLOCK.sub("", raw).strip()
    clean = re.sub(r"^```(?:json)?\s*\n?", "", clean)
    clean = re.sub(r"\n?```\s*$", "", clean)
    clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# VLM call helper (shared pattern with contrastive.py)
# ---------------------------------------------------------------------------

async def _call_vlm(
    client: AsyncOpenAI,
    model: str,
    content: list,
    max_tokens: int = 16384,
    temperature: float = 0,
    timeout: float = 300,
    max_attempts: int = 2,
) -> Optional[str]:
    """Call VLM with retries, respecting global LLM semaphore."""
    for attempt in range(1, max_attempts + 1):
        try:
            if _repair_base._LLM_SEM:
                async with _repair_base._LLM_SEM:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": content}],
                            max_tokens=max_tokens,
                            temperature=temperature,
                        ),
                        timeout=timeout,
                    )
            else:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                    timeout=timeout,
                )
            msg = resp.choices[0].message
            raw = msg.content or getattr(msg, "reasoning_content", None) or ""
            return raw.strip()
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(
                f"visual_diagnosis VLM attempt {attempt}/{max_attempts} failed: "
                f"{type(e).__name__}: {e}"
            )
            if attempt < max_attempts:
                await asyncio.sleep(2)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def diagnose_visual_issues(
    ctx: EvalContext,
    query: str,
    client: AsyncOpenAI,
    model: str,
) -> VisualDiagnosis:
    """
    VLM examines current screenshot + query → concrete visual issues and suggestions.

    Returns VisualDiagnosis (may have empty issues if VLM fails or page looks good).
    """
    t0 = time.time()

    # Get a representative screenshot (prefer "stable" frame)
    screenshots = ctx.all_screenshots or []
    if not screenshots:
        logger.debug("visual_diagnosis: no screenshots available")
        return VisualDiagnosis()

    # Pick the stable/first paint screenshot
    render_phase = ctx.get_phase("render_test")
    frame_annots = render_phase.data.get("frame_annotations", []) if render_phase else []
    chosen_path = screenshots[0]
    for i, fa in enumerate(frame_annots):
        if fa.get("label") in ("stable", "first_paint") and i < len(screenshots):
            chosen_path = screenshots[i]
            break

    # Get scores
    fs = ctx.final_score or {}
    vd = fs.get("visual_design", {})
    vd_score = vd.get("score", 0) if isinstance(vd, dict) else int(vd or 0)
    r = fs.get("rendering", {})
    r_score = r.get("score", 0) if isinstance(r, dict) else int(r or 0)

    prompt_text = _DIAGNOSIS_PROMPT.format(
        query=query,
        score=ctx.total_score,
        visual_design=vd_score,
        rendering=r_score,
    )

    # Encode screenshot
    try:
        b64 = await asyncio.to_thread(_encode_screenshot, chosen_path, 960)
    except Exception as e:
        logger.warning(f"visual_diagnosis: failed to encode screenshot: {e}")
        return VisualDiagnosis()

    content = [
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": "[Current page state]"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]

    raw = await _call_vlm(client, model, content)
    if not raw:
        return VisualDiagnosis()

    data = _parse_json_response(raw)
    if not data:
        logger.warning(f"visual_diagnosis: failed to parse JSON (len={len(raw)})")
        return VisualDiagnosis(raw_response=raw)

    result = VisualDiagnosis(
        issues=data.get("issues", [])[:8],
        suggestions=data.get("suggestions", [])[:8],
        css_focus_areas=data.get("css_focus_areas", [])[:6],
        raw_response=raw,
    )
    logger.info(
        f"visual_diagnosis: {len(result.issues)} issues, "
        f"{len(result.suggestions)} suggestions "
        f"({time.time() - t0:.1f}s)"
    )
    return result


async def verify_visual_change(
    before_ctx: EvalContext,
    after_ctx: EvalContext,
    query: str,
    client: AsyncOpenAI,
    model: str,
) -> VisualVerification:
    """
    VLM compares before/after screenshots → did visual quality improve? Any regression?

    Acceptance gate: improved=True AND functional_regression=False.
    """
    t0 = time.time()

    before_ss = before_ctx.all_screenshots or []
    after_ss = after_ctx.all_screenshots or []
    if not before_ss or not after_ss:
        logger.debug("visual_verification: missing screenshots")
        return VisualVerification()

    # Pick first available screenshot from each
    before_path = before_ss[0]
    after_path = after_ss[0]

    # Try to pick "stable" frames
    for ctx_obj, ss_list in [(before_ctx, before_ss), (after_ctx, after_ss)]:
        rp = ctx_obj.get_phase("render_test")
        if rp:
            annots = rp.data.get("frame_annotations", [])
            for i, fa in enumerate(annots):
                if fa.get("label") in ("stable", "first_paint") and i < len(ss_list):
                    if ctx_obj is before_ctx:
                        before_path = ss_list[i]
                    else:
                        after_path = ss_list[i]
                    break

    prompt_text = _VERIFICATION_PROMPT.format(
        query=query,
        score_before=before_ctx.total_score,
        score_after=after_ctx.total_score,
    )

    # Encode both screenshots
    try:
        b64_before, b64_after = await asyncio.gather(
            asyncio.to_thread(_encode_screenshot, before_path, 768),
            asyncio.to_thread(_encode_screenshot, after_path, 768),
        )
    except Exception as e:
        logger.warning(f"visual_verification: failed to encode screenshots: {e}")
        return VisualVerification()

    content = [
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": "BEFORE:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_before}"}},
        {"type": "text", "text": "AFTER:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_after}"}},
    ]

    raw = await _call_vlm(client, model, content)
    if not raw:
        return VisualVerification()

    data = _parse_json_response(raw)
    if not data:
        logger.warning(f"visual_verification: failed to parse JSON (len={len(raw)})")
        return VisualVerification(raw_response=raw)

    result = VisualVerification(
        improved=bool(data.get("improved", False)),
        functional_regression=bool(data.get("functional_regression", False)),
        improvements=data.get("improvements", [])[:5],
        regressions=data.get("regressions", [])[:5],
        raw_response=raw,
    )
    logger.info(
        f"visual_verification: improved={result.improved} "
        f"regression={result.functional_regression} "
        f"({time.time() - t0:.1f}s)"
    )
    return result
