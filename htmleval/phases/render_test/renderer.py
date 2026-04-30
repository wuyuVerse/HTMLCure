"""
Phase 2: RenderTestPhase — dynamic experience evaluation.

Four-layer approach that lets the VLM "experience" HTML like a human:
  Layer 1 (Observation):  time-lapse screenshots capturing load + animation
  Layer 2 (Interaction):  scroll, hover, click (burst micro-clips), keyboard, canvas
  Layer 3 (Deep):         5-second gameplay for games (canvas+rAF+keyboard)
  Layer 4 (Responsive):   mobile + tablet viewport screenshots

Each screenshot is an AnnotatedFrame with label + description that flows
through to the VLM prompt, so the evaluator knows what each frame shows.

Probe logic lives in probes.py; evidence aggregation in evidence.py.
This module is the orchestrator only.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from htmleval.concurrency.browser_pool import BrowserPool, _BROWSER_ARGS, VIEWPORT_W, VIEWPORT_H
from htmleval.concurrency.rate_limiter import TokenBucketRateLimiter
from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext, PhaseResult
from htmleval.core.phase import Phase
from htmleval.core.page_safety import install_page_safety
from htmleval.core.screenshot import safe_page_screenshot
from htmleval.phases.render_test.census import run_element_census
from htmleval.phases.render_test.evidence import (
    compute_evidence, annotate_gameplay_smoothness, annotate_interaction_outcomes,
)
from htmleval.phases.render_test.frame_types import AnnotatedFrame, frame_diff_score
from htmleval.phases.render_test.js_snapshot import collect_js_snapshot
from htmleval.phases.render_test.keyframe_selector import select_keyframes
from htmleval.phases.render_test.probes import PROBE_REGISTRY

logger = logging.getLogger("htmleval")

RENDER_RETRY_MAX = 5
RENDER_RETRY_BACKOFF = [5, 10, 15, 20, 30]

_WAIT_STRATEGIES = [
    ("networkidle",      8_000),
    ("load",            12_000),
    ("domcontentloaded", 6_000),
    ("commit",          15_000),
]

_FALLBACK_SCREENSHOT_TIMEOUT_MS = 8_000
_PROBE_TIMEOUT_S = 12
_FALLBACK_TIMEOUT_S = 6
_JS_SNAPSHOT_TIMEOUT_S = 3
_ELEMENT_CENSUS_TIMEOUT_S = 3


def _is_resource_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        isinstance(exc, BlockingIOError)
        or "resource temporarily unavailable" in msg
        or "errno 11" in msg
        or "pthread_create" in msg
        or "cannot allocate memory" in msg
        or "connection closed while reading from the driver" in msg
        or "target page, context or browser has been closed" in msg
        or "target crashed" in msg
        or ("spawn" in msg and "eagain" in msg)
        or "browser closed" in msg
        or "browser has been closed" in msg
        or ("browserstartevent" in msg and "timed out" in msg)
        or ("cdp connection" in msg and "timed out" in msg)
        or "timed out during opening handshake" in msg
        or "connecttimeout" in msg
        or "connectionerror" in msg
        or "remotedisconnected" in msg
        or "connection refused" in msg
    )


# ── Pure helper ────────────────────────────────────────────────────

def _compute_frame_diffs(all_frames: List[AnnotatedFrame]) -> None:
    """Compute inter-frame SSIM diffs in-place."""
    prev_path: str | None = None
    for f in all_frames:
        if prev_path:
            try:
                a = Path(prev_path).read_bytes()
                b = Path(f.screenshot_path).read_bytes()
                f.diff_from_prev = frame_diff_score(a, b)
            except Exception:
                f.diff_from_prev = 0.0
        prev_path = f.screenshot_path


async def _capture_fallback_frame(
    page,
    output_dir: Path,
    *,
    label: str,
    description: str,
    timestamp: float,
) -> AnnotatedFrame:
    """Capture one last-resort screenshot when probes fail to produce frames."""
    path = output_dir / f"frame_{label}.png"
    await safe_page_screenshot(
        page,
        path=str(path),
        animations="disabled",
        timeout=_FALLBACK_SCREENSHOT_TIMEOUT_MS,
    )
    return AnnotatedFrame(
        screenshot_path=str(path),
        label=label,
        description=description,
        timestamp=timestamp,
        layer="fallback",
    )


class RenderTestPhase(Phase):
    """Phase 2 — dynamic experience evaluation via headless Chromium."""

    name = "render_test"

    def __init__(self, config: EvalConfig, pool: Optional[BrowserPool] = None):
        super().__init__(config)
        self._pool = pool
        self._rate_limiter = TokenBucketRateLimiter(
            rate=config.processing.browser_launch_rate, burst=1,
        )

    def gate(self, ctx: EvalContext) -> bool:
        return not ctx.should_skip and bool(ctx.html_code)

    def should_stop_pipeline(self, result: PhaseResult, ctx: EvalContext) -> bool:
        if result.success:
            return False
        return result.data.get("resource_error", False)

    async def execute(self, ctx: EvalContext) -> PhaseResult:
        game_url = ctx.game_url_file or ctx.game_url_http
        if not game_url or not ctx.output_dir:
            return PhaseResult(
                phase_name=self.name, success=False,
                errors=["Missing game_url or output_dir"],
                data={"rendered": False, "resource_error": False},
            )

        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        static_phase = ctx.get_phase("static_analysis")
        static_data = static_phase.data if static_phase else {}
        data, screenshots, errors = await self._render_with_retries(game_url, ctx.output_dir, static_data)
        error_msg = data.pop("error", None)
        if error_msg:
            errors.append(error_msg)

        return PhaseResult(
            phase_name=self.name,
            success=data.get("rendered", False),
            data=data,
            screenshots=screenshots,
            errors=errors,
        )

    # ── Retry wrapper ──────────────────────────────────────────────

    async def _render_with_retries(self, game_url, output_dir, static_data=None):
        data: Dict[str, Any] = {
            "rendered": False, "page_title": "",
            "console_errors": [], "page_errors": [], "resource_error": False,
        }
        screenshots: List[str] = []
        errors: List[str] = []
        last_error = None

        for attempt in range(1 + RENDER_RETRY_MAX):
            try:
                if self._pool:
                    await self._render_via_pool(game_url, output_dir, data, screenshots, static_data)
                else:
                    await self._render_standalone(game_url, output_dir, data, screenshots, static_data)
                data["resource_error"] = False
                return data, screenshots, errors
            except Exception as exc:
                last_error = exc
                if _is_resource_error(exc) and attempt < RENDER_RETRY_MAX:
                    wait = RENDER_RETRY_BACKOFF[min(attempt, len(RENDER_RETRY_BACKOFF) - 1)]
                    logger.warning(f"[render_test] EAGAIN attempt {attempt+1}, retry in {wait}s: {exc}")
                    data["resource_error"] = True
                    data["console_errors"] = []
                    data["page_errors"] = []
                    screenshots.clear()
                    await asyncio.sleep(wait)
                else:
                    data["resource_error"] = _is_resource_error(exc)
                    errors.append(str(exc))
                    return data, screenshots, errors

        data["resource_error"] = _is_resource_error(last_error) if last_error else False
        errors.append(str(last_error) if last_error else "unknown error")
        return data, screenshots, errors

    async def _render_via_pool(self, game_url, output_dir, data, screenshots, static_data=None):
        async with self._pool.acquire() as (_browser, _ctx, page):
            await self._drive_page(page, game_url, output_dir, data, screenshots, static_data)

    async def _render_standalone(self, game_url, output_dir, data, screenshots, static_data=None):
        from playwright.async_api import async_playwright
        pw = browser = None
        try:
            await self._rate_limiter.acquire()
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True, args=_BROWSER_ARGS)
            ctx = await browser.new_context(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
            await install_page_safety(ctx)
            page = await ctx.new_page()
            await self._drive_page(page, game_url, output_dir, data, screenshots, static_data)
        finally:
            for obj in (browser, pw):
                if obj:
                    try:
                        await obj.close() if hasattr(obj, 'close') else await obj.stop()
                    except Exception:
                        pass

    # ── Orchestrator ───────────────────────────────────────────────

    async def _drive_page(self, page, game_url, output_dir, data, screenshots, static_data=None):
        """Full dynamic experience: probe loop → diffs → snapshot → evidence → keyframes."""
        page.set_default_timeout(15_000)
        page.on("console", lambda msg: data["console_errors"].append(
            {"type": msg.type, "text": msg.text}) if msg.type == "error" else None)
        page.on("pageerror", lambda err: data["page_errors"].append(str(err)))

        # ── Navigate with progressive fallback ─────────────────────
        for wait, timeout_ms in _WAIT_STRATEGIES:
            try:
                await page.goto(game_url, timeout=timeout_ms, wait_until=wait)
                break
            except Exception:
                if wait == "commit":
                    raise

        data["rendered"] = True
        data["page_title"] = await page.title()
        input_types = set((static_data or {}).get("input_types", []))
        out = Path(output_dir)
        sd = static_data or {}
        probe_errors: List[Dict[str, str]] = []

        # ── 1. Probe loop (all layers via registry) ────────────────
        all_frames: List[AnnotatedFrame] = []
        probe_evidence: Dict[str, Any] = {}
        t = 0.0
        for probe_fn, condition in PROBE_REGISTRY:
            if condition(sd, input_types):
                try:
                    result = await asyncio.wait_for(
                        probe_fn(page, out, t, sd, input_types),
                        timeout=_PROBE_TIMEOUT_S,
                    )
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, asyncio.TimeoutError):
                        msg = f"TimeoutError: exceeded {_PROBE_TIMEOUT_S}s probe budget"
                    probe_errors.append({"probe": probe_fn.__name__, "error": msg})
                    logger.warning(
                        "[render_test] probe %s failed for %s: %s",
                        probe_fn.__name__,
                        out.name,
                        msg,
                    )
                    continue

                all_frames.extend(result.frames)
                t = result.timestamp
                probe_evidence.update(result.evidence)

        if not all_frames:
            try:
                fallback = await asyncio.wait_for(
                    _capture_fallback_frame(
                        page,
                        out,
                        label="fallback_current",
                        description="Fallback desktop screenshot captured after probe failures",
                        timestamp=t,
                    ),
                    timeout=_FALLBACK_TIMEOUT_S,
                )
                all_frames.append(fallback)
                data["fallback_screenshot_used"] = True
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, asyncio.TimeoutError):
                    msg = f"TimeoutError: exceeded {_FALLBACK_TIMEOUT_S}s fallback screenshot budget"
                probe_errors.append({"probe": "fallback_screenshot", "error": msg})
                logger.warning("[render_test] fallback screenshot failed for %s: %s", out.name, msg)
                data["fallback_screenshot_used"] = False

        # ── 2. Compute inter-frame diffs ───────────────────────────
        _compute_frame_diffs(all_frames)

        # ── 3. Annotate interaction outcomes (must be after diffs) ──
        annotate_interaction_outcomes(all_frames)

        # ── 4. JS snapshot (game vars, DOM inventory, scroll dims) ──
        try:
            await asyncio.wait_for(
                collect_js_snapshot(page, data),
                timeout=_JS_SNAPSHOT_TIMEOUT_S,
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, asyncio.TimeoutError):
                msg = f"TimeoutError: exceeded {_JS_SNAPSHOT_TIMEOUT_S}s JS snapshot budget"
            probe_errors.append({"probe": "collect_js_snapshot", "error": msg})
            logger.warning("[render_test] js snapshot failed for %s: %s", out.name, msg)

        # ── 5. Evidence computation ────────────────────────────────
        compute_evidence(all_frames, data, probe_evidence)

        # ── 6. Gameplay annotation (mutates frame descriptions) ────
        deep_frames = [f for f in all_frames if f.layer == "deep"]
        annotate_gameplay_smoothness(deep_frames)

        # ── 7. Select keyframes (after descriptions are finalized) ──
        keyframes = select_keyframes(all_frames, max_frames=14)
        for kf in keyframes:
            screenshots.append(kf.screenshot_path)

        data["frame_annotations"] = [
            {"label": kf.label, "description": kf.description,
             "timestamp": kf.timestamp, "layer": kf.layer,
             "diff_from_prev": round(kf.diff_from_prev, 3),
             "screenshot_name": Path(kf.screenshot_path).name}
            for kf in keyframes
        ]
        data["total_frames_captured"] = len(all_frames)
        data["keyframes_selected"] = len(keyframes)
        data["render_screenshot_count"] = len(screenshots)
        data["probe_errors"] = probe_errors

        # ── 8. Element census (conditional — only when 3+ buttons) ──
        dom_btns = (data.get("dom_inventory") or {}).get("buttons", 0)
        if dom_btns >= 3:
            try:
                await asyncio.wait_for(
                    run_element_census(page, data),
                    timeout=_ELEMENT_CENSUS_TIMEOUT_S,
                )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, asyncio.TimeoutError):
                    msg = f"TimeoutError: exceeded {_ELEMENT_CENSUS_TIMEOUT_S}s element census budget"
                probe_errors.append({"probe": "run_element_census", "error": msg})
                logger.warning("[render_test] element census failed for %s: %s", out.name, msg)

        # ── 9. Log ─────────────────────────────────────────────────
        logger.info(
            f"[render_test] dynamic experience: {len(all_frames)} frames → "
            f"{len(keyframes)} keyframes  "
            f"animation={data.get('animation_detected', '?')}  "
            f"btn_responsive={data.get('button_responsive', '?')}  "
            f"kb_change={data.get('keyboard_visual_change', '?')}  "
            f"hover_fx={data.get('hover_effects_detected', '?')}  "
            f"responsive={data.get('responsive_viewports_tested', '?')}  "
            f"avg_latency={data.get('avg_interaction_latency_ms', 'n/a')}ms"
        )
