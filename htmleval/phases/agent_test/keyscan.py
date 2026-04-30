"""
KeyScanner — pre-agent key discovery for Phase 3.

Opens the page with a bare Playwright browser (no LLM, no browser-use),
presses each candidate key, and detects which ones:
  a) produce a visual change (full-page screenshot hash comparison), OR
  b) are received by the game via window.__probe.keysReceived

Why this works:
  The interaction helper (Phase 0) injects window.__probe which listens for
  keydown events on the capture phase (useCapture=true), so any key the
  game's own listener would receive also increments __probe.keysReceived.
  Combined with screenshot comparison we catch both JS-only and canvas-visual
  responses.

Total cost: ~14 keys × 350 ms = ~5-8 s, zero LLM tokens.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("htmleval")

# Keys to probe — covers arrow keys, space, enter, WASD, digits, escape
CANDIDATE_KEYS: List[str] = [
    "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
    " ", "Enter", "Escape",
    "w", "a", "s", "d",
    "1", "2", "3",
]

# Human-readable display names (used in the task prompt)
_KEY_DISPLAY: Dict[str, str] = {" ": "Space"}

# Playwright keyboard.press() names (some differ from key values)
_PLAYWRIGHT_KEY: Dict[str, str] = {" ": "Space"}


async def discover_keys(
    url: str,
    browser_args: List[str],
    viewport_w: int = 1280,
    viewport_h: int = 720,
    timeout_s: float = 15.0,
) -> Dict[str, Any]:
    """
    Launch a headless browser, load *url*, and probe which keys produce a
    response.  Returns a dict:

        {
          "working":   ["ArrowLeft", "ArrowRight", "Space"],  # display names
          "received":  {"ArrowLeft": 3, " ": 1},              # from __probe
          "game_vars": {"score": 0, "lives": 3},              # window.* vars
          "error":     None,                                   # or error str
        }
    """
    result: Dict[str, Any] = {
        "working": [], "received": {}, "game_vars": {},
        "is_animated": False, "error": None,
    }
    pw = browser = None

    try:
        from playwright.async_api import async_playwright

        pw = await asyncio.wait_for(async_playwright().start(), timeout=10)
        browser = await asyncio.wait_for(
            pw.chromium.launch(headless=True, args=browser_args),
            timeout=20,
        )
        ctx = await browser.new_context(
            viewport={"width": viewport_w, "height": viewport_h}
        )
        page = await ctx.new_page()

        # ── Load page ────────────────────────────────────────────────────
        try:
            await asyncio.wait_for(
                page.goto(url, wait_until="domcontentloaded"),
                timeout=8,
            )
        except Exception:
            pass  # partial load is fine — game may start without full load

        # Let game initialise (first RAF, timers, auto-start)
        await page.wait_for_timeout(900)

        # Focus canvas (use the injected helper if available, else direct JS)
        await page.evaluate("""() => {
            var c = document.querySelector('canvas');
            if (c) {
                if (!c.hasAttribute('tabindex')) c.setAttribute('tabindex', '0');
                c.focus();
            }
        }""")
        await page.wait_for_timeout(100)

        # ── Detect if canvas is continuously animating ───────────────────
        # Animated games change visually every frame (~16ms) regardless of input.
        # For those, screenshot comparison gives false positives for every key.
        # Strategy: if animated → rely on __probe.keysReceived; else → screenshots.
        ss_anim_a = _ss_hash(await page.screenshot())
        await page.wait_for_timeout(350)
        ss_anim_b = _ss_hash(await page.screenshot())
        is_animated = ss_anim_a != ss_anim_b
        logger.debug("[keyscan] is_animated=%s", is_animated)

        # ── Baseline screenshot hash ─────────────────────────────────────
        h_prev = _ss_hash(await page.screenshot())

        # ── Key scan ─────────────────────────────────────────────────────
        working: List[str] = []

        for raw_key in CANDIDATE_KEYS:
            display  = _KEY_DISPLAY.get(raw_key, raw_key)
            pw_key   = _PLAYWRIGHT_KEY.get(raw_key, raw_key)
            escaped  = raw_key.replace("'", "\\'").replace("\\", "\\\\")

            # Re-focus canvas before each key so focus never drifts
            await page.evaluate(
                "() => { var c = document.querySelector('canvas'); if(c) c.focus(); }"
            )

            # Dispatch via injected helper (fires on window + document + canvas)
            try:
                await page.evaluate(
                    f"() => window.__pressKey && window.__pressKey('{escaped}', 100)"
                )
            except Exception:
                pass

            # Also fire via Playwright's native keyboard (targets focused element)
            try:
                await page.keyboard.press(pw_key)
            except Exception:
                pass

            await page.wait_for_timeout(220)

            # Screenshot comparison (unreliable for animated canvases)
            if not is_animated:
                h_now = _ss_hash(await page.screenshot())
                changed = h_now != h_prev
                h_prev = h_now  # rolling baseline
            else:
                changed = False  # ignore for animated games

            # Check probe counter
            try:
                received: int = await page.evaluate(
                    f"() => (window.__probe && window.__probe.keysReceived['{escaped}']) || 0"
                )
            except Exception:
                received = 0

            if changed or received > 0:
                working.append(display)
                logger.debug(
                    "[keyscan] key=%r  changed=%s  received=%d", display, changed, received
                )

        result["working"]      = working
        result["is_animated"]  = is_animated

        # ── Read final probe snapshot ─────────────────────────────────────
        try:
            snap_json: Optional[str] = await page.evaluate(
                "() => window.__probe ? window.__probe.snapshot() : null"
            )
            if snap_json:
                snap = json.loads(snap_json)
                result["received"]  = snap.get("keysReceived", {})
                result["game_vars"] = snap.get("gameVars", {})
        except Exception as e:
            logger.debug("[keyscan] probe read failed: %s", e)

    except Exception as e:
        result["error"] = str(e)
        logger.warning("[keyscan] error: %s", e)
    finally:
        for obj in (browser, pw):
            if obj:
                try:
                    await asyncio.wait_for(obj.close(), timeout=5)
                except Exception:
                    pass

    return result


def _ss_hash(data: bytes) -> str:
    """MD5 of screenshot bytes — fast enough for comparison, not security."""
    return hashlib.md5(data).hexdigest()
