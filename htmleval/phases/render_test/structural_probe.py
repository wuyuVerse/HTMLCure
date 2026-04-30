"""
Structural validation probe — runtime detection of structural bugs.

Detects:
  - Event binding count (monkey-patches addEventListener)
  - requestAnimationFrame call count over 2 seconds
  - Visible overlays blocking >50% of the viewport on first screen
  - Buttons without any event handlers (onclick or addEventListener)

Registered in PROBE_REGISTRY with condition: always run.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from htmleval.phases.render_test.frame_types import ProbeResult

logger = logging.getLogger("htmleval")

# JavaScript injected into page.evaluate() to collect structural signals.
# Must run EARLY (before other probes interact) to capture baseline state.
_STRUCTURAL_JS = """() => {
    const result = {
        event_listener_count: 0,
        raf_calls_2s: 0,
        visible_overlays: [],
        unbound_buttons: 0,
        total_buttons: 0
    };

    // 1. Count existing addEventListener calls via monkey-patch
    //    (only counts NEW bindings after this script runs)
    const origAEL = EventTarget.prototype.addEventListener;
    let listenerCount = 0;
    EventTarget.prototype.addEventListener = function(type, fn, opts) {
        listenerCount++;
        return origAEL.call(this, type, fn, opts);
    };

    // 2. Count rAF calls over 2 seconds
    let rafCount = 0;
    const origRAF = window.requestAnimationFrame;
    window.requestAnimationFrame = function(cb) {
        rafCount++;
        return origRAF.call(window, cb);
    };

    // 3. Detect visible overlays covering >50% of viewport
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const vpArea = vw * vh;
    const allEls = document.querySelectorAll('*');
    for (let i = 0; i < allEls.length; i++) {
        const el = allEls[i];
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) continue;
        const pos = style.position;
        if (pos !== 'fixed' && pos !== 'absolute') continue;
        const zIndex = parseInt(style.zIndex, 10);
        if (isNaN(zIndex) || zIndex < 10) continue;
        const rect = el.getBoundingClientRect();
        const overlapX = Math.max(0, Math.min(rect.right, vw) - Math.max(rect.left, 0));
        const overlapY = Math.max(0, Math.min(rect.bottom, vh) - Math.max(rect.top, 0));
        const overlapArea = overlapX * overlapY;
        if (overlapArea > vpArea * 0.5) {
            const tag = el.tagName.toLowerCase();
            const id = el.id ? '#' + el.id : '';
            const cls = el.className && typeof el.className === 'string'
                ? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.')
                : '';
            const text = (el.textContent || '').trim().substring(0, 60);
            result.visible_overlays.push({
                selector: tag + id + cls,
                coverage: Math.round(overlapArea / vpArea * 100),
                z_index: zIndex,
                text_preview: text
            });
        }
    }

    // 4. Count buttons without event handlers
    const btnSelectors = 'button, [role="button"], .btn, input[type="button"], input[type="submit"]';
    const buttons = document.querySelectorAll(btnSelectors);
    result.total_buttons = buttons.length;
    let unbound = 0;
    for (let i = 0; i < buttons.length; i++) {
        const btn = buttons[i];
        // Check onclick attribute
        if (btn.onclick || btn.getAttribute('onclick')) continue;
        // Check if parent <a> or <form> handles it
        if (btn.closest('a[href]') || btn.closest('form')) continue;
        // Cannot detect addEventListener from outside, but we can check for
        // common patterns: data-* attributes suggesting framework binding
        const attrs = btn.attributes;
        let hasBinding = false;
        for (let a = 0; a < attrs.length; a++) {
            const name = attrs[a].name.toLowerCase();
            if (name.startsWith('data-') || name.startsWith('ng-') ||
                name.startsWith('v-') || name.startsWith('@') ||
                name.startsWith('on')) {
                hasBinding = true;
                break;
            }
        }
        if (!hasBinding) {
            unbound++;
        }
    }
    result.unbound_buttons = unbound;

    // Store counters for retrieval after 2s delay
    window.__structural_probe = {
        getResult: function() {
            result.event_listener_count = listenerCount;
            result.raf_calls_2s = rafCount;
            return result;
        }
    };

    return true;
}"""

_STRUCTURAL_COLLECT_JS = """() => {
    if (!window.__structural_probe) return null;
    return window.__structural_probe.getResult();
}"""


async def probe_structural_validation(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Runtime structural validation: event bindings, rAF, overlays, unbound buttons.

    Injects monkey-patches early, waits 2s for JS to execute, then collects results.
    Returns ProbeResult with evidence["structural_validation"] dict.
    """
    evidence: Dict[str, Any] = {}

    try:
        # Inject monkey-patches
        await page.evaluate(_STRUCTURAL_JS)

        # Wait 2 seconds for rAF counting and new addEventListener calls
        await asyncio.sleep(2.0)
        t += 2.0

        # Collect results
        raw = await page.evaluate(_STRUCTURAL_COLLECT_JS)
        if raw and isinstance(raw, dict):
            evidence["structural_validation"] = {
                "event_listener_count": raw.get("event_listener_count", 0),
                "raf_calls_2s": raw.get("raf_calls_2s", 0),
                "visible_overlays": raw.get("visible_overlays", []),
                "unbound_buttons": raw.get("unbound_buttons", 0),
                "total_buttons": raw.get("total_buttons", 0),
            }
            sv = evidence["structural_validation"]
            logger.info(
                f"[render_test] structural: listeners={sv['event_listener_count']} "
                f"raf_2s={sv['raf_calls_2s']} overlays={len(sv['visible_overlays'])} "
                f"unbound_btn={sv['unbound_buttons']}/{sv['total_buttons']}"
            )
    except Exception as exc:
        logger.debug(f"[render_test] structural probe error: {exc}")

    return ProbeResult(frames=[], timestamp=t, evidence=evidence)
