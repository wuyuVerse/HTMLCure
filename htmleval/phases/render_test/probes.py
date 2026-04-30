"""
Interaction probes for render_test phase.

Every probe has a uniform signature:

    async def probe_xxx(page, out, t, static_data, input_types) -> ProbeResult

PROBE_REGISTRY lists *all* probes (Layer 1-4) with their activation conditions.
Adding a new probe = write 1 function + add 1 line to PROBE_REGISTRY.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from htmleval.concurrency.browser_pool import VIEWPORT_W, VIEWPORT_H
from htmleval.core.screenshot import safe_page_screenshot
from htmleval.phases.render_test.frame_types import AnnotatedFrame, FPSResult, ProbeResult, frame_changed
from htmleval.phases.render_test.structural_probe import probe_structural_validation

logger = logging.getLogger("htmleval")

# ── Shared constants ──────────────────────────────────────────────

BUTTON_SELECTORS = (
    'button, [role="button"], .btn, [onclick], '
    'input[type="button"], input[type="submit"]'
)

_RESPONSIVE_VIEWPORTS = [
    (375, 667, "mobile", "Mobile viewport (iPhone SE — 375×667)"),
    (768, 1024, "tablet", "Tablet viewport (iPad — 768×1024)"),
]

# FPS quality thresholds (out of 9 visual changes across 10 rapid frames)
_FPS_SAMPLE_COUNT = 10
_FPS_SAMPLE_INTERVAL = 0.1  # seconds
_FPS_THRESHOLDS = [
    (7, "smooth"),      # >= 7 visual changes
    (4, "acceptable"),  # >= 4
    (2, "choppy"),      # >= 2
    (0, "frozen"),      # < 2
]


# =====================================================================
# Shared helpers
# =====================================================================

async def _mark_action(page, name: str) -> None:
    """Call window.__probe.markAction(name) for latency measurement.

    Uses json.dumps for safe JS string interpolation.
    """
    try:
        escaped = _json.dumps(name)
        await page.evaluate(
            f"() => {{ if (window.__probe) window.__probe.markAction({escaped}); }}"
        )
    except Exception:
        pass


async def _dismiss_blocking_overlays(page) -> None:
    try:
        await page.evaluate("""() => {
            const selectors = [
                '[id*="overlay"]', '[class*="overlay"]',
                '[id*="modal"]', '[class*="modal"]',
                '[id*="dialog"]', '[class*="dialog"]',
                '[id*="popup"]', '[class*="popup"]',
                '[data-modal]', '[data-overlay]', '[role="dialog"]', '[aria-modal="true"]'
            ];
            document.querySelectorAll(selectors.join(',')).forEach((el) => {
                try {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    const text = String(el.textContent || '').toLowerCase();
                    const big = rect.width > window.innerWidth * 0.35 || rect.height > window.innerHeight * 0.25;
                    const blocking = style.position === 'fixed' || style.position === 'absolute' || Number(style.zIndex || 0) >= 10;
                    const looksLikeIntro = /(start|play|begin|help|how to|instructions|tutorial|continue|close|ok|开始|帮助|教程|继续|关闭)/.test(text);
                    if (big && blocking && looksLikeIntro) {
                        el.style.display = 'none';
                        el.setAttribute('aria-hidden', 'true');
                    }
                } catch (err) {}
            });
            try { document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); } catch (err) {}
            try { document.dispatchEvent(new KeyboardEvent('keyup', { key: 'Escape', bubbles: true })); } catch (err) {}
        }""")
        await asyncio.sleep(0.15)
    except Exception:
        pass


def _is_game(sd: dict, it: set) -> bool:
    """Condition: page looks like a game (canvas+rAF/Three.js or keyboard inputs)."""
    return bool(
        (sd.get("has_canvas") and (sd.get("has_requestanimationframe") or sd.get("has_threejs")))
        or ("keyboard" in it)
    )


async def _prioritize_buttons(btns) -> list:
    """Score and prioritize buttons by text/attributes, return up to 6 sorted by relevance.

    Priority keywords (case-insensitive):
      high (100):   start, play, begin, submit, initiate, 开始
      medium (50):  next, continue, ok, confirm, 新建, 继续
      low (10):     cancel, close, back, return, 菜单
      default:      50
    """
    try:
        _HIGH = {"start", "play", "begin", "submit", "initiate", "开始"}
        _MEDIUM = {"next", "continue", "ok", "confirm", "新建", "继续"}
        _LOW = {"cancel", "close", "back", "return", "菜单"}

        scored = []
        for btn in btns:
            try:
                # Gather text + attributes for keyword matching
                text = ""
                try:
                    text = ((await btn.inner_text()) or "").strip().lower()
                except Exception:
                    pass
                attrs = ""
                try:
                    attrs = (await btn.evaluate(
                        "el => [el.value, el.getAttribute('aria-label'), el.id, el.className].join(' ')"
                    ) or "").lower()
                except Exception:
                    pass

                combined = text + " " + attrs
                # Priority: HIGH (100) > MEDIUM (50) > LOW (10) > default (50)
                # MEDIUM and default share score 50, so effectively:
                # HIGH > LOW-only > everything else
                if any(kw in combined for kw in _HIGH):
                    score = 100
                elif any(kw in combined for kw in _MEDIUM):
                    score = 50  # explicitly MEDIUM, not demotable by LOW
                elif any(kw in combined for kw in _LOW):
                    score = 10
                else:
                    score = 50  # default
                scored.append((score, btn))
            except Exception:
                scored.append((50, btn))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [btn for _, btn in scored[:6]]
    except Exception:
        return list(btns[:6])


def _classify_fps(visual_changes: int) -> str:
    """Map visual change count to quality label."""
    for threshold, label in _FPS_THRESHOLDS:
        if visual_changes >= threshold:
            return label
    return "frozen"


# =====================================================================
# Layer 1: Observation
# =====================================================================

async def probe_observation(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Time-lapse: capture page lifecycle at 0.5s, 1.5s, 3s, 5s.

    Evidence: fps_result (FPSResult) if animation detected, else absent.
    Note: observation always takes ~5s regardless of incoming t.
    """
    frames: List[AnnotatedFrame] = []
    t_start = t

    async def _snap(name, ts, desc):
        p = out / f"frame_{name}.png"
        try:
            await safe_page_screenshot(page, path=p)
        except Exception as exc:
            logger.debug(f"[render_test] observation snapshot {name} failed: {exc}")
            return None
        frames.append(AnnotatedFrame(
            screenshot_path=str(p), label=name, description=desc,
            timestamp=ts, layer="observation",
        ))
        return p

    # Capture snapshots at scheduled times
    await asyncio.sleep(0.5)
    await _snap("early_load", t_start + 0.5, "Page state 0.5s after load — early render")

    await asyncio.sleep(1.0)
    await _snap("first_paint", t_start + 1.5, "First meaningful paint (t=1.5s)")

    await asyncio.sleep(1.5)
    p3 = await _snap("stable", t_start + 3.0, "Stable state — JS initialization complete (t=3s)")

    await asyncio.sleep(2.0)
    t = t_start + 5.0
    has_animation = False
    idle_bytes = None
    if p3 is not None:
        try:
            idle_bytes = await safe_page_screenshot(page)
            stable_bytes = Path(p3).read_bytes()
            has_animation = frame_changed(stable_bytes, idle_bytes)
        except Exception as exc:
            logger.debug(f"[render_test] observation idle snapshot failed: {exc}")

    if has_animation and idle_bytes is not None:
        p5 = out / "frame_idle.png"
        p5.write_bytes(idle_bytes)
        frames.append(AnnotatedFrame(
            screenshot_path=str(p5), label="idle_animation",
            description="Continuous animation detected (t=5s differs from t=3s)",
            timestamp=t, layer="observation",
        ))

    # FPS sampling: rapid frames at fixed intervals
    fps_result = None
    if has_animation:
        try:
            fps_samples = [idle_bytes]
            for _ in range(_FPS_SAMPLE_COUNT - 1):
                await asyncio.sleep(_FPS_SAMPLE_INTERVAL)
                fps_samples.append(await safe_page_screenshot(page))
            visual_changes = sum(
                1 for i in range(1, len(fps_samples))
                if frame_changed(fps_samples[i - 1], fps_samples[i])
            )
            fps_result = FPSResult(visual_changes, _classify_fps(visual_changes))
        except Exception:
            pass

    evidence = {}
    if fps_result:
        evidence["fps_result"] = fps_result
    return ProbeResult(frames=frames, timestamp=t, evidence=evidence)


# =====================================================================
# Layer 2: Interaction probes
# =====================================================================

async def probe_scroll(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Progressive scroll to 50% and 100% of page height."""
    frames: List[AnnotatedFrame] = []
    try:
        page_height = await page.evaluate("document.body.scrollHeight")
        viewport_h = await page.evaluate("window.innerHeight")
        if page_height > viewport_h * 1.2:
            for frac, label in [(0.5, "scroll_mid"), (1.0, "scroll_bottom")]:
                target_y = int(page_height * frac)
                await page.evaluate(
                    f"window.scrollTo({{top: {target_y}, behavior: 'smooth'}})"
                )
                await asyncio.sleep(0.8)
                t += 0.8
                p = str(out / f"frame_{label}.png")
                await safe_page_screenshot(page, path=p)
                frames.append(AnnotatedFrame(
                    screenshot_path=p, label=label,
                    description=f"Scrolled to {int(frac*100)}% (y={target_y}/{page_height}px)",
                    timestamp=t, layer="interaction",
                ))
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.3)
            t += 0.3
    except Exception:
        pass
    return ProbeResult(frames=frames, timestamp=t)


async def probe_buttons(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Hover + click burst on up to 6 prioritized visible buttons."""
    frames: List[AnnotatedFrame] = []
    try:
        await _dismiss_blocking_overlays(page)
        btns = await page.query_selector_all(BUTTON_SELECTORS)
        prioritized = await _prioritize_buttons(btns)
        for i, btn in enumerate(prioritized):
            try:
                if not await btn.is_visible():
                    continue
                box = await btn.bounding_box()
                text = ""
                try:
                    text = ((await btn.inner_text()) or "")[:30].strip()
                except Exception:
                    pass
                if box:
                    # Hover
                    await page.mouse.move(
                        box["x"] + box["width"] / 2,
                        box["y"] + box["height"] / 2,
                    )
                    await asyncio.sleep(0.5)
                    t += 0.5
                    p = str(out / f"frame_hover_btn_{i}.png")
                    await safe_page_screenshot(page, path=p)
                    frames.append(AnnotatedFrame(
                        screenshot_path=p, label=f"hover_btn_{i}",
                        description=f"Mouse hovering over button '{text}'",
                        timestamp=t, layer="interaction",
                    ))
                # Mark action for latency measurement, then click
                await _mark_action(page, f"click_btn_{i}")
                try:
                    await btn.click(timeout=3000)
                except Exception as exc:
                    if "intercepts pointer events" not in str(exc):
                        raise
                    await _dismiss_blocking_overlays(page)
                    await btn.click(timeout=3000)

                # Micro-clip burst: 3 rapid frames after click
                burst_times = [0.15, 0.4, 1.0]
                for bi, bt in enumerate(burst_times):
                    if bi == 0:
                        await asyncio.sleep(bt)
                    else:
                        await asyncio.sleep(bt - burst_times[bi - 1])
                    t += (bt - (burst_times[bi - 1] if bi > 0 else 0))
                    suffix = ["early", "mid", "settled"][bi]
                    p = str(out / f"frame_click_btn_{i}_{suffix}.png")
                    await safe_page_screenshot(page, path=p)
                    frames.append(AnnotatedFrame(
                        screenshot_path=p,
                        label=f"after_click_btn_{i}_{suffix}",
                        description=f"After clicking '{text}' +{bt:.2f}s ({suffix})",
                        timestamp=t, layer="interaction",
                    ))
            except Exception:
                pass
    except Exception:
        pass
    return ProbeResult(frames=frames, timestamp=t)


async def probe_keyboard(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Keyboard interaction test.

    Evidence: kb_data = {keyboard_probed, keys_responded}.
    """
    frames: List[AnnotatedFrame] = []
    kb_data = {"keyboard_probed": False, "keys_responded": []}

    try:
        await page.evaluate("""() => {
            const c = document.querySelector('canvas');
            if (c) { c.setAttribute('tabindex', '0'); c.focus(); }
            else { document.body.focus(); }
        }""")
        await asyncio.sleep(0.2)

        # Reset keysReceived
        await page.evaluate(
            "() => { if (window.__probe) window.__probe.keysReceived = {}; }"
        )

        key_seqs = [
            (["ArrowRight"] * 3, "3× Right arrow"),
            (["ArrowUp", "Space"], "Up + Space (jump/shoot)"),
            (["ArrowLeft"] * 3, "3× Left arrow"),
        ]
        for si, (keys, desc) in enumerate(key_seqs):
            await _mark_action(page, f"keyboard_{si}")
            for key in keys:
                try:
                    await page.keyboard.press(key)
                    await asyncio.sleep(0.3)
                except Exception:
                    pass
            await asyncio.sleep(0.8)
            t += len(keys) * 0.3 + 0.8
            p = str(out / f"frame_keyboard_{si}.png")
            await safe_page_screenshot(page, path=p)
            frames.append(AnnotatedFrame(
                screenshot_path=p, label=f"keyboard_{si}",
                description=f"After keyboard: {desc}",
                timestamp=t, layer="interaction",
            ))

        # Read keyboard probe results
        keys_received = await page.evaluate(
            "() => window.__probe ? window.__probe.keysReceived : {}"
        ) or {}
        keys_responded = [k for k, v in keys_received.items() if v]
        kb_data = {
            "keyboard_probed": True,
            "keys_responded": keys_responded,
        }
    except Exception as exc:
        logger.debug(f"[render_test] keyboard interaction error: {exc}")

    return ProbeResult(frames=frames, timestamp=t, evidence={"kb_data": kb_data})


async def probe_canvas_click(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Click the center of a canvas element."""
    frames: List[AnnotatedFrame] = []
    try:
        canvas = await page.query_selector("canvas")
        if canvas:
            box = await canvas.bounding_box()
            if box:
                cx = box["x"] + box["width"] / 2
                cy = box["y"] + box["height"] / 2
                await page.mouse.click(cx, cy)
                await asyncio.sleep(1.0)
                t += 1.0
                p = str(out / "frame_canvas_click.png")
                await safe_page_screenshot(page, path=p)
                frames.append(AnnotatedFrame(
                    screenshot_path=p, label="canvas_click",
                    description=f"Clicked canvas center ({cx:.0f}, {cy:.0f})",
                    timestamp=t, layer="interaction",
                ))
    except Exception:
        pass
    return ProbeResult(frames=frames, timestamp=t)


async def probe_form(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Fill text inputs and submit forms."""
    frames: List[AnnotatedFrame] = []
    try:
        inputs = await page.query_selector_all(
            'input[type="text"], input[type="email"], input[type="search"], '
            'input[type="number"], input[type="password"], input[type="tel"], '
            'input[type="url"], input:not([type]):not([hidden]), textarea'
        )
        for j, inp in enumerate(inputs[:2]):
            try:
                if not await inp.is_visible():
                    continue
                await _mark_action(page, f"form_input_{j}")
                input_type = (await inp.get_attribute("type")) or "text"
                test_values = {
                    "email": "test@example.com", "number": "42",
                    "search": "hello", "password": "test123",
                    "tel": "5551234567", "url": "https://example.com",
                }
                test_val = test_values.get(input_type, "Hello World")
                tag = await inp.evaluate("el => el.tagName.toLowerCase()")
                if tag == "textarea":
                    test_val = "This is a test message for form validation."
                await inp.click()
                await inp.fill(test_val)
                await asyncio.sleep(0.5)
                t += 0.5
                p = str(out / f"frame_form_input_{j}.png")
                await safe_page_screenshot(page, path=p)
                frames.append(AnnotatedFrame(
                    screenshot_path=p, label=f"form_input_{j}",
                    description=f"After typing '{test_val[:20]}' into {tag}[{input_type}] #{j}",
                    timestamp=t, layer="interaction",
                ))
            except Exception:
                pass

        # Try to submit the form
        submit_btn = await page.query_selector(
            'input[type="submit"], button[type="submit"], '
            'form button:not([type="button"]):not([type="reset"])'
        )
        if submit_btn:
            try:
                if await submit_btn.is_visible():
                    await _mark_action(page, "form_submit")
                    await submit_btn.click(timeout=3000)
                    await asyncio.sleep(1.0)
                    t += 1.0
                    p = str(out / "frame_form_submitted.png")
                    await safe_page_screenshot(page, path=p)
                    frames.append(AnnotatedFrame(
                        screenshot_path=p, label="form_submitted",
                        description="After submitting form (+1s)",
                        timestamp=t, layer="interaction",
                    ))
            except Exception:
                pass
    except Exception:
        pass
    return ProbeResult(frames=frames, timestamp=t)


async def probe_select(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Test dropdown selects."""
    frames: List[AnnotatedFrame] = []
    try:
        selects = await page.query_selector_all("select")
        for j, sel in enumerate(selects[:1]):
            try:
                if not await sel.is_visible():
                    continue
                options = await sel.query_selector_all("option")
                if len(options) > 1:
                    await sel.select_option(index=1)
                    await asyncio.sleep(0.5)
                    t += 0.5
                    p = str(out / f"frame_select_{j}.png")
                    await safe_page_screenshot(page, path=p)
                    frames.append(AnnotatedFrame(
                        screenshot_path=p, label=f"select_{j}",
                        description=f"Selected 2nd option in dropdown #{j}",
                        timestamp=t, layer="interaction",
                    ))
            except Exception:
                pass
    except Exception:
        pass
    return ProbeResult(frames=frames, timestamp=t)


async def probe_range(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Test range sliders."""
    frames: List[AnnotatedFrame] = []
    try:
        ranges = await page.query_selector_all('input[type="range"]')
        for j, rng in enumerate(ranges[:1]):
            try:
                if not await rng.is_visible():
                    continue
                box = await rng.bounding_box()
                if box:
                    await _mark_action(page, f"range_{j}")
                    await page.mouse.click(
                        box["x"] + box["width"] * 0.75,
                        box["y"] + box["height"] / 2,
                    )
                    await asyncio.sleep(0.5)
                    t += 0.5
                    p = str(out / f"frame_range_{j}.png")
                    await safe_page_screenshot(page, path=p)
                    frames.append(AnnotatedFrame(
                        screenshot_path=p, label=f"range_{j}",
                        description=f"Moved range slider #{j} to 75%",
                        timestamp=t, layer="interaction",
                    ))
            except Exception:
                pass
    except Exception:
        pass
    return ProbeResult(frames=frames, timestamp=t)


async def probe_links(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Click visible links (up to 2) to test navigation/interaction.

    Uses JS click with preventDefault to avoid leaving the page,
    then checks for visual changes (SPA routing, tab switches, modals).
    """
    frames: List[AnnotatedFrame] = []
    try:
        # Find visible <a> links with href (skip anchors and empty)
        links = await page.query_selector_all(
            'a[href]:not([href=""]):not([href="#"]):not([href^="javascript:"])'
        )
        tested = 0
        for i, link in enumerate(links[:5]):
            if tested >= 2:
                break
            try:
                if not await link.is_visible():
                    continue
                text = ""
                try:
                    text = ((await link.inner_text()) or "")[:30].strip()
                except Exception:
                    pass
                if not text:
                    continue  # skip links with no text (icons, images)

                await _mark_action(page, f"link_click_{tested}")
                # Intercept navigation, then click normally
                await link.evaluate("""el => {
                    el.addEventListener('click', e => e.preventDefault(), {once: true, capture: true});
                    el.click();
                }""")
                await asyncio.sleep(1.0)
                t += 1.0
                p = str(out / f"frame_link_click_{tested}.png")
                await safe_page_screenshot(page, path=p)
                frames.append(AnnotatedFrame(
                    screenshot_path=p, label=f"link_click_{tested}",
                    description=f"After clicking link '{text}'",
                    timestamp=t, layer="interaction",
                ))
                tested += 1
            except Exception:
                pass
    except Exception:
        pass
    return ProbeResult(frames=frames, timestamp=t)


async def probe_checkbox(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Toggle visible checkboxes and radio buttons (up to 2)."""
    frames: List[AnnotatedFrame] = []
    try:
        checks = await page.query_selector_all(
            'input[type="checkbox"], input[type="radio"]'
        )
        tested = 0
        for j, chk in enumerate(checks[:4]):
            if tested >= 2:
                break
            try:
                if not await chk.is_visible():
                    continue
                input_type = (await chk.get_attribute("type")) or "checkbox"
                label_text = ""
                try:
                    label_text = await chk.evaluate("""el => {
                        const id = el.id;
                        if (id) {
                            const label = document.querySelector('label[for="' + id + '"]');
                            if (label) return label.textContent.trim().substring(0, 30);
                        }
                        const parent = el.closest('label');
                        if (parent) return parent.textContent.trim().substring(0, 30);
                        return '';
                    }""")
                except Exception:
                    pass

                await _mark_action(page, f"checkbox_{tested}")
                await chk.click()
                await asyncio.sleep(0.5)
                t += 0.5
                p = str(out / f"frame_checkbox_{tested}.png")
                await safe_page_screenshot(page, path=p)
                desc = f"Toggled {input_type}"
                if label_text:
                    desc += f" '{label_text}'"
                frames.append(AnnotatedFrame(
                    screenshot_path=p, label=f"checkbox_{tested}",
                    description=desc,
                    timestamp=t, layer="interaction",
                ))
                tested += 1
            except Exception:
                pass
    except Exception:
        pass
    return ProbeResult(frames=frames, timestamp=t)


async def probe_drag(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Drag interaction on canvas or draggable elements."""
    frames: List[AnnotatedFrame] = []
    try:
        target = await page.query_selector("canvas")
        if not target:
            target = await page.query_selector(
                '[draggable="true"], .draggable, [class*="drag"]'
            )
        if target:
            box = await target.bounding_box()
            if box and await target.is_visible():
                await _mark_action(page, "drag")
                x1 = box["x"] + box["width"] * 0.25
                y1 = box["y"] + box["height"] * 0.5
                x2 = box["x"] + box["width"] * 0.75
                y2 = y1
                await page.mouse.move(x1, y1)
                await page.mouse.down()
                for step in range(1, 6):
                    frac = step / 5
                    await page.mouse.move(
                        x1 + (x2 - x1) * frac,
                        y1 + (y2 - y1) * frac,
                    )
                    await asyncio.sleep(0.08)
                await page.mouse.up()
                await asyncio.sleep(0.5)
                t += 1.0
                p = str(out / "frame_after_drag.png")
                await safe_page_screenshot(page, path=p)
                frames.append(AnnotatedFrame(
                    screenshot_path=p, label="after_drag",
                    description="After horizontal drag (left→right across element)",
                    timestamp=t, layer="interaction",
                ))
    except Exception:
        pass
    return ProbeResult(frames=frames, timestamp=t)


# =====================================================================
# Layer 3: Deep gameplay — action sequences & helpers
# =====================================================================

# 15 actions × 1.0s interval = ~15s of diverse keyboard gameplay
_KEYBOARD_GAMEPLAY_ACTIONS = [
    # Phase 1: exploration (5s)
    ("ArrowRight", "Move right"),
    ("ArrowRight", "Move right"),
    ("ArrowUp",    "Move up / jump"),
    ("Space",      "Action / jump"),
    ("ArrowLeft",  "Move left"),
    # Phase 2: combat/action (5s)
    ("ArrowRight", "Move right"),
    ("Space",      "Action"),
    ("ArrowDown",  "Move down / crouch"),
    ("ArrowUp",    "Jump"),
    ("ArrowRight", "Move right"),
    # Phase 3: varied input (5s)
    ("KeyW",       "Alt up"),
    ("KeyD",       "Alt right"),
    ("Space",      "Action"),
    ("KeyA",       "Alt left"),
    ("ArrowRight", "Move right"),
]

# Shorter fallback for pages without clear input type
_DEFAULT_GAMEPLAY_ACTIONS = _KEYBOARD_GAMEPLAY_ACTIONS[:10]  # ~10s

# Mouse click positions on canvas (fractions of width/height)
_MOUSE_CLICK_POSITIONS = [
    (0.50, 0.50, "center"),
    (0.25, 0.25, "upper-left"),
    (0.75, 0.25, "upper-right"),
    (0.25, 0.75, "lower-left"),
    (0.75, 0.75, "lower-right"),
]

# Drag gestures on canvas: (start_frac, end_frac, label)
_MOUSE_DRAG_GESTURES = [
    ((0.25, 0.50), (0.75, 0.50), "horizontal-drag"),
    ((0.50, 0.25), (0.50, 0.75), "vertical-drag"),
]

_GAMEPLAY_ACTION_INTERVAL = 1.0  # seconds between actions


def _select_gameplay_actions(
    static_data: dict, input_types: set,
) -> tuple[str, list]:
    """Select gameplay action sequence based on page input capabilities.

    Returns (mode, actions) where mode is 'keyboard', 'mouse', or 'default'.
    """
    has_canvas = static_data.get("has_canvas")
    has_mouse = "mouse_click" in input_types

    if has_mouse and has_canvas:
        return "mouse", []  # mouse gameplay handled separately
    elif "keyboard" in input_types:
        return "keyboard", _KEYBOARD_GAMEPLAY_ACTIONS
    else:
        return "default", _DEFAULT_GAMEPLAY_ACTIONS


async def _take_game_snapshot(page) -> dict | None:
    """Read window.__probe.snapshot() game state; return parsed dict or None."""
    try:
        raw = await page.evaluate(
            "() => window.__probe ? window.__probe.snapshot() : null"
        )
        if raw is None:
            return None
        return _json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None


def _diff_game_vars(
    pre_snap: dict | None, post_snap: dict | None,
) -> tuple[bool, dict]:
    """Compare gameVars from two snapshots.

    Returns (state_changed, vars_diff) where vars_diff maps
    variable names to (old_value, new_value) tuples.
    """
    if not pre_snap or not post_snap:
        return False, {}
    pre_vars = pre_snap.get("gameVars", {})
    post_vars = post_snap.get("gameVars", {})
    changed = {
        k: (pre_vars.get(k), post_vars[k])
        for k in post_vars
        if post_vars[k] != pre_vars.get(k)
    }
    return bool(changed), changed


async def _detect_game_completion(page) -> dict:
    """Detect game completion state via visible modals and JS globals.

    Returns {completed: bool, won: bool, state: str, overlay_text: str}.
    """
    try:
        result = await page.evaluate("""() => {
            const result = {completed: false, won: false, state: 'playing', overlay_text: ''};

            // Check JS globals
            const winVars = ['gameOver', 'isGameOver', 'gameWon', 'levelCleared',
                             'gameEnd', 'isWin', 'isLose', 'gameover'];
            for (const v of winVars) {
                if (typeof window[v] !== 'undefined' && window[v]) {
                    result.completed = true;
                    if (['gameWon', 'levelCleared', 'isWin'].includes(v)) {
                        result.won = true;
                        result.state = 'won';
                    } else {
                        result.state = 'game_over';
                    }
                    break;
                }
            }

            // Check visible modals/overlays with win/lose keywords
            const keywords = /\\b(game\\s*over|you\\s*(win|lose|lost|won|died)|victory|defeat|congratulations|level\\s*clear|通关|胜利|失败|游戏结束)\\b/i;
            const candidates = document.querySelectorAll(
                '.modal, .overlay, .popup, .game-over, .gameover, .endscreen, '
                + '[class*="modal"], [class*="overlay"], [class*="popup"], '
                + '[class*="game-over"], [class*="end"], [class*="result"]'
            );
            for (const el of candidates) {
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) < 0.1) continue;
                const text = (el.textContent || '').trim().substring(0, 200);
                if (keywords.test(text)) {
                    result.completed = true;
                    result.overlay_text = text.substring(0, 100);
                    if (/\\b(you\\s*(win|won)|victory|congratulations|level\\s*clear|通关|胜利)\\b/i.test(text)) {
                        result.won = true;
                        result.state = 'won';
                    } else {
                        result.state = 'game_over';
                    }
                    break;
                }
            }

            return result;
        }""")
        return result or {"completed": False, "won": False, "state": "playing", "overlay_text": ""}
    except Exception:
        return {"completed": False, "won": False, "state": "unknown", "overlay_text": ""}


async def _run_keyboard_gameplay(
    page, out: Path, t: float, actions: list,
) -> tuple[List[AnnotatedFrame], float]:
    """Execute keyboard actions with per-action screenshots."""
    frames: List[AnnotatedFrame] = []
    for idx, (key, desc) in enumerate(actions):
        try:
            await _mark_action(page, f"gameplay_{idx}")
            await page.keyboard.press(key)
        except Exception:
            pass
        await asyncio.sleep(_GAMEPLAY_ACTION_INTERVAL)
        t += _GAMEPLAY_ACTION_INTERVAL
        p = str(out / f"frame_gameplay_{idx}.png")
        try:
            await safe_page_screenshot(page, path=p)
        except Exception as exc:
            logger.debug(f"[render_test] keyboard gameplay frame {idx} failed: {exc}")
            continue
        frames.append(AnnotatedFrame(
            screenshot_path=p, label=f"gameplay_{idx}",
            description=f"Gameplay t={idx+1}s — {desc} ({key})",
            timestamp=t, layer="deep",
        ))
    return frames, t


async def _run_mouse_gameplay(
    page, out: Path, t: float,
) -> tuple[List[AnnotatedFrame], float]:
    """Execute mouse clicks and drags on canvas with screenshots."""
    frames: List[AnnotatedFrame] = []
    try:
        canvas = await page.query_selector("canvas")
        if not canvas:
            return frames, t
        box = await canvas.bounding_box()
        if not box:
            return frames, t
    except Exception:
        return frames, t

    cx, cy, cw, ch = box["x"], box["y"], box["width"], box["height"]

    # Phase 1: click at 5 positions (~5s)
    for idx, (fx, fy, label) in enumerate(_MOUSE_CLICK_POSITIONS):
        x, y = cx + cw * fx, cy + ch * fy
        try:
            await _mark_action(page, f"gameplay_click_{idx}")
            await page.mouse.click(x, y)
        except Exception:
            pass
        await asyncio.sleep(_GAMEPLAY_ACTION_INTERVAL)
        t += _GAMEPLAY_ACTION_INTERVAL
        p = str(out / f"frame_gameplay_click_{idx}.png")
        try:
            await safe_page_screenshot(page, path=p)
        except Exception as exc:
            logger.debug(f"[render_test] mouse gameplay click frame {idx} failed: {exc}")
            continue
        frames.append(AnnotatedFrame(
            screenshot_path=p, label=f"gameplay_click_{idx}",
            description=f"Mouse click {label} ({x:.0f}, {y:.0f})",
            timestamp=t, layer="deep",
        ))

    # Phase 2: drag gestures (~4s)
    for idx, ((sx, sy), (ex, ey), label) in enumerate(_MOUSE_DRAG_GESTURES):
        x1, y1 = cx + cw * sx, cy + ch * sy
        x2, y2 = cx + cw * ex, cy + ch * ey
        try:
            await _mark_action(page, f"gameplay_drag_{idx}")
            await page.mouse.move(x1, y1)
            await page.mouse.down()
            for step in range(1, 6):
                frac = step / 5
                await page.mouse.move(
                    x1 + (x2 - x1) * frac,
                    y1 + (y2 - y1) * frac,
                )
                await asyncio.sleep(0.1)
            await page.mouse.up()
        except Exception:
            pass
        await asyncio.sleep(0.5)
        t += _GAMEPLAY_ACTION_INTERVAL
        p = str(out / f"frame_gameplay_drag_{idx}.png")
        try:
            await safe_page_screenshot(page, path=p)
        except Exception as exc:
            logger.debug(f"[render_test] mouse gameplay drag frame {idx} failed: {exc}")
            continue
        frames.append(AnnotatedFrame(
            screenshot_path=p, label=f"gameplay_drag_{idx}",
            description=f"Mouse {label} ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})",
            timestamp=t, layer="deep",
        ))

    # Phase 3: a few more clicks with keyboard intermixed (~4s)
    mixed_actions = [
        ("click", 0.50, 0.50, "center re-click"),
        ("key",   "Space", "", "Action"),
        ("click", 0.60, 0.40, "offset click"),
        ("key",   "ArrowRight", "", "Move right"),
    ]
    for idx, action in enumerate(mixed_actions):
        try:
            await _mark_action(page, f"gameplay_mixed_{idx}")
            if action[0] == "click":
                await page.mouse.click(cx + cw * action[1], cy + ch * action[2])
            else:
                await page.keyboard.press(action[1])
        except Exception:
            pass
        await asyncio.sleep(_GAMEPLAY_ACTION_INTERVAL)
        t += _GAMEPLAY_ACTION_INTERVAL
        p = str(out / f"frame_gameplay_mixed_{idx}.png")
        try:
            await safe_page_screenshot(page, path=p)
        except Exception as exc:
            logger.debug(f"[render_test] mixed gameplay frame {idx} failed: {exc}")
            continue
        frames.append(AnnotatedFrame(
            screenshot_path=p, label=f"gameplay_mixed_{idx}",
            description=f"Mixed gameplay — {action[3]}",
            timestamp=t, layer="deep",
        ))

    return frames, t


async def probe_deep_gameplay(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """~15-second gameplay: input-type-adaptive actions with snapshot diff."""
    frames: List[AnnotatedFrame] = []
    evidence: Dict[str, Any] = {}

    # Ensure canvas is focusable
    try:
        await page.evaluate("""() => {
            const c = document.querySelector('canvas');
            if (c) { c.setAttribute('tabindex', '0'); c.focus(); }
        }""")
    except Exception:
        pass

    # Pre-gameplay game state snapshot
    pre_snap = await _take_game_snapshot(page)

    # Select action sequence based on page capabilities
    mode, actions = _select_gameplay_actions(static_data, input_types)
    evidence["gameplay_mode"] = mode

    if mode == "mouse":
        new_frames, t = await _run_mouse_gameplay(page, out, t)
    else:
        new_frames, t = await _run_keyboard_gameplay(page, out, t, actions)
    frames.extend(new_frames)

    # Post-gameplay game state snapshot & diff
    post_snap = await _take_game_snapshot(page)
    state_changed, vars_diff = _diff_game_vars(pre_snap, post_snap)
    evidence["gameplay_state_changed"] = state_changed
    evidence["gameplay_vars_diff"] = vars_diff
    if vars_diff:
        logger.info(
            f"[render_test] gameplay state diff: {len(vars_diff)} var(s) changed"
        )

    # Game completion detection
    game_completion = await _detect_game_completion(page)
    evidence["game_completion"] = game_completion
    if game_completion.get("completed"):
        logger.info(
            f"[render_test] game completion detected: state={game_completion.get('state')}"
        )

    return ProbeResult(frames=frames, timestamp=t, evidence=evidence)


# =====================================================================
# Layer 4: Responsive viewport testing
# =====================================================================

async def probe_responsive(
    page, out: Path, t: float, static_data: dict, input_types: set,
) -> ProbeResult:
    """Capture stable screenshots at mobile and tablet viewports."""
    frames: List[AnnotatedFrame] = []

    for width, height, label, desc in _RESPONSIVE_VIEWPORTS:
        try:
            await page.set_viewport_size({"width": width, "height": height})
            await asyncio.sleep(0.8)
            t += 0.8
            p = str(out / f"frame_responsive_{label}.png")
            await safe_page_screenshot(page, path=p)
            frames.append(AnnotatedFrame(
                screenshot_path=p, label=f"responsive_{label}",
                description=desc,
                timestamp=t, layer="responsive",
            ))
        except Exception as e:
            logger.debug(f"[render_test] responsive {label} failed: {e}")

    # Restore original viewport
    try:
        await page.set_viewport_size({"width": VIEWPORT_W, "height": VIEWPORT_H})
        await asyncio.sleep(0.3)
    except Exception:
        pass

    return ProbeResult(frames=frames, timestamp=t)


# =====================================================================
# Probe registry: (probe_fn, condition)
# condition(static_data, input_types) -> bool
# ALL layers (1-4) in execution order.
# =====================================================================

PROBE_REGISTRY = [
    (probe_structural_validation, lambda sd, it: True),                              # L0 structural
    (probe_observation,   lambda sd, it: True),                                     # L1
    (probe_scroll,        lambda sd, it: True),                                     # L2
    (probe_buttons,       lambda sd, it: True),
    (probe_links,         lambda sd, it: True),
    (probe_keyboard,      lambda sd, it: "keyboard" in it),
    (probe_canvas_click,  lambda sd, it: "mouse_click" in it or sd.get("has_canvas")),
    (probe_form,          lambda sd, it: sd.get("has_text_input") or sd.get("has_textarea")),
    (probe_select,        lambda sd, it: sd.get("has_select")),
    (probe_range,         lambda sd, it: sd.get("has_range")),
    (probe_checkbox,      lambda sd, it: sd.get("has_checkbox")),
    (probe_drag,          lambda sd, it: "mouse_drag" in it),
    (probe_deep_gameplay, _is_game),                                                # L3
    (probe_responsive,    lambda sd, it: True),                                     # L4
]
