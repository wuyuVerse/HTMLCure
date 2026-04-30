"""
Test-runner action implementations — browser actions + assertions.

Every action has a uniform async signature:

    async def action_xxx(page: Page, step: TestStep, ctx: _RunCtx) -> StepResult

ACTION_DISPATCH maps action name → handler function.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from htmleval.core.screenshot import safe_page_screenshot
from htmleval.phases.test_runner.schema import TestStep, StepResult

logger = logging.getLogger("htmleval")


# ---------------------------------------------------------------------------
# Run context — accumulated state across steps within one test case
# ---------------------------------------------------------------------------

@dataclass
class _RunCtx:
    """Mutable context shared across steps within a single test case."""

    output_dir: Path
    screenshot_idx: int = 0
    last_screenshot: Optional[bytes] = None
    console_errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(action: str, t0: float) -> StepResult:
    return StepResult(action=action, passed=True, duration_ms=(time.monotonic() - t0) * 1000)


def _fail(action: str, t0: float, msg: str) -> StepResult:
    return StepResult(action=action, passed=False, error=msg, duration_ms=(time.monotonic() - t0) * 1000)


async def _first_visible_match(locator, *, target: str) -> Tuple[Any, int, int]:
    """Return the first visible match from a Playwright locator."""
    count = await locator.count()
    if count <= 0:
        raise LookupError(f"No matches for {target}")

    last_exc: Exception | None = None
    for idx in range(count):
        candidate = locator.nth(idx)
        try:
            if await candidate.is_visible():
                return candidate, idx, count
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise LookupError(
            f"No visible matches for {target} (checked {count}; last error: {last_exc})"
        )
    raise LookupError(f"No visible matches for {target} (checked {count})")


def _interaction_settle_ms(step: TestStep) -> int:
    """Return an action-aware settle window for post-interaction UI updates."""
    action = (step.action or "").lower()
    if action in {"hover", "focus", "contextmenu"}:
        return min(max(step.timeout // 30, 40), 80)
    if action in {"type", "press_key", "scroll", "eval_js"}:
        return min(max(step.timeout // 16, 70), 160)
    if action in {"click", "click_text", "select_option", "check", "drag"}:
        return min(max(step.timeout // 10, 120), 240)
    return min(max(step.timeout // 20, 50), 120)


async def _post_interaction_settle(page, step: TestStep) -> None:
    """Give the UI a short time slice to apply state changes."""
    settle_ms = _interaction_settle_ms(step)
    try:
        await page.wait_for_timeout(settle_ms)
    except Exception:
        await asyncio.sleep(settle_ms / 1000)


# ---------------------------------------------------------------------------
# Navigation / wait actions
# ---------------------------------------------------------------------------

async def action_wait_for(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        await page.wait_for_selector(step.selector, timeout=step.timeout)
        return _ok("wait_for", t0)
    except Exception as e:
        return _fail("wait_for", t0, f"Selector {step.selector!r} not found: {e}")


async def action_wait(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    await asyncio.sleep(step.ms / 1000)
    return _ok("wait", t0)


# ---------------------------------------------------------------------------
# Interaction actions
# ---------------------------------------------------------------------------

async def action_click(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        await target.click(timeout=step.timeout)
        await _post_interaction_settle(page, step)
        return _ok("click", t0)
    except Exception as e:
        return _fail("click", t0, f"Click failed on {step.selector!r}: {e}")

async def action_click_text(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        pattern = re.compile(step.text_pattern) if step.text_pattern else re.compile(re.escape(step.text))
        locator = page.get_by_text(pattern, exact=False)
        visible_matches = []
        count = await locator.count()
        for idx in range(count):
            candidate = locator.nth(idx)
            try:
                if await candidate.is_visible():
                    visible_matches.append(candidate)
            except Exception:
                continue
        if len(visible_matches) == 0:
            return _fail(
                "click_text",
                t0,
                f"No visible text match for pattern {pattern.pattern!r}",
            )
        if len(visible_matches) > 1:
            return _fail(
                "click_text",
                t0,
                f"Ambiguous text match for pattern {pattern.pattern!r}: {len(visible_matches)} visible matches",
            )
        await visible_matches[0].click(timeout=step.timeout)
        await _post_interaction_settle(page, step)
        return _ok("click_text", t0)
    except re.error as e:
        return _fail("click_text", t0, f"Invalid text pattern {step.text_pattern!r}: {e}")
    except Exception as e:
        return _fail("click_text", t0, f"click_text failed for pattern {step.text_pattern!r}: {e}")

async def action_eval_js(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        await page.evaluate(step.expression)
        return _ok("eval_js", t0)
    except Exception as e:
        return _fail("eval_js", t0, f"eval_js error: {e}")


async def action_type(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        await target.fill(step.text, timeout=step.timeout)
        await _post_interaction_settle(page, step)
        return _ok("type", t0)
    except Exception as e:
        return _fail("type", t0, f"Type failed on {step.selector!r}: {e}")

async def action_press_key(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        for _ in range(step.repeat):
            await page.keyboard.press(step.key)
            await asyncio.sleep(0.05)
        await _post_interaction_settle(page, step)
        return _ok("press_key", t0)
    except Exception as e:
        return _fail("press_key", t0, f"press_key {step.key!r} failed: {e}")

async def action_scroll(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        direction = step.direction.lower()
        dx, dy = 0, 0
        if direction == "down":
            dy = step.amount
        elif direction == "up":
            dy = -step.amount
        elif direction == "right":
            dx = step.amount
        elif direction == "left":
            dx = -step.amount
        else:
            dy = step.amount  # default: scroll down
        await page.evaluate(f"window.scrollBy({dx}, {dy})")
        await _post_interaction_settle(page, step)
        return _ok("scroll", t0)
    except Exception as e:
        return _fail("scroll", t0, f"scroll failed: {e}")

async def action_drag(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        fx, fy = step.from_pos or [0, 0]
        tx, ty = step.to_pos or [0, 0]
        await page.mouse.move(fx, fy)
        await page.mouse.down()
        # Move in small steps for smoother drag
        steps = 10
        for i in range(1, steps + 1):
            cx = fx + (tx - fx) * i / steps
            cy = fy + (ty - fy) * i / steps
            await page.mouse.move(cx, cy)
            await asyncio.sleep(0.02)
        await page.mouse.up()
        await _post_interaction_settle(page, step)
        return _ok("drag", t0)
    except Exception as e:
        return _fail("drag", t0, f"drag failed: {e}")

async def action_hover(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        await target.hover(timeout=step.timeout)
        await _post_interaction_settle(page, step)
        return _ok("hover", t0)
    except Exception as e:
        return _fail("hover", t0, f"hover failed on {step.selector!r}: {e}")

async def action_focus(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        await target.focus(timeout=step.timeout)
        await _post_interaction_settle(page, step)
        return _ok("focus", t0)
    except Exception as e:
        return _fail("focus", t0, f"focus failed on {step.selector!r}: {e}")

async def action_contextmenu(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    selector = step.selector or "body"
    try:
        locator = page.locator(selector)
        target, _, _ = await _first_visible_match(locator, target=selector)
        await target.click(button="right", timeout=step.timeout)
        await _post_interaction_settle(page, step)
        return _ok("contextmenu", t0)
    except Exception as e:
        return _fail("contextmenu", t0, f"contextmenu failed on {selector!r}: {e}")

async def action_select_option(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        await target.select_option(value=step.value, timeout=step.timeout)
        await _post_interaction_settle(page, step)
        return _ok("select_option", t0)
    except Exception as e:
        return _fail("select_option", t0, f"select_option failed: {e}")

async def action_check(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        await target.check(timeout=step.timeout)
        await _post_interaction_settle(page, step)
        return _ok("check", t0)
    except Exception as e:
        return _fail("check", t0, f"check failed on {step.selector!r}: {e}")

async def action_screenshot(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        label = step.label or f"step_{ctx.screenshot_idx}"
        path = ctx.output_dir / f"{label}.png"
        shot = await safe_page_screenshot(page, path=path, timeout=step.timeout)
        ctx.last_screenshot = shot
        ctx.screenshot_idx += 1
        return _ok("screenshot", t0)
    except Exception as e:
        return _fail("screenshot", t0, f"screenshot failed: {e}")


async def action_resize(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        await page.set_viewport_size({"width": step.width, "height": step.height})
        return _ok("resize", t0)
    except Exception as e:
        return _fail("resize", t0, f"resize failed: {e}")


# ---------------------------------------------------------------------------
# Assertion actions
# ---------------------------------------------------------------------------

async def action_assert_visible(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        visible = await target.is_visible(timeout=step.timeout)
        if visible:
            return _ok("assert_visible", t0)
        return _fail("assert_visible", t0, f"Element {step.selector!r} is not visible")
    except Exception as e:
        return _fail("assert_visible", t0, f"assert_visible error: {e}")

async def action_assert_not_visible(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        count = await locator.count()
        if count == 0:
            return _ok("assert_not_visible", t0)
        for idx in range(count):
            candidate = locator.nth(idx)
            try:
                if await candidate.is_visible():
                    return _fail("assert_not_visible", t0, f"Element {step.selector!r} match #{idx} is visible")
            except Exception as exc:
                return _fail("assert_not_visible", t0, f"assert_not_visible error on {step.selector!r}: {exc}")
        return _ok("assert_not_visible", t0)
    except Exception as e:
        return _fail("assert_not_visible", t0, f"assert_not_visible error: {e}")

async def action_assert_text_contains(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        text = await target.inner_text(timeout=step.timeout)
        pattern = step.text_pattern or step.text
        if re.search(pattern, text, re.IGNORECASE):
            return _ok("assert_text_contains", t0)
        return _fail("assert_text_contains", t0,
                      f"Text {pattern!r} not found in element {step.selector!r}")
    except Exception as e:
        return _fail("assert_text_contains", t0, f"assert_text_contains error: {e}")

async def action_assert_text_not_contains(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        text = await target.inner_text(timeout=step.timeout)
        pattern = step.text_pattern or step.text
        if not re.search(pattern, text, re.IGNORECASE):
            return _ok("assert_text_not_contains", t0)
        return _fail("assert_text_not_contains", t0,
                      f"Text {pattern!r} unexpectedly found in element {step.selector!r}")
    except Exception as e:
        return _fail("assert_text_not_contains", t0, f"assert_text_not_contains error: {e}")

async def action_assert_count(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        count = await page.locator(step.selector).count()
        if step.eq is not None and count != step.eq:
            return _fail("assert_count", t0, f"Expected count={step.eq}, got {count}")
        if step.gte is not None and count < step.gte:
            return _fail("assert_count", t0, f"Expected count>={step.gte}, got {count}")
        if step.lte is not None and count > step.lte:
            return _fail("assert_count", t0, f"Expected count<={step.lte}, got {count}")
        return _ok("assert_count", t0)
    except Exception as e:
        return _fail("assert_count", t0, f"assert_count error: {e}")


async def action_assert_attribute(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        actual = await target.get_attribute(step.attr, timeout=step.timeout)
        if actual == step.value:
            return _ok("assert_attribute", t0)
        return _fail("assert_attribute", t0,
                      f"Attribute {step.attr!r}: expected {step.value!r}, got {actual!r}")
    except Exception as e:
        return _fail("assert_attribute", t0, f"assert_attribute error: {e}")

async def action_assert_console_clean(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    if not ctx.console_errors:
        return _ok("assert_console_clean", t0)
    return _fail("assert_console_clean", t0,
                  f"Found {len(ctx.console_errors)} console error(s): {ctx.console_errors[:3]}")


async def action_assert_screenshot_changed(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        if ctx.last_screenshot is None:
            return _fail(
                "assert_screenshot_changed",
                t0,
                "assert_screenshot_changed requires a prior screenshot step in the same test case",
            )

        new_shot = await safe_page_screenshot(page, timeout=step.timeout)

        from htmleval.phases.render_test.frame_types import frame_diff_score
        diff = frame_diff_score(ctx.last_screenshot, new_shot)
        ctx.last_screenshot = new_shot
        # diff > (1 - threshold) means enough change
        min_diff = 1.0 - step.threshold
        if diff >= min_diff:
            return _ok("assert_screenshot_changed", t0)
        return _fail("assert_screenshot_changed", t0,
                      f"Screenshot diff={diff:.4f} < min_diff={min_diff:.4f}")
    except Exception as e:
        return _fail("assert_screenshot_changed", t0, f"assert_screenshot_changed error: {e}")


async def action_assert_screenshot_not_blank(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        import io
        from PIL import Image

        shot = await safe_page_screenshot(page, timeout=step.timeout)
        img = Image.open(io.BytesIO(shot)).convert("L")
        extrema = img.getextrema()
        # If min == max, the image is a single solid color = blank
        if extrema[0] == extrema[1]:
            return _fail("assert_screenshot_not_blank", t0, "Screenshot is a single solid color (blank)")
        # Also check variance — very low variance means nearly blank
        pixels = list(img.getdata())
        mean = sum(pixels) / len(pixels)
        variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
        if variance < 10:
            return _fail("assert_screenshot_not_blank", t0, f"Screenshot nearly blank (variance={variance:.1f})")
        return _ok("assert_screenshot_not_blank", t0)
    except Exception as e:
        return _fail("assert_screenshot_not_blank", t0, f"assert_screenshot_not_blank error: {e}")


async def action_assert_js_value(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        if not step.expression.strip():
            return _fail("assert_js_value", t0, "assert_js_value requires a non-empty expression")
        actual = await page.evaluate(step.expression)
        if actual == step.expected:
            return _ok("assert_js_value", t0)
        return _fail("assert_js_value", t0,
                      f"JS expression returned {actual!r}, expected {step.expected!r}")
    except Exception as e:
        return _fail("assert_js_value", t0, f"assert_js_value error: {e}")

async def action_assert_style(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        prop = step.property
        locator = page.locator(step.selector)
        target, _, _ = await _first_visible_match(locator, target=step.selector)
        actual = await target.evaluate(
            f"el => getComputedStyle(el)[{prop!r}]", timeout=step.timeout
        )
        if actual == step.expected:
            return _ok("assert_style", t0)
        return _fail("assert_style", t0,
                      f"Style {prop!r}: expected {step.expected!r}, got {actual!r}")
    except Exception as e:
        return _fail("assert_style", t0, f"assert_style error: {e}")


async def action_assert_no_horizontal_scroll(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        has_hscroll = await page.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )
        if not has_hscroll:
            return _ok("assert_no_horizontal_scroll", t0)
        return _fail("assert_no_horizontal_scroll", t0, "Page has horizontal scroll")
    except Exception as e:
        return _fail("assert_no_horizontal_scroll", t0, f"assert_no_horizontal_scroll error: {e}")


async def action_assert_semantic_html(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        missing: list[str] = []
        for tag in (step.tags or []):
            count = await page.locator(tag).count()
            if count == 0:
                missing.append(tag)
        if not missing:
            return _ok("assert_semantic_html", t0)
        return _fail("assert_semantic_html", t0, f"Missing semantic tags: {missing}")
    except Exception as e:
        return _fail("assert_semantic_html", t0, f"assert_semantic_html error: {e}")


async def action_assert_a11y_basic(page, step: TestStep, ctx: _RunCtx) -> StepResult:
    t0 = time.monotonic()
    try:
        issues: list[str] = []

        # Check: all <img> have alt
        imgs_without_alt = await page.evaluate(
            "document.querySelectorAll('img:not([alt])').length"
        )
        if imgs_without_alt > 0:
            issues.append(f"{imgs_without_alt} img(s) missing alt attribute")

        # Check: page has at least one <h1>
        h1_count = await page.locator("h1").count()
        if h1_count == 0:
            issues.append("No <h1> heading found")

        # Check: <html> has lang attribute
        lang = await page.evaluate("document.documentElement.getAttribute('lang')")
        if not lang:
            issues.append("<html> missing lang attribute")

        if not issues:
            return _ok("assert_a11y_basic", t0)
        return _fail("assert_a11y_basic", t0, "; ".join(issues))
    except Exception as e:
        return _fail("assert_a11y_basic", t0, f"assert_a11y_basic error: {e}")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

ACTION_DISPATCH = {
    "wait_for":                     action_wait_for,
    "wait":                         action_wait,
    "click":                        action_click,
    "click_text":                   action_click_text,
    "eval_js":                      action_eval_js,
    "type":                         action_type,
    "press_key":                    action_press_key,
    "scroll":                       action_scroll,
    "drag":                         action_drag,
    "hover":                        action_hover,
    "focus":                        action_focus,
    "contextmenu":                  action_contextmenu,
    "select_option":                action_select_option,
    "check":                        action_check,
    "screenshot":                   action_screenshot,
    "resize":                       action_resize,
    "assert_visible":               action_assert_visible,
    "assert_not_visible":           action_assert_not_visible,
    "assert_text_contains":         action_assert_text_contains,
    "assert_text_not_contains":     action_assert_text_not_contains,
    "assert_count":                 action_assert_count,
    "assert_attribute":             action_assert_attribute,
    "assert_console_clean":         action_assert_console_clean,
    "assert_screenshot_changed":    action_assert_screenshot_changed,
    "assert_screenshot_not_blank":  action_assert_screenshot_not_blank,
    "assert_js_value":              action_assert_js_value,
    "assert_style":                 action_assert_style,
    "assert_no_horizontal_scroll":  action_assert_no_horizontal_scroll,
    "assert_semantic_html":         action_assert_semantic_html,
    "assert_a11y_basic":            action_assert_a11y_basic,
}
