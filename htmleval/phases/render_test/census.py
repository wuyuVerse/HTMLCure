"""
Element census for render_test phase.

Post-probe analysis that clicks visible buttons to measure response rate.
Not an interaction probe — runs after keyframe selection as optional enrichment.
"""
from __future__ import annotations

import asyncio
import logging

from htmleval.core.screenshot import safe_page_screenshot
from htmleval.phases.render_test.frame_types import frame_changed
from htmleval.phases.render_test.probes import BUTTON_SELECTORS

logger = logging.getLogger("htmleval")


async def run_element_census(page, data: dict) -> None:
    """Click visible buttons to measure response rate. Mutates *data* in-place."""
    try:
        if page.is_closed():
            return
        all_btns = await page.query_selector_all(BUTTON_SELECTORS)
        visible_btns = []
        for btn in all_btns[:8]:
            try:
                if page.is_closed():
                    break
                if await btn.is_visible():
                    visible_btns.append(btn)
            except Exception:
                pass

        responsive = 0
        tested = 0
        for btn in visible_btns[:6]:
            try:
                if page.is_closed():
                    break
                before_bytes = await safe_page_screenshot(page)
                await btn.click(timeout=2000, no_wait_after=True)
                await asyncio.sleep(0.3)
                if page.is_closed():
                    break
                after_bytes = await safe_page_screenshot(page)
                tested += 1
                if frame_changed(before_bytes, after_bytes):
                    responsive += 1
            except Exception:
                pass

        data["buttons_total"] = len(visible_btns)
        data["buttons_tested"] = tested
        data["buttons_responsive_census"] = responsive
        data["button_response_rate"] = (
            round(responsive / tested, 2) if tested > 0 else -1
        )
    except Exception as e:
        logger.debug(f"[render_test] element census failed: {e}")
