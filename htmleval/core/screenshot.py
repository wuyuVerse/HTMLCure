"""
Shared Playwright screenshot helpers.

Playwright's native page.screenshot() is occasionally brittle under heavy
benchmark load. Two failure modes matter in practice:

1. The capture path stalls long enough to hit the Playwright timeout.
2. One timeout inside a probe causes that entire probe to lose all frames.

This module adds a small global concurrency gate for screenshot captures and
falls back to Chrome DevTools Protocol capture when the primary API times out.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("htmleval")

_MAX_CONCURRENT_SCREENSHOTS = max(
    1, int(os.getenv("HTMLEVAL_SCREENSHOT_MAX_CONCURRENCY", "8"))
)
_PRIMARY_SCREENSHOT_TIMEOUT_MS = max(
    500, int(os.getenv("HTMLEVAL_SCREENSHOT_PRIMARY_TIMEOUT_MS", "4000"))
)
_CDP_SCREENSHOT_TIMEOUT_MS = max(
    500, int(os.getenv("HTMLEVAL_SCREENSHOT_CDP_TIMEOUT_MS", "5000"))
)
_STEP_TIMEOUT_HEADROOM_MS = max(
    0, int(os.getenv("HTMLEVAL_SCREENSHOT_STEP_TIMEOUT_HEADROOM_MS", "1000"))
)
_SCREENSHOT_SEMAPHORE: asyncio.Semaphore | None = None


def _get_screenshot_semaphore() -> asyncio.Semaphore:
    global _SCREENSHOT_SEMAPHORE
    if _SCREENSHOT_SEMAPHORE is None:
        _SCREENSHOT_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_SCREENSHOTS)
    return _SCREENSHOT_SEMAPHORE


def _primary_timeout_ms(timeout: int | None) -> int:
    if timeout is None or timeout <= 0:
        return _PRIMARY_SCREENSHOT_TIMEOUT_MS
    return max(500, min(timeout, _PRIMARY_SCREENSHOT_TIMEOUT_MS + _STEP_TIMEOUT_HEADROOM_MS))


def _cdp_timeout_ms(timeout: int | None) -> int:
    if timeout is None or timeout <= 0:
        return _CDP_SCREENSHOT_TIMEOUT_MS
    primary_timeout = _primary_timeout_ms(timeout)
    remaining = max(500, timeout - primary_timeout)
    return max(500, min(remaining, _CDP_SCREENSHOT_TIMEOUT_MS + _STEP_TIMEOUT_HEADROOM_MS))


async def _read_font_status(page) -> str:
    try:
        status = await page.evaluate(
            "() => document.fonts ? document.fonts.status : 'unsupported'"
        )
        return str(status)
    except Exception:
        return "unknown"


async def _capture_via_cdp(
    page,
    *,
    path: str | Path | None,
    timeout_ms: int,
    disable_animations: bool,
    capture_beyond_viewport: bool,
) -> bytes:
    session = await page.context.new_cdp_session(page)
    style_handle = None
    try:
        if disable_animations:
            style_handle = await page.add_style_tag(
                content=(
                    "*, *::before, *::after { "
                    "animation: none !important; "
                    "transition: none !important; "
                    "caret-color: transparent !important; "
                    "scroll-behavior: auto !important; "
                    "}"
                )
            )

        resp = await asyncio.wait_for(
            session.send(
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "fromSurface": True,
                    "captureBeyondViewport": capture_beyond_viewport,
                },
            ),
            timeout=timeout_ms / 1000,
        )
        image_bytes = base64.b64decode(resp["data"])
        if path is not None:
            Path(path).write_bytes(image_bytes)
        return image_bytes
    finally:
        if style_handle is not None:
            try:
                await style_handle.evaluate("el => el.remove()")
            except Exception:
                pass
        try:
            await session.detach()
        except Exception:
            pass


async def safe_page_screenshot(
    page,
    *,
    path: str | Path | None = None,
    timeout: int | None = None,
    animations: str | None = None,
    full_page: bool | None = None,
    **kwargs: Any,
) -> bytes:
    """
    Best-effort screenshot capture.

    Strategy:
      1. Limit concurrent captures across the process.
      2. Try native Playwright screenshot with a short timeout budget.
      3. Fall back to CDP capture if the primary path fails.
    """
    options: dict[str, Any] = dict(kwargs)
    if path is not None:
        options["path"] = str(path)
    if animations is not None:
        options["animations"] = animations
    if full_page is not None:
        options["full_page"] = full_page

    primary_timeout = _primary_timeout_ms(timeout)
    cdp_timeout = _cdp_timeout_ms(timeout)

    async with _get_screenshot_semaphore():
        try:
            return await page.screenshot(timeout=primary_timeout, **options)
        except Exception as primary_exc:
            font_status = await _read_font_status(page)
            try:
                result = await _capture_via_cdp(
                    page,
                    path=path,
                    timeout_ms=cdp_timeout,
                    disable_animations=animations == "disabled",
                    capture_beyond_viewport=bool(full_page),
                )
                logger.debug(
                    "[screenshot] CDP fallback succeeded (fonts=%s, primary=%s)",
                    font_status,
                    type(primary_exc).__name__,
                )
                return result
            except Exception as cdp_exc:
                raise RuntimeError(
                    "screenshot failed "
                    f"(primary={type(primary_exc).__name__}: {primary_exc}; "
                    f"cdp={type(cdp_exc).__name__}: {cdp_exc}; "
                    f"fonts={font_status})"
                ) from primary_exc
