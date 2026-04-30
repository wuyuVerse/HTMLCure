"""
BrowserPool — pre-warmed, reusable Playwright browser instances with automatic recycling.

Maintains a pool of live Browser instances; each acquire() returns a
fresh BrowserContext (isolated cookies/storage) without spawning a new
Chromium process for every evaluation.

Recycling: each browser is automatically closed and replaced after max_uses
page loads to prevent memory leaks from accumulating across thousands of
evaluations. Without recycling, Chromium processes grow from ~50MB to 500MB+
after a few hundred page loads, eventually triggering OOM/EAGAIN.

Staggered recycling: each browser gets a randomized max_uses (±30%) so they
don't all recycle simultaneously and cause a burst of chromium.launch() calls.
"""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import Optional

from htmleval.concurrency.rate_limiter import TokenBucketRateLimiter
from htmleval.core.page_safety import install_page_safety

logger = logging.getLogger("htmleval")

_BROWSER_ARGS = [
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    # Keep software WebGL available for benchmark categories such as three_3d
    # and particle/scene prompts while still avoiding reliance on host GPU.
    "--ignore-gpu-blocklist",
    "--enable-webgl",
    "--use-angle=swiftshader",
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-ui-for-media-stream",
    "--blink-settings=imagesEnabled=false",  # skip image downloads (faster load)
]

VIEWPORT_W, VIEWPORT_H = 1280, 720


class _TrackedBrowser:
    """Thin wrapper tracking how many times a Browser has been used."""
    __slots__ = ("browser", "uses", "recycle_at")

    def __init__(self, browser, max_uses: int = 200):
        self.browser = browser
        self.uses = 0
        # Stagger recycling: ±30% jitter so browsers don't all recycle at once
        jitter = int(max_uses * 0.3)
        self.recycle_at = max_uses + random.randint(-jitter, jitter)


class BrowserPool:
    """
    Async pool of Playwright Browser instances with staggered automatic recycling.

    Each browser is closed and replaced after a randomized number of page loads
    (max_uses ± 30%). This prevents both Chromium memory leaks AND the cascade
    failure where all browsers recycle simultaneously.

    Usage:
        pool = BrowserPool(max_size=8, launch_rate=2.0)
        await pool.start()
        try:
            async with pool.acquire() as (browser, context, page):
                await page.goto(url)
                ...
        finally:
            await pool.stop()
    """

    def __init__(self, max_size: int = 16, launch_rate: float = 2.0, max_uses: int = 200):
        self.max_size = max_size
        self.max_uses = max_uses
        self._rate_limiter = TokenBucketRateLimiter(rate=launch_rate, burst=min(16, max_size))
        self._playwright   = None
        self._browsers: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._created = 0
        self._recycled = 0
        self._lock    = asyncio.Lock()

    async def start(self) -> None:
        """Initialize Playwright. Browsers are created lazily on acquire()."""
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        logger.info(f"BrowserPool started (max_size={self.max_size}, max_uses={self.max_uses})")

    async def stop(self) -> None:
        """Close all pooled browsers and stop Playwright."""
        while not self._browsers.empty():
            try:
                tb = self._browsers.get_nowait()
                await tb.browser.close()
            except Exception:
                pass
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info(f"BrowserPool stopped (recycled={self._recycled})")
        self._created = 0
        self._recycled = 0

    @asynccontextmanager
    async def acquire(self):
        """Yield (browser, context, page); closes context on exit, recycles if needed."""
        tb = await self._get_browser()
        context = page = None
        recycle_browser = False
        try:
            context = await tb.browser.new_context(
                viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
            )
            await install_page_safety(context)
            page = await context.new_page()
            yield tb.browser, context, page
        except BaseException:
            # asyncio.wait_for cancellation often leaves heavy canvas/WebGL pages
            # burning CPU in Chromium. Do not return that browser to the pool.
            recycle_browser = True
            raise
        finally:
            for obj in (page, context):
                if obj:
                    try:
                        await asyncio.wait_for(obj.close(), timeout=5)
                    except Exception:
                        pass
            tb.uses += 1
            if recycle_browser:
                await self._discard_browser(tb)
            else:
                await self._return_browser(tb)

    async def _get_browser(self) -> _TrackedBrowser:
        # Try to grab an idle browser from the pool
        try:
            return self._browsers.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # Atomically reserve a creation slot, then create OUTSIDE the lock
        should_create = False
        async with self._lock:
            if self._created < self.max_size:
                self._created += 1
                should_create = True

        if should_create:
            try:
                return await self._create_browser()
            except Exception:
                async with self._lock:
                    self._created -= 1
                raise

        # Pool at capacity — wait for a browser to be returned
        try:
            return await asyncio.wait_for(self._browsers.get(), timeout=60)
        except asyncio.TimeoutError:
            logger.error(
                f"BrowserPool: timed out waiting for browser "
                f"(created={self._created}/{self.max_size}, "
                f"available={self._browsers.qsize()})"
            )
            raise

    async def _return_browser(self, tb: _TrackedBrowser) -> None:
        # Recycle if used too many times or process died
        if tb.uses >= tb.recycle_at or not tb.browser.is_connected():
            if tb.uses >= tb.recycle_at:
                self._recycled += 1
                logger.info(
                    f"BrowserPool: recycled browser after {tb.uses} uses "
                    f"(recycle_at={tb.recycle_at}, total recycled={self._recycled})"
                )
            # Close old browser, decrement count.
            # Next _get_browser() call will see _created < max_size and create fresh.
            async with self._lock:
                self._created -= 1
            try:
                await tb.browser.close()
            except Exception:
                pass
            return

        # Browser still healthy — return to pool
        try:
            self._browsers.put_nowait(tb)
        except asyncio.QueueFull:
            async with self._lock:
                self._created -= 1
            try:
                await tb.browser.close()
            except Exception:
                pass

    async def _discard_browser(self, tb: _TrackedBrowser) -> None:
        """Close and remove a browser that was interrupted mid-record."""
        async with self._lock:
            self._created -= 1
            self._recycled += 1
        try:
            await asyncio.wait_for(tb.browser.close(), timeout=5)
        except Exception:
            pass

    async def _create_browser(self) -> _TrackedBrowser:
        """Launch a new Chromium with retry on transient failures."""
        last_err = None
        for attempt in range(3):
            try:
                await self._rate_limiter.acquire()
                browser = await self._playwright.chromium.launch(
                    headless=True, args=_BROWSER_ARGS,
                )
                logger.debug(f"BrowserPool: launched browser ({self._created}/{self.max_size})")
                return _TrackedBrowser(browser, max_uses=self.max_uses)
            except Exception as e:
                last_err = e
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        f"BrowserPool: launch failed (attempt {attempt+1}/3), "
                        f"retry in {wait}s: {e}"
                    )
                    await asyncio.sleep(wait)
        raise last_err  # type: ignore[misc]
