"""
Test-case execution engine — runs test cases against a loaded page.

execute_test_case()  → run one TestCase, stop on first assert failure
execute_all()        → run all test cases, page.reload() between each
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, List

from htmleval.core.page_safety import install_page_safety
from htmleval.phases.test_runner.actions import ACTION_DISPATCH, _RunCtx
from htmleval.phases.test_runner.schema import TestCase, TestCaseResult, StepResult

logger = logging.getLogger("htmleval")

_RELOAD_WAIT_STRATEGIES = [
    ("domcontentloaded", 15_000),
    ("load",            20_000),
    ("commit",           8_000),
]
_POST_RELOAD_SETTLE_S = 0.2
_RETRYABLE_PAGE_ERROR_SNIPPETS = (
    "target page, context or browser has been closed",
    "target closed",
    "page crashed",
    "browser has been closed",
    "browser closed",
    "page is closed",
    "context has been closed",
)


async def execute_test_case(
    page,
    tc: TestCase,
    output_dir: Path,
    timeout_ms: int = 15000,
    game_url: str | None = None,
    retry_on_page_failure: bool = True,
) -> tuple[TestCaseResult, Any]:
    """Execute a single test case step-by-step.

    On assert failure → mark the case failed, skip remaining steps.
    Non-assert steps that fail → also mark failed and stop.
    If the failure indicates a page crash / closed target, retry once on a
    fresh page when game_url is available.
    """
    active_page = page
    step_results: List[StepResult] = []
    passed = True
    error_msg = ""

    for attempt in range(2 if (retry_on_page_failure and game_url) else 1):
        ctx = _RunCtx(output_dir=output_dir)
        step_results = []
        passed = True
        error_msg = ""

        try:
            result = await asyncio.wait_for(
                _run_steps(active_page, tc, ctx, step_results),
                timeout=timeout_ms / 1000,
            )
            passed = result
        except asyncio.TimeoutError:
            passed = False
            error_msg = f"Test case timed out after {timeout_ms}ms"
        except Exception as e:
            passed = False
            error_msg = f"Unexpected error: {e}"

        if not passed and not error_msg and step_results:
            for sr in step_results:
                if not sr.passed:
                    error_msg = sr.error
                    break

        if passed:
            return (
                TestCaseResult(
                    id=tc.id,
                    name=tc.name,
                    weight=tc.weight,
                    passed=True,
                    error="",
                    step_results=step_results,
                ),
                active_page,
            )

        retryable = _is_retryable_page_failure(error_msg)
        if not retryable and step_results:
            retryable = any(
                _is_retryable_page_failure(sr.error)
                for sr in step_results
                if not sr.passed
            )

        if attempt == 0 and retryable and game_url:
            logger.warning(
                f"[test_runner] retrying {tc.id} after page failure: {error_msg or 'retryable step failure'}"
            )
            try:
                active_page = await _recover_fresh_page(active_page, game_url)
                continue
            except Exception as recovery_exc:
                error_msg = (
                    f"{error_msg} (page recovery failed: {recovery_exc})"
                    if error_msg
                    else f"Page recovery failed: {recovery_exc}"
                )
        break

    if not passed and not error_msg and step_results:
        for sr in step_results:
            if not sr.passed:
                error_msg = sr.error
                break

    return (
        TestCaseResult(
            id=tc.id,
            name=tc.name,
            weight=tc.weight,
            passed=passed,
            error=error_msg,
            step_results=step_results,
        ),
        active_page,
    )


async def _run_steps(
    page,
    tc: TestCase,
    ctx: _RunCtx,
    step_results: List[StepResult],
) -> bool:
    """Run all steps in a test case. Returns True if all passed."""
    for step in tc.steps:
        handler = ACTION_DISPATCH.get(step.action)
        if handler is None:
            sr = StepResult(action=step.action, passed=False, error=f"Unknown action: {step.action}")
            step_results.append(sr)
            return False

        sr = await handler(page, step, ctx)
        step_results.append(sr)

        if not sr.passed:
            return False
    return True


async def execute_all(
    page,
    test_cases: List[TestCase],
    output_dir: Path,
    game_url: str,
) -> List[TestCaseResult]:
    """Execute all test cases, reloading the page between each.

    Flow:
    1. For each test case:
       a. Reload page to reset JS state
       b. Execute the test case
    2. Return all results
    """
    results: List[TestCaseResult] = []

    for tc in test_cases:
        try:
            page = await _reload_or_recover(page, game_url)
        except Exception as e:
            logger.warning(f"[test_runner] page reload failed for {tc.id}: {e}")
            results.append(TestCaseResult(
                id=tc.id, name=tc.name, weight=tc.weight,
                passed=False, error=f"Page reload failed: {e}",
            ))
            continue

        tc_dir = output_dir / tc.id
        tc_dir.mkdir(parents=True, exist_ok=True)

        result, page = await execute_test_case(page, tc, tc_dir, game_url=game_url)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        logger.debug(f"[test_runner] {tc.id} ({tc.name}): {status}")

    return results


async def _reload_or_recover(page, game_url: str):
    try:
        await _reload_page(page, game_url)
        return page
    except Exception as exc:
        if not _is_retryable_page_failure(str(exc)):
            raise
        logger.warning(f"[test_runner] reload hit page failure, creating fresh page: {exc}")
        return await _recover_fresh_page(page, game_url)


async def _recover_fresh_page(page, game_url: str):
    """Create a new page in the same context and navigate to the benchmark URL."""
    context = getattr(page, "context", None)
    if callable(context):
        context = context()
    if context is None:
        raise RuntimeError("Cannot recover page without an attached browser context")

    try:
        await install_page_safety(context)
    except Exception:
        pass

    new_page = await context.new_page()
    try:
        await _reload_page(new_page, game_url)
    except Exception:
        try:
            await new_page.close()
        except Exception:
            pass
        raise

    try:
        await page.close()
    except Exception:
        pass

    return new_page


async def _reload_page(page, game_url: str) -> None:
    """Reload with progressive fallback.

    Benchmark pages often include heavy canvas loops, CDN scripts, and webfonts.
    Waiting for full `load` with a short timeout is unnecessarily brittle for
    per-test resets; `domcontentloaded` is usually sufficient because the key
    blocking scripts have already executed by then.
    """
    last_exc = None
    for wait_until, timeout_ms in _RELOAD_WAIT_STRATEGIES:
        try:
            await page.goto(game_url, wait_until=wait_until, timeout=timeout_ms)
            try:
                await page.wait_for_selector("body", timeout=2_000)
            except Exception:
                pass
            await asyncio.sleep(_POST_RELOAD_SETTLE_S)
            return
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Page reload failed without an exception")


def _is_retryable_page_failure(error_text: str) -> bool:
    text = (error_text or "").lower()
    return any(snippet in text for snippet in _RETRYABLE_PAGE_ERROR_SNIPPETS)
