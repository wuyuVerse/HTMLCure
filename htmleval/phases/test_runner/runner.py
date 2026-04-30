"""
Phase 2.5: TestRunnerPhase — execute benchmark test cases against rendered HTML.

Gated on ctx.test_cases being non-None, so existing 97K data without test_cases
is completely unaffected. Uses BrowserPool when available, falls back to standalone.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext, PhaseResult
from htmleval.core.phase import Phase
from htmleval.core.page_safety import install_page_safety
from htmleval.phases.test_runner.executor import execute_all
from htmleval.phases.test_runner.schema import TestCaseResult, parse_test_cases

logger = logging.getLogger("htmleval")


def _is_resource_error(exc: Exception) -> bool:
    """Check if error is a transient browser resource issue."""
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
        or "browser closed" in msg
        or "browser has been closed" in msg
    )


class TestRunnerPhase(Phase):
    """Phase 2.5 — Execute benchmark test cases via Playwright."""

    @property
    def name(self) -> str:
        return "test_runner"

    def __init__(self, config: EvalConfig, pool=None):
        super().__init__(config)
        self._pool = pool

    def gate(self, ctx: EvalContext) -> bool:
        """Only run when test_cases are present and pipeline hasn't been stopped."""
        return not ctx.should_skip and ctx.test_cases is not None

    async def execute(self, ctx: EvalContext) -> PhaseResult:
        game_url = ctx.game_url_file or ctx.game_url_http
        if not game_url:
            return PhaseResult(
                phase_name=self.name, success=False,
                errors=["No game URL available"],
                data={"test_pass_rate": 0.0, "tests_total": 0, "tests_passed": 0, "tests_failed": 0},
            )

        tc_dir = Path(ctx.output_dir) / "test_runner" if ctx.output_dir else Path("/tmp/test_runner")
        tc_dir.mkdir(parents=True, exist_ok=True)

        test_cases = ctx.test_cases or []
        if test_cases and isinstance(test_cases[0], dict):
            test_cases = parse_test_cases(test_cases)
            ctx.test_cases = test_cases

        results = await self._run_with_retries(game_url, test_cases, tc_dir)

        total_w = sum(r.weight for r in results)
        pass_w = sum(r.weight for r in results if r.passed)
        pass_rate = pass_w / total_w if total_w > 0 else 0.0
        serialized_results = [asdict(r) for r in results]
        serialized_specs = [_serialize_test_case(tc) for tc in test_cases]
        sidecar_payload = {
            "test_pass_rate": round(pass_rate, 4),
            "tests_total": len(results),
            "tests_passed": sum(1 for r in results if r.passed),
            "tests_failed": sum(1 for r in results if not r.passed),
            "results": serialized_results,
            "test_case_specs": serialized_specs,
        }
        sidecar_path = tc_dir / "results.json"
        sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return PhaseResult(
            phase_name=self.name,
            success=True,
            data={
                "test_pass_rate": round(pass_rate, 4),
                "tests_total": len(results),
                "tests_passed": sum(1 for r in results if r.passed),
                "tests_failed": sum(1 for r in results if not r.passed),
                "results": serialized_results,
                "test_case_specs": serialized_specs,
                "results_path": str(sidecar_path),
            },
        )

    async def _run_with_retries(
        self,
        game_url: str,
        test_cases: list,
        output_dir: Path,
    ) -> List[TestCaseResult]:
        """Execute with retry/backoff for transient browser errors."""
        for attempt in range(3):
            try:
                if self._pool:
                    async with self._pool.acquire() as (_, _, page):
                        return await execute_all(page, test_cases, output_dir, game_url)
                else:
                    return await self._run_standalone(game_url, test_cases, output_dir)
            except Exception as exc:
                if _is_resource_error(exc) and attempt < 2:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        f"[test_runner] resource error (attempt {attempt + 1}/3): {exc} "
                        f"— retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
        # Should not reach here
        return []

    async def _run_standalone(
        self,
        game_url: str,
        test_cases: list,
        output_dir: Path,
    ) -> List[TestCaseResult]:
        """Run without BrowserPool — launch a temporary browser."""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                context = await browser.new_context()
                await install_page_safety(context)
                page = await context.new_page()
                return await execute_all(page, test_cases, output_dir, game_url)
            finally:
                await browser.close()


def _serialize_test_case(test_case) -> dict:
    """Return a compact, score-friendly view of the original testcase spec."""
    return {
        "id": test_case.id,
        "name": test_case.name,
        "weight": test_case.weight,
        "steps": [
            {
                "action": step.action,
                "selector": step.selector,
                "expression": step.expression,
                "key": step.key,
                "text": step.text,
                "text_pattern": step.text_pattern,
                "tags": list(step.tags or []),
            }
            for step in test_case.steps
        ],
    }
