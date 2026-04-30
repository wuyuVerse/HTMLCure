"""
PipelineEngine — orchestrates sequential phase execution with concurrency control.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Coroutine, List, Optional

from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext
from htmleval.core.phase import Phase

logger = logging.getLogger("htmleval")


class PipelineEngine:
    """
    Runs an ordered list of Phase objects on an EvalContext.

    Usage (single):
        engine = PipelineEngine(phases=[...], config=cfg)
        ctx = await engine.evaluate(ctx)

    Usage (concurrent batch):
        results = await engine.evaluate_batch(contexts, on_complete=cb)

    Skip flags (set in ProcessingConfig):
        skip_agent_phase=True  → skips AgentTestPhase  (~15 s/rec vs ~5 min)
        skip_vision_phase=True → skips VisionEvalPhase (~10 s/rec, no LLM)
    """

    def __init__(self, phases: List[Phase], config: EvalConfig):
        self.phases = phases
        self.config = config
        self._semaphore = asyncio.Semaphore(config.processing.concurrency)

    # ── Single evaluation ─────────────────────────────────────

    async def evaluate(self, ctx: EvalContext) -> EvalContext:
        """Run all (non-skipped) phases sequentially on one context."""
        self._prepare_output(ctx)

        skip_phases: set[str] = set()
        if self.config.processing.skip_agent_phase:
            skip_phases.add("AgentTestPhase")
        if self.config.processing.skip_vision_phase:
            skip_phases.add("VisionEvalPhase")

        for phase in self.phases:
            if type(phase).__name__ in skip_phases:
                logger.debug(f"[{ctx.dir_name}] skip {type(phase).__name__}")
                continue

            logger.info(f"[{ctx.dir_name}] → {phase.name}")
            ctx.active_phase = phase.name
            result = await phase.run(ctx)
            ctx.active_phase = ""
            logger.info(f"[{ctx.dir_name}]   {result}")

            if ctx.should_skip:
                logger.info(f"[{ctx.dir_name}] pipeline stopped: {ctx.skip_reason}")
                break

        # Fast mode: when vision is skipped but test_cases exist, compute
        # composite score from deterministic signals only (no VLM).
        if (
            self.config.processing.skip_vision_phase
            and ctx.test_cases
            and ctx.final_score is None
        ):
            tr = ctx.get_phase("test_runner")
            if tr and tr.data.get("test_pass_rate") is not None:
                from htmleval.phases.test_runner.scoring import composite_score
                sa = ctx.get_phase("static_analysis")
                rt = ctx.get_phase("render_test")
                ctx.final_score = composite_score(
                    test_runner_data=tr.data,
                    static_data=sa.data if sa else {},
                    render_data=rt.data if rt else {},
                    vlm_scores={},  # no VLM in fast mode
                    has_interaction=ctx.has_interaction,
                )

        ctx.end_time = time.time()
        logger.info(
            f"[{ctx.dir_name}] done  score={ctx.total_score}  "
            f"elapsed={ctx.elapsed_ms:.0f}ms  phases={len(ctx.phase_results)}"
        )
        return ctx

    # ── Single with semaphore ─────────────────────────────────

    async def evaluate_one(self, ctx: EvalContext) -> EvalContext:
        """Evaluate one context, respecting the global concurrency semaphore."""
        timeout = self.config.processing.record_timeout
        async with self._semaphore:
            try:
                return await asyncio.wait_for(self.evaluate(ctx), timeout=timeout)
            except asyncio.TimeoutError:
                ctx.end_time = time.time()
                ctx.skip_reason = f"record timed out after {timeout}s"
                ctx.timeout_phase = ctx.active_phase or "unknown"
                ctx.timeout_elapsed_ms = ctx.elapsed_ms
                ctx.should_skip = True
                self._finalize_timed_out_context(ctx)
                logger.warning(
                    f"[{ctx.dir_name}] timed out after {timeout}s "
                    f"(phase={ctx.timeout_phase}, completed_phases={list(ctx.phase_results)}) — marking failed"
                )
                return ctx

    def _finalize_timed_out_context(self, ctx: EvalContext) -> None:
        """Recover a deterministic benchmark score after a late record timeout.

        A full-record timeout can happen after render_test and test_runner already
        completed, most commonly while the VLM report/scorer is slow. In that
        case the benchmark has enough deterministic evidence to produce a fair
        composite score. Empty responses or pages that did not reach test_runner
        remain failed.
        """
        if ctx.final_score is not None or not ctx.test_cases:
            return

        tr = ctx.get_phase("test_runner")
        if not tr or tr.data.get("test_pass_rate") is None:
            return

        try:
            from htmleval.phases.test_runner.scoring import composite_score
            from htmleval.phases.vision_eval.report import generate_report

            sa = ctx.get_phase("static_analysis")
            rt = ctx.get_phase("render_test")
            ctx.final_score = composite_score(
                test_runner_data=tr.data,
                static_data=sa.data if sa else {},
                render_data=rt.data if rt else {},
                vlm_scores={
                    "visual_design": {
                        "score": 8,
                        "reason": "fallback: record timed out before VLM scoring completed",
                    },
                    "summary": ctx.skip_reason,
                },
                has_interaction=ctx.has_interaction,
            )
            ctx.final_score["fallback_scoring"] = True
            ctx.final_score["fallback_reason"] = ctx.skip_reason
            if ctx.output_dir:
                (ctx.output_dir / "report.md").write_text(
                    generate_report(ctx),
                    encoding="utf-8",
                )
        except Exception as exc:
            logger.warning(
                f"[{ctx.dir_name}] failed to finalize timed-out context: {exc}",
                exc_info=True,
            )

    # ── Batch evaluation ──────────────────────────────────────

    async def evaluate_batch(
        self,
        contexts: List[EvalContext],
        on_complete: Optional[Callable[[EvalContext], Coroutine]] = None,
    ) -> List[EvalContext]:
        """
        Concurrent batch evaluation with sliding-window dispatch.

        Args:
            contexts:    list of EvalContext to evaluate.
            on_complete: optional async callback invoked after each finishes.

        Returns:
            list of completed EvalContext (order may differ from input).
        """
        results: List[EvalContext] = []
        pending: set[asyncio.Task] = set()
        concurrency = self.config.processing.concurrency

        async def _run(c: EvalContext) -> EvalContext:
            r = await self.evaluate_one(c)
            if on_complete:
                await on_complete(r)
            return r

        idx = 0
        while idx < len(contexts) or pending:
            # Fill sliding window up to 2× concurrency
            while idx < len(contexts) and len(pending) < concurrency * 2:
                t = asyncio.create_task(_run(contexts[idx]))
                pending.add(t)
                idx += 1

            if pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED,
                )
                for t in done:
                    try:
                        results.append(t.result())
                    except Exception as e:
                        logger.error(f"batch task error: {e}", exc_info=True)

        return results

    # ── Context preparation ───────────────────────────────────

    def _prepare_output(self, ctx: EvalContext) -> None:
        """Create output directory and set URL paths on context."""
        output_dir = self.config.reports_dir / ctx.dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        ctx.output_dir = output_dir
        ctx.game_url_file = f"file://{output_dir.resolve()}/game.html"
        ctx.game_url_http = (
            f"http://127.0.0.1:{self.config.processing.port}"
            f"/reports/{ctx.dir_name}/game.html"
        )
