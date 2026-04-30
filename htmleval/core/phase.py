"""
Phase — abstract base class for all evaluation phases.

A Phase is a pluggable, self-contained evaluation step.
Implement execute() and optionally override gate() / should_stop_pipeline().
Config is injected at construction time.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext, PhaseResult

logger = logging.getLogger("htmleval")


class Phase(ABC):
    """
    Abstract base for evaluation phases.

    Lifecycle (managed by Phase.run() — do not override run()):
        1. gate(ctx)                    → should this phase execute?
        2. execute(ctx)                 → perform work, return PhaseResult
        3. should_stop_pipeline(result) → abort remaining phases?

    To add a custom phase:
        class MyPhase(Phase):
            @property
            def name(self): return "my_phase"

            async def execute(self, ctx):
                # read ctx.html_code, ctx.phase_results, etc.
                return PhaseResult(phase_name=self.name, success=True, data={...})
    """

    def __init__(self, config: EvalConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case identifier, e.g. 'static_analysis'."""

    @abstractmethod
    async def execute(self, ctx: EvalContext) -> PhaseResult:
        """
        Core phase logic.  Read from ctx; do NOT call ctx.add_result() —
        that is done by run().
        """

    def gate(self, ctx: EvalContext) -> bool:
        """Return False to skip this phase entirely. Default: run unless stopped."""
        return not ctx.should_skip

    def should_stop_pipeline(self, result: PhaseResult, ctx: EvalContext) -> bool:
        """Return True to abort all subsequent phases. Default: never."""
        return False

    # ── Runner — do not override ──────────────────────────────

    async def run(self, ctx: EvalContext) -> PhaseResult:
        """Execute the full phase lifecycle. Called by PipelineEngine."""
        if not self.gate(ctx):
            skipped = PhaseResult(
                phase_name=self.name,
                success=True,
                data={"skipped": True, "reason": ctx.skip_reason or "gated"},
            )
            ctx.add_result(skipped)
            return skipped

        t0 = time.monotonic()
        try:
            result = await self.execute(ctx)
        except Exception as e:
            logger.error(f"[{self.name}] unhandled error: {e}", exc_info=True)
            result = PhaseResult(
                phase_name=self.name,
                success=False,
                errors=[f"{type(e).__name__}: {e}"],
            )

        result.duration_ms = (time.monotonic() - t0) * 1000
        ctx.add_result(result)

        if self.should_stop_pipeline(result, ctx):
            ctx.should_skip = True
            ctx.skip_reason = f"stopped by {self.name}"

        return result
