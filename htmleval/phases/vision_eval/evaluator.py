"""
Phase 4: VisionEvalPhase — Multi-agent Vision LLM scoring.

Orchestrates a pipeline of EvalAgent instances (Observer → Scorer) that
collectively produce the 5-dimension evaluation scores (0–100 total).

Architecture (perception–judgment separation):
    1. ObserverAgent: sees all screenshots + probe data → outputs factual report
    2. ScorerAgent:   reads Observer report + key screenshots → outputs scores

To extend: add an EvalAgent subclass in agents.py and append to EVAL_AGENTS.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext, PhaseResult
from htmleval.core.phase import Phase
from htmleval.phases.vision_eval.agents import (
    EVAL_AGENTS,
    AgentContext,
    EvalAgent,
    SCORE_DIMENSIONS,
)
from htmleval.phases.vision_eval.llm import (
    call_vlm,
    encode_image_b64,
    set_vision_llm_semaphore,
)
from htmleval.phases.vision_eval.report import generate_report, resolve_report_screenshots

logger = logging.getLogger("htmleval")


# ---------------------------------------------------------------------------
# Backward-compat re-exports (scripts import these from evaluator.py)
# ---------------------------------------------------------------------------
# encode_image_b64 and set_vision_llm_semaphore are imported above and thus
# importable as `from htmleval.phases.vision_eval.evaluator import ...`
__all__ = [
    "VisionEvalPhase",
    "encode_image_b64",
    "set_vision_llm_semaphore",
    "SCORE_DIMENSIONS",
]


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def _empty_static() -> Dict[str, Any]:
    return {
        "html_size": 0, "has_canvas": False, "has_script": False,
        "has_style": False, "has_svg": False,
        "has_requestanimationframe": False,
        "external_resources": [], "issues": [], "input_types": [],
    }


def _empty_eval(msg: str = "Evaluation failed") -> Dict[str, Any]:
    return {
        d: {"score": 0, "reason": msg} for d in SCORE_DIMENSIONS
    } | {
        "total_score": 0, "agent_phase_run": False,
        "bugs": [], "missing_features": [], "highlights": [],
        "improvement_hints": [],
        "summary": f"LLM evaluation failed: {msg}", "error": msg,
    }


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------

class VisionEvalPhase(Phase):
    """Phase 4 — Multi-agent Vision LLM scoring (Observer → Scorer)."""

    def __init__(self, config: EvalConfig):
        super().__init__(config)
        ev = config.evaluator
        self.client = AsyncOpenAI(base_url=ev.base_url, api_key=ev.api_key)
        self.model = ev.model
        self.max_screenshots = ev.max_screenshots

    @property
    def name(self) -> str:
        return "vision_eval"

    async def execute(self, ctx: EvalContext) -> PhaseResult:
        static = self._phase_data(ctx, "static_analysis", _empty_static())
        render = self._phase_data(ctx, "render_test", {})
        agent  = self._phase_data(ctx, "agent_test", {})
        agent_ran = ctx.get_phase("agent_test") is not None
        screenshots = resolve_report_screenshots(ctx, limit=self.max_screenshots)

        eval_result = await self._llm_evaluate(
            query=ctx.query,
            static=static, render=render, agent=agent,
            agent_ran=agent_ran,
            screenshots=screenshots,
        )

        # Benchmark composite scoring: override with deterministic scores
        # when test_runner phase produced results
        test_runner = ctx.get_phase("test_runner")
        if test_runner and test_runner.data.get("test_pass_rate") is not None:
            from htmleval.phases.test_runner.scoring import composite_score
            eval_result = composite_score(
                test_runner_data=test_runner.data,
                static_data=static,
                render_data=render,
                vlm_scores=eval_result,
                has_interaction=ctx.has_interaction,
            )

        ctx.final_score = eval_result

        report_md = generate_report(ctx)
        if ctx.output_dir:
            (Path(ctx.output_dir) / "report.md").write_text(report_md, encoding="utf-8")

        return PhaseResult(
            phase_name=self.name,
            success="error" not in eval_result,
            data=eval_result,
        )

    # ------------------------------------------------------------------
    # Multi-agent pipeline
    # ------------------------------------------------------------------

    async def _llm_evaluate(
        self,
        query: str,
        static: Dict[str, Any],
        render: Dict[str, Any],
        agent: Dict[str, Any],
        agent_ran: bool,
        screenshots: List[str],
    ) -> Dict[str, Any]:
        """Run the agent pipeline sequentially: Observer → TaskAuditor → Scorer."""

        # Build shared context
        agent_ctx = AgentContext(
            query=query,
            static=static,
            render=render,
            agent=agent,
            agent_ran=agent_ran,
            screenshots=screenshots,
            max_screenshots=self.max_screenshots,
            input_types=static.get("input_types", []),
            kb_probed=render.get("keyboard_probed", False),
            kb_responded=render.get("keys_responded", []),
            kb_vis_change=render.get("keyboard_visual_change", False),
            kb_responsive=render.get("keyboard_responsive", False),
            frame_annotations=render.get("frame_annotations", []),
        )

        try:
            # Sequential pipeline: Observer → TaskAuditor → Scorer
            # Scorer's build_content reads task_auditor output from
            # ctx.stage_outputs, so TaskAuditor MUST complete first.
            for eval_agent in EVAL_AGENTS:
                if eval_agent.name == "scorer":
                    analyst_report = agent_ctx.stage_outputs.get("analyst")
                    observer_report = agent_ctx.stage_outputs.get("observer")
                    if not analyst_report and not observer_report:
                        logger.warning(
                            "[vision_eval] skipping scorer because no upstream visual report is available"
                        )
                        continue
                try:
                    result = await self._call_agent(eval_agent, agent_ctx)
                    agent_ctx.stage_outputs[eval_agent.name] = result
                except Exception as e:
                    logger.warning(f"[vision_eval] {eval_agent.name} failed: {e}")
                    agent_ctx.stage_outputs[eval_agent.name] = {}

            return self._merge_results(agent_ctx)

        except Exception as e:
            logger.error(f"Agent pipeline failed: {e}")
            return _empty_eval(str(e))

    async def _call_agent(
        self,
        eval_agent: EvalAgent,
        ctx: AgentContext,
    ) -> Dict[str, Any]:
        """Call a single agent: build content → VLM call → parse response.

        For agents with images (Observer): build_content offloaded to thread pool.
        For text-only agents (TaskAuditor, Scorer): build_content runs inline.
        """
        logger.info(f"[vision_eval] running {eval_agent.name} agent")

        # Only offload to thread when there's CPU-heavy image encoding
        if eval_agent.name in ("observer", "analyst"):
            content = await asyncio.to_thread(eval_agent.build_content, ctx)
        else:
            content = eval_agent.build_content(ctx)

        # Retry loop with consistency validation for scorer
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                raw_json = await call_vlm(
                    client=self.client,
                    model=self.model,
                    content=content,
                    max_tokens=eval_agent.max_tokens,
                    agent_name=eval_agent.name,
                )

                # Consistency check: total=0 but highlights → retry (scorer only)
                if eval_agent.name == "scorer":
                    pre_total = sum(
                        raw_json[k].get("score", 0) for k in SCORE_DIMENSIONS
                        if isinstance(raw_json.get(k), dict)
                    )
                    highlights = raw_json.get("highlights", [])
                    if pre_total == 0 and len(highlights) > 0 and attempt < max_retries:
                        logger.warning(
                            f"[{eval_agent.name}] inconsistent: all scores=0 but "
                            f"{len(highlights)} highlights — retrying"
                        )
                        await asyncio.sleep(2)
                        continue

                return eval_agent.parse_response(raw_json, ctx)

            except RuntimeError:
                # call_vlm exhausted its own retries — propagate
                raise
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"[{eval_agent.name}] parse attempt {attempt}/{max_retries} "
                        f"failed: {e} — retrying"
                    )
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        # Should not reach here, but satisfy type checker
        raise RuntimeError(f"{eval_agent.name} failed after {max_retries} attempts")

    @staticmethod
    def _merge_results(ctx: AgentContext) -> Dict[str, Any]:
        """Merge agent outputs into the final eval result dict.

        Scorer output is the primary result (backward-compatible format).
        Observation + audit reports are attached for downstream use.
        Supports both 2-agent (analyst) and 3-agent (observer+task_auditor) pipelines.
        """
        scorer_result = ctx.stage_outputs.get("scorer", {})

        analyst_report = ctx.stage_outputs.get("analyst")
        if analyst_report:
            # 2-agent pipeline: extract observer-compatible and auditor-compatible parts
            observer_report = analyst_report
            task_auditor_report = {
                "requirements": analyst_report.get("requirements", []),
                "summary": analyst_report.get("summary", {}),
            }
        else:
            # Legacy 3-agent pipeline
            observer_report = ctx.stage_outputs.get("observer", {})
            task_auditor_report = ctx.stage_outputs.get("task_auditor", {})

        # Attach upstream reports for repair system / debugging
        scorer_result["observer_report"] = observer_report
        scorer_result["task_auditor_report"] = task_auditor_report

        return scorer_result

    @staticmethod
    def _phase_data(ctx: EvalContext, name: str, default: Dict) -> Dict:
        r = ctx.get_phase(name)
        return r.data if r is not None else default
