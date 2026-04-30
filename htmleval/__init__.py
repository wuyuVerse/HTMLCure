"""
htmleval — Universal HTML evaluation framework.

Quick start:
    from htmleval import EvalConfig, build_pipeline, run_batch

    config = EvalConfig()          # or load_config("configs/eval.example.yaml")
    pipeline = build_pipeline(config)
    summary = asyncio.run(run_batch(pipeline, config, dataset_name="my_data"))
"""

import os

# Playwright screenshots wait on document.fonts.ready by default, which is
# unnecessarily brittle for benchmark pages that load remote fonts/CDN assets.
# Skipping this extra wait removes a major source of false timeouts while the
# page render itself is still gated by normal navigation readiness.
os.environ.setdefault("PW_TEST_SCREENSHOT_NO_FONTS_READY", "1")

from htmleval.core.config import EvalConfig, load_config
from htmleval.core.context import EvalContext, PhaseResult
from htmleval.core.pipeline import PipelineEngine
from htmleval.batch.orchestrator import run_batch

__version__ = "0.1.0"
__all__ = [
    "EvalConfig",
    "load_config",
    "EvalContext",
    "PhaseResult",
    "PipelineEngine",
    "run_batch",
]


def build_pipeline(config: "EvalConfig", browser_pool=None) -> PipelineEngine:
    """
    Build the standard 5-phase evaluation pipeline.

    Phases:
        0. ExtractPhase        — extract HTML from LLM response
        1. StaticAnalysisPhase — structural analysis (no browser)
        2. RenderTestPhase     — headless Playwright rendering
      2.5. TestRunnerPhase     — benchmark test case execution (gated on test_cases)
        3. AgentTestPhase      — autonomous browser-use agent testing
        4. VisionEvalPhase     — Vision LLM multi-dimensional scoring

    Args:
        config:       EvalConfig (from load_config or constructed directly).
        browser_pool: optional pre-warmed BrowserPool for RenderTestPhase.
                      If None, a standalone browser is launched per record.

    Returns:
        Configured PipelineEngine ready to call .evaluate() or .evaluate_batch().
    """
    from htmleval.phases.extract import ExtractPhase
    from htmleval.phases.static_analysis.analyzer import StaticAnalysisPhase
    from htmleval.phases.render_test.renderer import RenderTestPhase
    from htmleval.phases.test_runner.runner import TestRunnerPhase
    from htmleval.phases.agent_test.runner import AgentTestPhase
    from htmleval.phases.vision_eval.evaluator import VisionEvalPhase

    phases = [
        ExtractPhase(config),                              # Phase 0
        StaticAnalysisPhase(config),                       # Phase 1
        RenderTestPhase(config, pool=browser_pool),        # Phase 2
        TestRunnerPhase(config, pool=browser_pool),        # Phase 2.5
        AgentTestPhase(config),                            # Phase 3
        VisionEvalPhase(config),                           # Phase 4
    ]
    return PipelineEngine(phases=phases, config=config)
