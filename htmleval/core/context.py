"""
EvalContext — shared mutable state that flows through the evaluation pipeline.

Each phase reads previous phase results from ctx and appends its own.
The context also carries extracted artifacts (HTML, paths, screenshots).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PhaseResult:
    """Structured output from a single evaluation phase."""

    phase_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    screenshots: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def __repr__(self) -> str:
        tag = "OK" if self.success else "FAIL"
        return f"<PhaseResult {self.phase_name} {tag} {self.duration_ms:.0f}ms>"


@dataclass
class EvalContext:
    """
    Mutable context that accumulates state across pipeline phases.

    Lifecycle:
        1. Created with (query, response, game_id).
        2. PipelineEngine prepares output_dir and URL paths.
        3. Each phase reads ctx, writes its PhaseResult via add_result().
        4. After all phases, final_score is set by VisionEvalPhase.
    """

    # ── Immutable inputs ──────────────────────────────────────
    query: str                # original user request / task description
    response: str             # LLM-generated HTML response to evaluate
    game_id: str              # stable unique identifier for this record
    variant: str = "default"  # variant label (e.g. "default", "repaired")
    title: str = ""           # optional human-readable title

    # ── Artifacts set during pipeline ─────────────────────────
    html_code: Optional[str] = None        # extracted HTML (set by ExtractPhase)
    html_path: Optional[Path] = None       # path to written game.html
    game_url_file: Optional[str] = None    # file:// URL
    game_url_http: Optional[str] = None    # http:// URL (for browser-use agent)
    output_dir: Optional[Path] = None      # per-record output directory

    # ── Benchmark test cases (set by loader / orchestrator) ──
    test_cases: Optional[list] = None

    # ── Benchmark interaction flag (set by loader / orchestrator) ──
    has_interaction: bool = True

    # ── Phase results (keyed by phase.name) ───────────────────
    phase_results: Dict[str, PhaseResult] = field(default_factory=dict)

    # ── Aggregated screenshots from all phases ────────────────
    all_screenshots: List[str] = field(default_factory=list)

    # ── Final evaluation score (set by VisionEvalPhase) ───────
    final_score: Optional[Dict[str, Any]] = None

    # ── Pipeline control flags ────────────────────────────────
    should_skip: bool = False
    skip_reason: str = ""
    timeout_phase: str = ""
    timeout_elapsed_ms: float = 0.0
    active_phase: str = ""

    # ── Timing ────────────────────────────────────────────────
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    # ── Derived properties ────────────────────────────────────

    @property
    def dir_name(self) -> str:
        """Subdirectory name for this evaluation's artifacts."""
        return f"{self.game_id}_{self.variant}"

    @property
    def elapsed_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def total_score(self) -> int:
        if self.final_score:
            return self.final_score.get("total_score", 0)
        return 0

    @property
    def status(self) -> str:
        if self.final_score and self.skip_reason:
            return "completed_with_fallback"
        if self.should_skip:
            return "failed"
        if self.final_score:
            return "completed"
        # All ran phases succeeded (vision may have been skipped via config)
        if self.phase_results:
            return "completed" if all(v.success for v in self.phase_results.values()) else "failed"
        return "processing"

    # ── Mutations ─────────────────────────────────────────────

    def get_phase(self, name: str) -> Optional[PhaseResult]:
        return self.phase_results.get(name)

    def add_result(self, result: PhaseResult) -> None:
        self.phase_results[result.phase_name] = result
        self.all_screenshots.extend(result.screenshots)
