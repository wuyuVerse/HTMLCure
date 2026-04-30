"""
Centralized configuration for the htmleval evaluation framework.

Usage:
    # From YAML file:
    config = load_config("configs/eval.yaml")

    # Programmatically:
    config = EvalConfig(
        agent=AgentConfig(base_url="http://...", api_key="...", model="..."),
        processing=ProcessingConfig(concurrency=32, skip_agent_phase=True),
    )

    # Via environment variables (no config file needed):
    export HTMLEVAL_BASE_URL=http://...
    export HTMLEVAL_API_KEY=your-api-key
    export HTMLEVAL_MODEL=gpt-4o
    config = EvalConfig()
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """LLM config for the browser-use agent (Phase 3)."""
    base_url: str = Field(default_factory=lambda: os.getenv("HTMLEVAL_BASE_URL", ""))
    api_key: str  = Field(default_factory=lambda: os.getenv("HTMLEVAL_API_KEY", "empty"))
    model: str    = Field(default_factory=lambda: os.getenv("HTMLEVAL_MODEL", ""))
    max_steps: int = 30       # max agent steps per record
    timeout: int = 600        # outer wall-clock timeout (seconds)
    llm_timeout: int = 300    # per-step LLM call timeout (seconds)


class EvaluatorConfig(BaseModel):
    """Vision LLM config for final scoring (Phase 4)."""
    base_url: str = Field(default_factory=lambda: os.getenv("HTMLEVAL_BASE_URL", ""))
    api_key: str  = Field(default_factory=lambda: os.getenv("HTMLEVAL_API_KEY", "empty"))
    model: str    = Field(default_factory=lambda: os.getenv("HTMLEVAL_MODEL", ""))
    max_screenshots: int = 14  # max screenshots passed to vision LLM (matches keyframe limit)


class ProcessingConfig(BaseModel):
    """Runtime / concurrency settings."""
    concurrency: int = 8                # legacy total concurrency / evaluation alias
    generation_concurrency: int = 8     # benchmark generation parallelism
    evaluation_concurrency: int = 8     # benchmark evaluation / run_batch parallelism
    vlm_concurrency: int = 8            # vision-LM parallelism
    max_llm_concurrency: int = 8        # legacy alias for VLM concurrency
    overlap_mode: Literal["off", "chunked"] = "off"
    overlap_chunk_size: int = 32        # chunk size for generation/evaluation overlap
    browser_pool_size: int = 16         # pre-warmed browser instances (RenderTest)
    browser_launch_rate: float = 2.0    # max new browser launches per second
    port: int = 8890                    # local HTTP server port for serving HTML
    retry: int = 3                      # retries on transient failures (EAGAIN etc.)
    save_interval: int = 50             # flush output buffer every N records
    resume: bool = True                 # skip already-scored records on restart
    skip_agent_phase: bool = False      # skip Phase 3 (~15 s/rec instead of ~5 min)
    skip_vision_phase: bool = False     # skip Phase 4 (~10 s/rec, no LLM scoring)
    record_timeout: int = 300           # hard per-record wall-clock timeout (seconds)

    @model_validator(mode="after")
    def _sync_compatibility_fields(self):
        fields_set = getattr(self, "__pydantic_fields_set__", set())

        # Backward compatibility: legacy `concurrency` should still behave like
        # the old single knob and seed the split fields when only it is provided.
        if "concurrency" in fields_set:
            if "evaluation_concurrency" not in fields_set:
                self.evaluation_concurrency = self.concurrency
            if "generation_concurrency" not in fields_set:
                self.generation_concurrency = self.concurrency
            if "vlm_concurrency" not in fields_set and "max_llm_concurrency" not in fields_set:
                self.vlm_concurrency = self.concurrency
                self.max_llm_concurrency = self.concurrency

        # New evaluation knob becomes the canonical value for the legacy alias.
        if "evaluation_concurrency" in fields_set:
            self.concurrency = self.evaluation_concurrency

        # Keep the VLM alias pair aligned regardless of which one was supplied.
        if "vlm_concurrency" in fields_set and "max_llm_concurrency" not in fields_set:
            self.max_llm_concurrency = self.vlm_concurrency
        elif "max_llm_concurrency" in fields_set and "vlm_concurrency" not in fields_set:
            self.vlm_concurrency = self.max_llm_concurrency

        return self


class FilterConfig(BaseModel):
    """Thresholds for post-scoring tier assignment."""
    tier_a_threshold: int = 80   # score ≥ this → Tier A (high quality)
    tier_b_threshold: int = 40   # score ≥ this → Tier B (repairable)
    min_html_size: int = 200     # below this → reject before any browser work


class DataConfig(BaseModel):
    """Input / output paths (overridden per-dataset in score_all_data.py)."""
    input: str = ""
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class EvalConfig(BaseModel):
    """Root configuration — single source of truth for the evaluation pipeline."""

    agent:      AgentConfig      = Field(default_factory=AgentConfig)
    evaluator:  EvaluatorConfig  = Field(default_factory=EvaluatorConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    filter:     FilterConfig     = Field(default_factory=FilterConfig)
    data:       DataConfig       = Field(default_factory=DataConfig)
    workspace:  str = "./eval_results"

    # Derived paths --------------------------------------------------------

    @property
    def reports_dir(self) -> Path:
        return Path(self.workspace) / "reports"

    @property
    def completed_dir(self) -> Path:
        return Path(self.workspace) / "completed"

    @property
    def failed_dir(self) -> Path:
        return Path(self.workspace) / "failed"

    @property
    def logs_dir(self) -> Path:
        return Path(self.workspace) / "logs"

    def ensure_dirs(self) -> None:
        for d in (self.reports_dir, self.completed_dir, self.failed_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.data.output_dir:
            Path(self.data.output_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_value(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} placeholders in YAML-loaded config data."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_value(v) for k, v in value.items()}
    return value


def load_config(path: str) -> EvalConfig:
    """Load EvalConfig from a YAML file. Missing fields use class defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env_value(raw)
    return EvalConfig(**raw)
