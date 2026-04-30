"""
htmlrefine configuration — extends htmleval.EvalConfig with repair-specific settings.

Division of responsibility:
  htmleval.EvalConfig   — all evaluation pipeline config (agent, evaluator, processing,
                          filter, data, workspace, derived path properties)
  htmlrefine.AppConfig  — EvalConfig + RepairConfig + ClusterConfig + VolumeMount
"""

from __future__ import annotations

from typing import List

import yaml
from pydantic import BaseModel, Field

from htmleval.core.config import (  # re-export for backward compat
    EvalConfig,
    AgentConfig,
    EvaluatorConfig,
    ProcessingConfig,
    FilterConfig,
    DataConfig,
    _expand_env_value,
    load_config as _eval_load_config,
)

__all__ = [
    "AppConfig",
    "EvalConfig",
    "AgentConfig",
    "EvaluatorConfig",
    "ProcessingConfig",
    "FilterConfig",
    "DataConfig",
    "RepairConfig",
    "ClusterConfig",
    "VolumeMount",
    "load_config",
]


# ---------------------------------------------------------------------------
# htmlrefine-only sub-configs (not in htmleval)
# ---------------------------------------------------------------------------

class RepairConfig(BaseModel):
    """Iterative repair pipeline settings."""
    base_url: str = ""
    api_key: str = "empty"
    model: str = ""
    max_iterations: int = 8
    improvement_threshold: float = 2.0
    strategies: List[str] = Field(default_factory=lambda: [
        "bug_fix", "feature_complete", "visual_enhance", "holistic_rewrite",
        "fix_playability", "fix_interaction", "fix_game",
        "polish_visual", "enhance_interaction", "refine_functionality", "code_cleanup",
        "visual_enrichment",
    ])
    # Rejection sampling: generate N candidates per iteration, keep the best.
    # Trades cost for significantly higher quality. Disabled for score >= 70
    # (diminishing returns). Set to 1 to disable.
    n_candidates: int = 3
    # Only emit SFT pair when final score meets this threshold.
    quality_gate_score: int = 80
    # Minimum score improvement required before a repaired HTML qualifies as SFT data.
    min_improvement_for_sft: int = 10
    # Pass current-state screenshots to repair LLM (multimodal repair context).
    vision_in_repair: bool = True
    # Use VLM to compare before/after screenshots each iteration (contrastive feedback).
    contrastive_feedback: bool = True
    # VLM-driven visual enrichment after functional convergence (score >= 80).
    visual_enrichment: bool = True
    visual_enrichment_max_iters: int = 2


class ClusterConfig(BaseModel):
    """Optional distributed job settings for private deployments."""
    num_jobs: int = 64
    code_dir: str = ""
    name_prefix: str = "html-eval"
    image: str = ""
    image_version: str = ""
    image_url: str = ""
    image_type: str = "custom"
    priority: str = "high"
    compute_pool: str = ""
    instance: str = ""
    gpu_per_pod: int = 8


class VolumeMount(BaseModel):
    mount_dir: str
    volume_id: int


# ---------------------------------------------------------------------------
# Root config: eval + repair
# ---------------------------------------------------------------------------

class AppConfig(EvalConfig):
    """
    Full configuration for htmlrefine (eval pipeline + repair engine).

    Inherits all eval fields from htmleval.EvalConfig:
      agent, evaluator, processing, filter, data, workspace
      reports_dir, completed_dir, failed_dir, logs_dir, ensure_dirs()

    Adds repair-specific fields:
      repair, cluster, volumes
    """

    workspace: str = "./eval_workspace"   # override EvalConfig default

    repair:  RepairConfig  = Field(default_factory=RepairConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    volumes: List[VolumeMount] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> AppConfig:
    """Load AppConfig from a YAML file. Missing fields use class defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env_value(raw)
    return AppConfig(**raw)
