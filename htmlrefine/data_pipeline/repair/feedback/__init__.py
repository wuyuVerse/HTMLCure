"""repair.feedback — Iteration feedback and tracking."""

from htmlrefine.data_pipeline.repair.feedback.contrastive import (
    ContrastiveReport, generate_contrastive_feedback, format_contrastive_feedback,
)
from htmlrefine.data_pipeline.repair.feedback.tracker import RepairTracker

__all__ = [
    "ContrastiveReport", "generate_contrastive_feedback", "format_contrastive_feedback",
    "RepairTracker",
]
