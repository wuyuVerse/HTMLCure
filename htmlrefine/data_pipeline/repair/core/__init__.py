"""repair.core — Diagnostic infrastructure (evidence + diagnosis)."""

from htmlrefine.data_pipeline.repair.core.evidence import Evidence, collect_evidence
from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis, extract_diagnosis

__all__ = ["Evidence", "collect_evidence", "Diagnosis", "extract_diagnosis"]
