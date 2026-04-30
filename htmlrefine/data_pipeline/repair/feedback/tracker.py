"""
RepairTracker — writes repair traces to JSONL for DPO/RLHF training data.

Output format (one JSON per line):
{
  "record_id":      "query_4461",
  "query":          "...",
  "original_score": 52,
  "final_score":    78,
  "improvement":    26,
  "converged":      true,
  "elapsed_s":      45.2,
  "iterations": [
    {
      "iteration":    1,
      "strategy":     "feature_complete",
      "score_before": 52,
      "score_after":  71,
      "delta":        19,
      "html_before":  "...",
      "html_after":   "...",
      "elapsed_s":    18.3,
      "success":      true
    }, ...
  ],
  "best_html":      "...",
  "original_html":  "..."
}

This format directly supports:
  SFT data:    {query, best_html}  when improvement > 0
  DPO pairs:   {query, chosen=best_html, rejected=original_html}
  Repair data: full iteration trace
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import List

from htmlrefine.data_pipeline.repair.engine import RepairResult

logger = logging.getLogger("htmlrefine.repair")


class RepairTracker:
    """
    Thread-safe, buffered writer for repair traces.

    Usage:
        tracker = RepairTracker("eval_results/repair_traces.jsonl", flush_interval=10)
        tracker.record(result)
        tracker.flush()   # call at the end
    """

    def __init__(self, output_path: str, flush_interval: int = 10):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self._buf: List[str] = []
        self._lock = threading.Lock()
        self._count = 0
        self._improved = 0
        self._total_delta = 0

    def record(self, result: RepairResult) -> None:
        """Serialize a RepairResult and buffer it."""
        row = {
            "record_id":       result.record_id,
            "query":           result.query,
            "original_score":  result.original_score,
            "final_score":     result.final_score,
            "improvement":     result.improvement,
            "converged":       result.converged,
            "sft_eligible":    result.sft_eligible,        # reached quality gate + min delta
            "evidence_quality": result.evidence_quality,   # high/medium/low — WHY this strategy
            "elapsed_s":       round(result.elapsed_s, 1),
            "iterations": [
                {
                    "iteration":        it.iteration,
                    "strategy":         it.strategy,
                    "score_before":     it.score_before,
                    "score_after":      it.score_after,
                    "delta":            it.delta,
                    "composite_before": it.composite_before,  # anti-regression score before
                    "composite_after":  it.composite_after,   # anti-regression score after
                    "n_candidates":     it.n_candidates,      # rejection sampling count
                    "elapsed_s":        round(it.elapsed_s, 1),
                    "success":          it.success,
                    "stop_reason":      it.error or "",        # "quality_gate" / "converged" / ""
                    "contrastive":      it.contrastive_summary[:500] if it.contrastive_summary else "",
                    "html_before":      it.html_before,
                    "html_after":       it.html_after,
                }
                for it in result.iterations
            ],
            "best_html":     result.best_html,
            "original_html": result.original_html,
        }
        line = json.dumps(row, ensure_ascii=False)

        with self._lock:
            self._buf.append(line)
            self._count += 1
            if result.improvement > 0:
                self._improved += 1
                self._total_delta += result.improvement
            if self._count % self.flush_interval == 0:
                self._flush()

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for line in self._buf:
                f.write(line + "\n")
        self._buf.clear()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total":    self._count,
                "improved": self._improved,
                "avg_delta": round(self._total_delta / self._improved, 1) if self._improved else 0,
            }

    def export_sft(self, output_path: str, min_score: int = 80) -> int:
        """
        Export (query, html) pairs from repaired records that reached min_score.
        Returns the number of pairs exported.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        if not self.path.exists():
            return 0
        with open(self.path, encoding="utf-8") as fin, \
             open(out, "w", encoding="utf-8") as fout:
            for line in fin:
                row = json.loads(line)
                if row["final_score"] >= min_score and row["improvement"] > 0:
                    fout.write(json.dumps({
                        "query": row["query"],
                        "html":  row["best_html"],
                        "score": row["final_score"],
                        "source": "repaired",
                    }, ensure_ascii=False) + "\n")
                    count += 1
        logger.info(f"Exported {count} SFT pairs → {out}")
        return count
