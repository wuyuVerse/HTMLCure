"""
TieredFilter — assigns Tier A/B/C to scored records and splits them.

Tiers:
  A (score ≥ 80): high quality → use directly as SFT data
  B (40 ≤ score < 80): repairable → feed into repair engine
  C (score < 40): low quality → holistic rewrite or discard
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("htmlrefine.filter")


@dataclass
class TieredSplit:
    tier_a: List[dict] = field(default_factory=list)
    tier_b: List[dict] = field(default_factory=list)
    tier_c: List[dict] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        return {"A": len(self.tier_a), "B": len(self.tier_b), "C": len(self.tier_c)}

    def __repr__(self) -> str:
        c = self.counts
        total = sum(c.values())
        return (f"TieredSplit(A={c['A']} {c['A']/total*100:.1f}%  "
                f"B={c['B']} {c['B']/total*100:.1f}%  "
                f"C={c['C']} {c['C']/total*100:.1f}%  total={total})")


def _get_score(rec: dict) -> int:
    """Extract total score from a results.jsonl record."""
    s = rec.get("score", {})
    if isinstance(s, dict):
        return int(s.get("total", 0))
    return int(s or 0)


def split_tiers(
    records: List[dict],
    tier_a: int = 80,
    tier_b: int = 40,
    *,
    require_status: bool = True,
) -> TieredSplit:
    """
    Split records into Tier A / B / C.

    Args:
        records:        list of dicts from results.jsonl
        tier_a:         score threshold for Tier A
        tier_b:         score threshold for Tier B
        require_status: only include records with eval_status="completed"

    Returns:
        TieredSplit with three lists.
    """
    result = TieredSplit()
    for rec in records:
        if require_status and rec.get("eval_status") != "completed":
            continue
        score = _get_score(rec)
        if score >= tier_a:
            result.tier_a.append(rec)
        elif score >= tier_b:
            result.tier_b.append(rec)
        else:
            result.tier_c.append(rec)
    return result


def load_and_split(
    results_path: str,
    tier_a: int = 80,
    tier_b: int = 40,
) -> TieredSplit:
    """Load a results.jsonl file and split into tiers."""
    records = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    split = split_tiers(records, tier_a, tier_b)
    logger.info(f"Loaded {len(records)} records from {results_path} → {split}")
    return split
