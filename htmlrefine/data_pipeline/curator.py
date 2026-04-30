"""
Curator — orchestrates the full data pipeline:
  1. Load scored results across all datasets
  2. Split into Tier A / B / C
  3. Deduplicate Tier A (MinHash)
  4. Repair Tier B (and optionally Tier C) via RepairEngine
  5. Export SFT data + repair traces

Usage:
    python -m htmlrefine.data_pipeline.curator \
        --results-dir eval_results \
        --config configs/refine.example.yaml \
        --output-dir curated

Or programmatically:
    curator = Curator(config)
    await curator.run(results_dir="eval_results", output_dir="curated")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from htmleval import build_pipeline
from htmleval.core.context import EvalContext
from htmlrefine.core.config import AppConfig
from htmlrefine.data_pipeline.filter import HTMLDeduplicator, load_and_split
from htmlrefine.data_pipeline.repair import RepairEngine, RepairTracker
from htmlrefine.data_pipeline.repair.core.diagnosis import extract_diagnosis

logger = logging.getLogger("htmlrefine.curator")

DATASETS = ["spring", "html_data", "query", "html_query", "html", "three", "game_html"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_html(rec: dict) -> str:
    """Extract HTML string from a results.jsonl record."""
    data = rec.get("data", rec)
    messages = data.get("messages", []) if isinstance(data, dict) else []
    if messages:
        return next(
            (m["content"] for m in messages if m.get("role") == "assistant"), ""
        )
    return rec.get("response", rec.get("output", rec.get("completion", "")))


def _get_query(rec: dict) -> str:
    data = rec.get("data", rec)
    messages = data.get("messages", []) if isinstance(data, dict) else []
    if messages:
        return next((m["content"] for m in messages if m.get("role") == "user"), "")
    return rec.get("query", rec.get("instruction", rec.get("prompt", "")))


# ---------------------------------------------------------------------------
# Main Curator
# ---------------------------------------------------------------------------

class Curator:
    """
    Full data curation pipeline: filter → dedup → repair → export.

    Args:
        config: AppConfig with repair and eval settings.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    async def run(
        self,
        results_dir: str = "eval_results",
        output_dir: str = "curated",
        datasets: Optional[List[str]] = None,
        repair_tier_c: bool = False,
        dedup_threshold: float = 0.8,
        sft_min_score: int = 80,
        limit_repair: int = 0,
    ) -> dict:
        """
        Run the full curation pipeline.

        Args:
            results_dir:     directory with per-dataset results.jsonl files
            output_dir:      where to write curated outputs
            datasets:        which datasets to process (None = all)
            repair_tier_c:   also attempt to repair Tier C records
            dedup_threshold: Jaccard threshold for deduplication (default 0.8)
            sft_min_score:   minimum score to include in SFT export (default 80)
            limit_repair:    max records to repair (0 = all; useful for testing)

        Returns:
            Summary dict with counts and timings.
        """
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        datasets = datasets or DATASETS
        cfg = self.config

        # ── 1. Load & split all datasets ──────────────────────────
        all_tier_a, all_tier_b, all_tier_c = [], [], []
        for ds in datasets:
            f = Path(results_dir) / ds / "results.jsonl"
            if not f.exists():
                logger.warning(f"[curator] {ds}: no results.jsonl, skipping")
                continue
            split = load_and_split(str(f), cfg.filter.tier_a_threshold, cfg.filter.tier_b_threshold)
            all_tier_a.extend(split.tier_a)
            all_tier_b.extend(split.tier_b)
            all_tier_c.extend(split.tier_c)
            logger.info(f"[curator] {ds}: A={len(split.tier_a)} B={len(split.tier_b)} C={len(split.tier_c)}")

        logger.info(f"[curator] total: A={len(all_tier_a)} B={len(all_tier_b)} C={len(all_tier_c)}")

        # ── 2. Deduplicate Tier A ──────────────────────────────────
        deduper = HTMLDeduplicator(threshold=dedup_threshold)
        # Enrich records with html field for dedup
        for rec in all_tier_a:
            rec["_html"] = _get_html(rec)
        tier_a_deduped, removed = deduper.deduplicate(all_tier_a, html_key="_html")
        logger.info(f"[curator] Tier A after dedup: {len(tier_a_deduped)} ({removed} dupes removed)")

        # Write Tier A SFT data directly
        sft_a_path = out / "sft_tier_a.jsonl"
        _write_sft(tier_a_deduped, sft_a_path, source="tier_a")

        # ── 3. Repair Tier B (+ optionally C) ─────────────────────
        to_repair = list(all_tier_b)
        if repair_tier_c:
            to_repair.extend(all_tier_c)
        if limit_repair > 0:
            to_repair = to_repair[:limit_repair]

        logger.info(f"[curator] repairing {len(to_repair)} records...")

        tracker = RepairTracker(str(out / "repair_traces.jsonl"))
        pipeline = build_pipeline(cfg)

        # Start BrowserPool
        from htmleval.concurrency.browser_pool import BrowserPool
        pool = BrowserPool(
            max_size=cfg.processing.browser_pool_size,
            launch_rate=cfg.processing.browser_launch_rate,
        )
        await pool.start()
        pipeline.browser_pool = pool

        try:
            engine = RepairEngine(cfg.repair)
            semaphore = asyncio.Semaphore(min(cfg.processing.concurrency, 16))  # cap repair concurrency

            async def repair_one(rec: dict):
                async with semaphore:
                    query = _get_query(rec)
                    html  = _get_html(rec)
                    if not html or not query:
                        return None
                    # Build a minimal EvalContext from the scored record
                    score_obj = rec.get("score", {})
                    game_id   = rec.get("_eval_uid", rec.get("line_number", "unknown"))
                    # Create context with existing scores (skip re-eval of original)
                    ctx = EvalContext(query=query, response=html, game_id=str(game_id))
                    # Inject scores from existing results so diagnosis works without re-eval
                    _inject_scores(ctx, rec)
                    try:
                        result = await engine.repair(ctx, pipeline, cfg)
                        tracker.record(result)
                        return result
                    except Exception as e:
                        logger.error(f"[curator] repair failed for {game_id}: {e}")
                        return None

            tasks = [repair_one(rec) for rec in to_repair]
            results = []
            pbar = tqdm(total=len(tasks), desc="repairing", unit="rec")
            for coro in asyncio.as_completed(tasks):
                r = await coro
                results.append(r)
                pbar.update(1)
            pbar.close()
        finally:
            await pool.stop()

        tracker.flush()

        # ── 4. Export repaired SFT data ───────────────────────────
        repaired_count = tracker.export_sft(str(out / "sft_repaired.jsonl"), sft_min_score)

        # ── 5. Summary ────────────────────────────────────────────
        elapsed = time.time() - t0
        summary = {
            "tier_a_raw":     len(all_tier_a),
            "tier_a_deduped": len(tier_a_deduped),
            "tier_b":         len(all_tier_b),
            "tier_c":         len(all_tier_c),
            "repaired":       len(to_repair),
            "repaired_sft":   repaired_count,
            "tracker_stats":  tracker.stats,
            "elapsed_s":      round(elapsed),
        }
        (out / "curator_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"[curator] done in {elapsed:.0f}s — {summary}")
        return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sft(records: List[dict], path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            row = {
                "query":  _get_query(rec),
                "html":   rec.get("_html") or _get_html(rec),
                "score":  rec.get("score", {}).get("total", 0) if isinstance(rec.get("score"), dict) else 0,
                "source": source,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} SFT pairs → {path}")


def _inject_scores(ctx: EvalContext, rec: dict) -> None:
    """Inject existing eval scores + diagnosis into ctx.final_score so repair works."""
    score_obj = rec.get("score", {})
    if not isinstance(score_obj, dict):
        return

    # Pull structured diagnosis if stored in results.jsonl
    # (present when scored with new EVAL_PROMPT that outputs bugs/missing_features)
    bugs            = score_obj.get("bugs", [])
    missing         = score_obj.get("missing_features", [])
    summary         = score_obj.get("summary", "")
    highlights      = score_obj.get("highlights", [])
    improvement_hints = score_obj.get("improvement_hints", [])

    ctx.final_score = {
        "rendering":     {"score": score_obj.get("rendering",    0),
                          "reason": score_obj.get("rendering_reason", "")},
        "visual_design": {"score": score_obj.get("visual_design", 0),
                          "reason": score_obj.get("visual_design_reason", "")},
        "functionality": {"score": score_obj.get("functionality", 0),
                          "reason": score_obj.get("functionality_reason", "")},
        "interaction":   {"score": score_obj.get("interaction",  0),
                          "reason": score_obj.get("interaction_reason", "")},
        "code_quality":  {"score": score_obj.get("code_quality", 0),
                          "reason": score_obj.get("code_quality_reason", "")},
        "total_score":   score_obj.get("total", 0),
        "bugs":               bugs if isinstance(bugs, list) else [],
        "missing_features":   missing if isinstance(missing, list) else [],
        "highlights":         highlights if isinstance(highlights, list) else [],
        "improvement_hints":  improvement_hints if isinstance(improvement_hints, list) else [],
        "summary":            summary if isinstance(summary, str) else "",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from htmlrefine.core.config import load_config

    p = argparse.ArgumentParser(description="HTMLRefine data curation pipeline")
    p.add_argument("--config",       default="configs/refine.example.yaml")
    p.add_argument("--results-dir",  default="eval_results")
    p.add_argument("--output-dir",   default="curated")
    p.add_argument("--datasets",     nargs="*", default=None)
    p.add_argument("--repair-tier-c",action="store_true")
    p.add_argument("--limit-repair", type=int, default=0, help="Max records to repair (0=all)")
    p.add_argument("--dedup-threshold", type=float, default=0.8)
    p.add_argument("--sft-min-score",   type=int,   default=80)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = load_config(args.config)
    curator = Curator(cfg)
    asyncio.run(curator.run(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        repair_tier_c=args.repair_tier_c,
        dedup_threshold=args.dedup_threshold,
        sft_min_score=args.sft_min_score,
        limit_repair=args.limit_repair,
    ))
