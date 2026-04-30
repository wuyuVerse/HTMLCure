"""
Batch orchestrator — processes JSONL datasets with concurrency, checkpoint/resume,
and per-dataset progress tracking.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext
from htmleval.core.pipeline import PipelineEngine

logger = logging.getLogger("htmleval")


# ---------------------------------------------------------------------------
# Record parsing — supports multiple JSONL formats
# ---------------------------------------------------------------------------

def parse_record(rec: dict, idx: int) -> Tuple[str, str, str]:
    """Extract (query, response, stable_id) from any supported record format.

    Supported formats
    -----------------
    Primary (data pipeline output):
        {"data": {"messages": [{"role":"user","content":"..."},
                               {"role":"assistant","content":"..."}]},
         "line_number": N, ...}

    Fallback (legacy / other sources):
        {"query": "...", "response": "...", ...}
        {"instruction": "...", "output": "...", ...}
        {"prompt": "...", "completion": "...", ...}
    """
    data = rec.get("data", rec)
    messages = data.get("messages", []) if isinstance(data, dict) else []

    if messages:
        query    = next((m["content"] for m in messages if m.get("role") == "user"), "")
        response = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
    else:
        query    = rec.get("query",    rec.get("instruction", rec.get("prompt", "")))
        response = rec.get("response", rec.get("output",      rec.get("completion", "")))

    stable_id = (
        str(data.get("data_id", rec.get("line_number", idx)))
        if isinstance(data, dict) else str(idx)
    )
    return query, response, stable_id


def record_uid(rec: dict, idx: int) -> str:
    """Stable content-hash UID for checkpoint/resume."""
    query, response, _ = parse_record(rec, idx)
    raw = (query + response)[:2000]
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{idx:06d}_{h}"


def load_records_from_jsonl(input_path: str | Path) -> List[dict]:
    """Load JSONL records and annotate them with their original line index."""
    records: List[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
                if isinstance(loaded, dict):
                    rec = dict(loaded)
                    rec["_line_idx"] = i
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def load_done_set(output_path: str) -> Set[str]:
    """Return UIDs of already-completed records from ALL shard files in the same directory.

    Scans all shard_*.jsonl files (not just the current shard's file) so that
    resume works correctly even when num_shards changes between runs.
    Also reads results.jsonl (non-sharded output) if present.
    """
    done: Set[str] = set()
    p = Path(output_path)
    output_dir = p.parent

    if not output_dir.exists():
        return done

    # Scan ALL shard files + results.jsonl in the output directory
    candidates = list(output_dir.glob("shard_*.jsonl"))
    results_file = output_dir / "results.jsonl"
    if results_file.exists():
        candidates.append(results_file)

    for filepath in candidates:
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        uid = rec.get("_eval_uid", "")
                        # Only skip completed records on resume — failed records
                        # should be retried (e.g., browser timeout, VLM error).
                        if uid and rec.get("eval_status") in {"completed", "completed_with_fallback"}:
                            done.add(uid)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    return done


def build_eval_context_from_record(rec: dict, dataset_name: str = "") -> EvalContext:
    """Build an EvalContext from a raw benchmark record."""
    idx = int(rec.get("_line_idx", 0))
    query, response, stable_id = parse_record(rec, idx)
    game_id = f"{dataset_name}_{stable_id}" if dataset_name else stable_id

    ctx = EvalContext(
        query=query,
        response=response,
        game_id=game_id,
        title=rec.get("title", ""),
        variant=rec.get("variant", "default"),
        has_interaction=rec.get("has_interaction", True),
    )

    raw_tc = rec.get("test_cases")
    if raw_tc:
        from htmleval.phases.test_runner.schema import parse_test_cases

        ctx.test_cases = parse_test_cases(raw_tc)

    ctx._source_record = rec  # type: ignore[attr-defined]
    return ctx


def _build_score_payload(ctx: EvalContext) -> Dict[str, Any]:
    """Build the JSON-ready score payload for a completed context."""
    score_obj: Dict[str, Any] = {"total": ctx.total_score}
    extract_phase = ctx.get_phase("extract")
    render_phase = ctx.get_phase("render_test")
    test_runner_phase = ctx.get_phase("test_runner")

    if ctx.final_score:
        for k in ("rendering", "visual_design", "functionality", "interaction", "code_quality"):
            v = ctx.final_score.get(k, {})
            score_obj[k] = v.get("score", 0) if isinstance(v, dict) else 0
            if isinstance(v, dict) and v.get("reason"):
                score_obj[f"{k}_reason"] = v["reason"]

        for field in ("bugs", "missing_features", "highlights", "improvement_hints"):
            val = ctx.final_score.get(field, [])
            if isinstance(val, list) and val:
                score_obj[field] = val

        for field in ("summary",):
            val = ctx.final_score.get(field, "")
            if isinstance(val, str) and val:
                score_obj[field] = val

        for field in ("observer_report", "task_auditor_report"):
            val = ctx.final_score.get(field)
            if isinstance(val, dict) and val:
                score_obj[field] = val

        if test_runner_phase and test_runner_phase.data.get("test_pass_rate") is not None:
            score_obj["test_pass_rate"] = test_runner_phase.data["test_pass_rate"]
            score_obj["tests_passed"] = test_runner_phase.data.get("tests_passed", 0)
            score_obj["tests_total"] = test_runner_phase.data.get("tests_total", 0)

    return score_obj


def serialize_eval_context_result(ctx: EvalContext, source_record: dict, dataset_name: str = "") -> Dict[str, Any]:
    """Serialize an EvalContext into the JSON object written to output JSONL."""
    out = {
        **source_record,
        "score": _build_score_payload(ctx),
        "dataset": dataset_name,
        "eval_status": ctx.status,
        "report": str(ctx.output_dir / "report.md") if ctx.output_dir else None,
    }
    if ctx.skip_reason:
        out["skip_reason"] = ctx.skip_reason
        out["timeout_phase"] = ctx.timeout_phase
        out["timeout_elapsed_ms"] = round(ctx.timeout_elapsed_ms, 1)
        out["completed_phases_before_timeout"] = list(ctx.phase_results)
    if ctx.final_score and ctx.final_score.get("fallback_scoring"):
        out["fallback_scoring"] = True
        out["fallback_reason"] = ctx.final_score.get("fallback_reason", ctx.skip_reason)

    extract_phase = ctx.get_phase("extract")
    render_phase = ctx.get_phase("render_test")
    test_runner_phase = ctx.get_phase("test_runner")

    if extract_phase:
        out["extract_summary"] = {
            "html_size": extract_phase.data.get("html_size", 0),
            "html_recovered_partial": extract_phase.data.get("html_recovered_partial", False),
        }

    if render_phase:
        probe_errors = render_phase.data.get("probe_errors", []) if isinstance(render_phase.data, dict) else []
        out["render_summary"] = {
            "rendered": render_phase.data.get("rendered", False),
            "screenshot_count": len(ctx.all_screenshots),
            "keyframes_selected": render_phase.data.get("keyframes_selected", 0),
            "total_frames_captured": render_phase.data.get("total_frames_captured", 0),
            "probe_errors_count": len(probe_errors),
            "probe_errors": probe_errors[:5],
            "phase_errors": render_phase.errors[:5],
        }

    if test_runner_phase and isinstance(test_runner_phase.data, dict):
        out["test_runner_summary"] = {
            "test_pass_rate": test_runner_phase.data.get("test_pass_rate"),
            "tests_total": test_runner_phase.data.get("tests_total"),
            "tests_passed": test_runner_phase.data.get("tests_passed"),
            "tests_failed": test_runner_phase.data.get("tests_failed"),
            "results_path": test_runner_phase.data.get("results_path"),
        }

    phase_errors = {
        name: result.errors[:5]
        for name, result in ctx.phase_results.items()
        if result.errors
    }
    if phase_errors:
        out["phase_errors"] = phase_errors

    out.pop("_line_idx", None)
    return out


def serialize_eval_context_result_json(
    ctx: EvalContext,
    source_record: dict,
    dataset_name: str = "",
) -> str:
    """Serialize an EvalContext into the JSONL line written by the batch runner."""
    return json.dumps(
        serialize_eval_context_result(ctx, source_record, dataset_name=dataset_name),
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Buffered output writer
# ---------------------------------------------------------------------------

class OutputBuffer:
    """Thread-safe buffered JSONL writer."""

    def __init__(self, path: str, flush_interval: int = 50):
        self.path = path
        self.flush_interval = flush_interval
        self._buf: List[str] = []
        self._lock = threading.Lock()
        self._count = 0

    def append(self, line: str) -> None:
        with self._lock:
            self._buf.append(line)
            self._count += 1
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


async def run_incremental_evaluation(
    pipeline: PipelineEngine,
    config: EvalConfig,
    records: List[dict],
    *,
    output_path: str,
    dataset_name: str = "",
    resume: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Evaluate a provided record list with resume support and buffered output."""
    proc = config.processing

    if not records:
        return {"completed": 0, "failed": 0, "skipped": 0, "scores": []}

    done_set: Set[str] = set()
    if resume and not force:
        done_set = load_done_set(output_path)
        if done_set:
            logger.info(f"Resume: {len(done_set)} already scored (from all shards in {Path(output_path).parent}).")

    pending: List[dict] = []
    for rec in records:
        idx = rec.get("_line_idx", 0)
        uid = record_uid(rec, idx)
        rec["_eval_uid"] = uid
        if uid not in done_set:
            pending.append(rec)

    skipped = len(records) - len(pending)
    if not pending:
        return {"completed": 0, "failed": 0, "skipped": skipped, "scores": []}

    buf = OutputBuffer(output_path, proc.save_interval)
    stats: Dict[str, Any] = {"completed": 0, "failed": 0, "skipped": skipped, "scores": []}
    stats_lock = asyncio.Lock()
    pbar = tqdm(total=len(pending), desc=dataset_name or "eval", unit="rec", dynamic_ncols=True)

    async def on_complete(ctx: EvalContext) -> None:
        source_record = ctx._source_record  # type: ignore[attr-defined]
        buf.append(serialize_eval_context_result_json(ctx, source_record, dataset_name=dataset_name))
        # Benchmark resume runs are often small targeted repairs. Flush each
        # completed record so a later hung browser/page cannot hide progress.
        buf.flush()

        async with stats_lock:
            if ctx.status in {"completed", "completed_with_fallback"}:
                stats["completed"] += 1
                stats["scores"].append(ctx.total_score)
            else:
                stats["failed"] += 1
            avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            pbar.set_postfix(ok=stats["completed"], fail=stats["failed"], avg=f"{avg:.1f}", refresh=False)
            pbar.update(1)

    contexts = [build_eval_context_from_record(rec, dataset_name=dataset_name) for rec in pending]

    try:
        await pipeline.evaluate_batch(contexts, on_complete=on_complete)
    finally:
        pbar.close()
        buf.flush()

    return stats


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    pipeline: PipelineEngine,
    config: EvalConfig,
    *,
    shard_id: int = -1,
    num_shards: int = 1,
    limit: int = 0,
    force: bool = False,
    dataset_name: str = "",
) -> Dict[str, Any]:
    """
    Run batch evaluation on the JSONL file at config.data.input.

    Args:
        pipeline:      configured PipelineEngine.
        config:        EvalConfig (data.input and data.output_dir must be set).
        shard_id:      shard index; -1 = process all records.
        num_shards:    total shards for distributed runs.
        limit:         max records to process (0 = all).
        force:         re-evaluate already-scored records.
        dataset_name:  label embedded in output records and used as game_id prefix.

    Returns:
        Summary dict with counts and score statistics.
    """
    input_path = config.data.input
    output_dir = config.data.output_dir
    proc = config.processing

    out_name = f"shard_{shard_id:04d}.jsonl" if shard_id >= 0 else "results.jsonl"
    output_path = str(Path(output_dir) / out_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_records = load_records_from_jsonl(input_path)

    total_input = len(all_records)

    # Shard selection
    if shard_id >= 0 and num_shards > 1:
        chunk = (total_input + num_shards - 1) // num_shards
        records = all_records[shard_id * chunk: (shard_id + 1) * chunk]
        shard_label = f"shard {shard_id}/{num_shards}"
    else:
        records = all_records
        shard_label = "all"

    if limit > 0:
        records = records[:limit]

    if not records:
        return {"completed": 0, "failed": 0, "skipped": 0}

    done_set: Set[str] = set()
    if proc.resume and not force:
        done_set = load_done_set(output_path)

    skipped = sum(1 for rec in records if record_uid(rec, rec["_line_idx"]) in done_set)
    pending_count = len(records) - skipped

    logger.info("=" * 60)
    logger.info(f"  htmleval Batch — {dataset_name or '(unnamed)'}")
    logger.info(f"  Input : {Path(input_path).name}  ({total_input:,} total)")
    logger.info(f"  Shard : {shard_label}  Pending: {pending_count:,}  Skipped: {skipped:,}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Concurrency: {proc.concurrency}  "
                f"skip_agent={proc.skip_agent_phase}  skip_vision={proc.skip_vision_phase}")
    logger.info("=" * 60)

    if pending_count == 0:
        logger.info("All records already scored. Use force=True to re-evaluate.")
        return {"completed": 0, "failed": 0, "skipped": skipped}

    t0 = time.time()
    stats = await run_incremental_evaluation(
        pipeline,
        config,
        records,
        output_path=output_path,
        dataset_name=dataset_name,
        resume=proc.resume,
        force=force,
    )
    elapsed = time.time() - t0
    scores = stats.get("scores", [])
    avg     = sum(scores) / len(scores) if scores else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "shard_id": shard_id,
        "completed": stats["completed"],
        "skipped":   stats["skipped"],
        "failed":    stats["failed"],
        "avg_score": round(avg, 1),
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "elapsed_seconds": round(elapsed),
        "throughput_per_min": round(stats["completed"] / (elapsed / 60), 1) if elapsed > 0 else 0,
    }

    logger.info(f"Batch done — completed={stats['completed']} failed={stats['failed']} "
                f"avg={avg:.1f} elapsed={elapsed:.0f}s")

    summary_path = Path(config.workspace) / (
        f"summary_{dataset_name}.json" if dataset_name else "batch_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary
