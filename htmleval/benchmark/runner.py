"""
Benchmark runner — single command: generate (if needed) + evaluate + analyze.

Usage:
    # English benchmark — auto-generate + evaluate
    python -m htmleval benchmark run benchmark/en/ --config configs/eval.example.yaml --generate

    # Fast mode (skip VLM, deterministic scoring only)
    python -m htmleval benchmark run benchmark/en/ --config configs/eval.example.yaml --generate --mode fast

    # Multiple trials with pass@k metrics
    python -m htmleval benchmark run benchmark/en/ --config configs/eval.example.yaml --generate --trials 3

    # Items with responses → evaluate directly
    python -m htmleval benchmark run benchmark/responses/sample.jsonl --config configs/eval.example.yaml
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from htmleval.core.config import EvalConfig
from htmleval.phases.extract import extract_complete_html

logger = logging.getLogger("htmleval")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _valid_response(response: str) -> bool:
    return bool(response) and extract_complete_html(response) is not None


def _item_id(item: dict, fallback: int = 0) -> str:
    return str(item.get("id", fallback))


def _prompt_compatible(item: dict, record: dict) -> bool:
    """Only reuse a cached response when the source prompt still matches."""
    item_prompt = item.get("prompt")
    rec_prompt = record.get("prompt")
    if item_prompt is None:
        return True
    if rec_prompt is None:
        return False
    return item_prompt == rec_prompt


def _iter_chunks(seq: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in seq:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _infer_lang(benchmark_path: str, language: str) -> str:
    bp = Path(benchmark_path)
    if bp.name in ("en", "zh"):
        return bp.name
    if bp.parent.name in ("en", "zh"):
        return bp.parent.name
    return language or "mixed"


def _inspect_checkpoint_path(checkpoint_path: Path) -> dict[str, Any]:
    """Describe checkpoint readability so eval-only failures are actionable."""
    status: dict[str, Any] = {
        "path": str(checkpoint_path),
        "lexists": os.path.lexists(checkpoint_path),
        "exists": checkpoint_path.exists(),
        "is_symlink": checkpoint_path.is_symlink(),
        "target": None,
        "size_bytes": None,
        "issue": "",
        "readable": False,
    }

    if status["is_symlink"]:
        try:
            status["target"] = os.readlink(checkpoint_path)
        except OSError as exc:
            status["issue"] = f"symlink_target_unreadable:{exc}"
            return status

    if not status["lexists"]:
        status["issue"] = "missing"
        return status

    if not status["exists"]:
        status["issue"] = "broken_or_recursive_symlink" if status["is_symlink"] else "path_not_readable"
        return status

    try:
        status["size_bytes"] = checkpoint_path.stat().st_size
        with open(checkpoint_path, encoding="utf-8") as f:
            f.read(1)
        status["readable"] = True
    except OSError as exc:
        status["issue"] = f"unreadable:{exc.__class__.__name__}:{exc}"

    return status


def _load_existing_responses(items: list[dict], checkpoint_path: Path) -> tuple[int, int, int]:
    """Merge valid responses from checkpoint into items in-place."""
    if not os.path.lexists(checkpoint_path):
        return 0, 0, 0

    checkpoint: dict[str, dict] = {}
    invalid = 0
    prompt_mismatch = 0
    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rid = str(rec.get("id", ""))
                response = rec.get("response", "")
                if rid and _valid_response(response):
                    checkpoint[rid] = rec
                elif rid and response:
                    invalid += 1
    except OSError as exc:
        logger.warning("Checkpoint responses file is not readable: %s (%s)", checkpoint_path, exc)
        return 0, 0, 0

    merged = 0
    if checkpoint:
        for idx, item in enumerate(items):
            iid = _item_id(item, idx)
            if iid in checkpoint and not _valid_response(item.get("response", "")):
                rec = checkpoint[iid]
                if _prompt_compatible(item, rec):
                    item["response"] = rec["response"]
                    merged += 1
                else:
                    prompt_mismatch += 1

    return merged, invalid, prompt_mismatch


def _results_snapshot(results_path: Path) -> dict[str, Any]:
    """Return lightweight stats for partial progress and resume summaries."""
    latest_by_item: dict[str, dict[str, Any]] = {}

    if not results_path.exists():
        return {
            "completed_item_ids": set(),
            "failed_item_ids": set(),
            "terminal_item_ids": set(),
            "completed_scores": [],
            "avg_score_completed": 0.0,
            "avg_test_pass_rate_completed": None,
        }

    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = str(rec.get("line_number", ""))
            if not item_id:
                continue

            latest_by_item[item_id] = rec

    completed_item_ids: set[str] = set()
    failed_item_ids: set[str] = set()
    completed_scores: list[float] = []
    completed_test_pass_rates: list[float] = []
    for item_id, rec in latest_by_item.items():
        status = rec.get("eval_status", "")
        if status in {"completed", "completed_with_fallback"}:
            completed_item_ids.add(item_id)
            score_obj = rec.get("score", {})
            completed_scores.append(float(score_obj.get("total", 0) or 0))
            tpr = score_obj.get("test_pass_rate")
            if tpr is not None:
                completed_test_pass_rates.append(float(tpr))
        else:
            failed_item_ids.add(item_id)

    avg_score = (
        sum(completed_scores) / len(completed_scores) if completed_scores else 0.0
    )
    avg_tpr = (
        sum(completed_test_pass_rates) / len(completed_test_pass_rates)
        if completed_test_pass_rates else None
    )
    return {
        "completed_item_ids": completed_item_ids,
        "failed_item_ids": failed_item_ids,
        "terminal_item_ids": completed_item_ids | failed_item_ids,
        "completed_scores": completed_scores,
        "avg_score_completed": avg_score,
        "avg_test_pass_rate_completed": avg_tpr,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _config_fingerprint(config: EvalConfig, extras: dict[str, Any]) -> str:
    payload = {
        "evaluator_model": config.evaluator.model,
        "processing": config.processing.model_dump(mode="json"),
        "extras": extras,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest[:16]}"


def _runner_fingerprint() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    tracked_files = [
        "htmleval/benchmark/analysis.py",
        "htmleval/benchmark/runner.py",
        "htmleval/batch/orchestrator.py",
        "htmleval/phases/test_runner/actions.py",
        "htmleval/phases/test_runner/executor.py",
        "htmleval/phases/test_runner/runner.py",
        "htmleval/phases/test_runner/scoring.py",
        "htmleval/phases/vision_eval/prompts.py",
    ]
    payload: dict[str, str] = {}
    for rel_path in tracked_files:
        path = repo_root / rel_path
        try:
            payload[rel_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            payload[rel_path] = "missing"
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest[:16]}"


def _stamp_results_metadata(results_path: Path, metadata: dict[str, Any]) -> None:
    stamped: list[str] = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec.update(metadata)
            stamped.append(json.dumps(rec, ensure_ascii=False))
    results_path.write_text("\n".join(stamped) + ("\n" if stamped else ""), encoding="utf-8")


def _write_live_files(
    *,
    run_state_path: Path,
    summary_live_path: Path,
    model_slug: str,
    lang: str,
    mode_name: str,
    total_items: int,
    items: list[dict],
    attempted_generation_ids: set[str],
    generation_failed_ids: set[str],
    active_chunk_ids: list[str],
    queued_eval_ids: set[str],
    results_snapshot: dict[str, Any],
    started_at: str,
    started_ts: float,
    config_fingerprint: str,
    status: str,
) -> None:
    generated_ids = {_item_id(item, idx) for idx, item in enumerate(items) if _valid_response(item.get("response", ""))}
    terminal_ids = results_snapshot["terminal_item_ids"]
    completed_ids = results_snapshot["completed_item_ids"]
    failed_eval_ids = results_snapshot["failed_item_ids"]
    pending_generation_ids = [
        _item_id(item, idx)
        for idx, item in enumerate(items)
        if _item_id(item, idx) not in attempted_generation_ids and not _valid_response(item.get("response", ""))
    ]
    pending_evaluation_ids = sorted(generated_ids - terminal_ids - queued_eval_ids)

    elapsed_s = max(time.time() - started_ts, 0.001)
    generated_count = len(generated_ids)
    evaluated_count = len(terminal_ids)
    generation_failed_count = len(generation_failed_ids)
    evaluation_failed_count = len(failed_eval_ids)
    avg_score_completed = results_snapshot["avg_score_completed"]
    avg_tpr_completed = results_snapshot["avg_test_pass_rate_completed"]

    run_state = {
        "mode": mode_name,
        "total_items": total_items,
        "generated_count": generated_count,
        "evaluated_count": evaluated_count,
        "failed_generation_count": generation_failed_count,
        "failed_evaluation_count": evaluation_failed_count,
        "pending_generation_ids": pending_generation_ids,
        "pending_evaluation_ids": pending_evaluation_ids,
        "active_chunk_ids": active_chunk_ids,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "status": status,
        "config_fingerprint": config_fingerprint,
    }
    _write_json(run_state_path, run_state)

    summary_live = {
        "model": model_slug,
        "language": lang,
        "mode": mode_name,
        "generate": {
            "done": generated_count,
            "total": total_items,
            "failed": generation_failed_count,
        },
        "evaluate": {
            "done": evaluated_count,
            "total": total_items,
            "failed": evaluation_failed_count,
        },
        "partial_scores": {
            "avg_score_completed": round(avg_score_completed, 2),
            "avg_test_pass_rate_completed": (
                round(avg_tpr_completed, 4) if avg_tpr_completed is not None else None
            ),
        },
        "throughput": {
            "generated_per_min": round(generated_count / (elapsed_s / 60), 2),
            "evaluated_per_min": round(evaluated_count / (elapsed_s / 60), 2),
        },
        "started_at": started_at,
        "updated_at": _utc_now(),
        "status": status,
    }
    _write_json(summary_live_path, summary_live)


def _resolve_processing_defaults(config: EvalConfig, generate_concurrency: int) -> tuple[int, int]:
    proc = config.processing
    gen_conc = generate_concurrency or proc.generation_concurrency or proc.concurrency
    eval_conc = proc.evaluation_concurrency or proc.concurrency
    vlm_conc = proc.vlm_concurrency or proc.max_llm_concurrency or proc.concurrency
    proc.generation_concurrency = gen_conc
    proc.evaluation_concurrency = eval_conc
    proc.vlm_concurrency = vlm_conc
    proc.max_llm_concurrency = vlm_conc
    proc.concurrency = eval_conc
    return gen_conc, vlm_conc


def _set_vision_llm_concurrency(config: EvalConfig) -> None:
    from htmleval.phases.vision_eval.llm import set_vision_llm_semaphore

    vlm_conc = config.processing.vlm_concurrency or config.processing.max_llm_concurrency
    if vlm_conc > 0:
        set_vision_llm_semaphore(asyncio.Semaphore(vlm_conc))
    else:
        set_vision_llm_semaphore(None)


def _indexed_records(
    indexed_items: list[tuple[int, dict]],
    benchmark_item_to_record,
) -> list[dict]:
    records: list[dict] = []
    for idx, item in indexed_items:
        rec = benchmark_item_to_record(item)
        rec["_line_idx"] = idx
        records.append(rec)
    return records


async def _evaluate_records_once(
    *,
    config: EvalConfig,
    records: list[dict],
    results_path: Path,
    dataset_name: str,
    force: bool,
) -> dict[str, Any]:
    from htmleval import build_pipeline
    from htmleval.batch.orchestrator import run_incremental_evaluation
    from htmleval.concurrency.browser_pool import BrowserPool

    if force and results_path.exists():
        results_path.unlink()

    results_path.parent.mkdir(parents=True, exist_ok=True)

    pool = BrowserPool(
        max_size=config.processing.browser_pool_size,
        launch_rate=config.processing.browser_launch_rate,
    )
    await pool.start()
    try:
        _set_vision_llm_concurrency(config)
        pipeline = build_pipeline(config, browser_pool=pool)
        return await run_incremental_evaluation(
            pipeline,
            config,
            records,
            output_path=str(results_path),
            dataset_name=dataset_name,
            resume=config.processing.resume,
            force=force,
        )
    finally:
        await pool.stop()


async def _run_overlap_single_trial(
    *,
    items: list[dict],
    config: EvalConfig,
    model_dir: Path,
    eval_dir: Path,
    model_slug: str,
    lang: str,
    mode_name: str,
    generate: bool,
    generate_url: str,
    generate_model: str,
    generate_key: str,
    generate_concurrency: int,
    generate_temperature: float,
    generate_timeout: int,
    generate_max_tokens: int,
    disable_thinking: bool,
    seed: int,
    chunk_size: int,
    force: bool,
    benchmark_item_to_record,
    config_fingerprint: str,
) -> tuple[float, float, int, float | None]:
    from htmleval import build_pipeline
    from htmleval.batch.orchestrator import run_incremental_evaluation
    from htmleval.benchmark.generator import generate_responses
    from htmleval.concurrency.browser_pool import BrowserPool

    results_path = eval_dir / "results.jsonl"
    state_dir = eval_dir / ".state"
    run_state_path = state_dir / "run_state.json"
    summary_live_path = state_dir / "summary_live.json"
    state_dir.mkdir(parents=True, exist_ok=True)

    if force and results_path.exists():
        results_path.unlink()

    started_at = _utc_now()
    started_ts = time.time()
    t_gen = 0.0
    t_eval = 0.0
    first_terminal_at: float | None = None

    attempted_generation_ids: set[str] = set()
    generation_failed_ids: set[str] = set()
    queued_eval_ids: set[str] = set()
    active_chunk_ids: list[str] = []
    results_snapshot = _results_snapshot(results_path)

    indexed_items = list(enumerate(items))
    ready_items = [(idx, item) for idx, item in indexed_items if _valid_response(item.get("response", ""))]
    pending_gen = [(idx, item) for idx, item in indexed_items if not _valid_response(item.get("response", ""))]
    generation_batch_size = max(chunk_size, generate_concurrency, 1)

    queue: asyncio.Queue[tuple[str, list[tuple[int, dict]]] | None] = asyncio.Queue(maxsize=2)

    _write_live_files(
        run_state_path=run_state_path,
        summary_live_path=summary_live_path,
        model_slug=model_slug,
        lang=lang,
        mode_name=mode_name,
        total_items=len(items),
        items=items,
        attempted_generation_ids=attempted_generation_ids,
        generation_failed_ids=generation_failed_ids,
        active_chunk_ids=active_chunk_ids,
        queued_eval_ids=queued_eval_ids,
        results_snapshot=results_snapshot,
        started_at=started_at,
        started_ts=started_ts,
        config_fingerprint=config_fingerprint,
        status="running",
    )

    pool = BrowserPool(
        max_size=config.processing.browser_pool_size,
        launch_rate=config.processing.browser_launch_rate,
    )
    await pool.start()
    try:
        _set_vision_llm_concurrency(config)
        pipeline = build_pipeline(config, browser_pool=pool)

        async def producer() -> None:
            async def enqueue_eval_chunk(
                chunk_id: str,
                chunk: list[tuple[int, dict]],
            ) -> None:
                await queue.put((chunk_id, chunk))
                queued_eval_ids.update(_item_id(item, idx) for idx, item in chunk)
                _write_live_files(
                    run_state_path=run_state_path,
                    summary_live_path=summary_live_path,
                    model_slug=model_slug,
                    lang=lang,
                    mode_name=mode_name,
                    total_items=len(items),
                    items=items,
                    attempted_generation_ids=attempted_generation_ids,
                    generation_failed_ids=generation_failed_ids,
                    active_chunk_ids=active_chunk_ids,
                    queued_eval_ids=queued_eval_ids,
                    results_snapshot=results_snapshot,
                    started_at=started_at,
                    started_ts=started_ts,
                    config_fingerprint=config_fingerprint,
                    status="running",
                )

            ready_chunks = list(_iter_chunks(ready_items, chunk_size))
            ready_chunk_no = 0
            interleave_generation = bool(pending_gen and generate and ready_chunks)
            initial_ready_chunks = ready_chunks[:1] if interleave_generation else ready_chunks
            remaining_ready_chunks = ready_chunks[1:] if interleave_generation else []

            for chunk in initial_ready_chunks:
                ready_chunk_no += 1
                chunk_id = f"ready_{ready_chunk_no:04d}"
                await enqueue_eval_chunk(chunk_id, chunk)

            if pending_gen and not generate:
                await enqueue_eval_chunk("missing_0000", pending_gen)
                await queue.put(None)
                return

            gen_chunk_no = 0
            for gen_batch in _iter_chunks(pending_gen, generation_batch_size):
                gen_chunk_no += 1
                batch_id = f"gen_{gen_chunk_no:04d}"
                attempted_generation_ids.update(_item_id(item, idx) for idx, item in gen_batch)

                t0 = time.time()
                generated_chunk_items = await generate_responses(
                    [copy.deepcopy(item) for _, item in gen_batch],
                    generate_url,
                    generate_key,
                    generate_model,
                    concurrency=generate_concurrency,
                    temperature=generate_temperature,
                    timeout=generate_timeout,
                    seed=seed,
                    output_path=str(model_dir / "responses.jsonl"),
                    max_tokens=generate_max_tokens,
                    disable_thinking=disable_thinking,
                    checkpoint_flush_interval=max(1, config.processing.save_interval // 4),
                )
                t_gen_nonlocal = time.time() - t0
                nonlocal_t_gen[0] += t_gen_nonlocal

                generated_chunk: list[tuple[int, dict]] = []
                for (idx, _old_item), new_item in zip(gen_batch, generated_chunk_items):
                    items[idx] = new_item
                    iid = _item_id(new_item, idx)
                    if _valid_response(new_item.get("response", "")):
                        generation_failed_ids.discard(iid)
                    else:
                        generation_failed_ids.add(iid)
                    generated_chunk.append((idx, new_item))

                for sub_idx, eval_chunk in enumerate(_iter_chunks(generated_chunk, chunk_size), start=1):
                    chunk_id = f"{batch_id}_{sub_idx:02d}"
                    await enqueue_eval_chunk(chunk_id, eval_chunk)

                if remaining_ready_chunks:
                    ready_chunk_no += 1
                    chunk_id = f"ready_{ready_chunk_no:04d}"
                    await enqueue_eval_chunk(chunk_id, remaining_ready_chunks.pop(0))

            for chunk in remaining_ready_chunks:
                ready_chunk_no += 1
                chunk_id = f"ready_{ready_chunk_no:04d}"
                await enqueue_eval_chunk(chunk_id, chunk)

            await queue.put(None)

        async def consumer() -> None:
            nonlocal results_snapshot, t_eval, first_terminal_at
            while True:
                payload = await queue.get()
                if payload is None:
                    break
                chunk_id, chunk = payload
                active_chunk_ids.append(chunk_id)
                records = _indexed_records(chunk, benchmark_item_to_record)
                t0 = time.time()
                await run_incremental_evaluation(
                    pipeline,
                    config,
                    records,
                    output_path=str(results_path),
                    dataset_name=f"benchmark_{chunk_id}",
                    resume=config.processing.resume,
                    force=False,
                )
                t_eval += time.time() - t0
                active_chunk_ids.remove(chunk_id)
                queued_eval_ids.difference_update(_item_id(item, idx) for idx, item in chunk)
                results_snapshot = _results_snapshot(results_path)
                if first_terminal_at is None and results_snapshot["terminal_item_ids"]:
                    first_terminal_at = time.time()
                _write_live_files(
                    run_state_path=run_state_path,
                    summary_live_path=summary_live_path,
                    model_slug=model_slug,
                    lang=lang,
                    mode_name=mode_name,
                    total_items=len(items),
                    items=items,
                    attempted_generation_ids=attempted_generation_ids,
                    generation_failed_ids=generation_failed_ids,
                    active_chunk_ids=active_chunk_ids,
                    queued_eval_ids=queued_eval_ids,
                    results_snapshot=results_snapshot,
                    started_at=started_at,
                    started_ts=started_ts,
                    config_fingerprint=config_fingerprint,
                    status="running",
                )

        nonlocal_t_gen = [0.0]
        await asyncio.gather(producer(), consumer())
        t_gen = nonlocal_t_gen[0]
    except Exception:
        _write_live_files(
            run_state_path=run_state_path,
            summary_live_path=summary_live_path,
            model_slug=model_slug,
            lang=lang,
            mode_name=mode_name,
            total_items=len(items),
            items=items,
            attempted_generation_ids=attempted_generation_ids,
            generation_failed_ids=generation_failed_ids,
            active_chunk_ids=active_chunk_ids,
            queued_eval_ids=queued_eval_ids,
            results_snapshot=results_snapshot,
            started_at=started_at,
            started_ts=started_ts,
            config_fingerprint=config_fingerprint,
            status="failed",
        )
        raise
    finally:
        await pool.stop()

    results_snapshot = _results_snapshot(results_path)
    _write_live_files(
        run_state_path=run_state_path,
        summary_live_path=summary_live_path,
        model_slug=model_slug,
        lang=lang,
        mode_name=mode_name,
        total_items=len(items),
        items=items,
        attempted_generation_ids=attempted_generation_ids,
        generation_failed_ids=generation_failed_ids,
        active_chunk_ids=[],
        queued_eval_ids=set(),
        results_snapshot=results_snapshot,
        started_at=started_at,
        started_ts=started_ts,
        config_fingerprint=config_fingerprint,
        status="completed",
    )
    time_to_first_score = (
        round(first_terminal_at - started_ts, 1) if first_terminal_at is not None else None
    )
    return t_gen, t_eval, len(generation_failed_ids), time_to_first_score


async def run_benchmark(
    benchmark_path: str,
    config: EvalConfig,
    output_dir: str = "./benchmark_results",
    limit: int = 0,
    force: bool = False,
    language: str = "",
    category: str = "",
    difficulty: str = "",
    generate: bool = False,
    generate_url: str = "",
    generate_model: str = "",
    generate_key: str = "",
    generate_concurrency: int = 0,
    generate_temperature: float = 0.7,
    generate_timeout: int = 180,
    generate_max_tokens: int = 0,
    disable_thinking: bool = False,
    trials: int = 1,
    fast: bool = False,
    model_name: str = "",
    seed: int = 0,
    strict: bool = False,
) -> dict:
    """Load benchmark items, optionally generate responses, evaluate, and analyze."""
    from htmleval.benchmark.analysis import (
        VALIDATOR_MODE,
        analyze_results,
        analyze_with_trials,
        validate_items,
    )
    from htmleval.benchmark.generator import generate_responses
    from htmleval.benchmark.loader import benchmark_item_to_record, load_benchmark_items
    from htmleval.phases.test_runner.scoring import SCORE_VERSION

    t_total = time.time()
    run_started_ts = t_total
    run_started_at = _utc_now()
    config = config.model_copy(deep=True)

    raw_model = model_name or generate_model or config.evaluator.model
    model_slug = raw_model.rstrip("/").split("/")[-1].lower() if raw_model else "unknown"
    lang = _infer_lang(benchmark_path, language)
    mode_name = "fast" if fast else "full"

    model_dir = Path(output_dir) / model_slug / lang
    eval_dir = model_dir / mode_name
    state_dir = eval_dir / ".state"
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Model dir: %s  Eval dir: %s  (model=%s, lang=%s, mode=%s)",
        model_dir, eval_dir, model_slug, lang, mode_name,
    )

    if fast:
        config.processing.skip_vision_phase = True
        logger.info("Fast mode: VLM phase disabled (deterministic scoring only)")

    gen_conc, vlm_conc = _resolve_processing_defaults(config, generate_concurrency)
    chunk_size = max(1, config.processing.overlap_chunk_size or 32)

    items = load_benchmark_items(
        benchmark_path,
        language=language,
        category=category,
        difficulty=difficulty,
    )
    if limit > 0:
        items = items[:limit]
    if not items:
        logger.warning("No benchmark items found.")
        return {"overall": {"total": 0}}
    logger.info("Loaded %s benchmark items", len(items))

    val_errors = validate_items(items)
    if val_errors:
        for err in val_errors[:10]:
            logger.warning("Validation: %s", err)
        if len(val_errors) > 10:
            logger.warning("... and %s more validation errors", len(val_errors) - 10)
        if strict:
            raise ValueError(
                f"Schema validation failed with {len(val_errors)} error(s). "
                "Fix items or remove --strict to continue with warnings."
            )

    checkpoint_path = model_dir / "responses.jsonl"
    checkpoint_status = _inspect_checkpoint_path(checkpoint_path)
    if checkpoint_status["issue"] and checkpoint_status["issue"] != "missing":
        extra = []
        if checkpoint_status.get("target"):
            extra.append(f"target={checkpoint_status['target']}")
        if checkpoint_status.get("size_bytes") is not None:
            extra.append(f"size={checkpoint_status['size_bytes']}")
        detail = f" ({', '.join(extra)})" if extra else ""
        logger.warning(
            "Checkpoint responses path problem: %s [%s]%s",
            checkpoint_path,
            checkpoint_status["issue"],
            detail,
        )

    merged, invalid, prompt_mismatch = _load_existing_responses(items, checkpoint_path)
    if merged:
        logger.info(
            "Pre-loaded %s responses from checkpoint (%s)",
            merged,
            checkpoint_path.name,
        )
    elif checkpoint_status["readable"]:
        logger.info(
            "Checkpoint file was readable but yielded 0 reusable responses: %s",
            checkpoint_path.name,
        )
    if invalid:
        logger.info("Ignored %s invalid/truncated checkpoint responses from %s", invalid, checkpoint_path.name)
    if prompt_mismatch:
        logger.warning(
            "Ignored %s cached responses with prompt mismatches in %s",
            prompt_mismatch,
            checkpoint_path.name,
        )

    original_items = copy.deepcopy(items)
    need_gen_indices = [
        idx for idx, item in enumerate(items)
        if not _valid_response(item.get("response", ""))
    ]
    need_gen = [items[idx] for idx in need_gen_indices]
    config_fingerprint = _config_fingerprint(
        config,
        {
            "benchmark_path": benchmark_path,
            "model_slug": model_slug,
            "mode": mode_name,
            "generate_model": generate_model or config.evaluator.model,
            "generate_temperature": generate_temperature,
            "generate_timeout": generate_timeout,
            "generate_max_tokens": generate_max_tokens,
            "seed": seed,
            "gen_conc": gen_conc,
            "vlm_conc": vlm_conc,
            "chunk_size": chunk_size,
        },
    )
    validator_mode = f"{VALIDATOR_MODE}:{'strict' if strict else 'warn_only'}"
    runner_fingerprint = _runner_fingerprint()
    benchmark_metadata = {
        "score_version": SCORE_VERSION,
        "runner_fingerprint": runner_fingerprint,
        "validator_mode": validator_mode,
    }

    overlap_enabled = (
        trials == 1
        and config.processing.overlap_mode == "chunked"
        and generate
        and bool(need_gen)
    )
    if overlap_enabled:
        logger.info(
            "Benchmark overlap enabled: chunked generation/evaluation (chunk_size=%s, gen=%s, eval=%s, vlm=%s)",
            chunk_size,
            gen_conc,
            config.processing.concurrency,
            vlm_conc,
        )
    elif config.processing.overlap_mode == "chunked" and trials > 1:
        logger.info("Overlap mode requested, but trials=%s; falling back to standard trial loop", trials)

    t_gen = 0.0
    t_eval = 0.0
    generation_failed = 0
    time_to_first_score_s: float | None = None

    if need_gen and not generate:
        checkpoint_hint = ""
        if checkpoint_status["issue"] == "missing":
            checkpoint_hint = f" No checkpoint responses file found at {checkpoint_path}."
        elif checkpoint_status["issue"]:
            checkpoint_hint = (
                f" Checkpoint path issue: {checkpoint_status['issue']} at {checkpoint_path}."
            )
            if checkpoint_status.get("target"):
                checkpoint_hint += f" target={checkpoint_status['target']}."
        elif checkpoint_status["readable"] and merged == 0:
            checkpoint_hint = (
                f" Checkpoint file {checkpoint_path} was readable but none of its responses matched"
                " the current benchmark items/prompts."
            )

        logger.warning(
            "%s/%s items have no response.%s Use --generate to auto-generate via LLM.",
            len(need_gen),
            len(items),
            checkpoint_hint,
        )
        if len(need_gen) == len(items):
            raise RuntimeError(
                "No valid responses available for eval-only benchmark run. "
                f"checkpoint={checkpoint_path} issue={checkpoint_status['issue'] or 'no_usable_responses'}"
            )

    if overlap_enabled:
        url = generate_url or config.evaluator.base_url
        model = generate_model or config.evaluator.model
        key = generate_key or config.evaluator.api_key or "EMPTY"
        if not url or not model:
            raise ValueError(
                "Generation requires base_url and model "
                "(set in config evaluator section or via --generate-url/--generate-model)"
            )

        t_gen, t_eval, generation_failed, time_to_first_score_s = await _run_overlap_single_trial(
            items=items,
            config=config,
            model_dir=model_dir,
            eval_dir=eval_dir,
            model_slug=model_slug,
            lang=lang,
            mode_name=mode_name,
            generate=generate,
            generate_url=url,
            generate_model=model,
            generate_key=key,
            generate_concurrency=gen_conc,
            generate_temperature=generate_temperature,
            generate_timeout=generate_timeout,
            generate_max_tokens=generate_max_tokens,
            disable_thinking=disable_thinking,
            seed=seed,
            chunk_size=chunk_size,
            force=force,
            benchmark_item_to_record=benchmark_item_to_record,
            config_fingerprint=config_fingerprint,
        )
    else:
        if need_gen and generate:
            url = generate_url or config.evaluator.base_url
            model = generate_model or config.evaluator.model
            key = generate_key or config.evaluator.api_key or "EMPTY"
            if not url or not model:
                raise ValueError(
                    "Generation requires base_url and model "
                    "(set in config evaluator section or via --generate-url/--generate-model)"
                )

            logger.info("Generating %s responses (concurrency=%s)", len(need_gen), gen_conc)
            t0 = time.time()
            items = await generate_responses(
                items,
                url,
                key,
                model,
                concurrency=gen_conc,
                temperature=generate_temperature,
                timeout=generate_timeout,
                seed=seed,
                output_path=str(checkpoint_path),
                max_tokens=generate_max_tokens,
                disable_thinking=disable_thinking,
                checkpoint_flush_interval=max(1, config.processing.save_interval // 4),
            )
            t_gen = time.time() - t0
            logger.info("Generation done in %.0fs", t_gen)

        generation_failed = sum(1 for item in items if not _valid_response(item.get("response", "")))
        if generation_failed:
            logger.info(
                "%s/%s items have empty/invalid responses; they will be evaluated as failures",
                generation_failed,
                len(items),
            )

        all_trial_results: list[dict] = []
        t_eval_start = time.time()

        for trial in range(trials):
            if trials > 1:
                logger.info("=== Trial %s/%s ===", trial + 1, trials)

            if trial > 0 and generate and need_gen_indices:
                url = generate_url or config.evaluator.base_url
                model = generate_model or config.evaluator.model
                key = generate_key or config.evaluator.api_key or "EMPTY"
                logger.info("Trial %s: re-generating %s responses", trial + 1, len(need_gen_indices))
                regen_seed_items = copy.deepcopy(original_items)
                for idx in need_gen_indices:
                    regen_seed_items[idx]["response"] = ""
                t0 = time.time()
                trial_items = await generate_responses(
                    regen_seed_items,
                    url,
                    key,
                    model,
                    concurrency=gen_conc,
                    temperature=generate_temperature,
                    timeout=generate_timeout,
                    seed=(seed + trial) if seed else 0,
                    output_path=str(model_dir / f"responses_trial{trial}.jsonl"),
                    max_tokens=generate_max_tokens,
                    disable_thinking=disable_thinking,
                    checkpoint_flush_interval=max(1, config.processing.save_interval // 4),
                )
                t_gen += time.time() - t0
            else:
                trial_items = items

            trial_workspace = eval_dir / f"trial_{trial}" if trials > 1 else eval_dir
            trial_workspace.mkdir(parents=True, exist_ok=True)
            trial_results_path = trial_workspace / "results.jsonl"

            trial_cfg = config.model_copy(deep=True)
            trial_cfg.workspace = str(trial_workspace)
            trial_cfg.data.output_dir = str(trial_workspace)

            trial_records = _indexed_records(list(enumerate(trial_items)), benchmark_item_to_record)
            await _evaluate_records_once(
                config=trial_cfg,
                records=trial_records,
                results_path=trial_results_path,
                dataset_name=f"benchmark_trial{trial}" if trials > 1 else "benchmark",
                force=force,
            )

            if trials == 1:
                break

            with open(trial_results_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec["trial"] = trial
                    all_trial_results.append(rec)

        t_eval = time.time() - t_eval_start

        if trials > 1:
            results_path = eval_dir / "results.jsonl"
            with open(results_path, "w", encoding="utf-8") as f:
                for rec in all_trial_results:
                    rec["model"] = model_slug
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    results_path = eval_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Benchmark results were not written: {results_path}")
    _stamp_results_metadata(results_path, benchmark_metadata)

    if trials > 1:
        analysis = analyze_with_trials(str(results_path), num_trials=trials, items=items)
    else:
        analysis = analyze_results(str(results_path), items)

    t_total = time.time() - t_total
    analysis["timing"] = {
        "generate_s": round(t_gen, 1),
        "evaluate_s": round(t_eval, 1),
        "total_s": round(t_total, 1),
        "time_to_first_score_s": time_to_first_score_s,
        "items_generated": len(need_gen) if generate else 0,
        "items_evaluated": len(items),
        "items_generation_failed": generation_failed,
        "items_skipped": 0,
        "trials": trials,
        "mode": mode_name,
        "seed": seed,
    }
    analysis["generation_params"] = {
        "temperature": generate_temperature,
        "timeout": generate_timeout,
        "max_tokens": generate_max_tokens,
        "seed": seed,
        "concurrency": gen_conc,
        "disable_thinking": disable_thinking,
    }
    analysis["model"] = {
        "name": model_slug,
        "raw": raw_model,
        "evaluator": config.evaluator.model,
    }
    analysis["execution"] = {
        "overlap_mode": config.processing.overlap_mode,
        "overlap_enabled": overlap_enabled,
        "chunk_size": chunk_size if overlap_enabled else None,
        "generation_concurrency": gen_conc,
        "evaluation_concurrency": config.processing.concurrency,
        "vlm_concurrency": vlm_conc,
        "config_fingerprint": config_fingerprint,
        "score_version": benchmark_metadata["score_version"],
        "runner_fingerprint": benchmark_metadata["runner_fingerprint"],
        "validator_mode": benchmark_metadata["validator_mode"],
    }
    analysis.update(benchmark_metadata)

    analysis_path = eval_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Analysis written to %s", analysis_path)

    if not overlap_enabled:
        results_snapshot = _results_snapshot(results_path)
        _write_live_files(
            run_state_path=state_dir / "run_state.json",
            summary_live_path=state_dir / "summary_live.json",
            model_slug=model_slug,
            lang=lang,
            mode_name=mode_name,
            total_items=len(items),
            items=items,
            attempted_generation_ids={_item_id(item, idx) for idx, item in enumerate(items) if idx in need_gen_indices},
            generation_failed_ids={
                _item_id(item, idx)
                for idx, item in enumerate(items)
                if idx in need_gen_indices and not _valid_response(item.get("response", ""))
            },
            active_chunk_ids=[],
            queued_eval_ids=set(),
            results_snapshot=results_snapshot,
            started_at=run_started_at,
            started_ts=run_started_ts,
            config_fingerprint=config_fingerprint,
            status="completed",
        )

    return analysis
