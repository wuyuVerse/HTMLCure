"""
FastAPI application for HTMLRefine evaluation server.

Endpoints:
    GET  /health              — liveness check
    POST /evaluate            — evaluate a single HTML page
    POST /repair              — evaluate + iterative repair
    GET  /repair/{job_id}     — poll async repair job progress
    GET  /reports/{path}      — serve static report files
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from htmlrefine.core.config import AppConfig
from htmleval.core.context import EvalContext
from htmleval.core.pipeline import PipelineEngine
from htmlrefine.data_pipeline.repair import RepairEngine, RepairResult

logger = logging.getLogger("htmlrefine")


# ── Request / Response Models ─────────────────────────────────

class EvalRequest(BaseModel):
    game_id: str
    title: str = ""
    query: str
    html_code: str
    variant: str = "default"


class ScoreDetail(BaseModel):
    rendering: int = 0
    visual_design: int = 0
    functionality: int = 0
    interaction: int = 0
    code_quality: int = 0
    total: int = 0


class EvalResponse(BaseModel):
    game_id: str
    variant: str
    score: int
    score_detail: Optional[ScoreDetail] = None
    report_dir: str
    status: str
    error: Optional[str] = None


class RepairRequest(BaseModel):
    game_id: str
    query: str
    html_code: str
    variant: str = "default"
    async_mode: bool = Field(False, description="Return job_id immediately and run in background")


class IterationSummary(BaseModel):
    iteration: int
    strategy: str
    score_before: int
    score_after: int
    delta: int
    elapsed_s: float
    success: bool


class RepairResponse(BaseModel):
    game_id: str
    job_id: str
    status: str                            # "completed" | "running" | "failed"
    original_score: int = 0
    final_score: int = 0
    improvement: int = 0
    best_html: str = ""
    iterations: List[IterationSummary] = []
    elapsed_s: float = 0.0
    error: Optional[str] = None


# ── In-memory job store (for async mode) ──────────────────────

_jobs: Dict[str, RepairResponse] = {}


# ── App Factory ───────────────────────────────────────────────

def create_app(pipeline: PipelineEngine, config: AppConfig) -> FastAPI:
    """Create the FastAPI application with injected pipeline and config."""

    repair_engine = RepairEngine(config.repair)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.ensure_dirs()
        app.mount("/reports",   StaticFiles(directory=str(config.reports_dir)),   name="reports")
        app.mount("/completed", StaticFiles(directory=str(config.completed_dir)), name="completed")
        app.mount("/failed",    StaticFiles(directory=str(config.failed_dir)),    name="failed")
        logger.info(f"Server ready on port {config.processing.port}")
        yield
        logger.info("Server shutting down")

    app = FastAPI(title="HTMLRefine Server", version="0.2.0", lifespan=lifespan)

    # ── Health ────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "time": datetime.now().isoformat()}

    # ── Evaluate ──────────────────────────────────────────

    @app.post("/evaluate", response_model=EvalResponse)
    async def evaluate(req: EvalRequest):
        ctx = EvalContext(
            query=req.query, response=req.html_code,
            game_id=req.game_id, variant=req.variant, title=req.title,
        )
        try:
            ctx = await pipeline.evaluate_one(ctx)
        except Exception as e:
            logger.error(f"evaluate error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        return _eval_response(ctx)

    # ── Repair ────────────────────────────────────────────

    @app.post("/repair", response_model=RepairResponse)
    async def repair(req: RepairRequest):
        job_id = str(uuid.uuid4())[:8]

        if req.async_mode:
            # Return immediately; run in background
            _jobs[job_id] = RepairResponse(
                game_id=req.game_id, job_id=job_id, status="running"
            )
            asyncio.create_task(_run_repair(req, job_id, pipeline, repair_engine, config))
            return _jobs[job_id]

        # Synchronous mode
        return await _run_repair_sync(req, job_id, pipeline, repair_engine, config)

    @app.get("/repair/{job_id}", response_model=RepairResponse)
    async def get_repair_job(job_id: str):
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
        return _jobs[job_id]

    return app


# ── Internal helpers ──────────────────────────────────────────

async def _run_repair_sync(
    req: RepairRequest,
    job_id: str,
    pipeline: PipelineEngine,
    engine: RepairEngine,
    config: AppConfig,
) -> RepairResponse:
    # Step 1: full evaluation
    ctx = EvalContext(
        query=req.query, response=req.html_code,
        game_id=req.game_id, variant=req.variant,
    )
    try:
        ctx = await pipeline.evaluate_one(ctx)
    except Exception as e:
        logger.error(f"[repair:{job_id}] eval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

    original_score = ctx.total_score

    # Step 2: repair loop
    try:
        result: RepairResult = await engine.repair(ctx, pipeline, config)
    except Exception as e:
        logger.error(f"[repair:{job_id}] repair failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Repair failed: {e}")

    return _repair_response(job_id, req.game_id, result, "completed")


async def _run_repair(
    req: RepairRequest,
    job_id: str,
    pipeline: PipelineEngine,
    engine: RepairEngine,
    config: AppConfig,
) -> None:
    """Background task for async repair."""
    try:
        resp = await _run_repair_sync(req, job_id, pipeline, engine, config)
        _jobs[job_id] = resp
    except Exception as e:
        _jobs[job_id] = RepairResponse(
            game_id=req.game_id, job_id=job_id, status="failed", error=str(e)
        )


def _eval_response(ctx: EvalContext) -> EvalResponse:
    ev = ctx.final_score or {}
    def _dim(k): return ev.get(k, {}).get("score", 0) if isinstance(ev.get(k), dict) else 0
    detail = ScoreDetail(
        rendering=_dim("rendering"), visual_design=_dim("visual_design"),
        functionality=_dim("functionality"), interaction=_dim("interaction"),
        code_quality=_dim("code_quality"), total=ctx.total_score,
    )
    return EvalResponse(
        game_id=ctx.game_id, variant=ctx.variant,
        score=ctx.total_score, score_detail=detail,
        report_dir=str(ctx.output_dir) if ctx.output_dir else "",
        status=ctx.status,
        error=ctx.skip_reason if ctx.should_skip else None,
    )


def _repair_response(job_id: str, game_id: str, result: RepairResult, status: str) -> RepairResponse:
    return RepairResponse(
        game_id=game_id, job_id=job_id, status=status,
        original_score=result.original_score,
        final_score=result.final_score,
        improvement=result.improvement,
        best_html=result.best_html,
        iterations=[
            IterationSummary(
                iteration=it.iteration, strategy=it.strategy,
                score_before=it.score_before, score_after=it.score_after,
                delta=it.delta, elapsed_s=it.elapsed_s, success=it.success,
            )
            for it in result.iterations
        ],
        elapsed_s=result.elapsed_s,
    )


def run_server(pipeline: PipelineEngine, config: AppConfig) -> None:
    app = create_app(pipeline, config)
    uvicorn.run(app, host="0.0.0.0", port=config.processing.port, log_level="info")
