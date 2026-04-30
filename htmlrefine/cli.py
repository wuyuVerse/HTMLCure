"""
CLI entry point for HTMLRefine.

Usage:
    python -m htmlrefine serve  --config configs/refine.example.yaml
    python -m htmlrefine batch  --config configs/refine.example.yaml --limit 10
    python -m htmlrefine eval   --config configs/refine.example.yaml --query "..." --html "<html>..."
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def build_pipeline(config):
    """Build the standard 5-phase evaluation pipeline (delegates to htmleval)."""
    from htmleval import build_pipeline as _build
    return _build(config)


def cmd_serve(args):
    """Start the evaluation HTTP server."""
    from htmlrefine.core.config import load_config
    from htmlrefine.server.app import run_server

    config = load_config(args.config)
    if args.port:
        config.processing.port = args.port
    config.ensure_dirs()

    pipeline = build_pipeline(config)
    run_server(pipeline, config)


def cmd_batch(args):
    """Run batch evaluation on a JSONL file."""
    from htmlrefine.core.config import load_config
    from htmleval.batch.orchestrator import run_batch

    config = load_config(args.config)
    if args.input:
        config.data.input = args.input
    if args.output_dir:
        config.data.output_dir = args.output_dir
    config.ensure_dirs()

    pipeline = build_pipeline(config)
    asyncio.run(run_batch(
        pipeline=pipeline,
        config=config,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        limit=args.limit,
        force=args.force,
    ))


def cmd_eval(args):
    """Evaluate a single HTML game (for testing)."""
    from htmlrefine.core.config import load_config
    from htmleval.core.context import EvalContext

    config = load_config(args.config)
    config.ensure_dirs()
    pipeline = build_pipeline(config)

    ctx = EvalContext(
        query=args.query,
        response=args.html,
        game_id=args.game_id or "test_001",
        variant=args.variant or "default",
    )

    async def _run():
        result = await pipeline.evaluate(ctx)
        print(f"\nScore: {result.total_score}/100")
        print(f"Status: {result.status}")
        for name, pr in result.phase_results.items():
            print(f"  {name}: {'OK' if pr.success else 'FAIL'} ({pr.duration_ms:.0f}ms)")
        if result.output_dir:
            print(f"Report: {result.output_dir}/report.md")

    asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser(
        prog="htmlrefine",
        description="HTMLRefine — Agent-driven evaluation and refinement for HTML code",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    p_serve = sub.add_parser("serve", help="Start evaluation HTTP server")
    p_serve.add_argument("--config", required=True, help="Path to config.yaml")
    p_serve.add_argument("--port", type=int, help="Override server port")

    # batch
    p_batch = sub.add_parser("batch", help="Run batch evaluation")
    p_batch.add_argument("--config", required=True, help="Path to config.yaml")
    p_batch.add_argument("--input", help="Override input JSONL path")
    p_batch.add_argument("--output-dir", help="Override output directory")
    p_batch.add_argument("--shard-id", type=int, default=-1, help="Shard index (-1=all)")
    p_batch.add_argument("--num-shards", type=int, default=1, help="Total shards")
    p_batch.add_argument("--limit", type=int, default=0, help="Max records (0=all)")
    p_batch.add_argument("--force", action="store_true", help="Re-evaluate scored records")

    # eval (single)
    p_eval = sub.add_parser("eval", help="Evaluate a single game")
    p_eval.add_argument("--config", required=True, help="Path to config.yaml")
    p_eval.add_argument("--query", required=True, help="Game description")
    p_eval.add_argument("--html", required=True, help="HTML code or file path")
    p_eval.add_argument("--game-id", default="test_001")
    p_eval.add_argument("--variant", default="default")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
