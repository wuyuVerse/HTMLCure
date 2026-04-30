#!/usr/bin/env python3
"""Run batch evaluation on JSONL data."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from htmlrefine.cli import cmd_batch, setup_logging

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HTMLRefine Batch Evaluator")
    parser.add_argument("--config", default="configs/refine.example.yaml", help="Config file")
    parser.add_argument("--input", help="Override input JSONL path")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--shard-id", type=int, default=-1, help="Shard index (-1=all)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards")
    parser.add_argument("--limit", type=int, default=0, help="Max records (0=all)")
    parser.add_argument("--force", action="store_true", help="Re-evaluate scored records")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cmd_batch(args)
