#!/usr/bin/env python3
"""Start the HTMLRefine evaluation server."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from htmlrefine.cli import cmd_serve, setup_logging

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HTMLRefine Evaluation Server")
    parser.add_argument("--config", default="configs/refine.example.yaml", help="Config file")
    parser.add_argument("--port", type=int, help="Override server port")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cmd_serve(args)
