#!/usr/bin/env python3
"""Run a local HTMLCure smoke test without API credentials.

This script exercises extraction, static analysis, rendering, and benchmark
test-case execution on one built-in HTML response. It deliberately skips the
agent and vision phases so it can run offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from htmleval import EvalConfig, EvalContext, build_pipeline


DEMO_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>HTMLCure Smoke Dashboard</title>
  <style>
    body { font-family: sans-serif; margin: 24px; background: #f7fafc; color: #172033; }
    main { max-width: 840px; margin: auto; }
    .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .card { background: white; border: 1px solid #dbe3ef; border-radius: 12px; padding: 16px; }
    svg { width: 100%; height: 220px; margin-top: 24px; background: white; border-radius: 12px; }
    button { margin-top: 16px; padding: 10px 14px; border: 0; border-radius: 8px; background: #1769aa; color: white; }
  </style>
</head>
<body>
<main>
  <h1>Sales Dashboard</h1>
  <section class="cards">
    <div class="card"><strong>Total Revenue</strong><p>$45,230</p></div>
    <div class="card"><strong>Orders</strong><p>342</p></div>
    <div class="card"><strong>Customers</strong><p>156</p></div>
    <div class="card"><strong>Conversion Rate</strong><p>3.2%</p></div>
  </section>
  <svg role="img" aria-label="Monthly sales bar chart" viewBox="0 0 600 220">
    <text x="20" y="28">Monthly Sales</text>
    <rect x="60" y="140" width="44" height="50" fill="#1769aa"></rect>
    <rect x="140" y="110" width="44" height="80" fill="#1769aa"></rect>
    <rect x="220" y="80" width="44" height="110" fill="#1769aa"></rect>
    <rect x="300" y="95" width="44" height="95" fill="#1769aa"></rect>
    <rect x="380" y="60" width="44" height="130" fill="#1769aa"></rect>
    <rect x="460" y="70" width="44" height="120" fill="#1769aa"></rect>
    <text x="60" y="210">Jan</text><text x="140" y="210">Feb</text>
    <text x="220" y="210">Mar</text><text x="300" y="210">Apr</text>
    <text x="380" y="210">May</text><text x="460" y="210">Jun</text>
  </svg>
  <button id="refresh">Refresh dashboard</button>
  <p id="status">Ready</p>
</main>
<script>
document.getElementById('refresh').addEventListener('click', () => {
  document.getElementById('status').textContent = 'Updated';
});
</script>
</body>
</html>"""


TEST_CASES = [
    {
        "id": "render",
        "name": "Page renders",
        "weight": 1,
        "steps": [
            {"action": "wait_for", "selector": "body", "timeout": 3000},
            {"action": "assert_console_clean"},
        ],
    },
    {
        "id": "cards",
        "name": "Prompt values are visible",
        "weight": 1,
        "steps": [
            {"action": "assert_text_contains", "selector": "body", "text": "45,230|Orders|Customers|3.2%"},
        ],
    },
    {
        "id": "chart",
        "name": "SVG chart bars are present",
        "weight": 1,
        "steps": [
            {"action": "assert_count", "selector": "svg rect", "gte": 6},
        ],
    },
    {
        "id": "interaction",
        "name": "Refresh button updates status",
        "weight": 1,
        "steps": [
            {"action": "click_text", "text_pattern": "Refresh"},
            {"action": "wait", "ms": 100},
            {"action": "assert_text_contains", "selector": "#status", "text": "Updated"},
        ],
    },
]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/smoke_tool")
    args = parser.parse_args()

    config = EvalConfig()
    config.workspace = args.output_dir
    config.processing.skip_agent_phase = True
    config.processing.skip_vision_phase = True
    config.processing.browser_pool_size = 1
    config.processing.evaluation_concurrency = 1
    config.processing.concurrency = 1

    ctx = EvalContext(
        query="Create a sales dashboard with stat cards, a monthly bar chart, and a refresh button.",
        response=DEMO_HTML,
        game_id="smoke_tool",
    )
    ctx.test_cases = TEST_CASES

    pipeline = build_pipeline(config)
    result = await pipeline.evaluate(ctx)

    payload = {
        "status": result.status,
        "score": result.total_score,
        "phases": {
            name: {
                "success": phase.success,
                "duration_ms": round(phase.duration_ms, 1),
                "errors": phase.errors,
            }
            for name, phase in result.phase_results.items()
        },
        "output_dir": str(result.output_dir) if result.output_dir else None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
