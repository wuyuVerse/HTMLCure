#!/usr/bin/env python3
"""Build a per-test-case pass/fail matrix from benchmark result directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_benchmark_items(bench: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    files = [bench] if bench.is_file() else sorted(bench.glob("*.jsonl"))
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                item["_file"] = path.name
                out[str(item["id"])] = item
    return out


def latest_results(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            item_id = str(rec.get("item_id") or rec.get("line_number") or rec.get("id"))
            out[item_id] = rec
    return out


def load_tc_results(rec: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    summary = rec.get("test_runner_summary") or {}
    raw_path = summary.get("results_path")
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.is_absolute() and base_dir:
        path = base_dir / path
    if not path.exists() or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("results", []):
        tc_id = str(row.get("id") or row.get("test_id") or row.get("name"))
        out[tc_id] = row
    return out


def discover_results(results_root: Path) -> dict[str, Path]:
    """Return model label -> results.jsonl for common HTMLCure output layouts."""
    if results_root.is_file():
        return {results_root.parent.name: results_root}

    candidates = sorted(results_root.glob("*/en/full/results.jsonl"))
    if not candidates:
        candidates = sorted(results_root.glob("**/results.jsonl"))
    return {path.parents[2].name if len(path.parents) >= 3 else path.parent.name: path for path in candidates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark/en"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/tc_pass_matrix.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/tc_pass_matrix.csv"))
    args = parser.parse_args()

    bench = load_benchmark_items(args.benchmark)
    result_paths = discover_results(args.results_root)
    if not result_paths:
        raise SystemExit(f"No results.jsonl files found under {args.results_root}")

    model_results = {label: latest_results(path) for label, path in result_paths.items()}
    rows: list[dict[str, Any]] = []

    for item_id, item in sorted(bench.items()):
        test_cases = item.get("test_cases", [])
        per_model_tc = {
            label: load_tc_results(model_results[label].get(item_id, {}), base_dir=path.parent)
            for label, path in result_paths.items()
        }
        for idx, tc in enumerate(test_cases):
            tc_id = str(tc.get("id") or f"idx_{idx}")
            row: dict[str, Any] = {
                "item_id": item_id,
                "category": item.get("category"),
                "sub_type": item.get("sub_type"),
                "file": item.get("_file"),
                "tc_idx": idx,
                "tc_id": tc_id,
                "weight": tc.get("weight", 1),
                "name": tc.get("name", ""),
            }
            for label in sorted(result_paths):
                result = per_model_tc[label].get(tc_id)
                row[f"{label}_pass"] = None if result is None else bool(result.get("passed"))
            rows.append(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"models={len(result_paths)} rows={len(rows)}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
