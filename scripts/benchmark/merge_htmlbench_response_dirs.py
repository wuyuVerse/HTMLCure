#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from htmleval.phases.extract import extract_complete_html


def load_valid_records(path: Path) -> dict[str, dict]:
    kept: dict[str, dict] = {}
    if not path.exists():
        return kept
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            item_id = str(rec.get("id") or "")
            response = rec.get("response") or ""
            if item_id and extract_complete_html(response) is not None:
                kept[item_id] = rec
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Target responses.jsonl path")
    ap.add_argument("--source-dir", action="append", default=[], help="Directory containing responses.jsonl")
    ap.add_argument("--source-file", action="append", default=[], help="Explicit source responses.jsonl file")
    ap.add_argument("--backup-suffix", default="merge_backup_20260428.jsonl")
    args = ap.parse_args()

    target = Path(args.target)
    sources: list[Path] = []
    for src_dir in args.source_dir:
        sources.append(Path(src_dir) / "responses.jsonl")
    for src_file in args.source_file:
        sources.append(Path(src_file))

    if not sources:
        raise SystemExit("No sources provided")

    merged = load_valid_records(target)
    before = len(merged)
    for src in sources:
        merged.update(load_valid_records(src))

    if target.exists():
        backup_path = target.parent / f"{target.stem}.{args.backup_suffix}"
        if not backup_path.exists():
            shutil.copy2(target, backup_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for item_id in sorted(merged):
            f.write(json.dumps(merged[item_id], ensure_ascii=False) + "\n")

    print(f"target={target}")
    print(f"valid_before={before}")
    print(f"valid_after={len(merged)}")
    print(f"sources={len(sources)}")
    for src in sources:
        print(src)


if __name__ == "__main__":
    main()
