#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results.jsonl containing prompt + test_cases")
    ap.add_argument("--output", required=True, help="output benchmark jsonl")
    args = ap.parse_args()

    src = Path(args.results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    seen = set()
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            item_id = rec["line_number"]
            if item_id in seen:
                continue
            seen.add(item_id)
            user_msg = rec["data"]["messages"][0]["content"]
            records.append(
                {
                    "id": item_id,
                    "category": rec.get("category"),
                    "sub_type": rec.get("sub_type"),
                    "difficulty": rec.get("difficulty"),
                    "language": rec.get("language"),
                    "prompt": user_msg,
                    "test_cases": rec.get("test_cases", []),
                    "has_interaction": rec.get("has_interaction", False),
                }
            )

    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
