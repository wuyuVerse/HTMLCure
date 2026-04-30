#!/usr/bin/env python3
"""Repair missing benchmark responses for a current benchmark run.

The script rebuilds a clean successful-response checkpoint from an existing
results.jsonl and then generates only benchmark items that still lack a valid
HTML response. It is intended for post-run repair, not for changing scores.
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

from htmleval.benchmark.generator import generate_responses
from htmleval.benchmark.loader import load_benchmark_items
from htmleval.benchmark.models import load_model_profile
from htmleval.benchmark.runner import _valid_response


def _load_model_profile(path: Path, profile: str) -> dict:
    # Reuse the benchmark model loader so ${ENV_VAR} api keys are expanded
    # consistently with `python -m htmleval benchmark generate`.
    return load_model_profile(profile, str(path))


def _messages_response(rec: dict) -> str:
    messages = (rec.get("data") or {}).get("messages") or []
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return rec.get("response") or ""


def _load_valid_checkpoint_responses(responses_path: Path) -> dict[str, dict]:
    """Load valid existing responses so interrupted repairs can resume safely."""
    kept: dict[str, dict] = {}
    if not responses_path.exists():
        return kept
    with responses_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            item_id = str(rec.get("id") or "")
            response = rec.get("response") or ""
            if item_id and _valid_response(response):
                kept[item_id] = rec
    return kept


def _rebuild_checkpoint_from_results(
    results_path: Path,
    responses_path: Path,
    *,
    write: bool,
) -> set[str]:
    """Write only valid responses from results.jsonl to responses_path."""
    kept: dict[str, dict] = {}
    # Start with current checkpoint entries. This preserves responses generated
    # by a previous repair attempt even if they have not been evaluated yet.
    kept.update(_load_valid_checkpoint_responses(responses_path))

    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                item_id = str(rec.get("line_number") or rec.get("id") or "")
                response = _messages_response(rec)
                if item_id and _valid_response(response):
                    out = {
                        "id": item_id,
                        "category": rec.get("source_category") or rec.get("category"),
                        "sub_type": rec.get("sub_type"),
                        "difficulty": rec.get("difficulty"),
                        "language": rec.get("language"),
                        "prompt": ((rec.get("data") or {}).get("messages") or [{}])[0].get("content", rec.get("prompt", "")),
                        "response": response,
                    }
                    kept[item_id] = out

    if write:
        responses_path.parent.mkdir(parents=True, exist_ok=True)
        with responses_path.open("w", encoding="utf-8") as f:
            for item_id in sorted(kept):
                f.write(json.dumps(kept[item_id], ensure_ascii=False) + "\n")
    return set(kept)


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-path", required=True)
    ap.add_argument("--model-dir", required=True, help=".../<model>/en directory")
    ap.add_argument("--model-profile", default="")
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--generate-url", default="")
    ap.add_argument("--generate-model", default="")
    ap.add_argument("--generate-key", default="")
    ap.add_argument("--only-id", action="append", default=[])
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    results_path = model_dir / "full" / "results.jsonl"
    responses_path = model_dir / "responses.jsonl"
    kept_ids = _rebuild_checkpoint_from_results(
        results_path,
        responses_path,
        write=not args.dry_run,
    )

    items = load_benchmark_items(args.benchmark_path, language="en")
    only_ids = {
        part.strip()
        for raw in args.only_id
        for part in str(raw).split(",")
        if part.strip()
    }
    missing = [item for item in items if str(item.get("id", "")) not in kept_ids]
    if only_ids:
        missing = [item for item in missing if str(item.get("id", "")) in only_ids]
    print(f"checkpoint_rebuilt={responses_path}")
    print(f"valid_responses={len(kept_ids)} total_items={len(items)} missing={len(missing)}")
    if only_ids:
        print("only_ids=" + ",".join(sorted(only_ids)))
    if missing:
        print("missing_ids=" + ",".join(str(x.get("id")) for x in missing))
    if args.dry_run or not missing:
        return

    profile = (
        _load_model_profile(Path(args.models_config), args.model_profile)
        if args.model_profile else {}
    )
    base_url = args.generate_url or profile.get("base_url")
    model = args.generate_model or profile.get("model")
    if not base_url or not model:
        raise ValueError(
            "Generation requires --model-profile or both --generate-url and --generate-model"
        )
    generated = await generate_responses(
        missing,
        base_url=base_url,
        api_key=str(args.generate_key or profile.get("api_key") or "EMPTY"),
        model=model,
        concurrency=args.concurrency or int(profile.get("concurrency", 4)),
        temperature=args.temperature if args.temperature is not None else float(profile.get("temperature", 0.7)),
        timeout=args.timeout or int(profile.get("timeout", 300)),
        output_path=str(responses_path),
        max_tokens=args.max_tokens,
        disable_thinking=args.disable_thinking,
        checkpoint_flush_interval=1,
    )
    ok = sum(1 for item in generated if _valid_response(item.get("response", "")))
    print(f"generated_valid={ok}/{len(missing)}")


if __name__ == "__main__":
    asyncio.run(_main())
