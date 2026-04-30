#!/usr/bin/env python3
"""Freeze the 2026-04-30 6000-TC scoring pool into benchmark/en and zh.

The selection policy is intentionally centralized in
scripts/analysis/rescore_drop_cov_tc_rend_0430.py so the benchmark JSONL files,
the 0430 rescore package, and validation scripts use the same TC pool.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = BASE_DIR / "benchmark"
RESCORE_SCRIPT = BASE_DIR / "scripts" / "analysis" / "rescore_drop_cov_tc_rend_0430.py"
DEFAULT_MANIFEST = BENCHMARK_DIR / "tc_selection_20260430_6000.json"


def load_rescore_module():
    spec = importlib.util.spec_from_file_location("htmlbench_rescore_0430", RESCORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RESCORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_items(lang: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    items: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((BENCHMARK_DIR / lang).glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                item_id = str(item.get("id", ""))
                if not item_id:
                    raise ValueError(f"item missing id in {path}")
                if item_id in items:
                    raise ValueError(f"duplicate item id {item_id} in {path}")
                items[item_id] = (path, item)
    return items


def write_items(lang: str, items_by_id: dict[str, tuple[Path, dict[str, Any]]]) -> None:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for path, item in items_by_id.values():
        by_path[path].append(item)
    for path, items in sorted(by_path.items()):
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False, sort_keys=False) + "\n")


def test_case_id(tc: dict[str, Any]) -> str:
    return str(tc.get("id") or tc.get("name") or "")


def build_selection():
    mod = load_rescore_module()
    runs = mod.discover_runs(mod.DEFAULT_INPUT_ROOTS)
    stats = mod.collect_tc_stats(runs, target_model=mod.DEFAULT_TARGET_SFT_MODEL)
    tc_filter = mod.build_tc_filter(
        stats,
        min_observations=8,
        low_pass_rate=0.05,
        high_pass_rate=0.95,
        sft_contrast_enabled=True,
        target_model=mod.DEFAULT_TARGET_SFT_MODEL,
        target_tc_count=mod.DEFAULT_TARGET_TC_COUNT,
        hard_anchor_ratio=mod.DEFAULT_HARD_ANCHOR_RATIO,
    )
    return mod, tc_filter


def selected_by_item(tc_filter) -> dict[str, list[str]]:
    by_item: dict[str, list[str]] = defaultdict(list)
    for item_id, test_id in sorted(tc_filter.keep):
        by_item[item_id].append(test_id)
    return dict(by_item)


def apply_to_en(
    en_items: dict[str, tuple[Path, dict[str, Any]]],
    selected: dict[str, list[str]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], Counter]:
    source_cases: dict[str, dict[str, dict[str, Any]]] = {}
    counts: Counter = Counter()
    for item_id, test_ids in selected.items():
        if item_id not in en_items:
            raise ValueError(f"selected item not found in benchmark/en: {item_id}")
        _path, item = en_items[item_id]
        original = {test_case_id(tc): tc for tc in item.get("test_cases") or []}
        missing = [tc_id for tc_id in test_ids if tc_id not in original]
        if missing:
            raise ValueError(f"benchmark/en item {item_id} missing selected TC ids: {missing[:5]}")
        item["test_cases"] = [original[tc_id] for tc_id in test_ids]
        source_cases[item_id] = {tc_id: original[tc_id] for tc_id in test_ids}
        counts["items"] += 1
        counts["test_cases"] += len(test_ids)
    return source_cases, counts


def apply_to_zh(
    zh_items: dict[str, tuple[Path, dict[str, Any]]],
    selected: dict[str, list[str]],
    en_selected_cases: dict[str, dict[str, dict[str, Any]]],
) -> Counter:
    counts: Counter = Counter()
    for item_id, test_ids in selected.items():
        if item_id not in zh_items:
            raise ValueError(f"selected item not found in benchmark/zh: {item_id}")
        _path, item = zh_items[item_id]
        new_cases = []
        for tc_id in test_ids:
            tc = dict(en_selected_cases[item_id][tc_id])
            tc["_tc_source"] = "en_selected_aligned"
            counts["en_selected_aligned"] += 1
            new_cases.append(tc)
        item["test_cases"] = new_cases
        counts["items"] += 1
        counts["test_cases"] += len(new_cases)
    return counts


def build_manifest(mod, tc_filter, zh_counts: Counter) -> dict[str, Any]:
    rows = []
    for item_id, test_id in sorted(tc_filter.keep):
        st = tc_filter.stats[(item_id, test_id)]
        rows.append(
            {
                "item_id": item_id,
                "test_id": test_id,
                "keep_reason": tc_filter.keep_reasons.get((item_id, test_id), ""),
                "observations": st.observations,
                "pass_rate": round(st.pass_rate, 4),
                "sft_pass_rate": round(st.sft_pass_rate, 4),
                "non_sft_pass_rate": round(st.non_sft_pass_rate, 4),
                "target_pass_rate": round(st.target_pass_rate, 4),
                "name": st.name,
            }
        )
    keep_reason_counts = Counter(row["keep_reason"] for row in rows)
    return {
        "score_version": mod.SCORE_VERSION,
        "target_tc_count": mod.DEFAULT_TARGET_TC_COUNT,
        "hard_anchor_ratio": mod.DEFAULT_HARD_ANCHOR_RATIO,
        "target_sft_model": mod.DEFAULT_TARGET_SFT_MODEL,
        "total_tc": len(rows),
        "keep_reason_counts": dict(sorted(keep_reason_counts.items())),
        "zh_source_counts": dict(sorted(zh_counts.items())),
        "selection": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mod, tc_filter = build_selection()
    selected = selected_by_item(tc_filter)
    if sum(len(v) for v in selected.values()) != mod.DEFAULT_TARGET_TC_COUNT:
        raise SystemExit("selection did not produce exactly 6000 TC")

    en_items = load_items("en")
    zh_items = load_items("zh")
    if set(en_items) != set(zh_items):
        raise SystemExit("benchmark/en and benchmark/zh item ids do not match")

    en_selected_cases, en_counts = apply_to_en(en_items, selected)
    zh_counts = apply_to_zh(zh_items, selected, en_selected_cases)
    manifest = build_manifest(mod, tc_filter, zh_counts)

    print(f"en: items={en_counts['items']} tc={en_counts['test_cases']}")
    print(
        "zh: items={items} tc={test_cases} "
        "en_selected_aligned={en_selected_aligned}".format(**zh_counts)
    )
    print(f"manifest: {args.manifest}")

    if args.dry_run:
        return

    write_items("en", en_items)
    write_items("zh", zh_items)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
