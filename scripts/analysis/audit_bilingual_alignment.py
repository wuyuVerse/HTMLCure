#!/usr/bin/env python3
"""Audit structural alignment between English and Chinese benchmark splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


FILES = (
    "apps_tools.jsonl",
    "content_marketing.jsonl",
    "data_visualization.jsonl",
    "games_simulations.jsonl",
    "three_d_webgl.jsonl",
    "visual_art_animation.jsonl",
)

META_FIELDS = ("category", "source_category", "sub_type", "difficulty", "has_interaction")
ALLOWED_TOP_LEVEL_DIFFS = {"prompt", "language", "test_cases", "_line_no"}
ALLOWED_STEP_FIELD_DIFFS = {
    "assert_text_contains": {"text_pattern"},
    "assert_text_not_contains": {"text_pattern"},
    "click_text": {"text_pattern"},
    "assert_js_value": {"expression"},
    "eval_js": {"expression"},
}
INTERNAL_FIELD_PREFIXES = ("_",)


def is_internal_field(field: str) -> bool:
    """Return true for non-scoring provenance fields."""
    return field.startswith(INTERNAL_FIELD_PREFIXES)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            rows.append(item)
    return rows


def add_error(
    errors: list[str],
    counts: Counter[str],
    kind: str,
    message: str,
    *,
    max_examples: int,
) -> None:
    counts[kind] += 1
    if len(errors) < max_examples:
        errors.append(f"{kind}: {message}")


def audit(
    en_dir: Path,
    zh_dir: Path,
    *,
    max_examples: int = 20,
    strict_fields: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    counts: Counter[str] = Counter()
    examples: list[str] = []
    per_file: list[dict[str, Any]] = []
    total_items = 0

    for filename in FILES:
        en_path = en_dir / filename
        zh_path = zh_dir / filename
        file_counts: Counter[str] = Counter()

        if not en_path.exists():
            add_error(examples, counts, "missing_en_file", str(en_path), max_examples=max_examples)
            file_counts["missing_en_file"] += 1
            continue
        if not zh_path.exists():
            add_error(examples, counts, "missing_zh_file", str(zh_path), max_examples=max_examples)
            file_counts["missing_zh_file"] += 1
            continue

        en_rows = load_jsonl(en_path)
        zh_rows = load_jsonl(zh_path)
        total_items += min(len(en_rows), len(zh_rows))

        if len(en_rows) != len(zh_rows):
            msg = f"{filename}: en={len(en_rows)} zh={len(zh_rows)}"
            add_error(examples, counts, "line_count_mismatch", msg, max_examples=max_examples)
            file_counts["line_count_mismatch"] += 1

        for idx, (en_item, zh_item) in enumerate(zip(en_rows, zh_rows), start=1):
            item_label = f"{filename}:{idx}:{en_item.get('id')}:{zh_item.get('id')}"

            if en_item.get("id") != zh_item.get("id"):
                add_error(
                    examples,
                    counts,
                    "id_mismatch",
                    item_label,
                    max_examples=max_examples,
                )
                file_counts["id_mismatch"] += 1
                continue

            if zh_item.get("language") != "zh":
                add_error(
                    examples,
                    counts,
                    "language_mismatch",
                    f"{item_label}: zh.language={zh_item.get('language')!r}",
                    max_examples=max_examples,
                )
                file_counts["language_mismatch"] += 1

            for field in META_FIELDS:
                if en_item.get(field) != zh_item.get(field):
                    add_error(
                        examples,
                        counts,
                        "metadata_mismatch",
                        f"{item_label}: {field} en={en_item.get(field)!r} zh={zh_item.get(field)!r}",
                        max_examples=max_examples,
                    )
                    file_counts["metadata_mismatch"] += 1

            if strict_fields:
                for field in sorted(set(en_item) | set(zh_item)):
                    if is_internal_field(field):
                        continue
                    if field in ALLOWED_TOP_LEVEL_DIFFS:
                        continue
                    if field in META_FIELDS or field == "id":
                        continue
                    if en_item.get(field) != zh_item.get(field):
                        add_error(
                            examples,
                            counts,
                            "top_level_field_mismatch",
                            f"{item_label}: {field} en={en_item.get(field)!r} zh={zh_item.get(field)!r}",
                            max_examples=max_examples,
                        )
                        file_counts["top_level_field_mismatch"] += 1

            en_tcs = en_item.get("test_cases") or []
            zh_tcs = zh_item.get("test_cases") or []
            if len(en_tcs) != len(zh_tcs):
                add_error(
                    examples,
                    counts,
                    "tc_count_mismatch",
                    f"{item_label}: en={len(en_tcs)} zh={len(zh_tcs)}",
                    max_examples=max_examples,
                )
                file_counts["tc_count_mismatch"] += 1

            for tc_idx, (en_tc, zh_tc) in enumerate(zip(en_tcs, zh_tcs), start=1):
                tc_label = f"{item_label}:tc[{tc_idx}]"
                if en_tc.get("id") != zh_tc.get("id"):
                    add_error(
                        examples,
                        counts,
                        "tc_id_mismatch",
                        f"{tc_label}: en={en_tc.get('id')!r} zh={zh_tc.get('id')!r}",
                        max_examples=max_examples,
                    )
                    file_counts["tc_id_mismatch"] += 1
                if en_tc.get("weight") != zh_tc.get("weight"):
                    add_error(
                        examples,
                        counts,
                        "tc_weight_mismatch",
                        f"{tc_label}: {en_tc.get('id')} en={en_tc.get('weight')!r} zh={zh_tc.get('weight')!r}",
                        max_examples=max_examples,
                    )
                    file_counts["tc_weight_mismatch"] += 1

                if strict_fields:
                    for field in sorted(set(en_tc) | set(zh_tc)):
                        if is_internal_field(field):
                            continue
                        if field == "steps":
                            continue
                        if en_tc.get(field) != zh_tc.get(field):
                            add_error(
                                examples,
                                counts,
                                "tc_field_mismatch",
                                f"{tc_label}: {field} en={en_tc.get(field)!r} zh={zh_tc.get(field)!r}",
                                max_examples=max_examples,
                            )
                            file_counts["tc_field_mismatch"] += 1

                en_steps = en_tc.get("steps") or []
                zh_steps = zh_tc.get("steps") or []
                if len(en_steps) != len(zh_steps):
                    add_error(
                        examples,
                        counts,
                        "step_count_mismatch",
                        f"{tc_label}: {en_tc.get('id')} en={len(en_steps)} zh={len(zh_steps)}",
                        max_examples=max_examples,
                    )
                    file_counts["step_count_mismatch"] += 1
                    continue

                en_actions = [step.get("action") for step in en_steps]
                zh_actions = [step.get("action") for step in zh_steps]
                if en_actions != zh_actions:
                    add_error(
                        examples,
                        counts,
                        "step_action_sequence_mismatch",
                        f"{tc_label}: {en_tc.get('id')} en={en_actions} zh={zh_actions}",
                        max_examples=max_examples,
                    )
                    file_counts["step_action_sequence_mismatch"] += 1

                if strict_fields:
                    for step_idx, (en_step, zh_step) in enumerate(zip(en_steps, zh_steps), start=1):
                        action = en_step.get("action")
                        allowed_fields = ALLOWED_STEP_FIELD_DIFFS.get(str(action), set())
                        for field in sorted(set(en_step) | set(zh_step)):
                            if is_internal_field(field):
                                continue
                            if field in allowed_fields:
                                if en_step.get(field) != zh_step.get(field):
                                    counts[f"allowed_{action}_{field}_diff"] += 1
                                continue
                            if en_step.get(field) != zh_step.get(field):
                                add_error(
                                    examples,
                                    counts,
                                    "step_field_mismatch",
                                    f"{tc_label}: {en_tc.get('id')} step[{step_idx}] {field} "
                                    f"en={en_step.get(field)!r} zh={zh_step.get(field)!r}",
                                    max_examples=max_examples,
                                )
                                file_counts["step_field_mismatch"] += 1

        per_file.append({
            "file": filename,
            "en_items": len(en_rows),
            "zh_items": len(zh_rows),
            "mismatches": dict(file_counts),
        })

    summary = {
        "en_dir": str(en_dir),
        "zh_dir": str(zh_dir),
        "strict_fields": strict_fields,
        "files": per_file,
        "total_items_checked": total_items,
        "mismatch_counts": {k: v for k, v in counts.items() if not k.startswith("allowed_")},
        "allowed_difference_counts": {k: v for k, v in counts.items() if k.startswith("allowed_")},
        "mismatch_total": sum(v for k, v in counts.items() if not k.startswith("allowed_")),
    }
    return summary, examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-dir", type=Path, default=Path("benchmark/en"))
    parser.add_argument("--zh-dir", type=Path, default=Path("benchmark/zh"))
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    parser.add_argument(
        "--strict-fields",
        action="store_true",
        help="Also compare every non-language-sensitive test-case and step field.",
    )
    args = parser.parse_args()

    summary, examples = audit(
        args.en_dir,
        args.zh_dir,
        max_examples=args.max_examples,
        strict_fields=args.strict_fields,
    )
    if args.json:
        print(json.dumps({"summary": summary, "examples": examples}, indent=2, ensure_ascii=False))
    else:
        print("Bilingual alignment audit")
        print(f"en_dir={summary['en_dir']}")
        print(f"zh_dir={summary['zh_dir']}")
        print(f"strict_fields={summary['strict_fields']}")
        print(f"total_items_checked={summary['total_items_checked']}")
        print(f"mismatch_total={summary['mismatch_total']}")
        for row in summary["files"]:
            print(
                f"  {row['file']}: en={row['en_items']} zh={row['zh_items']} "
                f"mismatches={sum(row['mismatches'].values())}"
            )
        if summary["mismatch_counts"]:
            print("mismatch_counts:")
            for key, value in sorted(summary["mismatch_counts"].items()):
                print(f"  {key}: {value}")
        if summary["allowed_difference_counts"]:
            print("allowed_difference_counts:")
            for key, value in sorted(summary["allowed_difference_counts"].items()):
                print(f"  {key}: {value}")
        if examples:
            print("examples:")
            for ex in examples:
                print(f"  - {ex}")

    return 1 if summary["mismatch_total"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
