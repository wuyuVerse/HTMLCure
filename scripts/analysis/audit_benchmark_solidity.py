#!/usr/bin/env python3
"""Static solidity audit for HTMLBench benchmark test cases.

This script does not score models and does not decide whether a TC should be
changed. It produces a triage report for reviewer-facing benchmark hardening:
schema/logic errors, cross-template leftovers, duplicate checks, and broad
visual-only checks that need human review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from htmleval.benchmark.analysis import VALIDATOR_MODE, validate_items


SEVERITY_ORDER = {"error": 0, "high": 1, "medium": 2, "low": 3}

RISK_META = {
    "validator_error": (
        "error",
        "Fails the strict benchmark validator.",
    ),
    "screenshot_changed_without_prior_screenshot": (
        "error",
        "A screenshot-change assertion has no prior screenshot seed.",
    ),
    "exact_duplicate_steps": (
        "high",
        "A TC repeats the exact same step sequence as another TC in the same item.",
    ),
    "duplicate_tc_name": (
        "medium",
        "A TC name is reused inside the same item.",
    ),
    "cross_template_empty_submit": (
        "high",
        "Empty-submit validation appears on a prompt that does not ask for a form flow.",
    ),
    "cross_template_add_button": (
        "high",
        "An Add-button requirement appears on a prompt that does not ask for add/create behavior.",
    ),
    "cross_template_markdown_editor": (
        "high",
        "Markdown/editor/bold/preview language appears on a prompt that does not ask for it.",
    ),
    "cross_template_heatmap": (
        "medium",
        "Heatmap-specific language appears on a prompt that does not ask for a heatmap/matrix.",
    ),
    "visual_change_only_tc": (
        "medium",
        "A TC relies on screenshot-change only, without a semantic/state assertion.",
    ),
    "generic_screenshot_not_blank": (
        "medium",
        "A TC uses nonblank screenshot evidence as a generic visual/no-crash signal.",
    ),
    "broad_surface_visible": (
        "medium",
        "A TC checks only for a broad surface such as svg/canvas.",
    ),
    "high_tc_density": (
        "low",
        "An item has unusually many TCs and should be checked for overlap.",
    ),
}


FORM_PROMPT_TOKENS = {
    "form",
    "submit",
    "login",
    "register",
    "signup",
    "sign up",
    "send",
    "save",
    "email",
    "message",
    "contact",
    "booking",
    "application",
    "feedback",
    "survey",
    "rsvp",
    "checkout",
}

ADD_PROMPT_TOKENS = {
    "add",
    "new",
    "create",
    "insert",
    "task",
    "todo",
    "card",
    "item",
    "note",
    "event",
    "entry",
}

EDITOR_PROMPT_TOKENS = {
    "markdown",
    "editor",
    "preview",
    "rich text",
    "bold",
    "format",
    "document",
    "whiteboard",
    "canvas editor",
}

HEATMAP_PROMPT_TOKENS = {
    "heatmap",
    "matrix",
    "correlation",
    "calendar",
    "grid",
    "cell",
}

GENERIC_VISUAL_NAMES = (
    "click triggers interaction",
    "keyboard input changes",
    "button click changes",
    "hover button",
    "visual changes",
    "visible reaction",
    "changes canvas",
    "screenshot changed",
    "game click triggers interaction",
)

SEMANTIC_ASSERTIONS = {
    "assert_js_value",
    "assert_text_contains",
    "assert_count",
    "assert_visible",
    "assert_not_visible",
    "assert_attribute",
    "assert_style",
    "assert_semantic_html",
    "assert_a11y_basic",
    "assert_no_horizontal_scroll",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _has_any(text: str, needles: set[str] | tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _step_text(tc: dict[str, Any]) -> str:
    return _norm((tc.get("name") or "") + " " + json.dumps(tc.get("steps", []), ensure_ascii=False))


def _step_signature(tc: dict[str, Any]) -> str:
    return json.dumps(tc.get("steps", []), sort_keys=True, ensure_ascii=False)


def _iter_items(path: Path) -> list[tuple[Path, int, dict[str, Any]]]:
    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))

    rows: list[tuple[Path, int, dict[str, Any]]] = []
    for file_path in files:
        if file_path.suffix == ".jsonl":
            with file_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if line.strip():
                        rows.append((file_path, line_no, json.loads(line)))
        elif file_path.suffix == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for idx, item in enumerate(data, 1):
                    rows.append((file_path, idx, item))
            elif isinstance(data, dict):
                rows.append((file_path, 1, data))
    return rows


def _risk_entry(
    file_path: Path,
    line_no: int,
    item: dict[str, Any],
    tc: dict[str, Any] | None,
    detail: str,
) -> dict[str, Any]:
    return {
        "file": str(file_path),
        "line": line_no,
        "item_id": item.get("id"),
        "category": item.get("category"),
        "sub_type": item.get("sub_type"),
        "tc_id": None if tc is None else tc.get("id"),
        "tc_name": None if tc is None else tc.get("name"),
        "detail": detail,
    }


def _add_risk(
    risk_counts: Counter[str],
    risk_examples: dict[str, list[dict[str, Any]]],
    item_risks: Counter[str],
    risk: str,
    entry: dict[str, Any],
    max_examples: int,
) -> None:
    risk_counts[risk] += 1
    if entry.get("item_id"):
        item_risks[str(entry["item_id"])] += 1
    if len(risk_examples[risk]) < max_examples:
        risk_examples[risk].append(entry)


def _has_prior_screenshot(steps: list[dict[str, Any]], idx: int) -> bool:
    return any(step.get("action") == "screenshot" for step in steps[:idx])


def audit(path: Path, max_examples: int = 25) -> dict[str, Any]:
    rows = _iter_items(path)
    items = [item for _, _, item in rows]
    validator_errors = validate_items(items)

    risk_counts: Counter[str] = Counter()
    risk_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    item_risks: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    weight_counts: Counter[str] = Counter()
    file_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "items": 0,
        "test_cases": 0,
        "risk_counts": Counter(),
    })
    item_tc_counts: dict[str, int] = {}

    for err in validator_errors:
        # Validator messages already include item/tc context; keep them as report-level errors.
        risk_counts["validator_error"] += 1
        if len(risk_examples["validator_error"]) < max_examples:
            risk_examples["validator_error"].append({"detail": err})

    for file_path, line_no, item in rows:
        file_key = str(file_path)
        prompt = _norm(item.get("prompt", ""))
        test_cases = item.get("test_cases", []) or []
        item_id = str(item.get("id"))
        item_tc_counts[item_id] = len(test_cases)

        file_stats[file_key]["items"] += 1
        file_stats[file_key]["test_cases"] += len(test_cases)

        if len(test_cases) > 45:
            entry = _risk_entry(file_path, line_no, item, None, f"{len(test_cases)} test cases")
            _add_risk(risk_counts, risk_examples, item_risks, "high_tc_density", entry, max_examples)
            file_stats[file_key]["risk_counts"]["high_tc_density"] += 1

        seen_signatures: dict[str, str] = {}
        seen_names: dict[str, str] = {}

        for tc in test_cases:
            tc_text = _step_text(tc)
            tc_name = _norm(tc.get("name", ""))
            steps = tc.get("steps", []) or []
            weight_counts[str(tc.get("weight", 1.0))] += 1

            sig = _step_signature(tc)
            if sig in seen_signatures:
                entry = _risk_entry(
                    file_path,
                    line_no,
                    item,
                    tc,
                    f"duplicates steps from {seen_signatures[sig]}",
                )
                _add_risk(risk_counts, risk_examples, item_risks, "exact_duplicate_steps", entry, max_examples)
                file_stats[file_key]["risk_counts"]["exact_duplicate_steps"] += 1
            else:
                seen_signatures[sig] = str(tc.get("id"))

            if tc_name:
                if tc_name in seen_names:
                    entry = _risk_entry(
                        file_path,
                        line_no,
                        item,
                        tc,
                        f"duplicates name from {seen_names[tc_name]}",
                    )
                    _add_risk(risk_counts, risk_examples, item_risks, "duplicate_tc_name", entry, max_examples)
                    file_stats[file_key]["risk_counts"]["duplicate_tc_name"] += 1
                else:
                    seen_names[tc_name] = str(tc.get("id"))

            if "empty submit shows validation" in tc_text and not _has_any(prompt, FORM_PROMPT_TOKENS):
                entry = _risk_entry(file_path, line_no, item, tc, "prompt lacks form/submit flow")
                _add_risk(risk_counts, risk_examples, item_risks, "cross_template_empty_submit", entry, max_examples)
                file_stats[file_key]["risk_counts"]["cross_template_empty_submit"] += 1

            if ("'add' feature button present as prompt requires" in tc_text or "add feature button" in tc_text) and not _has_any(prompt, ADD_PROMPT_TOKENS):
                entry = _risk_entry(file_path, line_no, item, tc, "prompt lacks add/create behavior")
                _add_risk(risk_counts, risk_examples, item_risks, "cross_template_add_button", entry, max_examples)
                file_stats[file_key]["risk_counts"]["cross_template_add_button"] += 1

            editor_leftover = (
                "bold/formatting" in tc_text
                or "bold button" in tc_text
                or "preview updates in real-time" in tc_text
                or ("markdown" in tc_text and "markdown" not in prompt)
            )
            if editor_leftover and not _has_any(prompt, EDITOR_PROMPT_TOKENS):
                entry = _risk_entry(file_path, line_no, item, tc, "prompt lacks markdown/editor/preview requirement")
                _add_risk(risk_counts, risk_examples, item_risks, "cross_template_markdown_editor", entry, max_examples)
                file_stats[file_key]["risk_counts"]["cross_template_markdown_editor"] += 1

            if "heatmap" in tc_text and not _has_any(prompt, HEATMAP_PROMPT_TOKENS):
                entry = _risk_entry(file_path, line_no, item, tc, "prompt lacks heatmap/matrix requirement")
                _add_risk(risk_counts, risk_examples, item_risks, "cross_template_heatmap", entry, max_examples)
                file_stats[file_key]["risk_counts"]["cross_template_heatmap"] += 1

            has_screenshot_changed = any(step.get("action") == "assert_screenshot_changed" for step in steps)
            has_semantic_assertion = any(step.get("action") in SEMANTIC_ASSERTIONS for step in steps)
            if has_screenshot_changed and not has_semantic_assertion:
                if _has_any(tc_name, GENERIC_VISUAL_NAMES) or len(steps) <= 5:
                    entry = _risk_entry(file_path, line_no, item, tc, "visual-change-only interaction check")
                    _add_risk(risk_counts, risk_examples, item_risks, "visual_change_only_tc", entry, max_examples)
                    file_stats[file_key]["risk_counts"]["visual_change_only_tc"] += 1

            for idx, step in enumerate(steps):
                action = step.get("action", "")
                if action:
                    action_counts[action] += 1

                if action == "assert_screenshot_changed" and not _has_prior_screenshot(steps, idx):
                    entry = _risk_entry(file_path, line_no, item, tc, f"step {idx}")
                    _add_risk(
                        risk_counts,
                        risk_examples,
                        item_risks,
                        "screenshot_changed_without_prior_screenshot",
                        entry,
                        max_examples,
                    )
                    file_stats[file_key]["risk_counts"]["screenshot_changed_without_prior_screenshot"] += 1

                if action == "assert_screenshot_not_blank":
                    entry = _risk_entry(file_path, line_no, item, tc, "nonblank-only visual check")
                    _add_risk(risk_counts, risk_examples, item_risks, "generic_screenshot_not_blank", entry, max_examples)
                    file_stats[file_key]["risk_counts"]["generic_screenshot_not_blank"] += 1

                selector = _norm(step.get("selector", ""))
                if action == "assert_visible" and selector in {"svg, canvas", "canvas, svg", "svg,canvas", "canvas,svg"}:
                    entry = _risk_entry(file_path, line_no, item, tc, f"selector={step.get('selector')!r}")
                    _add_risk(risk_counts, risk_examples, item_risks, "broad_surface_visible", entry, max_examples)
                    file_stats[file_key]["risk_counts"]["broad_surface_visible"] += 1

    risk_table = []
    for risk, count in risk_counts.items():
        severity, description = RISK_META.get(risk, ("low", "Unclassified risk."))
        risk_table.append({
            "risk": risk,
            "severity": severity,
            "count": count,
            "description": description,
        })
    risk_table.sort(key=lambda r: (SEVERITY_ORDER.get(r["severity"], 99), -int(r["count"]), r["risk"]))

    file_table = []
    for file_key, stats in sorted(file_stats.items()):
        risks = dict(stats["risk_counts"])
        file_table.append({
            "file": file_key,
            "items": stats["items"],
            "test_cases": stats["test_cases"],
            "avg_test_cases_per_item": round(stats["test_cases"] / max(1, stats["items"]), 2),
            "risk_counts": risks,
            "risk_total": sum(risks.values()),
        })

    top_items = []
    for item_id, count in item_risks.most_common(50):
        top_items.append({
            "item_id": item_id,
            "risk_count": count,
            "test_cases": item_tc_counts.get(item_id, 0),
        })

    return {
        "benchmark_path": str(path),
        "validator_mode": VALIDATOR_MODE,
        "items": len(rows),
        "test_cases": sum(item_tc_counts.values()),
        "avg_test_cases_per_item": round(sum(item_tc_counts.values()) / max(1, len(rows)), 2),
        "action_counts": dict(action_counts.most_common()),
        "weight_counts": dict(sorted(weight_counts.items())),
        "risk_table": risk_table,
        "risk_examples": dict(risk_examples),
        "files": file_table,
        "top_risk_items": top_items,
    }


def write_markdown(report: dict[str, Any], out_path: Path, top_examples: int = 8) -> None:
    lines: list[str] = []
    lines.append("# HTMLBench Solidity Audit")
    lines.append("")
    lines.append(f"- Benchmark: `{report['benchmark_path']}`")
    lines.append(f"- Validator: `{report['validator_mode']}`")
    lines.append(f"- Items: {report['items']}")
    lines.append(f"- Test cases: {report['test_cases']}")
    lines.append(f"- Avg TC/item: {report['avg_test_cases_per_item']}")
    lines.append("")
    lines.append("## Risk Counts")
    lines.append("")
    lines.append("| Severity | Risk | Count | Meaning |")
    lines.append("|----------|------|-------|---------|")
    for row in report["risk_table"]:
        lines.append(
            f"| {row['severity']} | `{row['risk']}` | {row['count']} | {row['description']} |"
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| File | Items | TCs | Avg TC/item | Risk Total |")
    lines.append("|------|-------|-----|-------------|------------|")
    for row in report["files"]:
        lines.append(
            f"| `{row['file']}` | {row['items']} | {row['test_cases']} | "
            f"{row['avg_test_cases_per_item']} | {row['risk_total']} |"
        )
    lines.append("")
    lines.append("## Top Risk Items")
    lines.append("")
    lines.append("| Item | Risk Count | TC Count |")
    lines.append("|------|------------|----------|")
    for row in report["top_risk_items"][:25]:
        lines.append(f"| `{row['item_id']}` | {row['risk_count']} | {row['test_cases']} |")
    lines.append("")
    lines.append("## Examples")
    for row in report["risk_table"]:
        risk = row["risk"]
        examples = report["risk_examples"].get(risk, [])
        if not examples:
            continue
        lines.append("")
        lines.append(f"### `{risk}`")
        for ex in examples[:top_examples]:
            item = ex.get("item_id", "")
            tc = ex.get("tc_id", "")
            file_name = ex.get("file", "")
            detail = ex.get("detail", "")
            name = ex.get("tc_name", "")
            lines.append(f"- `{item}` / `{tc}` in `{file_name}`: {detail}; {name}")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark/en"))
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    report = audit(args.benchmark, max_examples=args.max_examples)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.out_md:
        write_markdown(report, args.out_md)

    print(
        f"audited {report['items']} items, {report['test_cases']} TCs, "
        f"{len(report['risk_table'])} risk types",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
