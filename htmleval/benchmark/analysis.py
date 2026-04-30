"""
Benchmark result analysis — task-family, difficulty, and dimension breakdowns.

Usage:
    from htmleval.benchmark.analysis import analyze_results, print_benchmark_report

    analysis = analyze_results("benchmark_results/results.jsonl")
    print_benchmark_report(analysis)
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from htmleval.benchmark.taxonomy import normalize_task_family, source_category_for

logger = logging.getLogger("htmleval")

# Score tier thresholds (match FilterConfig defaults)
_TIER_A = 80
_TIER_B = 40

# The 5 scoring dimensions
_DIMENSIONS = ("rendering", "visual_design", "functionality", "interaction", "code_quality")
_DIMENSION_LABELS = {
    "rendering": "rendering",
    "visual_design": "visual_design",
    "functionality": "functionality",
    "interaction": "interaction",
    "code_quality": "implementation_quality",
}
VALIDATOR_MODE = "strict_logic_v2"
_COMPLETED_STATUSES = {"completed", "completed_with_fallback"}

# ---------------------------------------------------------------------------
# Result analysis
# ---------------------------------------------------------------------------

def analyze_results(results_path: str, items: list[dict] | None = None) -> dict:
    """Parse results JSONL and produce benchmark-specific breakdown.

    Metadata (category, sub_type, difficulty) is read directly from each result
    record. If original items are passed, they serve as a fallback for metadata
    enrichment via ID lookup.

    Returns dict with keys: overall, by_category, by_difficulty, by_subtype,
    by_source_category, by_dimension, failed_tests.
    """
    results = _load_results(results_path)
    if not results:
        return _empty_analysis()

    # Fallback item lookup (for older results without inline metadata)
    item_map: dict[str, dict] = {}
    if items:
        for item in items:
            item_map[str(item.get("id", ""))] = item

    # Accumulators
    overall = _Bucket()
    by_category: dict[str, _Bucket] = defaultdict(_Bucket)
    by_source_category: dict[str, _Bucket] = defaultdict(_Bucket)
    by_difficulty: dict[str, _Bucket] = defaultdict(_Bucket)
    by_subtype: dict[str, _Bucket] = defaultdict(_Bucket)
    by_language: dict[str, _Bucket] = defaultdict(_Bucket)
    test_failures: dict[str, dict] = defaultdict(lambda: {"count": 0, "errors": []})

    for rec in results:
        score_obj = rec.get("score", {})
        total = score_obj.get("total", 0)
        status = rec.get("eval_status", "unknown")

        # Read metadata from record first, fallback to item_map
        line_num = rec.get("line_number", rec.get("data", {}).get("line_number", ""))
        item = item_map.get(str(line_num), {})
        raw_category = rec.get("category", item.get("category", "unknown"))
        category = normalize_task_family(raw_category)
        source_category = source_category_for(
            raw_category,
            rec.get("source_category", item.get("source_category", "")),
        )
        difficulty = rec.get("difficulty", item.get("difficulty", "unknown"))
        # Handle both "sub_type" (canonical) and "subtype" (legacy)
        subtype = (rec.get("sub_type") or rec.get("subtype")
                   or item.get("sub_type") or item.get("subtype") or "")
        language = rec.get("language", item.get("language", ""))

        overall.add(total, status, score_obj)
        by_category[category].add(total, status, score_obj)
        if source_category:
            by_source_category[source_category].add(total, status, score_obj)
        by_difficulty[difficulty].add(total, status, score_obj)
        if subtype:
            by_subtype[subtype].add(total, status, score_obj)
        if language:
            by_language[language].add(total, status, score_obj)

        test_pass_rate = score_obj.get("test_pass_rate")
        # Track test failures
        if test_pass_rate is not None and test_pass_rate < 1.0:
            test_id = f"{category}/{line_num}"
            test_failures[test_id]["count"] += 1
            tp = score_obj.get("tests_passed", 0)
            tt = score_obj.get("tests_total", 0)
            test_failures[test_id]["errors"].append(f"passed {tp}/{tt}")

    # Build failed_tests list sorted by frequency
    failed_tests = [
        {"test_id": tid, "frequency": info["count"],
         "common_error": info["errors"][0] if info["errors"] else ""}
        for tid, info in sorted(test_failures.items(), key=lambda x: -x[1]["count"])
    ]

    return {
        "overall": overall.finalize(),
        "by_category": {k: v.finalize() for k, v in sorted(by_category.items())},
        "by_source_category": {k: v.finalize() for k, v in sorted(by_source_category.items())},
        "by_difficulty": {k: v.finalize() for k, v in by_difficulty.items()},
        "by_subtype": {k: v.finalize() for k, v in sorted(by_subtype.items())},
        "by_language": {k: v.finalize() for k, v in sorted(by_language.items())},
        "failed_tests": failed_tests[:20],
    }


def analyze_with_trials(
    results_path: str,
    num_trials: int,
    tc_threshold: float = 0.8,
    items: list[dict] | None = None,
) -> dict:
    """Analyze multi-trial results: standard analysis + pass@k metrics.

    Combines the base analyze_results breakdown with per-item pass@k
    metrics from aggregate_trials.

    Args:
        results_path: path to results.jsonl (all trials mixed)
        num_trials: expected number of trials per item
        tc_threshold: pass threshold for is_pass()
        items: optional original benchmark items for metadata fallback

    Returns:
        Analysis dict with additional 'pass_at_k' key.
    """
    from htmleval.benchmark.metrics import aggregate_trials

    # Base analysis (uses median scores across all trials)
    base = analyze_results(results_path, items)

    # Pass@k metrics
    results = _load_results(results_path)
    if results:
        pk = aggregate_trials(results, num_trials=num_trials, tc_threshold=tc_threshold)
        base["pass_at_k"] = pk
    else:
        base["pass_at_k"] = {
            "pass_at_1": 0.0, "pass_at_3": 0.0,
            "num_items": 0, "num_trials": num_trials,
            "by_category": {}, "by_difficulty": {},
        }

    return base


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_benchmark_report(analysis: dict) -> None:
    """Print formatted benchmark analysis to terminal."""
    overall = analysis.get("overall", {})
    n = overall.get("total", 0)

    print("\n" + "=" * 72)
    print("  BENCHMARK RESULTS")
    print("=" * 72)

    if n == 0:
        print("\n  No results found.")
        print("=" * 72)
        return

    # Pass@k metrics (shown first when available)
    pk = analysis.get("pass_at_k")
    if pk and pk.get("num_trials", 1) > 1:
        print(f"\n  Pass@1: {pk['pass_at_1']:.1%}  "
              f"Pass@3: {pk['pass_at_3']:.1%}  "
              f"({pk['num_items']} items x {pk['num_trials']} trials)")

    # Overall summary
    print(f"\n  Total: {n}  "
          f"Avg: {overall.get('avg_score', 0):.1f}  "
          f"Min: {overall.get('min_score', 0)}  "
          f"Max: {overall.get('max_score', 0)}")
    completed_total = overall.get("completed_total", 0)
    if completed_total and completed_total != n:
        print(
            f"  Completed: {completed_total}/{n}  "
            f"Completion: {overall.get('completion_rate', 0):.1%}  "
            f"Avg(completed): {overall.get('avg_score_completed', 0):.1f}"
        )

    median = overall.get("median_score")
    ci_lo = overall.get("ci_95_low")
    ci_hi = overall.get("ci_95_high")
    if median is not None:
        stats_line = f"  Median: {median:.1f}"
        if ci_lo is not None and ci_hi is not None:
            stats_line += f"  95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]"
        print(stats_line)

    print(f"  Tier A (>={_TIER_A}): {overall.get('tier_a_pct', 0):.1%}  "
          f"Tier B ({_TIER_B}-{_TIER_A-1}): {overall.get('tier_b_pct', 0):.1%}  "
          f"Tier C (<{_TIER_B}): {overall.get('tier_c_pct', 0):.1%}")

    tpr = overall.get("avg_test_pass_rate")
    if tpr is not None:
        print(f"  Test pass rate: {tpr:.1%}")

    # Per-dimension averages
    dims = overall.get("dimensions", {})
    if dims:
        parts = [f"{_DIMENSION_LABELS.get(d, d)}: {dims[d]:.1f}" for d in _DIMENSIONS if d in dims]
        print(f"\n  Dimensions: {' | '.join(parts)}")

    # By task family
    _print_table("Task Family", analysis.get("by_category", {}),
                 pk_by_group=pk.get("by_category") if pk else None)

    # By language
    _print_table("Language", analysis.get("by_language", {}))

    # By difficulty (ordered)
    by_diff = analysis.get("by_difficulty", {})
    if by_diff:
        order = {"easy": 0, "medium": 1, "hard": 2}
        ordered = dict(sorted(by_diff.items(), key=lambda x: order.get(x[0], 99)))
        _print_table("Difficulty", ordered,
                     pk_by_group=pk.get("by_difficulty") if pk else None)

    # By subtype (top 10 by avg score)
    by_sub = analysis.get("by_subtype", {})
    if by_sub:
        top = dict(sorted(by_sub.items(), key=lambda x: -x[1].get("avg_score", 0))[:10])
        _print_table("Subtype", top)

    # Failed tests
    failed = analysis.get("failed_tests", [])
    if failed:
        print(f"\n  Top failed tests:")
        for ft in failed[:5]:
            print(f"    {ft['test_id']}: {ft['frequency']}x -- {ft['common_error']}")

    print("\n" + "=" * 72)


def _print_table(label: str, groups: dict, pk_by_group: dict | None = None) -> None:
    """Print a grouped stats table, optionally with pass@1 column."""
    if not groups:
        return
    has_pk = pk_by_group is not None
    header = f"  {label:<20} {'N':>5} {'Avg':>6} {'Min':>5} {'Max':>5} {'TierA%':>7} {'TestPR':>7}"
    if has_pk:
        header += f" {'Pass@1':>7}"
    print(f"\n{header}")
    sep_len = 58 + (8 if has_pk else 0)
    print("  " + "-" * sep_len)
    for name, b in groups.items():
        tpr = b.get("avg_test_pass_rate")
        tpr_s = f"{tpr:>6.1%}" if tpr is not None else "   n/a"
        line = (f"  {name:<20} {b['total']:>5} {b['avg_score']:>6.1f} "
                f"{b.get('min_score', 0):>5} {b.get('max_score', 0):>5} "
                f"{b.get('tier_a_pct', 0):>6.1%} {tpr_s}")
        if has_pk:
            p1 = pk_by_group.get(name)
            pk_s = f"{p1:>6.1%}" if p1 is not None else "   n/a"
            line += f" {pk_s}"
        print(line)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

def compare_analyses(paths: list[str]) -> dict:
    """Load multiple analysis.json files and build a comparison structure.

    Args:
        paths: List of paths — each can be an analysis.json file or a directory
               containing one.

    Returns:
        Dict with keys: models, overall, by_category, by_difficulty, by_dimension.
    """
    analyses: list[tuple[str, dict]] = []

    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            pp = pp / "analysis.json"
        if not pp.exists():
            logger.warning(f"compare: skipping {p} (not found)")
            continue
        data = json.loads(pp.read_text(encoding="utf-8"))
        name = data.get("model", {}).get("name") or pp.parent.name
        analyses.append((name, data))

    if not analyses:
        return {"models": [], "overall": [], "by_category": {}, "by_difficulty": {}, "by_dimension": {}}

    models = [name for name, _ in analyses]
    _FIELDS = ("total", "avg_score", "min_score", "max_score", "tier_a_pct",
               "avg_test_pass_rate", "median_score", "ci_95_low", "ci_95_high")

    def _extract(bucket: dict) -> dict:
        return {k: bucket.get(k) for k in _FIELDS if bucket.get(k) is not None}

    # Overall row per model
    overall = []
    for name, data in analyses:
        row = {"model": name, **_extract(data.get("overall", {}))}
        overall.append(row)

    # By task family: {family: [row_per_model]}
    all_cats: set[str] = set()
    for _, data in analyses:
        all_cats.update(data.get("by_category", {}).keys())
    by_category: dict[str, list[dict]] = {}
    for cat in sorted(all_cats):
        rows = []
        for name, data in analyses:
            bucket = data.get("by_category", {}).get(cat, {})
            rows.append({"model": name, **_extract(bucket)})
        by_category[cat] = rows

    # By-difficulty
    all_diffs: set[str] = set()
    for _, data in analyses:
        all_diffs.update(data.get("by_difficulty", {}).keys())
    by_difficulty: dict[str, list[dict]] = {}
    for diff in sorted(all_diffs, key=lambda d: {"easy": 0, "medium": 1, "hard": 2}.get(d, 99)):
        rows = []
        for name, data in analyses:
            bucket = data.get("by_difficulty", {}).get(diff, {})
            rows.append({"model": name, **_extract(bucket)})
        by_difficulty[diff] = rows

    # By-dimension
    all_dims: set[str] = set()
    for _, data in analyses:
        for dim_name, _val in data.get("overall", {}).get("dimensions", {}).items():
            all_dims.add(dim_name)
    by_dimension: dict[str, list[dict]] = {}
    for dim in sorted(all_dims):
        rows = []
        for name, data in analyses:
            val = data.get("overall", {}).get("dimensions", {}).get(dim)
            rows.append({"model": name, "avg": val})
        by_dimension[dim] = rows

    return {
        "models": models,
        "overall": overall,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_dimension": by_dimension,
    }


def print_comparison_report(comparison: dict) -> None:
    """Print a side-by-side comparison table for multiple models."""
    models = comparison.get("models", [])
    if not models:
        print("No analyses to compare.")
        return

    show_delta = len(models) == 2

    print("\n" + "=" * 72)
    print("  BENCHMARK COMPARISON")
    print("=" * 72)

    # --- Overall ---
    overall = comparison.get("overall", [])
    header = f"  {'Model':<20} {'N':>5} {'Avg':>6} {'Med':>5} {'Min':>5} {'Max':>5} {'TierA%':>7} {'TestPR':>7}"
    if show_delta:
        header += f" {'Delta':>7}"
    print(f"\n{header}")
    print("  " + "-" * (58 + (8 if show_delta else 0)))
    for i, row in enumerate(overall):
        tpr = row.get("avg_test_pass_rate")
        tpr_s = f"{tpr:>6.1%}" if tpr is not None else "   n/a"
        med = row.get("median_score")
        med_s = f"{med:>5.1f}" if med is not None else "  n/a"
        line = (f"  {row['model']:<20} {row.get('total', 0):>5} {row.get('avg_score', 0):>6.1f} "
                f"{med_s} {row.get('min_score', 0):>5} {row.get('max_score', 0):>5} "
                f"{row.get('tier_a_pct', 0):>6.1%} {tpr_s}")
        if show_delta and i == 1:
            delta = (row.get("avg_score", 0) or 0) - (overall[0].get("avg_score", 0) or 0)
            line += f" {delta:>+6.1f}"
            # CI overlap check
            ci0_lo = overall[0].get("ci_95_low")
            ci0_hi = overall[0].get("ci_95_high")
            ci1_lo = row.get("ci_95_low")
            ci1_hi = row.get("ci_95_high")
            if all(v is not None for v in (ci0_lo, ci0_hi, ci1_lo, ci1_hi)):
                if ci0_lo <= ci1_hi and ci1_lo <= ci0_hi:
                    print(line)
                    print("  " + " " * 20 + "(CIs overlap — difference may not be significant)")
                    continue
        print(line)

    # --- By task family ---
    by_cat = comparison.get("by_category", {})
    if by_cat:
        print(f"\n  By Task Family:")
        for cat, rows in by_cat.items():
            header = f"    {cat}:"
            parts = []
            for i, row in enumerate(rows):
                avg = row.get("avg_score", 0)
                part = f"{row['model']}={avg:.1f}"
                if show_delta and i == 1:
                    d = (avg or 0) - (rows[0].get("avg_score", 0) or 0)
                    part += f" ({d:+.1f})"
                parts.append(part)
            print(f"  {cat:<20} {' | '.join(parts)}")

    # --- By dimension ---
    by_dim = comparison.get("by_dimension", {})
    if by_dim:
        print(f"\n  By Dimension:")
        for dim, rows in by_dim.items():
            parts = []
            for i, row in enumerate(rows):
                val = row.get("avg")
                val_s = f"{val:.1f}" if val is not None else "n/a"
                part = f"{row['model']}={val_s}"
                if show_delta and i == 1 and val is not None and rows[0].get("avg") is not None:
                    d = val - rows[0]["avg"]
                    part += f" ({d:+.1f})"
                parts.append(part)
            print(f"  {_DIMENSION_LABELS.get(dim, dim):<20} {' | '.join(parts)}")

    # --- By difficulty ---
    by_diff = comparison.get("by_difficulty", {})
    if by_diff:
        print(f"\n  By Difficulty:")
        for diff, rows in by_diff.items():
            parts = []
            for i, row in enumerate(rows):
                avg = row.get("avg_score", 0)
                part = f"{row['model']}={avg:.1f}"
                if show_delta and i == 1:
                    d = (avg or 0) - (rows[0].get("avg_score", 0) or 0)
                    part += f" ({d:+.1f})"
                parts.append(part)
            print(f"  {diff:<20} {' | '.join(parts)}")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_ACTIONS_REQUIRING_SELECTOR = frozenset({
    "wait_for",
    "click",
    "type",
    "hover",
    "focus",
    "contextmenu",
    "select_option",
    "check",
    "assert_visible",
    "assert_not_visible",
    "assert_text_contains",
    "assert_text_not_contains",
    "assert_count",
    "assert_attribute",
    "assert_style",
})
_QUERY_SELECTOR_RE = re.compile(
    r"querySelector(?:All)?\(\s*(['\"])(.*?)\1",
    re.DOTALL,
)
_CLEAR_TRUNCATION_SNIPPETS = (
    "[href^=const",
    "[href*=const",
    "[src^=const",
)


def _selector_has_empty_group(selector: str) -> bool:
    stripped = selector.strip()
    if not stripped:
        return True
    return any(not part.strip() for part in stripped.split(","))


def _is_vacuous_body_count(step: dict[str, Any]) -> bool:
    selector = str(step.get("selector", "")).strip().lower()
    if selector != "body":
        return False
    gte = step.get("gte")
    eq = step.get("eq")
    return (isinstance(gte, int) and gte > 1) or (isinstance(eq, int) and eq > 1)


def _invalid_query_selector_literals(expression: str) -> list[str]:
    bad_selectors: list[str] = []
    for match in _QUERY_SELECTOR_RE.finditer(expression):
        selector = match.group(2)
        if _selector_has_empty_group(selector):
            bad_selectors.append(selector)
    return bad_selectors


def _has_unbalanced_delimiters(expression: str) -> bool:
    pairs = {")": "(",
             "]": "[",
             "}": "{"}
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    in_regex = False
    regex_char_class = False
    prev_sig = ""

    for ch in expression:
        if in_regex:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "[":
                regex_char_class = True
                continue
            if ch == "]":
                regex_char_class = False
                continue
            if ch == "/" and not regex_char_class:
                in_regex = False
                prev_sig = "/"
                continue
            continue

        if quote is not None:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                quote = None
            continue

        if ch in {"'", '"', "`"}:
            quote = ch
            continue
        if ch == "/" and prev_sig in {"", "(", "[", "{", "=", ":", ",", ";", "!", "?", "&", "|"}:
            in_regex = True
            regex_char_class = False
            continue
        if ch in "([{":
            stack.append(ch)
            prev_sig = ch
            continue
        if ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return True
            stack.pop()
            prev_sig = ch
            continue
        if not ch.isspace():
            prev_sig = ch

    return quote is not None or in_regex or bool(stack)


def validate_items(items: list[dict]) -> list[str]:
    """Validate benchmark items without running. Returns list of errors."""
    from htmleval.phases.test_runner.schema import VALID_ACTIONS

    errors: list[str] = []

    for i, item in enumerate(items):
        item_id = item.get("id", f"index:{i}")
        prefix = f"[{item_id}]"

        # Required fields
        if "id" not in item:
            errors.append(f"{prefix}: missing 'id'")
        if "prompt" not in item:
            errors.append(f"{prefix}: missing 'prompt'")

        # test_cases validation
        test_cases = item.get("test_cases", [])
        if not isinstance(test_cases, list):
            errors.append(f"{prefix}: 'test_cases' must be a list")
            continue

        for j, tc in enumerate(test_cases):
            tc_id = tc.get("id", f"tc[{j}]")
            tc_prefix = f"{prefix}.{tc_id}"
            if "id" not in tc:
                errors.append(f"{tc_prefix}: missing 'id'")
            if "name" not in tc:
                errors.append(f"{tc_prefix}: missing 'name'")

            steps = tc.get("steps", [])
            if not steps:
                errors.append(f"{tc_prefix}: no steps defined")

            screenshot_seed_seen = False
            for k, step in enumerate(steps):
                action = step.get("action", "")
                if not action:
                    errors.append(f"{tc_prefix}.steps[{k}]: missing 'action'")
                    continue
                if action not in VALID_ACTIONS:
                    errors.append(f"{tc_prefix}.steps[{k}]: unknown action '{action}'")
                    continue

                selector = str(step.get("selector", ""))
                if action in _ACTIONS_REQUIRING_SELECTOR and _selector_has_empty_group(selector):
                    errors.append(
                        f"{tc_prefix}.steps[{k}]: action '{action}' requires a non-empty selector"
                    )

                if action == "assert_count" and _is_vacuous_body_count(step):
                    errors.append(
                        f"{tc_prefix}.steps[{k}]: vacuous assert_count on 'body' with count > 1"
                    )

                if action == "screenshot":
                    screenshot_seed_seen = True
                elif action == "assert_screenshot_changed" and not screenshot_seed_seen:
                    errors.append(
                        f"{tc_prefix}.steps[{k}]: assert_screenshot_changed requires a prior screenshot seed"
                    )

                if action in {"assert_js_value", "eval_js"}:
                    expression = str(step.get("expression", ""))
                    if not expression.strip():
                        errors.append(
                            f"{tc_prefix}.steps[{k}]: action '{action}' requires a non-empty expression"
                        )
                        continue

                    bad_selectors = _invalid_query_selector_literals(expression)
                    if bad_selectors:
                        errors.append(
                            f"{tc_prefix}.steps[{k}]: malformed querySelector literal(s): {bad_selectors[:2]}"
                        )

                    if any(snippet in expression for snippet in _CLEAR_TRUNCATION_SNIPPETS):
                        errors.append(
                            f"{tc_prefix}.steps[{k}]: expression contains a clearly truncated selector/expression fragment"
                        )
                    elif _has_unbalanced_delimiters(expression):
                        errors.append(
                            f"{tc_prefix}.steps[{k}]: expression has unbalanced delimiters"
                        )

    return errors


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _median(sorted_scores: list[int | float]) -> float:
    """Return the median of a pre-sorted list."""
    n = len(sorted_scores)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_scores[mid])
    return round((sorted_scores[mid - 1] + sorted_scores[mid]) / 2, 1)


def _percentile(sorted_scores: list[int | float], pct: int) -> float:
    """Return the pct-th percentile using nearest-rank method."""
    n = len(sorted_scores)
    if n == 0:
        return 0.0
    rank = max(0, min(n - 1, int(pct / 100 * n)))
    return float(sorted_scores[rank])


def _bootstrap_ci(scores: list[int | float], n_bootstrap: int = 1000) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for the mean.

    Uses a deterministic seed for reproducibility.
    """
    rng = random.Random(42)
    n = len(scores)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(0.025 * n_bootstrap)
    hi_idx = int(0.975 * n_bootstrap) - 1
    return round(means[lo_idx], 2), round(means[hi_idx], 2)


# ---------------------------------------------------------------------------
# Internal — Bucket accumulator
# ---------------------------------------------------------------------------

class _Bucket:
    """Accumulates scores for a group of results."""

    __slots__ = (
        "total", "completed", "failed", "tier_a", "tier_b", "tier_c",
        "_scores", "_completed_scores", "_test_pass_rates",
        "_dim_sums", "_dim_counts",
    )

    def __init__(self) -> None:
        self.total = 0
        self.completed = 0
        self.failed = 0
        self.tier_a = 0
        self.tier_b = 0
        self.tier_c = 0
        self._scores: list[int] = []
        self._completed_scores: list[int] = []
        self._test_pass_rates: list[float] = []
        self._dim_sums: dict[str, float] = defaultdict(float)
        self._dim_counts: dict[str, int] = defaultdict(int)

    def add(self, total_score: int, status: str, score_obj: dict) -> None:
        self.total += 1
        self._scores.append(total_score)

        if status in _COMPLETED_STATUSES:
            self.completed += 1
            self._completed_scores.append(total_score)
        else:
            self.failed += 1

        if total_score >= _TIER_A:
            self.tier_a += 1
        elif total_score >= _TIER_B:
            self.tier_b += 1
        else:
            self.tier_c += 1

        tpr = score_obj.get("test_pass_rate")
        if tpr is not None:
            self._test_pass_rates.append(tpr)

        # Per-dimension scores
        for dim in _DIMENSIONS:
            val = score_obj.get(dim)
            if isinstance(val, (int, float)):
                self._dim_sums[dim] += val
                self._dim_counts[dim] += 1

    def finalize(self) -> dict:
        n = self.total
        if n == 0:
            return {"total": 0, "avg_score": 0, "completion_rate": 0,
                    "tier_a_pct": 0, "tier_b_pct": 0, "tier_c_pct": 0}

        sorted_scores = sorted(self._scores)
        result = {
            "total": n,
            "avg_score": round(sum(sorted_scores) / n, 1),
            "avg_score_all": round(sum(sorted_scores) / n, 1),
            "min_score": sorted_scores[0],
            "max_score": sorted_scores[-1],
            "median_score": _median(sorted_scores),
            "p25_score": _percentile(sorted_scores, 25),
            "p75_score": _percentile(sorted_scores, 75),
            "completion_rate": round(self.completed / n, 3),
            "tier_a_pct": round(self.tier_a / n, 3),
            "tier_b_pct": round(self.tier_b / n, 3),
            "tier_c_pct": round(self.tier_c / n, 3),
        }

        if self._completed_scores:
            completed_scores = sorted(self._completed_scores)
            result["completed_total"] = len(completed_scores)
            result["avg_score_completed"] = round(sum(completed_scores) / len(completed_scores), 1)
            result["median_score_completed"] = _median(completed_scores)
        else:
            result["completed_total"] = 0
            result["avg_score_completed"] = 0
            result["median_score_completed"] = 0

        if n >= 2:
            mean = sum(sorted_scores) / n
            variance = sum((s - mean) ** 2 for s in sorted_scores) / (n - 1)
            result["std_dev"] = round(variance ** 0.5, 2)

        if n >= 10:
            lo, hi = _bootstrap_ci(self._scores, 1000)
            result["ci_95_low"] = lo
            result["ci_95_high"] = hi

        if self._test_pass_rates:
            result["avg_test_pass_rate"] = round(
                sum(self._test_pass_rates) / len(self._test_pass_rates), 3
            )

        # Per-dimension averages
        dims = {}
        for dim in _DIMENSIONS:
            c = self._dim_counts.get(dim, 0)
            if c > 0:
                dims[dim] = round(self._dim_sums[dim] / c, 1)
        if dims:
            result["dimensions"] = dims

        return result


def _load_results(path: str) -> list[dict]:
    """Load results from JSONL file, keeping the latest record per item/trial."""
    dedup: dict[tuple[Any, str], dict] = {}
    p = Path(path)
    if not p.exists():
        logger.warning(f"Results file not found: {path}")
        return []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item_id = str(rec.get("line_number", rec.get("data", {}).get("line_number", "")))
                trial = rec.get("trial", 0)
                key = (trial, item_id)
                dedup[key] = rec
    return list(dedup.values())


def _empty_analysis() -> dict:
    return {
        "overall": {"total": 0, "avg_score": 0, "pass_rate": 0,
                     "tier_a_pct": 0, "tier_b_pct": 0, "tier_c_pct": 0},
        "by_category": {},
        "by_source_category": {},
        "by_difficulty": {},
        "by_subtype": {},
        "by_language": {},
        "failed_tests": [],
    }
