"""
Pass@k metrics and pass/fail judgment for benchmark evaluation.

Standard HumanEval-style pass@k:
    pass_at_k(n, c, k) = 1 - C(n-c, k) / C(n, k)

Usage:
    from htmleval.benchmark.metrics import pass_at_k, is_pass, aggregate_trials
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from htmleval.benchmark.taxonomy import normalize_task_family


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (HumanEval formula).

    Args:
        n: total number of trials
        c: number of passing trials
        k: k value (e.g. 1 or 3)

    Returns:
        Probability that at least one of k samples passes.
    """
    if n < k:
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def is_pass(result: dict, tc_threshold: float = 0.8) -> bool:
    """Determine if a single evaluation result is a pass.

    A result passes if:
    - eval_status is completed, including deterministic fallback completion
    - test_pass_rate >= threshold (when test_pass_rate is available)

    Args:
        result: single evaluation result dict from results.jsonl
        tc_threshold: minimum test_pass_rate to count as pass
    """
    if result.get("eval_status", "") not in {"completed", "completed_with_fallback"}:
        return False
    score_obj = result.get("score", {})
    tpr = score_obj.get("test_pass_rate")
    if tpr is not None:
        return tpr >= tc_threshold
    # No test_pass_rate available — fall back to score threshold
    return score_obj.get("total", 0) >= 40


def aggregate_trials(
    results: list[dict],
    num_trials: int,
    tc_threshold: float = 0.8,
) -> dict:
    """Aggregate multi-trial results into pass@k metrics.

    Groups results by item identifier (line_number or item_id), counts
    pass/fail per item, and computes pass@1 and pass@3.

    Args:
        results: list of result dicts (all trials mixed)
        num_trials: expected number of trials per item
        tc_threshold: pass threshold for is_pass()

    Returns:
        Dict with keys: pass_at_1, pass_at_3, num_items, num_trials,
        by_category, by_difficulty, per_item.
    """
    # Group by item
    by_item: dict[str, list[dict]] = defaultdict(list)
    for rec in results:
        item_id = str(
            rec.get("line_number",
                     rec.get("data", {}).get("line_number",
                             rec.get("item_id", "")))
        )
        by_item[item_id].append(rec)

    # Per-item pass@k
    per_item: dict[str, dict[str, Any]] = {}
    cat_groups: dict[str, list[float]] = defaultdict(list)  # category -> list of pass@1
    diff_groups: dict[str, list[float]] = defaultdict(list)  # difficulty -> list of pass@1

    for item_id, recs in by_item.items():
        n = len(recs)
        c = sum(1 for r in recs if is_pass(r, tc_threshold))
        p1 = pass_at_k(n, c, 1)
        p3 = pass_at_k(n, c, 3)

        # Get metadata from first result
        first = recs[0]
        category = normalize_task_family(first.get("category", "unknown"))
        difficulty = first.get("difficulty", "unknown")

        # Median score across trials
        scores = [r.get("score", {}).get("total", 0) for r in recs]
        scores.sort()
        median_score = scores[len(scores) // 2] if scores else 0

        per_item[item_id] = {
            "n": n, "c": c,
            "pass_at_1": round(p1, 4),
            "pass_at_3": round(p3, 4),
            "median_score": median_score,
            "category": category,
            "difficulty": difficulty,
        }

        cat_groups[category].append(p1)
        diff_groups[difficulty].append(p1)

    # Global averages
    all_p1 = [v["pass_at_1"] for v in per_item.values()]
    all_p3 = [v["pass_at_3"] for v in per_item.values()]

    global_pass_1 = round(sum(all_p1) / len(all_p1), 4) if all_p1 else 0.0
    global_pass_3 = round(sum(all_p3) / len(all_p3), 4) if all_p3 else 0.0

    # By-category and by-difficulty averages
    by_category = {
        cat: round(sum(vals) / len(vals), 4)
        for cat, vals in sorted(cat_groups.items())
    }
    by_difficulty = {
        diff: round(sum(vals) / len(vals), 4)
        for diff, vals in diff_groups.items()
    }

    return {
        "pass_at_1": global_pass_1,
        "pass_at_3": global_pass_3,
        "num_items": len(per_item),
        "num_trials": num_trials,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "per_item": per_item,
    }
