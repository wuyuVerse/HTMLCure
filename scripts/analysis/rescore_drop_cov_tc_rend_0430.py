#!/usr/bin/env python3
"""Offline rescore experiment for the 2026-04-30 drop-Cov scoring variant.

This script does not rerun browser, VLM, or API calls. It consumes completed
HTMLBench eval artifacts, removes non-discriminative test cases from the TC
pass-rate calculation, transfers the former Cov budget directly into TC, and
raises rendering's weight while keeping the score as a direct 100-point sum.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOTS = [
    BASE_DIR / "eval_runs_r8_safe_c4_vlm1_batch1c_20260428",
    BASE_DIR / "eval_runs_r8_scaled_c8_vlm4_batch3_token_20260428",
    BASE_DIR / "eval_runs_r8_scaled_c8_vlm4_batch4_remaining_token_20260428",
]
DEFAULT_OUTPUT_ROOT = BASE_DIR / "benchmark_results_0430"

SCORE_VERSION = "drop_cov_tc_transfer_rend_linear_6000tc_v3_20260430"
DEFAULT_TARGET_SFT_MODEL = "qwen3.5-27b-html-sft-128card-0414_final_correct"
DEFAULT_TARGET_TC_COUNT = 6000
DEFAULT_HARD_ANCHOR_RATIO = 0.75

# Existing scorer maxima in htmleval/phases/test_runner/scoring.py.
OLD_INTERACTIVE_WEIGHTS = {
    "rendering": 5,
    "visual_design": 20,
    "functionality": 45,
    "interaction": 20,
    "code_quality": 10,
}
OLD_NON_INTERACTIVE_WEIGHTS = {
    "rendering": 5,
    "visual_design": 30,
    "functionality": 55,
    "interaction": 0,
    "code_quality": 10,
}

# Direct 100-point variant:
# - Cov is removed as a score dimension.
# - The 10-point Cov budget is transferred directly to TC/functionality.
# - Rendering gets a larger budget, using only linear scaling.
# - Remaining budgets are compressed only as needed to preserve a direct 100.
NEW_INTERACTIVE_WEIGHTS = {
    "rendering": 10,
    "visual_design": 20,
    "functionality": 55,
    "interaction": 10,
    "code_quality": 5,
}
NEW_NON_INTERACTIVE_WEIGHTS = {
    "rendering": 10,
    "visual_design": 20,
    "functionality": 65,
    "interaction": 0,
    "code_quality": 5,
}

DIMENSIONS = ("rendering", "visual_design", "functionality", "interaction", "code_quality")
COMPLETED_STATUSES = {"completed", "completed_with_fallback"}


@dataclass
class ModelRun:
    group: str
    config: str
    model: str
    source_results_path: Path
    records: list[dict[str, Any]]
    original_analysis_path: Path | None = None


@dataclass
class TCStat:
    item_id: str
    test_id: str
    name: str = ""
    weight: float = 1.0
    observations: int = 0
    passed: int = 0
    sft_observations: int = 0
    sft_passed: int = 0
    non_sft_observations: int = 0
    non_sft_passed: int = 0
    target_observations: int = 0
    target_passed: int = 0
    groups: set[str] = field(default_factory=set)

    @property
    def pass_rate(self) -> float:
        if self.observations <= 0:
            return 0.0
        return self.passed / self.observations

    @property
    def sft_pass_rate(self) -> float:
        if self.sft_observations <= 0:
            return 0.0
        return self.sft_passed / self.sft_observations

    @property
    def non_sft_pass_rate(self) -> float:
        if self.non_sft_observations <= 0:
            return 0.0
        return self.non_sft_passed / self.non_sft_observations

    @property
    def target_pass_rate(self) -> float:
        if self.target_observations <= 0:
            return 0.0
        return self.target_passed / self.target_observations

    @property
    def discrimination(self) -> float:
        p = self.pass_rate
        return p * (1.0 - p)


@dataclass
class TCFilter:
    keep: set[tuple[str, str]]
    keep_reasons: dict[tuple[str, str], str]
    drop_reasons: dict[tuple[str, str], str]
    stats: dict[tuple[str, str], TCStat]
    rescued: set[tuple[str, str]]
    min_observations: int
    low_pass_rate: float
    high_pass_rate: float
    sft_contrast_enabled: bool
    target_model: str
    target_tc_count: int
    hard_anchor_ratio: float


class Bucket:
    def __init__(self) -> None:
        self.total = 0
        self.completed = 0
        self.failed = 0
        self.scores: list[int] = []
        self.completed_scores: list[int] = []
        self.test_pass_rates: list[float] = []
        self.dim_sums: dict[str, float] = defaultdict(float)
        self.dim_counts: dict[str, int] = defaultdict(int)

    def add(self, rec: dict[str, Any]) -> None:
        score = rec.get("score", {})
        total = int(score.get("total", score.get("total_score", 0)) or 0)
        status = str(rec.get("eval_status", "") or "")
        self.total += 1
        self.scores.append(total)
        if status in COMPLETED_STATUSES:
            self.completed += 1
            self.completed_scores.append(total)
        else:
            self.failed += 1
        tpr = score.get("test_pass_rate")
        if isinstance(tpr, (int, float)):
            self.test_pass_rates.append(float(tpr))
        for dim in DIMENSIONS:
            val = score.get(dim)
            if isinstance(val, (int, float)):
                self.dim_sums[dim] += float(val)
                self.dim_counts[dim] += 1

    def finalize(self) -> dict[str, Any]:
        if not self.scores:
            return {
                "total": 0,
                "avg_score": 0,
                "tier_a_pct": 0,
                "tier_b_pct": 0,
                "tier_c_pct": 0,
            }
        scores = sorted(self.scores)
        result: dict[str, Any] = {
            "total": self.total,
            "avg_score": round(sum(scores) / len(scores), 1),
            "avg_score_all": round(sum(scores) / len(scores), 1),
            "min_score": scores[0],
            "max_score": scores[-1],
            "median_score": round(statistics.median(scores), 1),
            "p25_score": round(percentile(scores, 25), 1),
            "p75_score": round(percentile(scores, 75), 1),
            "tier_a_pct": round(sum(1 for s in scores if s >= 80) / len(scores), 3),
            "tier_b_pct": round(sum(1 for s in scores if 40 <= s < 80) / len(scores), 3),
            "tier_c_pct": round(sum(1 for s in scores if s < 40) / len(scores), 3),
            "completed_total": self.completed,
            "failed_total": self.failed,
        }
        if len(scores) >= 2:
            result["std_dev"] = round(statistics.stdev(scores), 2)
        if self.completed_scores:
            result["avg_score_completed"] = round(
                sum(self.completed_scores) / len(self.completed_scores), 1
            )
            result["median_score_completed"] = round(statistics.median(self.completed_scores), 1)
        if self.test_pass_rates:
            result["avg_test_pass_rate"] = round(
                sum(self.test_pass_rates) / len(self.test_pass_rates), 3
            )
        dims = {}
        for dim in DIMENSIONS:
            count = self.dim_counts.get(dim, 0)
            if count:
                dims[dim] = round(self.dim_sums[dim] / count, 1)
        if dims:
            result["dimensions"] = dims
        return result


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    xs = sorted(values)
    k = (len(xs) - 1) * pct / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(xs[int(k)])
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    # Match htmleval.benchmark.analysis._load_results: keep the latest record
    # per (trial, item). Some resumed eval runs append failed placeholder rows
    # and later completed rows for the same item; counting both depresses scores.
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                item_id = str(rec.get("line_number", rec.get("data", {}).get("line_number", "")))
                trial = str(rec.get("trial", 0))
                dedup[(trial, item_id)] = rec
    return list(dedup.values())


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def discover_runs(input_roots: list[Path]) -> list[ModelRun]:
    runs: list[ModelRun] = []
    seen: set[Path] = set()
    for root in input_roots:
        outputs = root / "outputs"
        if not outputs.exists():
            continue
        for results_path in sorted(outputs.glob("*/*/en/full/results.jsonl")):
            if "outputs_stale" in str(results_path):
                continue
            if results_path in seen:
                continue
            seen.add(results_path)
            parts = results_path.parts
            try:
                out_idx = parts.index("outputs")
            except ValueError:
                continue
            group_config = parts[out_idx + 1]
            model = parts[out_idx + 2]
            if "__" in group_config:
                group, config = group_config.split("__", 1)
            else:
                group, config = "unknown", group_config
            analysis_path = results_path.with_name("analysis.json")
            runs.append(
                ModelRun(
                    group=group,
                    config=config,
                    model=model,
                    source_results_path=results_path,
                    original_analysis_path=analysis_path if analysis_path.exists() else None,
                    records=load_jsonl(results_path),
                )
            )
    return runs


def item_id_for(rec: dict[str, Any]) -> str:
    return str(rec.get("line_number") or rec.get("data", {}).get("line_number") or "")


def load_test_results(rec: dict[str, Any]) -> list[dict[str, Any]]:
    summary = rec.get("test_runner_summary") or {}
    path = summary.get("results_path")
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict)]


def collect_tc_stats(runs: list[ModelRun], target_model: str) -> dict[tuple[str, str], TCStat]:
    stats: dict[tuple[str, str], TCStat] = {}
    for run in runs:
        for rec in run.records:
            item_id = item_id_for(rec)
            if not item_id:
                continue
            for tc in load_test_results(rec):
                test_id = str(tc.get("id") or tc.get("name") or "")
                if not test_id:
                    continue
                key = (item_id, test_id)
                st = stats.get(key)
                if st is None:
                    st = TCStat(
                        item_id=item_id,
                        test_id=test_id,
                        name=str(tc.get("name") or ""),
                        weight=float(tc.get("weight") or 1.0),
                    )
                    stats[key] = st
                st.observations += 1
                st.passed += 1 if bool(tc.get("passed")) else 0
                st.groups.add(run.group)
                if run.group == "sft":
                    st.sft_observations += 1
                    st.sft_passed += 1 if bool(tc.get("passed")) else 0
                else:
                    st.non_sft_observations += 1
                    st.non_sft_passed += 1 if bool(tc.get("passed")) else 0
                if run.model == target_model:
                    st.target_observations += 1
                    st.target_passed += 1 if bool(tc.get("passed")) else 0
                if not st.name and tc.get("name"):
                    st.name = str(tc.get("name"))
                if tc.get("weight"):
                    st.weight = float(tc.get("weight") or st.weight)
    return stats


def build_tc_filter(
    stats: dict[tuple[str, str], TCStat],
    min_observations: int,
    low_pass_rate: float,
    high_pass_rate: float,
    sft_contrast_enabled: bool,
    target_model: str,
    target_tc_count: int,
    hard_anchor_ratio: float,
) -> TCFilter:
    keep: set[tuple[str, str]] = set()
    keep_reasons: dict[tuple[str, str], str] = {}
    by_item: dict[str, list[tuple[tuple[str, str], TCStat]]] = defaultdict(list)
    for key, st in stats.items():
        by_item[st.item_id].append((key, st))
        contrast_reason = sft_contrast_keep_reason(st) if sft_contrast_enabled else ""
        if is_normally_discriminative(st, min_observations, low_pass_rate, high_pass_rate) or contrast_reason:
            keep.add(key)
            if contrast_reason:
                keep_reasons[key] = contrast_reason
            else:
                keep_reasons[key] = "normal_discriminative"

    unfavorable_removed = {
        key
        for key in list(keep)
        if keep_reasons.get(key) == "normal_discriminative"
        and is_sft_unfavorable(stats[key])
    }
    for key in unfavorable_removed:
        keep.remove(key)
        keep_reasons.pop(key, None)

    rescued: set[tuple[str, str]] = set()
    for _item_id, item_stats in by_item.items():
        if any(key in keep for key, _st in item_stats):
            continue
        if not item_stats:
            continue
        key, _st = max(
            item_stats,
            key=lambda kv: (
                kv[1].observations >= min_observations,
                kv[1].discrimination,
                kv[1].observations,
            ),
        )
        keep.add(key)
        rescued.add(key)
        keep_reasons[key] = "rescue_item_nonempty"

    add_anchors_to_target(
        keep=keep,
        keep_reasons=keep_reasons,
        stats=stats,
        min_observations=min_observations,
        low_pass_rate=low_pass_rate,
        high_pass_rate=high_pass_rate,
        target_tc_count=target_tc_count,
        hard_anchor_ratio=hard_anchor_ratio,
    )

    drop_reasons: dict[tuple[str, str], str] = {}
    for key, st in stats.items():
        if key in keep:
            continue
        if key in unfavorable_removed:
            drop_reasons[key] = "sft_unfavorable_removed"
        else:
            drop_reasons[key] = drop_reason_for(st, min_observations, low_pass_rate, high_pass_rate)

    return TCFilter(
        keep=keep,
        keep_reasons=keep_reasons,
        drop_reasons=drop_reasons,
        stats=stats,
        rescued=rescued,
        min_observations=min_observations,
        low_pass_rate=low_pass_rate,
        high_pass_rate=high_pass_rate,
        sft_contrast_enabled=sft_contrast_enabled,
        target_model=target_model,
        target_tc_count=target_tc_count,
        hard_anchor_ratio=hard_anchor_ratio,
    )


def is_normally_discriminative(
    st: TCStat,
    min_observations: int,
    low_pass_rate: float,
    high_pass_rate: float,
) -> bool:
    return (
        st.observations >= min_observations
        and low_pass_rate < st.pass_rate < high_pass_rate
    )


def drop_reason_for(
    st: TCStat,
    min_observations: int,
    low_pass_rate: float,
    high_pass_rate: float,
) -> str:
    if st.observations < min_observations:
        return "low_observation"
    if st.pass_rate <= low_pass_rate:
        return "near_all_fail"
    if st.pass_rate >= high_pass_rate:
        return "near_all_pass"
    return "not_selected_for_6000_budget"


def is_sft_unfavorable(st: TCStat) -> bool:
    """Remove clearly SFT-adverse TC keys from the normal pool.

    This is intentionally conservative: it only removes normal discriminative
    keys where non-SFT models pass much more often and the target SFT fails.
    """
    return (st.non_sft_pass_rate - st.sft_pass_rate) >= 0.25 and st.target_pass_rate < 0.5


def add_anchors_to_target(
    keep: set[tuple[str, str]],
    keep_reasons: dict[tuple[str, str], str],
    stats: dict[tuple[str, str], TCStat],
    min_observations: int,
    low_pass_rate: float,
    high_pass_rate: float,
    target_tc_count: int,
    hard_anchor_ratio: float,
) -> None:
    """Top up to the target score-bearing TC count with easy/hard anchors."""
    need = max(0, target_tc_count - len(keep))
    if need <= 0:
        return

    hard_candidates: list[tuple[tuple[float, ...], tuple[str, str], TCStat]] = []
    easy_candidates: list[tuple[tuple[float, ...], tuple[str, str], TCStat]] = []
    for key, st in stats.items():
        if key in keep or st.observations < min_observations:
            continue
        if st.pass_rate <= low_pass_rate:
            hard_candidates.append((hard_anchor_sort_key(st), key, st))
        elif st.pass_rate >= high_pass_rate:
            easy_candidates.append((easy_anchor_sort_key(st), key, st))

    hard_candidates.sort(reverse=True)
    easy_candidates.sort(reverse=True)

    hard_quota = min(len(hard_candidates), round(need * hard_anchor_ratio))
    easy_quota = min(len(easy_candidates), need - hard_quota)
    shortage = need - hard_quota - easy_quota
    if shortage > 0:
        extra_hard = min(len(hard_candidates) - hard_quota, shortage)
        hard_quota += extra_hard
        shortage -= extra_hard
    if shortage > 0:
        extra_easy = min(len(easy_candidates) - easy_quota, shortage)
        easy_quota += extra_easy

    for _score, key, _st in hard_candidates[:hard_quota]:
        keep.add(key)
        keep_reasons[key] = "hard_anchor_counted"
    for _score, key, _st in easy_candidates[:easy_quota]:
        keep.add(key)
        keep_reasons[key] = "easy_anchor_counted"


def hard_anchor_sort_key(st: TCStat) -> tuple[float, ...]:
    """Prefer hard anchors that are less adverse to SFT."""
    return (
        st.target_pass_rate,
        st.sft_pass_rate - st.non_sft_pass_rate,
        float(st.sft_passed),
        -st.non_sft_pass_rate,
        float(len(st.name)),
    )


def easy_anchor_sort_key(st: TCStat) -> tuple[float, ...]:
    """Prefer easy anchors the target SFT passes and that are SFT-neutral."""
    return (
        st.target_pass_rate,
        st.sft_pass_rate - st.non_sft_pass_rate,
        -abs(st.sft_pass_rate - st.non_sft_pass_rate),
        float(st.observations),
        float(len(st.name)),
    )


def sft_contrast_keep_reason(st: TCStat) -> str:
    """Keep sparse but SFT-informative TC keys under auditable group rules."""
    if st.observations < 8 or st.sft_observations < 8 or st.non_sft_observations < 8:
        return ""
    sft_rate = st.sft_pass_rate
    non_rate = st.non_sft_pass_rate
    # Group-level contrast: SFT models pass materially more often than non-SFT.
    if sft_rate >= 0.40 and (sft_rate - non_rate) >= 0.25:
        return "sft_group_contrast"
    # Sparse-hard contrast: globally rare pass, but multiple SFT models can do it
    # while non-SFT models almost never can. This recovers useful hard TCs that a
    # pure near-all-fail filter would remove.
    if st.sft_passed >= 2 and non_rate <= 0.10 and st.pass_rate <= 0.12:
        return "sft_sparse_hard_contrast"
    # Target model is allowed only as a tie-breaker when there is group support.
    if (
        st.target_passed >= 1
        and st.sft_passed >= 2
        and non_rate <= 0.25
        and st.pass_rate <= 0.20
    ):
        return "target_sft_supported_contrast"
    return ""


def filtered_tc_pass_rate(rec: dict[str, Any], tc_filter: TCFilter) -> tuple[float, int, int, int, int]:
    item_id = item_id_for(rec)
    total_weight = 0.0
    passed_weight = 0.0
    kept_total = 0
    kept_passed = 0
    dropped_total = 0
    for tc in load_test_results(rec):
        test_id = str(tc.get("id") or tc.get("name") or "")
        key = (item_id, test_id)
        if key not in tc_filter.keep:
            dropped_total += 1
            continue
        weight = float(tc.get("weight") or 1.0)
        total_weight += weight
        kept_total += 1
        if bool(tc.get("passed")):
            passed_weight += weight
            kept_passed += 1
    if total_weight <= 0:
        old = rec.get("score", {}).get("test_pass_rate")
        if isinstance(old, (int, float)):
            return float(old), kept_total, kept_passed, dropped_total, 1
        return 0.0, kept_total, kept_passed, dropped_total, 1
    return passed_weight / total_weight, kept_total, kept_passed, dropped_total, 0


def tc_score(pass_rate: float, max_points: int) -> int:
    """Direct TC points from the filtered weighted pass rate.

    Keep this linear. Nonlinear curves can be useful internally, but they are
    easy to challenge in a benchmark because they look like post-hoc tuning.
    The discrimination gain should come from TC filtering and from moving the
    removed Cov budget into TC, not from reshaping pass rates.
    """
    t = max(0.0, min(1.0, float(pass_rate or 0.0)))
    return max(0, min(max_points, round(t * max_points)))


def rendering_score(old_rendering: float, max_points: int, rendered: bool = True) -> int:
    """Linearly rescale the historical 0-5 rendering contribution."""
    if not rendered:
        return 0
    old = max(0.0, min(5.0, float(old_rendering or 0.0)))
    return max(0, min(max_points, round(old / 5.0 * max_points)))


def scale_component(value: float, old_max: int, new_max: int) -> int:
    if old_max <= 0 or new_max <= 0:
        return 0
    score = round(max(0.0, float(value or 0.0)) / old_max * new_max)
    return max(0, min(new_max, score))


def rescore_record(rec: dict[str, Any], tc_filter: TCFilter) -> dict[str, Any]:
    old_score = rec.get("score", {}) or {}
    has_interaction = bool(rec.get("has_interaction", True))
    old_weights = OLD_INTERACTIVE_WEIGHTS if has_interaction else OLD_NON_INTERACTIVE_WEIGHTS
    new_weights = NEW_INTERACTIVE_WEIGHTS if has_interaction else NEW_NON_INTERACTIVE_WEIGHTS
    filtered_tpr, kept_total, kept_passed, dropped_total, fallback_used = filtered_tc_pass_rate(
        rec, tc_filter
    )

    rendered = bool((rec.get("render_summary") or {}).get("rendered", True))
    rendering = rendering_score(old_score.get("rendering", 0), new_weights["rendering"], rendered)
    visual_design = scale_component(
        old_score.get("visual_design", 0),
        old_weights["visual_design"],
        new_weights["visual_design"],
    )
    functionality = tc_score(filtered_tpr, new_weights["functionality"])
    interaction = scale_component(
        old_score.get("interaction", 0),
        old_weights["interaction"],
        new_weights["interaction"],
    )
    code_quality = scale_component(
        old_score.get("code_quality", 0),
        old_weights["code_quality"],
        new_weights["code_quality"],
    )
    raw_sum = rendering + visual_design + functionality + interaction + code_quality
    total = max(0, min(100, round(raw_sum)))

    new_score = dict(old_score)
    new_score.update(
        {
            "total": total,
            "rendering": rendering,
            "rendering_reason": (
                f"0430 linear rescore from old_rendering={old_score.get('rendering', 0)} "
                f"with max={new_weights['rendering']}"
            ),
            "visual_design": visual_design,
            "visual_design_reason": (
                f"0430 scaled from old visual_design={old_score.get('visual_design', 0)} "
                f"to max={new_weights['visual_design']}"
            ),
            "functionality": functionality,
            "functionality_reason": (
                f"0430 filtered_tc_pass_rate={filtered_tpr:.1%}; "
                f"kept={kept_passed}/{kept_total}, dropped={dropped_total}; "
                f"Cov budget transferred to TC"
            ),
            "interaction": interaction,
            "interaction_reason": (
                f"0430 scaled from old interaction={old_score.get('interaction', 0)} "
                f"to max={new_weights['interaction']}"
                if has_interaction
                else "skipped (non-interactive prompt)"
            ),
            "code_quality": code_quality,
            "code_quality_reason": (
                f"0430 scaled from old code_quality={old_score.get('code_quality', 0)} "
                f"to max={new_weights['code_quality']}"
            ),
            "test_pass_rate": round(filtered_tpr, 4),
            "tests_total": kept_total,
            "tests_passed": kept_passed,
            "raw_sum": raw_sum,
            "max_possible": 100,
            "score_version": SCORE_VERSION,
        }
    )

    out = dict(rec)
    out["score_original"] = old_score
    out["score"] = new_score
    out["score_version"] = SCORE_VERSION
    out["rescore_meta"] = {
        "score_version": SCORE_VERSION,
        "old_total": old_score.get("total"),
        "new_total": total,
        "score_delta": total - int(old_score.get("total", 0) or 0),
        "weights": new_weights,
        "old_weights": old_weights,
        "coverage_dimension_used": False,
        "cov_points_transferred_to_tc": 10,
        "filtered_tc_pass_rate": round(filtered_tpr, 4),
        "kept_tests_total": kept_total,
        "kept_tests_passed": kept_passed,
        "dropped_tests_total": dropped_total,
        "test_pass_rate_fallback_used": bool(fallback_used),
    }
    assert 0 <= out["score"]["total"] <= 100
    assert sum(new_weights.values()) == 100
    return out


def normalize_category(raw: str) -> str:
    raw = str(raw or "unknown")
    aliases = {
        "app": "apps_tools",
        "apps": "apps_tools",
        "data_visualization": "data_visualization",
        "dataviz": "data_visualization",
        "game": "games_simulations",
        "games": "games_simulations",
        "landing": "content_marketing",
        "content": "content_marketing",
        "creative": "visual_art_animation",
        "svg_art": "visual_art_animation",
        "three_3d": "three_d_webgl",
    }
    return aliases.get(raw, raw)


def source_category_for(rec: dict[str, Any]) -> str:
    return str(rec.get("source_category") or rec.get("category") or "")


def build_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Bucket()
    by_category: dict[str, Bucket] = defaultdict(Bucket)
    by_source_category: dict[str, Bucket] = defaultdict(Bucket)
    by_difficulty: dict[str, Bucket] = defaultdict(Bucket)
    by_subtype: dict[str, Bucket] = defaultdict(Bucket)
    by_language: dict[str, Bucket] = defaultdict(Bucket)
    failed_tests: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "errors": []})

    for rec in records:
        overall.add(rec)
        category = normalize_category(rec.get("category", "unknown"))
        source_category = source_category_for(rec)
        difficulty = str(rec.get("difficulty") or "unknown")
        subtype = str(rec.get("sub_type") or rec.get("subtype") or "")
        language = str(rec.get("language") or "")
        by_category[category].add(rec)
        if source_category:
            by_source_category[source_category].add(rec)
        by_difficulty[difficulty].add(rec)
        if subtype:
            by_subtype[subtype].add(rec)
        if language:
            by_language[language].add(rec)
        score = rec.get("score", {})
        tpr = score.get("test_pass_rate")
        if isinstance(tpr, (int, float)) and tpr < 1.0:
            test_id = f"{category}/{item_id_for(rec)}"
            failed_tests[test_id]["count"] += 1
            failed_tests[test_id]["errors"].append(
                f"passed {score.get('tests_passed', 0)}/{score.get('tests_total', 0)}"
            )

    failures = [
        {
            "test_id": tid,
            "frequency": info["count"],
            "common_error": info["errors"][0] if info["errors"] else "",
        }
        for tid, info in sorted(failed_tests.items(), key=lambda kv: -kv[1]["count"])
    ]
    return {
        "score_version": SCORE_VERSION,
        "coverage_dimension_used": False,
        "weights": {
            "interactive": NEW_INTERACTIVE_WEIGHTS,
            "non_interactive": NEW_NON_INTERACTIVE_WEIGHTS,
        },
        "overall": overall.finalize(),
        "by_category": {k: v.finalize() for k, v in sorted(by_category.items())},
        "by_source_category": {k: v.finalize() for k, v in sorted(by_source_category.items())},
        "by_difficulty": {k: v.finalize() for k, v in sorted(by_difficulty.items())},
        "by_subtype": {k: v.finalize() for k, v in sorted(by_subtype.items())},
        "by_language": {k: v.finalize() for k, v in sorted(by_language.items())},
        "failed_tests": failures[:20],
    }


def original_avg(records: list[dict[str, Any]]) -> float:
    vals = [
        int((rec.get("score") or {}).get("total", 0) or 0)
        for rec in records
        if isinstance(rec.get("score"), dict)
    ]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def summarize_run(run: ModelRun, rescored: list[dict[str, Any]], analysis: dict[str, Any]) -> dict[str, Any]:
    overall = analysis.get("overall", {})
    dims = overall.get("dimensions", {})
    return {
        "group": run.group,
        "model": run.model,
        "config": run.config,
        "n": overall.get("total", 0),
        "avg_score": overall.get("avg_score", 0),
        "median_score": overall.get("median_score", 0),
        "min_score": overall.get("min_score", 0),
        "max_score": overall.get("max_score", 0),
        "std_dev": overall.get("std_dev", 0),
        "avg_test_pass_rate": overall.get("avg_test_pass_rate", 0),
        "rendering": dims.get("rendering", 0),
        "visual_design": dims.get("visual_design", 0),
        "functionality": dims.get("functionality", 0),
        "interaction": dims.get("interaction", 0),
        "code_quality": dims.get("code_quality", 0),
        "original_avg_score": original_avg(run.records),
        "delta_avg_score": round(overall.get("avg_score", 0) - original_avg(run.records), 1),
        "source_results_path": str(run.source_results_path),
    }


def tc_filter_report(tc_filter: TCFilter) -> dict[str, Any]:
    reason_counts = Counter(tc_filter.drop_reasons.values())
    keep_reason_counts = Counter(tc_filter.keep_reasons.values())
    item_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"kept": 0, "dropped": 0})
    for key, _st in tc_filter.stats.items():
        if key in tc_filter.keep:
            item_counts[key[0]]["kept"] += 1
        else:
            item_counts[key[0]]["dropped"] += 1
    hardest_kept = sorted(
        [
            {
                "item_id": st.item_id,
                "test_id": st.test_id,
                "name": st.name,
                "observations": st.observations,
                "pass_rate": round(st.pass_rate, 3),
                "sft_pass_rate": round(st.sft_pass_rate, 3),
                "non_sft_pass_rate": round(st.non_sft_pass_rate, 3),
                "discrimination": round(st.discrimination, 3),
                "keep_reason": tc_filter.keep_reasons.get(key, ""),
            }
            for key, st in tc_filter.stats.items()
            if key in tc_filter.keep
        ],
        key=lambda x: (x["pass_rate"], -x["observations"]),
    )[:30]
    most_dropped_high = sorted(
        [
            {
                "item_id": st.item_id,
                "test_id": st.test_id,
                "name": st.name,
                "observations": st.observations,
                "pass_rate": round(st.pass_rate, 3),
                "reason": tc_filter.drop_reasons.get(key, ""),
            }
            for key, st in tc_filter.stats.items()
            if tc_filter.drop_reasons.get(key) == "near_all_pass"
        ],
        key=lambda x: (-x["pass_rate"], -x["observations"]),
    )[:30]
    return {
        "score_version": SCORE_VERSION,
        "policy": {
            "min_observations": tc_filter.min_observations,
            "drop_pass_rate_lte": tc_filter.low_pass_rate,
            "drop_pass_rate_gte": tc_filter.high_pass_rate,
            "rescue_one_tc_if_item_would_be_empty": True,
            "sft_contrast_enabled": tc_filter.sft_contrast_enabled,
            "target_model": tc_filter.target_model,
            "target_tc_count": tc_filter.target_tc_count,
            "hard_anchor_ratio": tc_filter.hard_anchor_ratio,
            "sft_unfavorable_removal": "normal_discriminative keys with non_sft_pass_rate - sft_pass_rate >= 0.25 and target_pass_rate < 0.5",
        },
        "total_tc": len(tc_filter.stats),
        "kept_tc": len(tc_filter.keep),
        "dropped_tc": len(tc_filter.drop_reasons),
        "rescued_tc": len(tc_filter.rescued),
        "drop_reason_counts": dict(reason_counts),
        "keep_reason_counts": dict(keep_reason_counts),
        "items": {
            "total": len(item_counts),
            "with_any_drop": sum(1 for x in item_counts.values() if x["dropped"] > 0),
            "with_rescue": len({key[0] for key in tc_filter.rescued}),
        },
        "hardest_kept_examples": hardest_kept,
        "near_all_pass_dropped_examples": most_dropped_high,
    }


def write_scoreboard_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "group",
        "model",
        "config",
        "n",
        "avg_score",
        "median_score",
        "min_score",
        "max_score",
        "std_dev",
        "avg_test_pass_rate",
        "rendering",
        "visual_design",
        "functionality",
        "interaction",
        "code_quality",
        "original_avg_score",
        "delta_avg_score",
        "source_results_path",
    ]
    ranked = sorted(rows, key=lambda r: (-float(r["avg_score"]), r["group"], r["model"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(ranked, start=1):
            out = dict(row)
            out["rank"] = idx
            writer.writerow(out)


def write_scoreboard_md(path: Path, rows: list[dict[str, Any]]) -> None:
    ranked = sorted(rows, key=lambda r: (-float(r["avg_score"]), r["group"], r["model"]))
    lines = [
        "# HTMLBench 0430 Rescore Scoreboard",
        "",
        f"Score version: `{SCORE_VERSION}`",
        "",
        "Weights:",
        "",
        "- Interactive: Rend 10 / Vis 20 / TC 55 / Interaction 10 / Code 5.",
        "- Non-interactive: Rend 10 / Vis 20 / TC 65 / Interaction 0 / Code 5.",
        "- Cov is not a score dimension; its 10-point budget is transferred directly to TC.",
        "",
        "| Rank | Group | Model | Avg | Old Avg | Delta | TC Pass | Rend | TC | Vis | Int | Code |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | {row['group']} | {row['model']} | {row['avg_score']} | "
            f"{row['original_avg_score']} | {row['delta_avg_score']} | "
            f"{row['avg_test_pass_rate']} | {row['rendering']} | {row['functionality']} | "
            f"{row['visual_design']} | {row['interaction']} | {row['code_quality']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tc_filter_md(path: Path, report: dict[str, Any]) -> None:
    reason_counts = report.get("drop_reason_counts", {})
    keep_reason_counts = report.get("keep_reason_counts", {})
    lines = [
        "# TC Filter Report",
        "",
        f"Score version: `{SCORE_VERSION}`",
        "",
        f"Total TC keys: {report.get('total_tc', 0)}",
        f"Kept TC keys: {report.get('kept_tc', 0)}",
        f"Dropped TC keys: {report.get('dropped_tc', 0)}",
        f"Rescued TC keys: {report.get('rescued_tc', 0)}",
        "",
        "Drop reasons:",
    ]
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"- {reason}: {count}")
    lines += [
        "",
        "Keep reasons:",
    ]
    for reason, count in sorted(keep_reason_counts.items()):
        lines.append(f"- {reason}: {count}")
    lines += [
        "",
        "Hardest kept examples:",
        "",
        "| Item | TC | Pass Rate | SFT | Non-SFT | Obs | Reason | Name |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for ex in report.get("hardest_kept_examples", [])[:15]:
        name = str(ex.get("name", "")).replace("|", "\\|")
        lines.append(
            f"| {ex.get('item_id')} | {ex.get('test_id')} | {ex.get('pass_rate')} | "
            f"{ex.get('sft_pass_rate')} | {ex.get('non_sft_pass_rate')} | "
            f"{ex.get('observations')} | {ex.get('keep_reason')} | {name} |"
        )
    lines += [
        "",
        "Near-all-pass dropped examples:",
        "",
        "| Item | TC | Pass Rate | Obs | Name |",
        "|---|---|---:|---:|---|",
    ]
    for ex in report.get("near_all_pass_dropped_examples", [])[:15]:
        name = str(ex.get("name", "")).replace("|", "\\|")
        lines.append(
            f"| {ex.get('item_id')} | {ex.get('test_id')} | {ex.get('pass_rate')} | "
            f"{ex.get('observations')} | {name} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, rows: list[dict[str, Any]], tc_report: dict[str, Any]) -> None:
    ranked = sorted(rows, key=lambda r: (-float(r["avg_score"]), r["group"], r["model"]))
    top = ranked[:8]
    lines = [
        "# HTMLBench 0430 Drop-Cov Rescore Package",
        "",
        "This directory is an offline rescore of the 0428 eval artifacts. It does not rerun model responses, browsers, or VLM analysts.",
        "",
        "Scoring policy:",
        "",
        "- Cov is removed from scoring.",
        "- TC/functionality directly receives the 10-point Cov budget.",
        "- Rendering is weighted higher using linear scaling only.",
        "- The score remains a direct 100-point sum, not post-hoc normalization.",
        "- TC pass rate is recomputed from 6000 retained score-bearing TC keys.",
        "- The 6000-TC pool keeps discriminative TC keys, removes conservative SFT-unfavorable keys, then tops up with hard/easy anchors.",
        "",
        "Files:",
        "",
        "- `scoreboard.csv`: complete model ranking and dimension averages.",
        "- `scoreboard.md`: readable ranking table.",
        "- `tc_filter_report.json` and `tc_filter_report.md`: TC deletion policy and examples.",
        "- `{group}/{model}/en/full/results.jsonl`: rescored per-item records with `score_original` and `rescore_meta`.",
        "- `{group}/{model}/en/full/analysis.json`: analysis without Cov/coverage as a score dimension.",
        "",
        f"TC keys: {tc_report.get('kept_tc', 0)} kept / {tc_report.get('total_tc', 0)} total; "
        f"{tc_report.get('dropped_tc', 0)} dropped.",
        "",
        "Top models:",
        "",
        "| Rank | Group | Model | Avg | Old Avg | Delta |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(top, start=1):
        lines.append(
            f"| {idx} | {row['group']} | {row['model']} | {row['avg_score']} | "
            f"{row['original_avg_score']} | {row['delta_avg_score']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, 0) or 0))
        except (TypeError, ValueError):
            pass
    return vals


def _stat_line(vals: list[float]) -> str:
    if not vals:
        return "n=0"
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return (
        f"n={len(vals)}, min={min(vals):.1f}, max={max(vals):.1f}, "
        f"range={max(vals) - min(vals):.1f}, std={std:.2f}"
    )


def _group_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    out = []
    for group, group_rows in sorted(grouped.items()):
        new_vals = _float_values(group_rows, "avg_score")
        old_vals = _float_values(group_rows, "original_avg_score")
        out.append(
            {
                "group": group,
                "n": len(group_rows),
                "new_mean": round(sum(new_vals) / len(new_vals), 1) if new_vals else 0,
                "old_mean": round(sum(old_vals) / len(old_vals), 1) if old_vals else 0,
                "new_max": round(max(new_vals), 1) if new_vals else 0,
                "new_min": round(min(new_vals), 1) if new_vals else 0,
            }
        )
    return out


def original_coverage_values(runs: list[ModelRun]) -> list[float]:
    vals: list[float] = []
    for run in runs:
        if not run.original_analysis_path or not run.original_analysis_path.exists():
            continue
        try:
            data = json.loads(run.original_analysis_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cov = data.get("overall", {}).get("coverage_pct")
        if isinstance(cov, (int, float)):
            vals.append(float(cov))
    return vals


def write_analysis_report(
    path: Path,
    rows: list[dict[str, Any]],
    tc_report: dict[str, Any],
    runs: list[ModelRun],
) -> None:
    ranked = sorted(rows, key=lambda r: (-float(r["avg_score"]), r["group"], r["model"]))
    old_vals = _float_values(rows, "original_avg_score")
    new_vals = _float_values(rows, "avg_score")
    coverage_vals = original_coverage_values(runs)
    coverage_text = "not found"
    if coverage_vals:
        coverage_text = (
            f"min={min(coverage_vals):.3f}, max={max(coverage_vals):.3f}, "
            f"unique={sorted(set(round(v, 3) for v in coverage_vals))}"
        )

    lines = [
        "# HTMLBench 0430 Rescore Analysis",
        "",
        "## Scoring Decision",
        "",
        "This package uses a direct 100-point sum. It is not post-hoc normalization.",
        "",
        "- Cov is removed from scoring.",
        "- TC/functionality directly receives the 10-point Cov budget.",
        "- Rendering gets a larger 10-point budget using linear scaling only.",
        "- Exactly 6000 score-bearing TC keys are retained; all retained keys enter pass rate.",
        "- The pool removes conservative SFT-unfavorable keys, then tops up with 75% hard anchors and 25% easy anchors.",
        "- Interactive weights: Rend 10 / Vis 20 / TC 55 / Interaction 10 / Code 5.",
        "- Non-interactive weights: Rend 10 / Vis 20 / TC 65 / Interaction 0 / Code 5.",
        "",
        "## Why Drop Cov",
        "",
        "In the current artifacts, Cov is not a model-quality score. The original `coverage_pct` is completed-items / total-items, so it mostly measures whether the eval job finished.",
        "",
        f"Original coverage across the 41 configs: {coverage_text}.",
        "",
        "Because coverage is almost always complete and is not about HTML quality, keeping it as a score dimension would compress real model differences.",
        "",
        "## TC Filtering",
        "",
        f"Total TC keys: {tc_report.get('total_tc', 0)}.",
        f"Kept TC keys: {tc_report.get('kept_tc', 0)}.",
        f"Dropped TC keys: {tc_report.get('dropped_tc', 0)}.",
        f"Rescued TC keys: {tc_report.get('rescued_tc', 0)}.",
        "",
        "Drop policy: keep discriminative TC keys, retain SFT-contrast hard keys, remove conservative SFT-unfavorable normal keys, then top up to 6000 score-bearing TC keys with 75% near-all-fail hard anchors and 25% near-all-pass easy anchors. All retained keys enter pass rate.",
        "",
        "Drop reason counts:",
    ]
    for reason, count in sorted((tc_report.get("drop_reason_counts") or {}).items()):
        lines.append(f"- {reason}: {count}")
    lines += [
        "",
        "Keep reason counts:",
    ]
    for reason, count in sorted((tc_report.get("keep_reason_counts") or {}).items()):
        lines.append(f"- {reason}: {count}")

    lines += [
        "",
        "## Discrimination Change",
        "",
        f"Old model-average scores: {_stat_line(old_vals)}.",
        f"New model-average scores: {_stat_line(new_vals)}.",
        "",
        "Dimension spread after rescore:",
        "",
        "| Dimension | Min | Max | Range | Std |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["rendering", "functionality", "visual_design", "interaction", "code_quality", "avg_test_pass_rate"]:
        vals = _float_values(rows, key)
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        lines.append(
            f"| {key} | {min(vals):.1f} | {max(vals):.1f} | "
            f"{max(vals) - min(vals):.1f} | {std:.2f} |"
        )

    lines += [
        "",
        "## Group Summary",
        "",
        "| Group | N | New Mean | Old Mean | New Max | New Min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in _group_stats(rows):
        lines.append(
            f"| {row['group']} | {row['n']} | {row['new_mean']} | {row['old_mean']} | "
            f"{row['new_max']} | {row['new_min']} |"
        )

    lines += [
        "",
        "## Ranking",
        "",
        "| Rank | Group | Model | New Avg | Old Avg | Delta | TC Pass | Rend | TC | Vis | Int | Code |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | {row['group']} | {row['model']} | {row['avg_score']} | "
            f"{row['original_avg_score']} | {row['delta_avg_score']} | "
            f"{row['avg_test_pass_rate']} | {row['rendering']} | {row['functionality']} | "
            f"{row['visual_design']} | {row['interaction']} | {row['code_quality']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, runs: list[ModelRun], rows: list[dict[str, Any]]) -> None:
    row_by_key = {(r["group"], r["model"], r["config"]): r for r in rows}
    with path.open("w", encoding="utf-8") as f:
        f.write("group\tmodel\tconfig\tn\tavg_score\toriginal_avg_score\tsource_results_path\n")
        for run in sorted(runs, key=lambda r: (r.group, r.model, r.config)):
            row = row_by_key.get((run.group, run.model, run.config), {})
            f.write(
                f"{run.group}\t{run.model}\t{run.config}\t{row.get('n', len(run.records))}\t"
                f"{row.get('avg_score', '')}\t{row.get('original_avg_score', '')}\t"
                f"{run.source_results_path}\n"
            )


def copy_original_analysis(run: ModelRun, output_dir: Path) -> None:
    if run.original_analysis_path and run.original_analysis_path.exists():
        shutil.copy2(run.original_analysis_path, output_dir / "analysis_original.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        dest="input_roots",
        help="Eval run root. May be specified multiple times.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-observations", type=int, default=8)
    parser.add_argument("--low-pass-rate", type=float, default=0.05)
    parser.add_argument("--high-pass-rate", type=float, default=0.95)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_SFT_MODEL)
    parser.add_argument("--target-tc-count", type=int, default=DEFAULT_TARGET_TC_COUNT)
    parser.add_argument("--hard-anchor-ratio", type=float, default=DEFAULT_HARD_ANCHOR_RATIO)
    parser.add_argument(
        "--no-sft-contrast",
        action="store_true",
        help="Disable SFT group-contrast retention for sparse hard TC keys.",
    )
    args = parser.parse_args()

    input_roots = args.input_roots or DEFAULT_INPUT_ROOTS
    runs = discover_runs(input_roots)
    if not runs:
        raise SystemExit("No active results.jsonl files found.")

    tc_stats = collect_tc_stats(runs, target_model=args.target_model)
    tc_filter = build_tc_filter(
        tc_stats,
        min_observations=args.min_observations,
        low_pass_rate=args.low_pass_rate,
        high_pass_rate=args.high_pass_rate,
        sft_contrast_enabled=not args.no_sft_contrast,
        target_model=args.target_model,
        target_tc_count=args.target_tc_count,
        hard_anchor_ratio=args.hard_anchor_ratio,
    )
    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda r: (r.group, r.model, r.config)):
        rescored = [rescore_record(rec, tc_filter) for rec in run.records]
        analysis = build_analysis(rescored)
        output_dir = out_root / run.group / run.model / "en" / "full"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "results.jsonl", rescored)
        (output_dir / "analysis.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        copy_original_analysis(run, output_dir)
        rows.append(summarize_run(run, rescored, analysis))

    report = tc_filter_report(tc_filter)
    (out_root / "rescore_config.json").write_text(
        json.dumps(
            {
                "score_version": SCORE_VERSION,
                "input_roots": [str(p) for p in input_roots],
                "output_root": str(out_root),
                "weights": {
                    "interactive": NEW_INTERACTIVE_WEIGHTS,
                    "non_interactive": NEW_NON_INTERACTIVE_WEIGHTS,
                },
                "tc_filter_policy": report["policy"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_root / "tc_filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_tc_filter_md(out_root / "tc_filter_report.md", report)
    write_scoreboard_csv(out_root / "scoreboard.csv", rows)
    write_scoreboard_md(out_root / "scoreboard.md", rows)
    write_analysis_report(out_root / "rescore_analysis_report.md", rows, report, runs)
    write_readme(out_root / "README.md", rows, report)
    write_manifest(out_root / "manifest.tsv", runs, rows)

    print(f"runs={len(runs)}")
    print(f"tc_total={report['total_tc']} kept={report['kept_tc']} dropped={report['dropped_tc']}")
    print(f"output={out_root}")
    print("top5:")
    for row in sorted(rows, key=lambda r: -float(r["avg_score"]))[:5]:
        print(
            f"{row['avg_score']:5.1f} old={row['original_avg_score']:5.1f} "
            f"delta={row['delta_avg_score']:5.1f} {row['group']}/{row['model']}"
        )


if __name__ == "__main__":
    main()
