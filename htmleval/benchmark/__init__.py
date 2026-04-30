"""htmleval.benchmark — Benchmark loading, generation, execution, and analysis."""

from htmleval.benchmark.loader import load_benchmark_items, benchmark_item_to_record
from htmleval.benchmark.analysis import (
    analyze_results, analyze_with_trials, validate_items, print_benchmark_report,
    compare_analyses, print_comparison_report,
)
from htmleval.benchmark.generator import generate_responses
from htmleval.benchmark.runner import run_benchmark
from htmleval.benchmark.metrics import pass_at_k, is_pass, aggregate_trials
from htmleval.benchmark.taxonomy import (
    TASK_FAMILIES,
    TASK_FAMILY_LABELS,
    LEGACY_CATEGORIES,
    LEGACY_CATEGORY_TO_FAMILY,
)

__all__ = [
    "load_benchmark_items",
    "benchmark_item_to_record",
    "generate_responses",
    "analyze_results",
    "analyze_with_trials",
    "validate_items",
    "print_benchmark_report",
    "compare_analyses",
    "print_comparison_report",
    "run_benchmark",
    "pass_at_k",
    "is_pass",
    "aggregate_trials",
    "TASK_FAMILIES",
    "TASK_FAMILY_LABELS",
    "LEGACY_CATEGORIES",
    "LEGACY_CATEGORY_TO_FAMILY",
]
