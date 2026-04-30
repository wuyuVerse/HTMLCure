"""
Benchmark loader — load benchmark items and convert to orchestrator records.

Usage:
    from htmleval.benchmark.loader import load_benchmark_items, benchmark_item_to_record

    items = load_benchmark_items("path/to/benchmark/")
    records = [benchmark_item_to_record(item) for item in items]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from htmleval.benchmark.taxonomy import (
    category_filter_matches,
    normalize_task_family,
    source_category_for,
)

logger = logging.getLogger("htmleval")

# Metadata fields preserved through the orchestrator pipeline.
# These are written at record-level so they appear in results.jsonl.
_META_FIELDS = (
    "category",
    "source_category",
    "sub_type",
    "difficulty",
    "language",
    "has_interaction",
)


def load_benchmark_items(
    path: str,
    *,
    language: str = "",
    category: str = "",
    difficulty: str = "",
) -> List[Dict]:
    """Load benchmark items from a directory (recursive) or single file.

    The benchmark is organized as ``benchmark/{en,zh}/*.jsonl``.  When *path*
    points to a language sub-directory (e.g. ``benchmark/en/``), only that
    language is loaded.  When *path* is the parent ``benchmark/`` directory,
    both languages are loaded — use *language* to select one.

    Args:
        path:       Directory or file path.
        language:   If set, keep only items matching this language ("en"/"zh").
        category:   If set, keep only items matching this category. Supports
                    canonical six-family slugs and legacy ten-way aliases.
        difficulty: If set, keep only items matching this difficulty.
    """
    p = Path(path)
    items: List[Dict] = []

    if p.is_file():
        items.extend(_load_file(p))
    elif p.is_dir():
        # If the directory contains en/ and zh/ sub-directories (benchmark root),
        # only load from those to avoid picking up responses/ or other artifacts.
        en_dir = p / "en"
        zh_dir = p / "zh"
        if en_dir.is_dir() or zh_dir.is_dir():
            for lang_dir in (en_dir, zh_dir):
                if lang_dir.is_dir():
                    for f in sorted(lang_dir.rglob("*.json")) + sorted(lang_dir.rglob("*.jsonl")):
                        items.extend(_load_file(f))
        else:
            for f in sorted(p.rglob("*.json")) + sorted(p.rglob("*.jsonl")):
                items.extend(_load_file(f))
    else:
        raise FileNotFoundError(f"Benchmark path not found: {path}")

    # Filter
    if language:
        items = [it for it in items if it.get("language", "") == language]
    if category:
        items = [it for it in items if category_filter_matches(it, category)]
    if difficulty:
        items = [it for it in items if it.get("difficulty", "") == difficulty]

    logger.info(f"Loaded {len(items)} benchmark items from {path}")
    return items


def _load_file(path: Path) -> List[Dict]:
    """Load items from a single .json or .jsonl file."""
    items: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    items.append(_normalize_item(json.loads(line)))
        else:
            data = json.load(f)
            if isinstance(data, list):
                items.extend(_normalize_item(item) for item in data)
            elif isinstance(data, dict):
                items.append(_normalize_item(data))
    return items


def _normalize_item(item: Dict) -> Dict:
    """Normalize legacy ten-way category values to six task families."""
    raw_category = item.get("category", "")
    source_category = source_category_for(raw_category, item.get("source_category"))
    family = normalize_task_family(raw_category)

    if source_category:
        item.setdefault("source_category", source_category)
    if family != "unknown":
        item["category"] = family
    return item


def benchmark_item_to_record(item: Dict) -> Dict:
    """Convert a benchmark item to an orchestrator-compatible record.

    Preserves metadata (category, sub_type, difficulty, language) at record
    level so they flow through to results.jsonl for self-contained analysis.
    """
    rec = {
        "data": {
            "messages": [
                {"role": "user", "content": item.get("prompt", "")},
                {"role": "assistant", "content": item.get("response", "")},
            ],
        },
        "line_number": item.get("id", 0),
        "test_cases": item.get("test_cases", []),
    }
    # Propagate metadata
    for key in _META_FIELDS:
        if key in item:
            rec[key] = item[key]
    return rec
