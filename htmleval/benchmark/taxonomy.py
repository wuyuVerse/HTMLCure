"""Canonical benchmark task-family taxonomy.

The paper groups the benchmark into six semantic task families.  Older
benchmark files used ten collection categories; keep that mapping centralized
so loaders, analysis, and scripts report the same six-way breakdown.
"""

from __future__ import annotations

from collections.abc import Iterable

TASK_FAMILIES: tuple[str, ...] = (
    "games_simulations",
    "apps_tools",
    "data_visualization",
    "visual_art_animation",
    "three_d_webgl",
    "content_marketing",
)

TASK_FAMILY_LABELS: dict[str, str] = {
    "games_simulations": "Games & Simulations",
    "apps_tools": "Apps & Tools",
    "data_visualization": "Data Visualization",
    "visual_art_animation": "Visual Art & Animation",
    "three_d_webgl": "3D/WebGL Scenes",
    "content_marketing": "Content & Marketing",
}

LEGACY_CATEGORY_TO_FAMILY: dict[str, str] = {
    "game": "games_simulations",
    "app": "apps_tools",
    "ui": "apps_tools",
    "dataviz": "data_visualization",
    "creative": "visual_art_animation",
    "svg_art": "visual_art_animation",
    "three_3d": "three_d_webgl",
    "content": "content_marketing",
    "landing": "content_marketing",
    "portfolio": "content_marketing",
}

LEGACY_CATEGORIES: tuple[str, ...] = tuple(LEGACY_CATEGORY_TO_FAMILY)


def normalize_task_family(category: str | None) -> str:
    """Return the six-family slug for a canonical or legacy category value."""
    value = (category or "").strip()
    if not value:
        return "unknown"
    return LEGACY_CATEGORY_TO_FAMILY.get(value, value)


def source_category_for(category: str | None, source_category: str | None = None) -> str:
    """Return the original ten-way source category when it can be inferred."""
    source = (source_category or "").strip()
    if source:
        return source
    value = (category or "").strip()
    if value in LEGACY_CATEGORY_TO_FAMILY:
        return value
    return ""


def category_filter_matches(
    item: dict,
    requested_categories: str | Iterable[str],
) -> bool:
    """Match either canonical six-family slugs or legacy ten-way aliases."""
    if isinstance(requested_categories, str):
        raw_values = [v.strip() for v in requested_categories.split(",") if v.strip()]
    else:
        raw_values = [str(v).strip() for v in requested_categories if str(v).strip()]
    if not raw_values:
        return True

    requested = set(raw_values)
    requested_legacy = {v for v in raw_values if v in LEGACY_CATEGORY_TO_FAMILY}
    requested_families = {
        normalize_task_family(v)
        for v in raw_values
        if v not in LEGACY_CATEGORY_TO_FAMILY
    }

    category = str(item.get("category", "")).strip()
    family = normalize_task_family(category)
    source = source_category_for(category, item.get("source_category"))
    return (
        category in requested_families
        or family in requested_families
        or source in requested_legacy
    )
