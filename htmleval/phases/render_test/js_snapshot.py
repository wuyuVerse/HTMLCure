"""
JS probe snapshot reader for render_test phase.

Reads the window.__probe snapshot injected by the extract phase,
plus scroll dimensions. Separated from probes.py because this is
post-probe data collection, not an interaction probe.
"""
from __future__ import annotations

import json as _json


async def collect_js_snapshot(page, data: dict) -> None:
    """Read window.__probe snapshot + scroll dimensions. Mutates *data*."""
    await _read_scroll_dimensions(page, data)
    snap = await _read_probe_snapshot(page)
    if snap is None:
        return

    data["game_vars"]     = snap.get("gameVars", {})
    data["keys_received"] = snap.get("keysReceived", {})

    _merge_js_errors(snap, data)
    _extract_latency_timings(snap, data)
    _extract_canvas_info(snap, data)
    _extract_audio_info(snap, data)
    _extract_dom_inventory(snap, data)
    _extract_visible_text(snap, data)


# ── Internal helpers ───────────────────────────────────────────────

async def _read_scroll_dimensions(page, data: dict) -> None:
    try:
        data["scroll_height"] = await page.evaluate("document.body.scrollHeight")
        data["viewport_height"] = await page.evaluate("window.innerHeight")
    except Exception:
        pass


async def _read_probe_snapshot(page) -> dict | None:
    try:
        snap_json: str = await page.evaluate(
            "() => window.__probe ? window.__probe.snapshot() : null"
        )
        return _json.loads(snap_json) if snap_json else None
    except Exception:
        return None


def _merge_js_errors(snap: dict, data: dict) -> None:
    js_errors = snap.get("errors", [])
    existing = set(data.get("page_errors", []))
    for e in js_errors:
        if e not in existing:
            data["page_errors"].append(e)


def _extract_latency_timings(snap: dict, data: dict) -> None:
    timings = snap.get("interactionTimings", [])
    if not timings:
        return
    data["interaction_timings"] = timings
    valid = [t for t in timings if t.get("latencyMs", -1) >= 0]
    timed_out = [t for t in timings if t.get("latencyMs", -1) < 0]
    if valid:
        latencies = [t["latencyMs"] for t in valid]
        data["avg_interaction_latency_ms"] = round(sum(latencies) / len(latencies))
        data["max_interaction_latency_ms"] = max(latencies)
        data["min_interaction_latency_ms"] = min(latencies)
    data["interactions_timed_out"] = len(timed_out)


def _extract_canvas_info(snap: dict, data: dict) -> None:
    canvas_info = snap.get("canvasInfo")
    if not canvas_info:
        return
    data["canvas_type"] = canvas_info.get("type", "unknown")
    data["canvas_has_content"] = canvas_info.get("hasContent", False)
    data["canvas_fill_ratio"] = float(canvas_info.get("fillRatio", 0))
    data["canvas_tainted"] = canvas_info.get("tainted", False)


def _extract_audio_info(snap: dict, data: dict) -> None:
    audio_info = snap.get("audioInfo")
    if not audio_info:
        return
    data["audio_elements"] = audio_info.get("audioElements", 0)
    data["has_audio_context"] = audio_info.get("hasAudioContext", False)
    data["media_sources"] = audio_info.get("mediaSources", [])


def _extract_dom_inventory(snap: dict, data: dict) -> None:
    dom_inv = snap.get("domInventory")
    if dom_inv:
        data["dom_inventory"] = dom_inv


def _extract_visible_text(snap: dict, data: dict) -> None:
    vis_text = snap.get("visibleText", "")
    if vis_text:
        data["visible_text"] = vis_text
