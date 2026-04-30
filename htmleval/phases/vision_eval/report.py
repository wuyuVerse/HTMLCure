"""
Report generation for the VisionEval phase.

Produces a Markdown evaluation report from a completed EvalContext.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from htmleval.core.context import EvalContext


_SCORE_META = {
    "rendering":     ("Rendering", 5),
    "visual_design": ("Visual Design", 15),
    "functionality": ("Functionality", 50),
    "interaction":   ("Interaction", 15),
    "code_quality":  ("Implementation Quality", 15),
}


def resolve_report_screenshots(ctx: EvalContext, limit: int | None = None) -> list[str]:
    """Resolve screenshots for reporting/evaluation with filesystem fallback.

    Normally screenshots flow through ctx.all_screenshots. If that aggregation is
    empty but frame files exist on disk (for example after a mid-phase warning or
    when regenerating a report), fall back to render_test annotations and then to
    any saved frame_*.png files in the record output directory.
    """
    resolved: list[str] = []
    seen: set[str] = set()

    def _push(path_str: str) -> None:
        if not path_str:
            return
        path = Path(path_str)
        if path.exists():
            key = str(path)
            if key not in seen:
                seen.add(key)
                resolved.append(key)

    for ss in ctx.all_screenshots:
        _push(ss)

    if not resolved and ctx.output_dir is not None:
        render = _phase_data(ctx, "render_test")
        for ann in render.get("frame_annotations", []):
            if isinstance(ann, dict):
                name = ann.get("screenshot_name", "")
                if name:
                    _push(str(Path(ctx.output_dir) / name))

        if not resolved:
            for path in sorted(Path(ctx.output_dir).glob("frame_*.png")):
                _push(str(path))

    if limit is not None:
        return resolved[:limit]
    return resolved


def generate_report(ctx: EvalContext) -> str:
    """Generate a Markdown evaluation report from a completed EvalContext."""
    ev      = ctx.final_score or {}
    static  = _phase_data(ctx, "static_analysis")
    render  = _phase_data(ctx, "render_test")
    agent   = _phase_data(ctx, "agent_test")
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total   = ev.get("total_score", 0)
    screenshots = resolve_report_screenshots(ctx)
    probe_errors = render.get("probe_errors", [])

    lines = [
        f"# HTML Evaluation Report",
        f"",
        f"| Field       | Value |",
        f"|-------------|-------|",
        f"| ID          | `{ctx.game_id}` |",
        f"| Variant     | {ctx.variant} |",
        f"| Evaluated   | {ts} |",
        f"| **Score**   | **{total} / 100** |",
        f"",
        f"## Task Description",
        f"",
        ctx.query,
        f"",
        f"## Scores",
        f"",
        f"| Dimension | Score | Max | Reason |",
        f"|-----------|------:|----:|--------|",
    ]

    for key, (label, max_score) in _SCORE_META.items():
        dim = ev.get(key, {})
        score  = dim.get("score", 0)  if isinstance(dim, dict) else 0
        reason = dim.get("reason", "") if isinstance(dim, dict) else ""
        lines.append(f"| {label} | {score} | {max_score} | {reason} |")

    lines += [
        f"| **Total** | **{total}** | **100** | |",
        f"",
        f"## Summary",
        f"",
        ev.get("summary", "N/A"),
        f"",
    ]

    # Bugs
    bugs = ev.get("bugs", [])
    lines.append(f"## Bugs ({len(bugs)})\n")
    lines.extend(f"{i}. {b}" for i, b in enumerate(bugs, 1)) if bugs else lines.append("None found.")
    lines.append("")

    # Missing features
    mf = ev.get("missing_features", [])
    lines.append(f"## Missing Features ({len(mf)})\n")
    lines.extend(f"{i}. {f}" for i, f in enumerate(mf, 1)) if mf else lines.append("None.")
    lines.append("")

    # Highlights
    hl = ev.get("highlights", [])
    lines.append(f"## Highlights ({len(hl)})\n")
    lines.extend(f"{i}. {h}" for i, h in enumerate(hl, 1)) if hl else lines.append("None.")
    lines.append("")

    # Screenshots
    lines.append(f"## Screenshots ({len(screenshots)})\n")
    for ss in screenshots:
        name = Path(ss).name
        lines.append(f"![{name}](./{name})")
    if not screenshots:
        lines.append("No render screenshots were available.")
    lines.append("")

    # Agent feedback
    agent_ran = ev.get("agent_phase_run", False)
    lines += [
        f"## Agent Test",
        f"",
        f"- Phase ran: {'yes' if agent_ran else 'no (skipped)'}",
        f"- Steps: {agent.get('steps_taken', 0)}",
        f"- Completed: {'yes' if agent.get('agent_completed') else 'no'}",
        f"",
        f"### Agent Summary",
        f"",
        agent.get("agent_summary", "(not run)")[:3000],
        f"",
    ]

    # Technical details
    lines += [
        f"## Technical Details",
        f"",
        f"- HTML size: {static.get('html_size', 0):,} chars",
        f"- Canvas: {static.get('has_canvas', False)}  |  "
        f"JS: {static.get('has_script', False)}  |  "
        f"CSS: {static.get('has_style', False)}  |  "
        f"SVG: {static.get('has_svg', False)}  |  "
        f"rAF: {static.get('has_requestanimationframe', False)}",
        f"- External resources: {len(static.get('external_resources', []))}",
        f"- Input types: {', '.join(static.get('input_types', [])) or 'none'}",
        f"- Rendered: {render.get('rendered', False)}",
        f"- Render screenshots: {len(screenshots)}",
        f"- Render probe errors: {len(probe_errors)}",
        f"- Console errors: {len(render.get('console_errors', []))}",
        f"- JS exceptions: {len(render.get('page_errors', []))}",
    ]

    if probe_errors:
        lines += [
            "",
            f"## Render Warnings",
            "",
        ]
        for i, item in enumerate(probe_errors[:8], 1):
            probe = item.get("probe", "unknown")
            err = item.get("error", "")
            lines.append(f"{i}. `{probe}`: {err}")

    return "\n".join(lines)


def _phase_data(ctx: EvalContext, name: str) -> Dict[str, Any]:
    r = ctx.get_phase(name)
    return r.data if r is not None else {}
