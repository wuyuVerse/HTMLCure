"""
HTML Comparison Renderer.

Generates a self-contained single-page viewer that shows the before/after
HTML for each repaired record side by side in live iframes.

Usage:
    python scripts/render_comparison.py              # use default test records
    python scripts/render_comparison.py --uid UID1 UID2 ...
    python scripts/render_comparison.py --output path/to/viewer.html

Output:
    A single self-contained HTML file with all content embedded.
    Open it in any browser — no server required.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Default test records (from repair test run)
# ---------------------------------------------------------------------------

DEFAULT_RECORDS = [
    dict(uid="000604_3183a56c79af", score_before=40, score_after=91,  delta=51, strategy="holistic_rewrite",   label="Low func (F=8, V=9)"),
    dict(uid="000613_fa1a78164ed5", score_before=40, score_after=91,  delta=52, strategy="holistic_rewrite",   label="Very low func (F=5)"),
    dict(uid="003645_c1fb64fb2133", score_before=63, score_after=72,  delta=16, strategy="feature_complete",   label="Renders OK, missing features"),
    dict(uid="003760_e418db952ce0", score_before=68, score_after=78,  delta=10, strategy="feature_complete",   label="Partial functionality"),
    dict(uid="003383_e180fd86aa04", score_before=0,  score_after=94,  delta=94, strategy="holistic_rewrite",   label="Render fail → full rewrite"),
]


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

@dataclass
class Record:
    uid: str
    label: str
    strategy: str
    score_before: int
    score_after: int
    delta: int
    sft_eligible: bool
    query: str
    html_before: str
    html_after: str
    score_dims_before: dict = field(default_factory=dict)
    score_dims_after:  dict = field(default_factory=dict)


def _load_score_dims(results_file: Path, uid: str) -> dict:
    """Load per-dimension scores from results JSONL."""
    for line in results_file.read_text().splitlines():
        r = json.loads(line)
        if r.get("_eval_uid") == uid:
            return r.get("score", {})
    return {}


def _load_query(results_file: Path, uid: str) -> str:
    """Load the original query for a record."""
    for line in results_file.read_text().splitlines():
        r = json.loads(line)
        if r.get("_eval_uid") != uid:
            continue
        data = r.get("data", {})
        msgs = data.get("messages", []) if isinstance(data, dict) else []
        return next((m["content"] for m in msgs if m["role"] == "user"), "")
    return ""


def load_records(
    meta_list: list[dict],
    reports_dir: Path,
    results_file: Path,
) -> list[Record]:
    records = []
    for m in meta_list:
        uid = m["uid"]

        # Original HTML (from re-eval at repair start)
        orig_dir = reports_dir / f"repairtest_{uid}_test"
        orig_html_path = orig_dir / "game.html"

        # Best repaired HTML (iter 2 preferred, fall back to iter 1)
        for iter_label in ("repair_2_repair", "repair_1_repair"):
            best_path = reports_dir / f"repairtest_{uid}_{iter_label}" / "game.html"
            if best_path.exists():
                break

        if not orig_html_path.exists() or not best_path.exists():
            print(f"  [skip] {uid}: missing HTML files", file=sys.stderr)
            continue

        records.append(Record(
            uid=uid,
            label=m.get("label", uid),
            strategy=m.get("strategy", ""),
            score_before=m["score_before"],
            score_after=m["score_after"],
            delta=m["delta"],
            sft_eligible=m.get("score_after", 0) >= 80 and m.get("delta", 0) >= 10,
            query=_load_query(results_file, uid),
            html_before=orig_html_path.read_text(encoding="utf-8", errors="replace"),
            html_after=best_path.read_text(encoding="utf-8", errors="replace"),
            score_dims_before=_load_score_dims(results_file, uid),
        ))

    return records


# ---------------------------------------------------------------------------
# HTML generator
# ---------------------------------------------------------------------------

_VIEWER_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HTML Repair Comparison Viewer</title>
<style>
  /* ── Reset & base ─────────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:        #0f1117;
    --surface:   #1a1d27;
    --border:    #2a2d3a;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --accent:    #6366f1;
    --green:     #22c55e;
    --red:       #ef4444;
    --yellow:    #f59e0b;
    --radius:    8px;
    --header-h:  56px;
    --footer-h:  52px;
  }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Header ────────────────────────────────────────────── */
  #header {
    height: var(--header-h);
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 0 16px;
    flex-shrink: 0;
  }
  #header h1 { font-size: 14px; font-weight: 600; color: var(--text); letter-spacing: .4px; }
  .nav-btn {
    background: var(--border);
    border: none;
    color: var(--text);
    padding: 6px 14px;
    border-radius: var(--radius);
    cursor: pointer;
    font-size: 13px;
    transition: background .15s;
  }
  .nav-btn:hover { background: var(--accent); }
  .nav-btn:disabled { opacity: .35; cursor: default; }
  #counter { color: var(--muted); font-size: 12px; min-width: 40px; text-align: center; }

  /* ── Score badge ───────────────────────────────────────── */
  .score-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-left: auto;
  }
  .score-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
  }
  .chip-before { background: #2a1a1a; color: var(--red); border: 1px solid #4a2020; }
  .chip-after  { background: #1a2a1a; color: var(--green); border: 1px solid #204a20; }
  .chip-delta  { background: #1a1a2a; color: var(--accent); border: 1px solid #20204a; }
  .chip-sft    { background: #2a2010; color: var(--yellow); border: 1px solid #4a3820; font-size: 10px; }

  .strategy-tag {
    background: #1e1e35;
    color: #818cf8;
    border: 1px solid #2d2d55;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: monospace;
  }

  /* ── Info bar (query + dims) ─────────────────────────────── */
  #info-bar {
    background: #141720;
    border-bottom: 1px solid var(--border);
    padding: 8px 16px;
    flex-shrink: 0;
    display: flex;
    gap: 24px;
    align-items: flex-start;
    overflow: hidden;
  }
  #query-text {
    flex: 1;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.5;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }
  #query-text strong { color: var(--text); }
  .dim-bars {
    display: flex;
    gap: 10px;
    flex-shrink: 0;
    align-items: center;
  }
  .dim-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }
  .dim-label { font-size: 10px; color: var(--muted); }
  .dim-track {
    width: 28px;
    height: 40px;
    background: var(--border);
    border-radius: 3px;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
  }
  .dim-bar-before {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 48%;
    background: #ef4444aa;
    border-radius: 2px 2px 0 0;
    transition: height .4s ease;
  }
  .dim-bar-after {
    position: absolute;
    bottom: 0;
    right: 0;
    width: 48%;
    background: #22c55eaa;
    border-radius: 2px 2px 0 0;
    transition: height .4s ease;
  }
  .dim-val { font-size: 9px; color: var(--muted); margin-top: 1px; }

  /* ── Pane labels ──────────────────────────────────────── */
  #pane-labels {
    display: flex;
    flex-shrink: 0;
    border-bottom: 1px solid var(--border);
  }
  .pane-label {
    flex: 1;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .pane-label-before { background: #200e0e; color: #fca5a5; border-right: 1px solid var(--border); }
  .pane-label-after  { background: #0e200e; color: #86efac; }
  .pane-score { font-size: 18px; font-weight: 700; }

  /* ── Iframe area ──────────────────────────────────────── */
  #frames {
    flex: 1;
    display: flex;
    overflow: hidden;
    min-height: 0;
  }
  .frame-wrap {
    flex: 1;
    position: relative;
    overflow: hidden;
  }
  .frame-wrap:first-child { border-right: 2px solid var(--accent); }
  .frame-wrap iframe {
    width: 100%;
    height: 100%;
    border: none;
    background: white;
  }
  .frame-loading {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg);
    color: var(--muted);
    font-size: 12px;
    pointer-events: none;
    transition: opacity .2s;
  }
  .frame-loading.hidden { opacity: 0; }

  /* ── Footer ────────────────────────────────────────────── */
  #footer {
    height: var(--footer-h);
    background: var(--surface);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    flex-shrink: 0;
  }
  #uid-display { font-family: monospace; font-size: 11px; color: var(--muted); }
  #kbd-hint { font-size: 11px; color: var(--muted); }
  .kbd {
    display: inline-block;
    padding: 1px 5px;
    background: var(--border);
    border-radius: 3px;
    font-size: 10px;
    font-family: monospace;
  }
</style>
</head>
<body>

<!-- ── Header ── -->
<div id="header">
  <h1>HTML Repair Viewer</h1>
  <button class="nav-btn" id="btn-prev" onclick="navigate(-1)">&#8592; Prev</button>
  <span id="counter">1 / N</span>
  <button class="nav-btn" id="btn-next" onclick="navigate(+1)">Next &#8594;</button>

  <div class="score-row">
    <span class="score-chip chip-before" id="chip-before">&#9660; 0</span>
    <span class="score-chip chip-after"  id="chip-after">&#9650; 0</span>
    <span class="score-chip chip-delta"  id="chip-delta">&#916; 0</span>
    <span class="score-chip chip-sft"    id="chip-sft" style="display:none">&#10003; SFT</span>
    <span class="strategy-tag"           id="chip-strategy"></span>
  </div>
</div>

<!-- ── Info bar ── -->
<div id="info-bar">
  <div id="query-text"><strong>Query:</strong> <span id="query-span"></span></div>
  <div class="dim-bars" id="dim-bars"></div>
</div>

<!-- ── Pane labels ── -->
<div id="pane-labels">
  <div class="pane-label pane-label-before">
    <span>Before</span>
    <span class="pane-score" id="score-before-label">0</span>
    <span style="color:#64748b;font-size:11px">/ 100</span>
  </div>
  <div class="pane-label pane-label-after">
    <span>After</span>
    <span class="pane-score" id="score-after-label">0</span>
    <span style="color:#64748b;font-size:11px">/ 100</span>
  </div>
</div>

<!-- ── Iframes ── -->
<div id="frames">
  <div class="frame-wrap">
    <iframe id="frame-before" title="Before repair" sandbox="allow-scripts allow-same-origin"></iframe>
    <div class="frame-loading" id="load-before">Loading…</div>
  </div>
  <div class="frame-wrap">
    <iframe id="frame-after" title="After repair" sandbox="allow-scripts allow-same-origin"></iframe>
    <div class="frame-loading" id="load-after">Loading…</div>
  </div>
</div>

<!-- ── Footer ── -->
<div id="footer">
  <span id="uid-display"></span>
  <span id="kbd-hint">
    Navigate: <span class="kbd">&#8592;</span> <span class="kbd">&#8594;</span> arrow keys
  </span>
</div>

<script>
// ==========================================================================
// Data — embedded by render_comparison.py
// ==========================================================================
const RECORDS = __RECORDS_JSON__;

// ==========================================================================
// State
// ==========================================================================
const state = { index: 0 };

// ==========================================================================
// Dimension metadata
// ==========================================================================
const DIMS = [
  { key: "rendering",    label: "Rend", max: 5 },
  { key: "visual_design",label: "Vis",  max: 25 },
  { key: "functionality",label: "Func", max: 18 },
  { key: "interaction",  label: "Int",  max: 7 },
  { key: "code_quality", label: "Impl", max: 45 },
];

// ==========================================================================
// Blob URL management — revoke previous to free memory
// ==========================================================================
const _blobUrls = { before: null, after: null };

function _setIframeSrc(frameId, loaderId, html) {
  const side = frameId === "frame-before" ? "before" : "after";
  if (_blobUrls[side]) URL.revokeObjectURL(_blobUrls[side]);

  const blob = new Blob([html], { type: "text/html" });
  const url  = URL.createObjectURL(blob);
  _blobUrls[side] = url;

  const loader = document.getElementById(loaderId);
  const frame  = document.getElementById(frameId);
  loader.classList.remove("hidden");
  frame.onload = () => loader.classList.add("hidden");
  frame.src = url;
}

// ==========================================================================
// Dim bars renderer
// ==========================================================================
function _renderDimBars(rec) {
  const container = document.getElementById("dim-bars");
  container.innerHTML = "";
  const before = rec.score_dims_before || {};
  const after  = rec.score_dims_after  || {};
  for (const dim of DIMS) {
    const bVal = before[dim.key] ?? 0;
    const aVal = after[dim.key]  ?? (dim.key === "total" ? rec.score_after : 0);
    const bPct = (bVal / dim.max * 100).toFixed(0);
    const aPct = (aVal / dim.max * 100).toFixed(0);
    container.insertAdjacentHTML("beforeend", `
      <div class="dim-item">
        <div class="dim-label">${dim.label}</div>
        <div class="dim-track">
          <div class="dim-bar-before" style="height:${bPct}%"></div>
          <div class="dim-bar-after"  style="height:${aPct}%"></div>
        </div>
        <div class="dim-val">${bVal}→${aVal}</div>
      </div>`);
  }
}

// ==========================================================================
// Render the current record
// ==========================================================================
function render() {
  const rec = RECORDS[state.index];
  const n   = RECORDS.length;

  // Navigation controls
  document.getElementById("counter").textContent = `${state.index + 1} / ${n}`;
  document.getElementById("btn-prev").disabled = state.index === 0;
  document.getElementById("btn-next").disabled = state.index === n - 1;

  // Score chips
  document.getElementById("chip-before").innerHTML   = `&#9660; ${rec.score_before}`;
  document.getElementById("chip-after").innerHTML    = `&#9650; ${rec.score_after}`;
  const sign = rec.delta >= 0 ? "+" : "";
  document.getElementById("chip-delta").textContent  = `Δ${sign}${rec.delta}`;
  document.getElementById("chip-strategy").textContent = rec.strategy;
  const sftChip = document.getElementById("chip-sft");
  sftChip.style.display = rec.sft_eligible ? "inline-flex" : "none";

  // Pane labels
  document.getElementById("score-before-label").textContent = rec.score_before;
  document.getElementById("score-after-label").textContent  = rec.score_after;

  // Query text
  document.getElementById("query-span").textContent = rec.query || "(no query)";

  // Dim bars
  _renderDimBars(rec);

  // Footer
  document.getElementById("uid-display").textContent = `uid: ${rec.uid}  |  ${rec.label}`;

  // Iframes — only reload if content changed
  _setIframeSrc("frame-before", "load-before", rec.html_before);
  _setIframeSrc("frame-after",  "load-after",  rec.html_after);
}

// ==========================================================================
// Navigation
// ==========================================================================
function navigate(dir) {
  const newIdx = state.index + dir;
  if (newIdx < 0 || newIdx >= RECORDS.length) return;
  state.index = newIdx;
  render();
}

document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft")  navigate(-1);
  if (e.key === "ArrowRight") navigate(+1);
});

// ==========================================================================
// Init
// ==========================================================================
render();
</script>
</body>
</html>
"""


def build_viewer(records: list[Record]) -> str:
    """Serialize records to JSON and inject into the HTML template."""
    data = []
    for r in records:
        data.append({
            "uid":              r.uid,
            "label":            r.label,
            "strategy":         r.strategy,
            "score_before":     r.score_before,
            "score_after":      r.score_after,
            "delta":            r.delta,
            "sft_eligible":     r.sft_eligible,
            "query":            r.query[:400],   # truncate for display
            "html_before":      r.html_before,
            "html_after":       r.html_after,
            "score_dims_before": r.score_dims_before,
            "score_dims_after":  r.score_dims_after,
        })
    json_payload = json.dumps(data, ensure_ascii=False)
    return _VIEWER_TEMPLATE.replace("__RECORDS_JSON__", json_payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML before/after comparison viewer")
    parser.add_argument("--uid", nargs="*", help="UIDs to include (default: 5 test records)")
    parser.add_argument("--output", default="repair_comparison.html",
                        help="Output path (default: repair_comparison.html)")
    parser.add_argument("--reports-dir", default="eval_results/reports")
    parser.add_argument("--results",     default="eval_results/html_data/results.jsonl")
    args = parser.parse_args()

    reports_dir  = Path(args.reports_dir)
    results_file = Path(args.results)

    # Determine which records to include
    if args.uid:
        meta = [dict(uid=u, score_before=0, score_after=0, delta=0, strategy="") for u in args.uid]
    else:
        meta = DEFAULT_RECORDS

    print(f"Loading {len(meta)} records from {reports_dir} …")
    records = load_records(meta, reports_dir, results_file)
    print(f"  Loaded {len(records)} records successfully")

    html = build_viewer(records)

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"  Written: {out}  ({size_kb:.0f} KB)")
    print(f"  Open in browser:  file://{out.resolve()}")


if __name__ == "__main__":
    main()
