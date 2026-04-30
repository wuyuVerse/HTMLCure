"""Single-page app HTML — embedded as a module constant."""

APP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>htmleval viewer</title>
<style>
/* ── Reset ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d0f16;--surface:#13161f;--surface2:#1a1d29;--border:#252836;
  --text:#e2e8f0;--muted:#64748b;--accent:#6366f1;--accent2:#818cf8;
  --green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#38bdf8;
  --cyan:#06b6d4;
  --sidebar:320px;--header:48px;--radius:6px;
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13px}

/* ── Layout ── */
#app{display:grid;grid-template-rows:var(--header) 1fr;height:100vh}
#header{background:var(--surface);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:12px;padding:0 14px;flex-shrink:0}
#header h1{font-size:14px;font-weight:700;color:var(--text);letter-spacing:.3px;
  white-space:nowrap}
#header h1 span{color:var(--accent2)}
#body{display:grid;grid-template-columns:var(--sidebar) 1fr;overflow:hidden;min-height:0}

/* ── Sidebar ── */
#sidebar{background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;min-height:0}
#filters{padding:10px 10px 6px;border-bottom:1px solid var(--border);flex-shrink:0;
  display:flex;flex-direction:column;gap:7px}
.filter-row{display:flex;align-items:center;gap:6px}
.filter-label{font-size:11px;color:var(--muted);min-width:42px}
select,input[type=number]{background:var(--bg);border:1px solid var(--border);
  color:var(--text);padding:3px 7px;border-radius:4px;font-size:12px;cursor:pointer}
select:hover,input[type=number]:hover{border-color:var(--accent)}
select{width:100%}
.score-inputs{display:flex;align-items:center;gap:4px}
.score-inputs input{width:52px}
.score-sep{color:var(--muted)}
.toggle-btn{background:var(--bg);border:1px solid var(--border);color:var(--muted);
  padding:2px 8px;border-radius:4px;cursor:pointer;font-size:11px;transition:.15s}
.toggle-btn.active{background:#1e1e40;border-color:var(--accent);color:var(--accent2)}
.toggle-btn:hover{border-color:var(--accent);color:var(--accent2)}

/* ── Search ── */
#search-wrap{padding:8px 10px;flex-shrink:0;border-bottom:1px solid var(--border)}
#search{width:100%;background:var(--bg);border:1px solid var(--border);
  color:var(--text);padding:5px 10px;border-radius:var(--radius);font-size:12px;outline:none}
#search:focus{border-color:var(--accent)}

/* ── Sort + count ── */
#list-header{padding:5px 10px;display:flex;align-items:center;justify-content:space-between;
  flex-shrink:0;border-bottom:1px solid var(--border)}
#record-count{font-size:11px;color:var(--muted)}
#sort-select{background:var(--bg);border:1px solid var(--border);color:var(--text);
  padding:2px 6px;border-radius:4px;font-size:11px}

/* ── Record list ── */
#record-list{overflow-y:auto;flex:1;padding:4px 0}
#record-list::-webkit-scrollbar{width:5px}
#record-list::-webkit-scrollbar-track{background:transparent}
#record-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

.rec-item{padding:7px 10px;cursor:pointer;border-left:3px solid transparent;
  transition:background .1s,border-color .1s;border-bottom:1px solid #1a1d2a}
.rec-item:hover{background:var(--surface2)}
.rec-item.selected-a{border-left-color:var(--accent);background:#1a1a35}
.rec-item.selected-b{border-left-color:var(--green);background:#1a351a}

.rec-top{display:flex;align-items:center;gap:6px}
.rec-score{font-size:16px;font-weight:700;min-width:30px}
.score-hi{color:var(--green)}
.score-mid{color:var(--yellow)}
.score-lo{color:var(--red)}
.rec-uid{font-family:monospace;font-size:11px;color:var(--muted);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;flex:1}
.rec-tags{display:flex;gap:4px;margin-top:3px;flex-wrap:wrap}
.tag{font-size:10px;padding:1px 5px;border-radius:3px;font-weight:500}
.tag-dataset{background:#1e2035;color:#818cf8}
.tag-repair{background:#1a2a1a;color:var(--green)}
.tag-refine{background:#0a2a2a;color:var(--cyan)}
.tag-sft{background:#2a2010;color:var(--yellow)}
.tag-fail{background:#2a1a1a;color:var(--red)}
.rec-query{font-size:10px;color:var(--muted);margin-top:3px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}

/* ── Load more ── */
#load-more-wrap{padding:8px 10px;flex-shrink:0;border-top:1px solid var(--border)}
#load-more{width:100%;background:var(--surface2);border:1px solid var(--border);
  color:var(--muted);padding:5px;border-radius:var(--radius);cursor:pointer;font-size:12px}
#load-more:hover{color:var(--text);border-color:var(--accent)}
#load-more:disabled{opacity:.4;cursor:default}

/* ── Compare hint ── */
#compare-hint{padding:5px 10px;font-size:11px;color:var(--accent2);background:#13132a;
  border-top:1px solid #2020508;flex-shrink:0;display:none}
#compare-hint.visible{display:block}

/* ── Main viewer ── */
#viewer{display:flex;flex-direction:column;overflow:hidden;min-height:0;background:var(--bg)}
#viewer-toolbar{background:var(--surface);border-bottom:1px solid var(--border);
  padding:7px 14px;display:flex;align-items:center;gap:10px;flex-shrink:0;min-height:40px}
.vt-uid{font-family:monospace;font-size:11px;color:var(--muted)}
.vt-score-pair{display:flex;align-items:center;gap:6px}
.vt-chip{padding:2px 9px;border-radius:12px;font-size:12px;font-weight:600}
.chip-a{background:#200e0e;color:#fca5a5;border:1px solid #4a2020}
.chip-b{background:#0e200e;color:#86efac;border:1px solid #204a20}
.chip-d{background:#1a1a3a;color:var(--accent2);border:1px solid #303060}
.chip-sft{background:#2a2010;color:var(--yellow);border:1px solid #4a3820;font-size:10px}
.vt-strategy{font-family:monospace;font-size:11px;color:#818cf8;
  background:#1a1a35;padding:2px 7px;border-radius:3px}
.vt-dims{display:flex;gap:8px;margin-left:auto}
.dim-pill{display:flex;flex-direction:column;align-items:center;gap:1px}
.dim-pill-name{font-size:9px;color:var(--muted)}
.dim-pill-val{font-size:11px;font-weight:600}

/* ── View mode buttons ── */
.view-mode-group{display:flex;gap:0;border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
.view-mode-btn{background:var(--bg);color:var(--muted);border:none;
  padding:3px 10px;cursor:pointer;font-size:11px;font-weight:600;transition:.15s;
  border-right:1px solid var(--border)}
.view-mode-btn:last-child{border-right:none}
.view-mode-btn:hover{color:var(--text);background:var(--surface2)}
.view-mode-btn.active-repair{background:#1a2a1a;color:var(--green)}
.view-mode-btn.active-refine{background:#0a2a2a;color:var(--cyan)}
.view-mode-btn.active-original{background:var(--surface2);color:var(--text)}

/* ── Iteration bar ── */
#iter-bar{display:flex;gap:0;flex-wrap:wrap;padding:0 14px 6px;background:var(--surface);
  border-bottom:1px solid var(--border);flex-shrink:0}
.iter-btn{background:var(--bg);color:var(--muted);border:1px solid var(--border);
  padding:3px 10px;cursor:pointer;font-size:11px;font-weight:500;transition:.15s;
  border-radius:var(--radius);margin:2px 3px 2px 0}
.iter-btn:hover{color:var(--text);background:var(--surface2);border-color:var(--accent)}
.iter-btn.iter-active{background:#0a2a2a;color:var(--cyan);border-color:var(--cyan)}
.iter-btn .iter-delta{font-weight:700}
.iter-btn .iter-delta.pos{color:var(--green)}
.iter-btn .iter-delta.neg{color:var(--red)}
.iter-btn .iter-delta.zero{color:var(--muted)}

/* ── Pane labels ── */
#pane-labels{display:flex;flex-shrink:0}
.pane-label{flex:1;padding:5px 14px;font-size:12px;font-weight:600;
  display:flex;align-items:center;gap:8px}
.pane-label-a{background:#200e0e;color:#fca5a5;border-right:1px solid var(--border)}
.pane-label-b{background:#0e200e;color:#86efac}
.pane-label-refine{background:#0a2020;color:var(--cyan)}
.pane-score-num{font-size:17px;font-weight:700}

/* ── Frames ── */
#frames{display:flex;flex:1;overflow:hidden;min-height:0}
.frame-wrap{flex:1;position:relative;overflow:hidden}
.frame-wrap+.frame-wrap{border-left:2px solid var(--accent)}
.frame-wrap iframe{width:100%;height:100%;border:none;background:#fff}
.frame-loading{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;background:var(--bg);color:var(--muted);font-size:12px;
  pointer-events:none;transition:opacity .2s}
.frame-loading.gone{opacity:0;pointer-events:none}

/* ── Empty state ── */
#empty-state{display:flex;flex-direction:column;align-items:center;
  justify-content:center;flex:1;gap:12px;color:var(--muted)}
#empty-state .big{font-size:48px;line-height:1}
#empty-state p{font-size:13px;max-width:260px;text-align:center;line-height:1.6}

/* ── Header controls ── */
.header-sep{width:1px;height:20px;background:var(--border);flex-shrink:0}
.cmp-btn{background:#1a1a35;border:1px solid var(--accent);color:var(--accent2);
  padding:4px 12px;border-radius:var(--radius);cursor:pointer;font-size:12px}
.cmp-btn:hover{background:#252555}
.cmp-btn.active{background:var(--accent);color:#fff}
#clear-btn{background:transparent;border:1px solid var(--border);color:var(--muted);
  padding:4px 10px;border-radius:var(--radius);cursor:pointer;font-size:12px}
#clear-btn:hover{color:var(--text)}
.kbhint{font-size:11px;color:var(--muted);margin-left:auto}
.kbd{display:inline-block;padding:0 4px;background:var(--border);border-radius:3px;
  font-family:monospace;font-size:10px}
</style>
</head>
<body>
<div id="app">

<!-- ── Header ── -->
<header id="header">
  <h1>htmleval <span>viewer</span></h1>
  <div class="header-sep"></div>
  <button class="cmp-btn" id="cmp-toggle" onclick="toggleCompareMode()">
    Compare mode
  </button>
  <button id="clear-btn" onclick="clearSelection()">Clear</button>
  <span class="kbhint">
    <span class="kbd">&#x2191;&#x2193;</span> navigate &nbsp;
    <span class="kbd">Enter</span> select &nbsp;
    <span class="kbd">C</span> compare mode &nbsp;
    <span class="kbd">Esc</span> clear
  </span>
</header>

<div id="body">

<!-- ── Sidebar ── -->
<aside id="sidebar">
  <div id="filters">
    <div class="filter-row">
      <span class="filter-label">Dataset</span>
      <select id="ds-select" onchange="resetAndLoad()">
        <option value="all">All datasets</option>
      </select>
    </div>
    <div class="filter-row">
      <span class="filter-label">Score</span>
      <div class="score-inputs">
        <input type="number" id="score-min" value="0" min="0" max="100"
               onchange="resetAndLoad()">
        <span class="score-sep">&#x2013;</span>
        <input type="number" id="score-max" value="100" min="0" max="100"
               onchange="resetAndLoad()">
      </div>
    </div>
    <div class="filter-row">
      <span class="filter-label">Filter</span>
      <button class="toggle-btn" id="repair-toggle" onclick="toggleRepairFilter()">
        Has repair
      </button>
    </div>
  </div>

  <div id="search-wrap">
    <input id="search" type="text" placeholder="Search uid or query&#x2026;"
           oninput="onSearch(this.value)" autocomplete="off">
  </div>

  <div id="list-header">
    <span id="record-count">Loading&#x2026;</span>
    <select id="sort-select" onchange="resetAndLoad()">
      <option value="score_asc">Score &#x2191;</option>
      <option value="score_desc">Score &#x2193;</option>
      <option value="uid_asc">UID</option>
      <option value="dataset_asc">Dataset</option>
    </select>
  </div>

  <div id="record-list"></div>

  <div id="load-more-wrap">
    <button id="load-more" onclick="loadMore()" disabled>Load more</button>
  </div>
  <div id="compare-hint" class="visible" style="display:none">
    Click a record to set <b>left pane</b>. Then click another for <b>right pane</b>.
  </div>
</aside>

<!-- ── Main viewer ── -->
<main id="viewer">
  <div id="viewer-toolbar"></div>
  <div id="iter-bar" style="display:none"></div>
  <div id="pane-labels" style="display:none"></div>
  <div id="frames"></div>
</main>

</div><!-- #body -->
</div><!-- #app -->

<script>
// ==========================================================================
// State
// ==========================================================================
const state = {
  page: 0, total: 0, loading: false, allLoaded: false,
  records: [],       // all loaded records (accumulated across pages)
  filtered: [],      // client-side search subset
  searchQuery: "",
  selectedA: null,   // primary selection (left pane)
  selectedB: null,   // secondary (right pane, compare mode)
  compareMode: false,
  viewMode: "original",  // "original" | "repair" | "refine"
  cursor: -1,        // keyboard cursor index in filtered list
  iterations: [],    // fetched from /api/iterations
  activeIter: null,  // null = overall, 1/2/3... = specific iteration
};

// ==========================================================================
// Blob URL manager
// ==========================================================================
const blobs = { a: null, b: null };
function setIframe(id, htmlUrl) {
  const side = id === "frame-a" ? "a" : "b";
  if (blobs[side]) URL.revokeObjectURL(blobs[side]);
  const frame   = document.getElementById(id);
  const loadDiv = document.getElementById("load-" + side);
  loadDiv?.classList.remove("gone");
  if (frame) {
    frame.onload = () => loadDiv?.classList.add("gone");
    frame.src = htmlUrl;
  }
}

// ==========================================================================
// API helpers
// ==========================================================================
async function apiFetch(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

function htmlUrl(uid, kind) {
  return `/html/${encodeURIComponent(uid)}/${kind}`;
}

// ==========================================================================
// Score color helper
// ==========================================================================
function scoreClass(n) {
  if (n >= 80) return "score-hi";
  if (n >= 55) return "score-mid";
  return "score-lo";
}

function scoreColor(n) {
  if (n >= 80) return "#22c55e";
  if (n >= 55) return "#f59e0b";
  return "#ef4444";
}

// ==========================================================================
// Data loading
// ==========================================================================
async function loadPage() {
  if (state.loading || state.allLoaded) return;
  state.loading = true;
  document.getElementById("load-more").disabled = true;

  const ds       = document.getElementById("ds-select").value;
  const minScore = document.getElementById("score-min").value || 0;
  const maxScore = document.getElementById("score-max").value || 100;
  const sort     = document.getElementById("sort-select").value;
  const repair   = document.getElementById("repair-toggle").classList.contains("active") ? 1 : 0;

  const params = new URLSearchParams({
    page: state.page, limit: 80,
    dataset: ds, min_score: minScore, max_score: maxScore,
    sort, has_repair: repair,
  });

  try {
    const data = await apiFetch(`/api/records?${params}`);
    state.records.push(...data.records);
    state.total = data.total;
    state.page++;
    state.allLoaded = state.records.length >= data.total;
  } catch (e) {
    console.error("Load error:", e);
  } finally {
    state.loading = false;
  }

  applySearch();
  updateListUI();
}

function resetAndLoad() {
  state.records = [];
  state.page = 0;
  state.allLoaded = false;
  state.cursor = -1;
  loadPage();
}

// ==========================================================================
// Client-side search
// ==========================================================================
let _searchTimer = null;
function onSearch(q) {
  state.searchQuery = q.toLowerCase().trim();
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(() => { applySearch(); updateListUI(); }, 150);
}

function applySearch() {
  const q = state.searchQuery;
  state.filtered = q
    ? state.records.filter(r =>
        r.uid.includes(q) || r.query.toLowerCase().includes(q))
    : state.records;
}

// ==========================================================================
// Render record list
// ==========================================================================
function updateListUI() {
  const list    = document.getElementById("record-list");
  const countEl = document.getElementById("record-count");
  const lmBtn   = document.getElementById("load-more");

  const recs = state.filtered;
  countEl.textContent = `${recs.length} / ${state.total} records`;

  list.innerHTML = recs.map((r, i) => {
    const sc     = r.score.total ?? 0;
    const cls    = scoreClass(sc);
    const isA    = state.selectedA?.uid === r.uid;
    const isB    = state.selectedB?.uid === r.uid;
    const selCls = isA ? "selected-a" : isB ? "selected-b" : "";
    const curCls = state.cursor === i ? "cursor" : "";

    const tags = [
      `<span class="tag tag-dataset">${r.dataset}</span>`,
      r.has_repair ? `<span class="tag tag-repair">repair +${r.repair_delta||"?"}</span>` : "",
      r.has_refine ? `<span class="tag tag-refine">refine +${r.refine_delta||0}</span>` : "",
      r.has_improve ? `<span class="tag tag-refine">iters</span>` : "",
      r.eval_status === "failed" ? `<span class="tag tag-fail">fail</span>` : "",
    ].filter(Boolean).join("");

    return `<div class="rec-item ${selCls} ${curCls}" data-idx="${i}" onclick="selectRecord(${i})">
      <div class="rec-top">
        <span class="rec-score ${cls}">${sc}</span>
        <span class="rec-uid">${r.uid}</span>
      </div>
      <div class="rec-tags">${tags}</div>
      ${r.query ? `<div class="rec-query">${escapeHtml(r.query.slice(0, 80))}</div>` : ""}
    </div>`;
  }).join("");

  lmBtn.disabled = state.allLoaded || state.loading;
  lmBtn.textContent = state.allLoaded
    ? "All loaded"
    : `Load more (${state.total - state.records.length} remaining)`;
}

function loadMore() { loadPage(); }

// ==========================================================================
// Selection logic
// ==========================================================================
function selectRecord(idx) {
  const rec = state.filtered[idx];
  if (!rec) return;
  state.cursor = idx;

  if (state.compareMode) {
    // First click -> A, second -> B
    if (!state.selectedA) {
      state.selectedA = rec;
    } else if (state.selectedA.uid === rec.uid) {
      state.selectedA = null;
    } else {
      state.selectedB = rec;
    }
  } else {
    if (state.selectedA?.uid === rec.uid) {
      state.selectedA = null;
      state.selectedB = null;
      state.viewMode = "original";
      state.iterations = [];
      state.activeIter = null;
    } else {
      state.selectedA = rec;
      state.selectedB = null;
      state.activeIter = null;
      // Auto-select best available view mode
      if (rec.has_repair) {
        state.viewMode = "repair";
        state.selectedB = rec;
      } else if (rec.has_refine) {
        state.viewMode = "refine";
        state.selectedB = rec;
      } else {
        state.viewMode = "original";
      }
    }
  }

  updateListUI();
  updateViewer();

  // Fetch iterations if record has improve traces
  if (!state.compareMode && state.selectedA?.has_improve) {
    fetchIterations(state.selectedA.uid);
  } else {
    state.iterations = [];
    state.activeIter = null;
    renderIterBar();
  }
}

function setViewMode(mode) {
  const A = state.selectedA;
  if (!A) return;
  state.viewMode = mode;
  state.activeIter = null;
  if (mode === "original") {
    state.selectedB = null;
  } else {
    state.selectedB = A;
  }
  updateListUI();
  updateViewer();
  renderIterBar();
}

function clearSelection() {
  state.selectedA = null;
  state.selectedB = null;
  state.viewMode = "original";
  state.cursor = -1;
  state.iterations = [];
  state.activeIter = null;
  updateListUI();
  updateViewer();
  renderIterBar();
}

function toggleCompareMode() {
  state.compareMode = !state.compareMode;
  state.selectedA = null;
  state.selectedB = null;
  state.viewMode = "original";
  state.iterations = [];
  state.activeIter = null;
  document.getElementById("cmp-toggle").classList.toggle("active", state.compareMode);
  const hint = document.getElementById("compare-hint");
  hint.style.display = state.compareMode ? "block" : "none";
  updateListUI();
  updateViewer();
}

function toggleRepairFilter() {
  document.getElementById("repair-toggle").classList.toggle("active");
  resetAndLoad();
}

// ==========================================================================
// Viewer rendering
// ==========================================================================
function updateViewer() {
  const toolbar  = document.getElementById("viewer-toolbar");
  const labels   = document.getElementById("pane-labels");
  const frames   = document.getElementById("frames");
  const A = state.selectedA;
  const B = state.selectedB;

  if (!A) {
    toolbar.innerHTML = "";
    labels.style.display = "none";
    frames.innerHTML = `
      <div id="empty-state" style="display:flex;flex-direction:column;
           align-items:center;justify-content:center;flex:1;gap:12px;color:var(--muted)">
        <div style="font-size:48px">&#x1f310;</div>
        <p>Select a record from the list to preview it here.<br>
        Records with repairs show <b style="color:var(--green)">before / after</b> automatically.<br>
        Records with refines show <b style="color:var(--cyan)">before / after</b> too.</p>
      </div>`;
    return;
  }

  const isDual = !!B;
  const mode   = state.viewMode;  // "original" | "repair" | "refine"

  // ── Toolbar ──
  const sa = A.score;
  const dims = ["rendering","visual_design","functionality","interaction","code_quality"];
  const dimLabels = ["Rend","Vis","Func","Int","Impl"];
  const dimPills = dims.map((d, i) => `
    <div class="dim-pill">
      <span class="dim-pill-name">${dimLabels[i]}</span>
      <span class="dim-pill-val" style="color:${scoreColor(sa[d]??0)}">${sa[d]??0}</span>
    </div>`).join("");

  // View mode buttons (only in non-compare mode)
  let viewBtns = "";
  if (!state.compareMode && (A.has_repair || A.has_refine)) {
    const origCls  = mode === "original" ? "active-original" : "";
    const repCls   = mode === "repair"   ? "active-repair"   : "";
    const refCls   = mode === "refine"   ? "active-refine"   : "";
    viewBtns = `<div class="view-mode-group">
      <button class="view-mode-btn ${origCls}" onclick="setViewMode('original')">Original</button>
      ${A.has_repair ? `<button class="view-mode-btn ${repCls}" onclick="setViewMode('repair')">Repair</button>` : ""}
      ${A.has_refine ? `<button class="view-mode-btn ${refCls}" onclick="setViewMode('refine')">Refine</button>` : ""}
    </div>`;
  }

  // Delta chip
  let deltaChip = "";
  if (mode === "repair" && A.has_repair) {
    deltaChip = `<span class="vt-chip chip-d" id="delta-chip">&Delta; &#x2026;</span>`;
  } else if (mode === "refine" && A.has_refine) {
    deltaChip = `<span class="vt-chip chip-d" id="delta-chip">&Delta; &#x2026;</span>`;
  }

  toolbar.innerHTML = `
    <span class="vt-uid">${A.uid}</span>
    <span class="tag tag-dataset" style="font-size:11px">${A.dataset}</span>
    <span class="vt-chip chip-a">${sa.total??0}</span>
    ${isDual && B.uid !== A.uid ? `<span class="vt-chip chip-b">${(B.score?.total??0)}</span>` : ""}
    ${deltaChip}
    ${viewBtns}
    <span class="vt-dims">${dimPills}</span>`;

  // ── Pane labels ──
  const iterN = state.activeIter;
  const iterData = iterN !== null ? state.iterations.find(it => it.iteration === iterN) : null;

  if (iterData && mode === "repair") {
    // Iteration-specific pane labels: left=original, right=after iter N
    labels.style.display = "flex";
    const origScore = sa.total ?? 0;
    const totalDelta = iterData.score_after - origScore;
    labels.innerHTML = `
      <div class="pane-label pane-label-a">
        Original&nbsp;<span class="pane-score-num">${origScore}</span>
        <span style="color:#64748b;font-size:11px">/ 100</span>
      </div>
      <div class="pane-label pane-label-refine">
        After iter ${iterN}: ${iterData.strategy}&nbsp;
        <span class="pane-score-num">${iterData.score_after}</span>
        <span style="color:#64748b;font-size:11px">/ 100</span>
        <span style="margin-left:8px;font-size:12px;color:${totalDelta > 0 ? 'var(--green)' : totalDelta < 0 ? 'var(--red)' : 'var(--muted)'}">\u0394${totalDelta >= 0 ? '+' : ''}${totalDelta} vs original</span>
      </div>`;
  } else if (isDual && A.uid === B.uid && (mode === "repair" || mode === "refine")) {
    labels.style.display = "flex";
    const aScore = sa.total ?? 0;
    const isRefine = mode === "refine";
    const labelClass = isRefine ? "pane-label-refine" : "pane-label-b";
    const labelText  = isRefine ? "After refine" : "After repair";
    labels.innerHTML = `
      <div class="pane-label pane-label-a">
        Before&nbsp;<span class="pane-score-num">${aScore}</span>
        <span style="color:#64748b;font-size:11px">/ 100</span>
      </div>
      <div class="pane-label ${labelClass}">
        ${labelText}&nbsp;<span class="pane-score-num" id="score-b-num">&#x2026;</span>
        <span style="color:#64748b;font-size:11px">/ 100</span>
      </div>`;
  } else if (isDual && A.uid !== B.uid) {
    labels.style.display = "flex";
    labels.innerHTML = `
      <div class="pane-label pane-label-a">
        <span>${A.uid.slice(0,8)}</span>
        <span class="pane-score-num">${sa.total??0}</span>
      </div>
      <div class="pane-label pane-label-b">
        <span>${B.uid.slice(0,8)}</span>
        <span class="pane-score-num">${B.score?.total??0}</span>
      </div>`;
  } else {
    labels.style.display = "none";
  }

  // ── Frames ──
  // When activeIter is set, always show dual panes
  const showDual = isDual || iterData;
  frames.innerHTML = `
    <div class="frame-wrap">
      <iframe id="frame-a" sandbox="allow-scripts allow-same-origin"></iframe>
      <div class="frame-loading" id="load-a">Loading&#x2026;</div>
    </div>
    ${showDual ? `
    <div class="frame-wrap">
      <iframe id="frame-b" sandbox="allow-scripts allow-same-origin"></iframe>
      <div class="frame-loading" id="load-b">Loading&#x2026;</div>
    </div>` : ""}`;

  // Determine what to load in each frame
  if (iterData && mode === "repair") {
    // Iteration-specific: left=original, right=iter_N_after
    setIframe("frame-a", htmlUrl(A.uid, "repair_orig"));
    setIframe("frame-b", htmlUrl(A.uid, `iter_${iterN}_after`));
  } else if (!isDual) {
    // Single: show original HTML
    setIframe("frame-a", htmlUrl(A.uid, "original"));
  } else if (A.uid === B.uid && mode === "repair") {
    // Same record: repair before/after
    setIframe("frame-a", htmlUrl(A.uid, "repair_orig"));
    setIframe("frame-b", htmlUrl(A.uid, "repaired"));
    fetchScoreInfo(A.uid, "repair");
  } else if (A.uid === B.uid && mode === "refine") {
    // Same record: refine before/after
    setIframe("frame-a", htmlUrl(A.uid, "refine_orig"));
    setIframe("frame-b", htmlUrl(A.uid, "refined"));
    fetchScoreInfo(A.uid, "refine");
  } else {
    // Two different records (compare mode)
    setIframe("frame-a", htmlUrl(A.uid, "original"));
    setIframe("frame-b", htmlUrl(B.uid, "original"));
  }
}

async function fetchScoreInfo(uid, type) {
  try {
    const endpoint = type === "refine" ? "refine_info" : "repair_info";
    const d = await apiFetch(`/api/${endpoint}?uid=${encodeURIComponent(uid)}`);
    const el = document.getElementById("score-b-num");
    if (el) el.textContent = d.score_after ?? "?";
    const dc = document.getElementById("delta-chip");
    if (dc && d.delta != null) dc.textContent = `\u0394${d.delta >= 0 ? "+" : ""}${d.delta}`;
  } catch {}
}

async function fetchIterations(uid) {
  state.iterations = [];
  state.activeIter = null;
  try {
    const d = await apiFetch(`/api/iterations?uid=${encodeURIComponent(uid)}`);
    state.iterations = d.iterations || [];
    renderIterBar();
  } catch {
    renderIterBar();
  }
}

function renderIterBar() {
  const bar = document.getElementById("iter-bar");
  if (!state.iterations.length) {
    bar.style.display = "none";
    return;
  }
  bar.style.display = "flex";
  const overallCls = state.activeIter === null ? "iter-active" : "";
  let html = `<button class="iter-btn ${overallCls}" onclick="setActiveIter(null)">Overall</button>`;
  for (const it of state.iterations) {
    const cls = state.activeIter === it.iteration ? "iter-active" : "";
    const dCls = it.delta > 0 ? "pos" : it.delta < 0 ? "neg" : "zero";
    const arrow = it.delta > 0 ? "\u25B2" : it.delta < 0 ? "\u25BC" : "";
    html += `<button class="iter-btn ${cls}" onclick="setActiveIter(${it.iteration})">` +
      `Iter ${it.iteration}: ${it.strategy} ` +
      `${it.score_before}\u2192${it.score_after} ` +
      `<span class="iter-delta ${dCls}">${arrow}${it.delta > 0 ? "+" : ""}${it.delta}</span></button>`;
  }
  bar.innerHTML = html;
}

function setActiveIter(n) {
  state.activeIter = n;
  renderIterBar();
  updateViewer();
}

// ==========================================================================
// Keyboard navigation
// ==========================================================================
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const recs = state.filtered;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    state.cursor = Math.max(0, Math.min(recs.length - 1,
      state.cursor + (e.key === "ArrowDown" ? 1 : -1)));
    updateListUI();
    const el = document.querySelector(`.rec-item[data-idx="${state.cursor}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }
  if (e.key === "Enter" && state.cursor >= 0) {
    selectRecord(state.cursor);
  }
  if (e.key === "c" || e.key === "C") {
    toggleCompareMode();
  }
  if (e.key === "Escape") {
    clearSelection();
    if (state.compareMode) toggleCompareMode();
  }
  // R key to toggle repair view, F key to toggle refine view
  if (e.key === "r" || e.key === "R") {
    if (state.selectedA?.has_repair) setViewMode(state.viewMode === "repair" ? "original" : "repair");
  }
  if (e.key === "f" || e.key === "F") {
    if (state.selectedA?.has_refine) setViewMode(state.viewMode === "refine" ? "original" : "refine");
  }
});

// ==========================================================================
// Utilities
// ==========================================================================
function escapeHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
          .replace(/"/g,"&quot;");
}

// ==========================================================================
// Init &#x2014; load datasets then first page
// ==========================================================================
(async () => {
  try {
    const { datasets } = await apiFetch("/api/meta");
    const sel = document.getElementById("ds-select");
    for (const ds of datasets) {
      sel.insertAdjacentHTML("beforeend",
        `<option value="${ds}">${ds}</option>`);
    }
  } catch {}
  resetAndLoad();
  updateViewer();
})();
</script>
</body>
</html>
"""
