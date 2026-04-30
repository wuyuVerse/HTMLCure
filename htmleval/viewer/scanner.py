"""
ResultsScanner — indexes eval_results/ and locates HTML files.

Scans all results.jsonl files, builds an in-memory index of records,
and resolves HTML file paths for both scored originals and repaired versions.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RecordMeta:
    uid:         str
    dataset:     str
    line_number: int
    score:       Dict      # {total, rendering, visual_design, ...}
    eval_status: str       # "completed" | "failed"
    has_html:    bool      # original HTML exists on disk
    has_repair:  bool      # repaired HTML exists (from repair test)
    query:       str = ""  # truncated task description
    repair_delta: int = 0  # best_score - original if repaired
    has_refine:  bool = False  # refined HTML exists (from refine pipeline)
    refine_delta: int = 0     # best_score - original if refined
    has_improve: bool = False  # improve traces exist (iteration history)

    @property
    def total_score(self) -> int:
        return self.score.get("total", 0)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class ResultsScanner:
    """
    Scans an eval_results directory and builds a queryable in-memory record index.

    Thread-safe: all mutations are protected by _lock.
    Call scanner.load() once at startup; subsequent calls are no-ops.
    """

    DATASET_NAMES = [
        "spring", "html_data", "query", "html_query",
        "html", "three", "game_html",
    ]

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.reports_dir = self.results_dir / "reports"
        self._records: List[RecordMeta] = []
        self._by_uid:  Dict[str, RecordMeta] = {}
        self._lock = threading.Lock()
        self._loaded = False
        # repair_traces index: {record_id -> {final_score, improvement}}
        self._repair_traces: Dict[str, dict] = {}
        # refine_traces index: {record_id -> {final_score, improvement}}
        self._refine_traces: Dict[str, dict] = {}
        # improve_traces index: {record_id -> (file_path, byte_offset)}
        self._improve_index: Dict[str, tuple[Path, int]] = {}
        # improve_traces summary: {record_id -> {"improvement": int, "final_score": int}}
        self._improve_summary: Dict[str, dict] = {}
        self._load_repair_traces()
        self._load_refine_traces()
        self._load_improve_index()

    # ── Public API ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all results from disk. Safe to call multiple times."""
        with self._lock:
            if self._loaded:
                return
            self._scan_all()
            self._loaded = True

    @property
    def datasets(self) -> List[str]:
        seen: dict[str, int] = {}
        for r in self._records:
            seen[r.dataset] = seen.get(r.dataset, 0) + 1
        return [d for d in self.DATASET_NAMES if d in seen]

    def query(
        self,
        *,
        dataset:      str = "all",
        min_score:    int = 0,
        max_score:    int = 100,
        has_repair:   bool = False,
        has_refine:   bool = False,
        has_html_only: bool = True,   # only show records with HTML on disk
        sort:         str = "score_asc",    # score_asc|score_desc|uid_asc|dataset_asc
        page:         int = 0,
        limit:        int = 80,
    ) -> tuple[List[RecordMeta], int]:
        """Return (page_records, total_matching)."""
        recs = self._records

        # Filter
        if has_html_only:
            recs = [r for r in recs if r.has_html]
        if dataset != "all":
            recs = [r for r in recs if r.dataset == dataset]
        recs = [r for r in recs if min_score <= r.total_score <= max_score]
        if has_repair:
            recs = [r for r in recs if r.has_repair]
        if has_refine:
            recs = [r for r in recs if r.has_refine]

        # Sort
        key_fn = {
            "score_asc":   lambda r: r.total_score,
            "score_desc":  lambda r: -r.total_score,
            "uid_asc":     lambda r: r.uid,
            "dataset_asc": lambda r: (r.dataset, r.total_score),
        }.get(sort, lambda r: r.total_score)
        recs = sorted(recs, key=key_fn)

        total = len(recs)
        start = page * limit
        return recs[start: start + limit], total

    def get_by_uid(self, uid: str) -> Optional[RecordMeta]:
        return self._by_uid.get(uid)

    def html_path(self, uid: str, kind: str = "original") -> Optional[Path]:
        """
        Resolve HTML file path for a record.

        kind:
          "original"    — scored HTML from eval pipeline
          "repaired"    — best repaired version from repair engine
          "repair_orig" — re-eval snapshot used as repair baseline
          "refined"     — best refined version from refine pipeline
          "refine_orig" — original HTML used as refine baseline
        """
        rec = self._by_uid.get(uid)
        if rec is None:
            return None

        if kind == "original":
            p = self.reports_dir / f"{rec.dataset}_{rec.line_number}_default" / "game.html"
            return p if p.exists() else None

        if kind == "repair_orig":
            # test_repair.py style
            p = self.reports_dir / f"repairtest_{uid}_test" / "game.html"
            if p.exists():
                return p
            # full pipeline: use the original scored HTML as baseline
            p = self.reports_dir / f"{rec.dataset}_{rec.line_number}_default" / "game.html"
            return p if p.exists() else None

        if kind == "repaired":
            # test_repair.py style: repairtest_{uid}_repair_N_repair
            for tag in ("repair_2_repair", "repair_1_repair"):
                p = self.reports_dir / f"repairtest_{uid}_{tag}" / "game.html"
                if p.exists():
                    return p
            # full pipeline style: {dataset}_{uid}_repair_N_repair
            # Only return if this record is actually a repair (not refine)
            record_id = f"{rec.dataset}_{uid}"
            if record_id in self._repair_traces:
                for tag in ("repair_3_repair", "repair_2_repair", "repair_1_repair"):
                    p = self.reports_dir / f"{record_id}_{tag}" / "game.html"
                    if p.exists():
                        return p
            return None

        if kind == "refine_orig":
            # Refine baseline is the original scored HTML (Tier A, already 80+)
            p = self.reports_dir / f"{rec.dataset}_{rec.line_number}_default" / "game.html"
            return p if p.exists() else None

        if kind == "refined":
            # Refine uses same dir pattern: {record_id}_repair_N_repair
            # Distinguish from repair via refine_traces index
            record_id = f"{rec.dataset}_{uid}"
            if record_id in self._refine_traces:
                for tag in ("repair_3_repair", "repair_2_repair", "repair_1_repair"):
                    p = self.reports_dir / f"{record_id}_{tag}" / "game.html"
                    if p.exists():
                        return p
            return None

        return None

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_traces_file(filepath: Path) -> Dict[str, dict]:
        """
        Load record_id/final_score/improvement from a traces JSONL file.

        First checks for a lightweight cache file ({filepath}.index) which is a
        TSV of record_id, final_score, improvement. If missing or stale, builds
        it by streaming the JSONL line-by-line, only parsing the first ~500
        chars of each line (the big best_html field comes much later).
        """
        import re

        result: Dict[str, dict] = {}
        if not filepath.exists():
            return result

        cache = Path(str(filepath) + ".index")

        # Use cache if it exists and is newer than the traces file
        if cache.exists() and cache.stat().st_mtime >= filepath.stat().st_mtime:
            try:
                for line in cache.read_text().splitlines():
                    parts = line.split("\t")
                    if len(parts) == 3:
                        result[parts[0]] = {
                            "final_score": int(parts[1]),
                            "improvement": int(parts[2]),
                        }
                return result
            except Exception:
                pass  # Fall through to rebuild

        # Build index by reading only first 5000 bytes of each line.
        # 500 was too small: long Chinese queries in UTF-8 can push
        # final_score/improvement fields past the 500-byte window.
        _RE_RID = re.compile(r'"record_id"\s*:\s*"([^"]+)"')
        _RE_FS  = re.compile(r'"final_score"\s*:\s*(\d+)')
        _RE_IMP = re.compile(r'"improvement"\s*:\s*(-?\d+)')

        try:
            with open(filepath, "rb") as fh:
                for raw_line in fh:
                    head = raw_line[:5000].decode("utf-8", errors="replace")
                    m_rid = _RE_RID.search(head)
                    if not m_rid:
                        continue
                    rid = m_rid.group(1)
                    m_fs = _RE_FS.search(head)
                    m_imp = _RE_IMP.search(head)
                    result[rid] = {
                        "final_score": int(m_fs.group(1)) if m_fs else 0,
                        "improvement": int(m_imp.group(1)) if m_imp else 0,
                    }
        except Exception:
            pass

        # Write cache for next time
        try:
            lines = [f"{rid}\t{v['final_score']}\t{v['improvement']}" for rid, v in result.items()]
            cache.write_text("\n".join(lines))
        except Exception:
            pass

        return result

    def _load_repair_traces(self) -> None:
        """Load pipeline_output/repair_traces.jsonl for fast delta lookup."""
        candidates = [
            self.results_dir.parent / "pipeline_output" / "repair_traces.jsonl",
            Path("pipeline_output") / "repair_traces.jsonl",
        ]
        for p in candidates:
            if p.exists():
                self._repair_traces = self._load_traces_file(p)
                return

    def _load_refine_traces(self) -> None:
        """Load pipeline_output/refine_traces.jsonl for fast delta lookup."""
        candidates = [
            self.results_dir.parent / "pipeline_output" / "refine_traces.jsonl",
            Path("pipeline_output") / "refine_traces.jsonl",
        ]
        for p in candidates:
            if p.exists():
                self._refine_traces = self._load_traces_file(p)
                return

    def _load_improve_index(self) -> None:
        """
        Build byte-offset index for all improve_traces*.jsonl files.

        Only record_id, file offset, and summary stats are stored in memory.
        Full JSON (30KB+ per line) is read on demand via get_iterations().
        """
        import re

        _RE_RID = re.compile(r'"record_id"\s*:\s*"([^"]+)"')
        _RE_FS  = re.compile(r'"final_score"\s*:\s*(\d+)')
        _RE_IMP = re.compile(r'"improvement"\s*:\s*(-?\d+)')
        candidates = [
            self.results_dir.parent / "pipeline_output",
            Path("pipeline_output"),
        ]
        for output_dir in candidates:
            if not output_dir.is_dir():
                continue
            for p in sorted(output_dir.glob("improve_traces*.jsonl")):
                try:
                    with open(p, "rb") as fh:
                        while True:
                            offset = fh.tell()
                            raw = fh.readline()
                            if not raw:
                                break
                            head = raw[:5000].decode("utf-8", errors="replace")
                            m = _RE_RID.search(head)
                            if m:
                                rid = m.group(1)
                                if rid not in self._improve_index:
                                    self._improve_index[rid] = (p, offset)
                                    m_fs = _RE_FS.search(head)
                                    m_imp = _RE_IMP.search(head)
                                    self._improve_summary[rid] = {
                                        "final_score": int(m_fs.group(1)) if m_fs else 0,
                                        "improvement": int(m_imp.group(1)) if m_imp else 0,
                                    }
                except Exception:
                    continue
            if self._improve_index:
                return  # found in this candidate dir

    def get_iterations(self, uid: str, dataset: str) -> Optional[dict]:
        """
        Return iteration metadata (no HTML) for a record.

        Returns dict with top-level summary + iterations list, or None.
        """
        record_id = f"{dataset}_{uid}"
        entry = self._improve_index.get(record_id)
        if entry is None:
            return None

        filepath, offset = entry
        try:
            with open(filepath, "rb") as fh:
                fh.seek(offset)
                line = fh.readline()
                data = json.loads(line)
        except Exception:
            return None

        iterations = []
        for it in data.get("iterations", []):
            iterations.append({
                "iteration":    it.get("iteration", 0),
                "strategy":     it.get("strategy", ""),
                "score_before": it.get("score_before", 0),
                "score_after":  it.get("score_after", 0),
                "delta":        it.get("delta", 0),
                "n_candidates": it.get("n_candidates", 0),
                "elapsed_s":    round(it.get("elapsed_s", 0), 1),
                "success":      it.get("success", False),
            })

        return {
            "record_id":        data.get("record_id", record_id),
            "original_score":   data.get("original_score", 0),
            "final_score":      data.get("final_score", 0),
            "improvement":      data.get("improvement", 0),
            "evidence_quality": data.get("evidence_quality", ""),
            "converged":        data.get("converged", False),
            "iterations":       iterations,
        }

    def get_iteration_html(
        self, uid: str, dataset: str, iteration: int, kind: str
    ) -> Optional[str]:
        """
        Return html_before or html_after for a specific iteration.

        kind: "before" | "after"
        iteration: 1-based iteration number
        """
        record_id = f"{dataset}_{uid}"
        entry = self._improve_index.get(record_id)
        if entry is None:
            return None

        filepath, offset = entry
        try:
            with open(filepath, "rb") as fh:
                fh.seek(offset)
                line = fh.readline()
                data = json.loads(line)
        except Exception:
            return None

        for it in data.get("iterations", []):
            if it.get("iteration") == iteration:
                field = "html_before" if kind == "before" else "html_after"
                html = it.get(field)
                if html:
                    html = self._strip_markdown_wrapper(html)
                return html
        return None

    @staticmethod
    def _strip_markdown_wrapper(html: str) -> str:
        """Strip markdown description + ```html fences wrapping raw HTML."""
        import re
        # Pattern: optional text ... ```html\n<actual html>\n```
        m = re.search(r"```html?\s*\n", html)
        if m:
            html = html[m.end():]
            # Strip trailing ``` if present
            if html.rstrip().endswith("```"):
                html = html.rstrip()[:-3].rstrip()
        return html

    def get_improve_best_html(self, uid: str, dataset: str) -> Optional[str]:
        """Return the best_html from improve traces (final best version)."""
        return self._get_improve_field(uid, dataset, "best_html")

    def get_improve_original_html(self, uid: str, dataset: str) -> Optional[str]:
        """Return the original_html from improve traces."""
        return self._get_improve_field(uid, dataset, "original_html")

    def _get_improve_field(self, uid: str, dataset: str, field: str) -> Optional[str]:
        record_id = f"{dataset}_{uid}"
        entry = self._improve_index.get(record_id)
        if entry is None:
            return None

        filepath, offset = entry
        try:
            with open(filepath, "rb") as fh:
                fh.seek(offset)
                line = fh.readline()
                data = json.loads(line)
        except Exception:
            return None

        html = data.get(field)
        if html:
            html = self._strip_markdown_wrapper(html)
        return html

    def _read_report_score(self, report_md: Path) -> Optional[int]:
        """Parse the total score from a report.md file.  Returns None on failure."""
        import re
        try:
            text = report_md.read_text(errors="replace")
            m = re.search(r"\*\*Score\*\*\s*\|\s*\*\*(\d+)\s*/\s*100\*\*", text)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    def _scan_all(self) -> None:
        import os
        records: List[RecordMeta] = []
        by_uid:  Dict[str, RecordMeta] = {}

        # One readdir to get all report dir names — replaces O(N) Path.exists() calls.
        try:
            report_dirs: set = set(os.listdir(self.reports_dir))
        except OSError:
            report_dirs = set()

        for dataset in self.DATASET_NAMES:
            jl = self.results_dir / dataset / "results.jsonl"
            if not jl.exists():
                continue
            for row in self._load_results_meta(jl):
                rec = self._make_record(row, dataset, report_dirs)
                if rec:
                    records.append(rec)
                    by_uid[rec.uid] = rec

        self._records = records
        self._by_uid  = by_uid

    def _load_results_meta(self, jl: Path) -> List[dict]:
        """
        Return lightweight metadata rows for a results.jsonl file.

        Builds a .meta cache (JSON-lines of small fields only) next to jl on
        first load, then reuses it on subsequent loads.  This avoids re-reading
        1–2 GB JSONL files that embed full HTML in every record.
        """
        cache = Path(str(jl) + ".meta")

        if cache.exists() and cache.stat().st_mtime >= jl.stat().st_mtime:
            try:
                rows = []
                for line in cache.read_text(errors="replace").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
                return rows
            except Exception:
                pass  # Fall through to rebuild

        rows = []
        try:
            with open(jl, "r", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    uid     = r.get("_eval_uid") or r.get("uid", "")
                    if not uid:
                        continue
                    score   = r.get("score", {})
                    data    = r.get("data", {})
                    msgs    = data.get("messages", []) if isinstance(data, dict) else []
                    query   = next(
                        (m.get("content", "")[:200] for m in msgs if m.get("role") == "user"), ""
                    )
                    rows.append({
                        "uid":         uid,
                        "line_number": r.get("line_number", 0),
                        "score":       score,
                        "eval_status": r.get("eval_status", ""),
                        "query":       query,
                        "data_id":     data.get("data_id", "") if isinstance(data, dict) else "",
                    })
        except Exception:
            pass

        # Write cache
        try:
            cache.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
        except Exception:
            pass

        return rows

    def _make_record(self, row: dict, dataset: str, report_dirs: set) -> Optional[RecordMeta]:
        uid     = row.get("uid", "")
        line_no = row.get("line_number", 0)
        score   = row.get("score", {})
        status  = row.get("eval_status", "")
        query_text = row.get("query", "")
        if not uid:
            return None

        def dir_exists(name: str) -> bool:
            return name in report_dirs

        # Check HTML on disk
        # Primary: line_number based (game_html, spring, query, etc.)
        # Fallback: data_id based (html dataset uses MD5 hash as stable_id)
        has_html = dir_exists(f"{dataset}_{line_no}_default")
        if not has_html:
            data_id = row.get("data_id", "")
            if data_id:
                has_html = dir_exists(f"{dataset}_{data_id}_default")

        # Check repair
        has_repair   = False
        repair_delta = 0
        orig_score   = score.get("total", 0) if isinstance(score, dict) else 0

        # 1) test_repair.py style
        for tag in ("repair_2_repair", "repair_1_repair"):
            rp_dir = f"repairtest_{uid}_{tag}"
            if dir_exists(rp_dir):
                has_repair = True
                repair_score = self._read_report_score(self.reports_dir / rp_dir / "report.md")
                if repair_score is not None:
                    repair_delta = repair_score - orig_score
                break

        # 2) full pipeline style — only if record is in repair_traces
        # (refine also produces _repair_N_repair dirs with same naming)
        if not has_repair:
            record_id = f"{dataset}_{uid}"
            if record_id in self._repair_traces:
                for tag in ("repair_3_repair", "repair_2_repair", "repair_1_repair"):
                    rp_dir = f"{record_id}_{tag}"
                    if dir_exists(rp_dir):
                        has_repair = True
                        trace = self._repair_traces.get(record_id)
                        if trace and trace["improvement"] > 0:
                            repair_delta = trace["improvement"]
                        else:
                            repair_score = self._read_report_score(self.reports_dir / rp_dir / "report.md")
                            if repair_score is not None:
                                repair_delta = repair_score - orig_score
                        break

        # Check refine (disjoint from repair, trace-only detection)
        has_refine   = False
        refine_delta = 0
        refine_trace = self._refine_traces.get(f"{dataset}_{uid}")
        if refine_trace:
            has_refine   = True
            refine_delta = refine_trace.get("improvement", 0)

        # Check improve traces (iteration history)
        has_improve = f"{dataset}_{uid}" in self._improve_index
        # Improve traces also imply has_repair if not already set
        if has_improve and not has_repair:
            has_repair = True
            summary = self._improve_summary.get(f"{dataset}_{uid}", {})
            repair_delta = summary.get("improvement", 0)

        return RecordMeta(
            uid=uid,
            dataset=dataset,
            line_number=line_no,
            score=score,
            eval_status=status,
            has_html=has_html,
            has_repair=has_repair,
            query=query_text,
            repair_delta=repair_delta,
            has_refine=has_refine,
            refine_delta=refine_delta,
            has_improve=has_improve,
        )
