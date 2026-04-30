"""
ViewerServer — single-threaded HTTP server with JSON API.

Routes:
  GET /                          → SPA (app.html)
  GET /api/meta                  → {"datasets": [...]}
  GET /api/records?...           → paginated record list
  GET /api/repair_info?uid=...   → repair score info for a record
  GET /html/{uid}/{kind}         → serve HTML file directly (for iframes)
"""

from __future__ import annotations

import json
import mimetypes
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from htmleval.viewer._app import APP_HTML
from htmleval.viewer.scanner import ResultsScanner


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ViewerHandler(BaseHTTPRequestHandler):
    """HTTP handler — scanner is injected as a class attribute by ViewerServer."""

    scanner: ResultsScanner  # set by ViewerServer before use
    log_requests: bool = False

    # ── Routing ─────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"
        params = dict(urllib.parse.parse_qsl(parsed.query))

        try:
            if path in ("/", "/index.html"):
                self._serve_text(APP_HTML, "text/html")
            elif path == "/api/meta":
                self._api_meta()
            elif path == "/api/records":
                self._api_records(params)
            elif path == "/api/repair_info":
                self._api_repair_info(params)
            elif path == "/api/refine_info":
                self._api_refine_info(params)
            elif path == "/api/iterations":
                self._api_iterations(params)
            elif path.startswith("/html/"):
                self._serve_html(path)
            else:
                self._error(404, "Not found")
        except Exception as exc:
            self._error(500, str(exc))

    # ── API handlers ─────────────────────────────────────────────────────

    def _api_meta(self) -> None:
        self._serve_json({"datasets": self.scanner.datasets})

    def _api_records(self, params: dict) -> None:
        recs, total = self.scanner.query(
            dataset    = params.get("dataset", "all"),
            min_score  = int(params.get("min_score", 0)),
            max_score  = int(params.get("max_score", 100)),
            has_repair = params.get("has_repair", "0") == "1",
            has_refine = params.get("has_refine", "0") == "1",
            sort       = params.get("sort", "score_asc"),
            page       = int(params.get("page", 0)),
            limit      = int(params.get("limit", 80)),
        )
        payload = {
            "records": [_serialize_record(r) for r in recs],
            "total":   total,
            "page":    int(params.get("page", 0)),
        }
        self._serve_json(payload)

    def _api_repair_info(self, params: dict) -> None:
        uid = params.get("uid", "")
        rec = self.scanner.get_by_uid(uid)
        if not rec:
            self._error(404, f"uid not found: {uid}")
            return

        score_before = rec.total_score
        score_after  = (score_before + rec.repair_delta) if rec.has_repair else None
        delta        = rec.repair_delta if rec.has_repair and rec.repair_delta else None
        self._serve_json({
            "uid":          uid,
            "score_before": score_before,
            "score_after":  score_after,
            "delta":        delta,
            "has_repair":   rec.has_repair,
        })

    def _api_refine_info(self, params: dict) -> None:
        uid = params.get("uid", "")
        rec = self.scanner.get_by_uid(uid)
        if not rec:
            self._error(404, f"uid not found: {uid}")
            return

        score_before = rec.total_score
        score_after  = (score_before + rec.refine_delta) if rec.has_refine else None
        delta        = rec.refine_delta if rec.has_refine and rec.refine_delta else None
        self._serve_json({
            "uid":          uid,
            "score_before": score_before,
            "score_after":  score_after,
            "delta":        delta,
            "has_refine":   rec.has_refine,
        })

    def _api_iterations(self, params: dict) -> None:
        uid = params.get("uid", "")
        rec = self.scanner.get_by_uid(uid)
        if not rec:
            self._error(404, f"uid not found: {uid}")
            return
        data = self.scanner.get_iterations(uid, rec.dataset)
        if data is None:
            self._serve_json({"uid": uid, "iterations": []})
            return
        data["uid"] = uid
        self._serve_json(data)

    # ── HTML serving ────────────────────────────────────────────────────

    def _serve_html(self, path: str) -> None:
        # path format: /html/{uid}/{kind}
        # kind: original | repaired | repair_orig
        parts = path.lstrip("/").split("/")
        if len(parts) < 3:
            self._error(400, "Bad path")
            return
        uid, kind = parts[1], parts[2]

        # Handle iteration HTML: iter_N_before / iter_N_after
        import re as _re
        iter_match = _re.match(r"iter_(\d+)_(before|after)$", kind)
        if iter_match:
            iteration = int(iter_match.group(1))
            ba = iter_match.group(2)  # "before" or "after"
            rec = self.scanner.get_by_uid(uid)
            if rec is None:
                self._error(404, f"uid not found: {uid}")
                return
            html_str = self.scanner.get_iteration_html(uid, rec.dataset, iteration, ba)
            if html_str is None:
                msg = f"No iteration HTML&nbsp;(uid={uid}, iter={iteration}, {ba})"
                page = (
                    "<!DOCTYPE html><html><body style='margin:0;display:flex;align-items:center;"
                    "justify-content:center;height:100vh;background:#0d0f16;color:#64748b;"
                    "font-family:sans-serif;font-size:13px'>"
                    f"<span>{msg}</span></body></html>"
                )
                self._serve_text(page, "text/html")
                return
            self._respond(200, html_str.encode("utf-8"), "text/html; charset=utf-8")
            return

        html_path = self.scanner.html_path(uid, kind)
        if html_path is None:
            # Fallback: for "repaired" kind, try improve_traces best_html
            if kind == "repaired":
                rec = self.scanner.get_by_uid(uid)
                if rec and rec.has_improve:
                    html_str = self.scanner.get_improve_best_html(uid, rec.dataset)
                    if html_str:
                        self._respond(200, html_str.encode("utf-8"), "text/html; charset=utf-8")
                        return
            # Fallback: for "repair_orig" kind, try improve_traces original_html
            if kind == "repair_orig":
                rec = self.scanner.get_by_uid(uid)
                if rec and rec.has_improve:
                    html_str = self.scanner.get_improve_original_html(uid, rec.dataset)
                    if html_str:
                        self._respond(200, html_str.encode("utf-8"), "text/html; charset=utf-8")
                        return
            # Serve a friendly HTML page so the iframe shows a message, not a JSON error
            msg = f"No HTML available&nbsp;(uid={uid}, kind={kind})"
            page = (
                "<!DOCTYPE html><html><body style='margin:0;display:flex;align-items:center;"
                "justify-content:center;height:100vh;background:#0d0f16;color:#64748b;"
                "font-family:sans-serif;font-size:13px'>"
                f"<span>{msg}</span></body></html>"
            )
            self._serve_text(page, "text/html")
            return

        content = html_path.read_bytes()
        self._respond(200, content, "text/html; charset=utf-8")

    # ── Low-level helpers ────────────────────────────────────────────────

    def _serve_json(self, data: object) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self._respond(200, body, "application/json")

    def _serve_text(self, text: str, content_type: str = "text/plain") -> None:
        self._respond(200, text.encode("utf-8"), content_type + "; charset=utf-8")

    def _error(self, code: int, msg: str) -> None:
        body = json.dumps({"error": msg}).encode()
        self._respond(code, body, "application/json")

    def _respond(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:  # type: ignore[override]
        if self.log_requests:
            super().log_message(fmt, *args)


def _serialize_record(r) -> dict:
    return {
        "uid":         r.uid,
        "dataset":     r.dataset,
        "line_number": r.line_number,
        "score":       r.score,
        "eval_status": r.eval_status,
        "has_html":    r.has_html,
        "has_repair":  r.has_repair,
        "repair_delta": r.repair_delta,
        "has_refine":  r.has_refine,
        "refine_delta": r.refine_delta,
        "has_improve": r.has_improve,
        "query":       r.query,
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class ViewerServer:
    """
    Wraps ThreadingHTTPServer to inject the scanner and provide start/stop.

    Usage:
        server = ViewerServer(scanner, host="127.0.0.1", port=7860)
        server.start()          # non-blocking (background thread)
        server.serve_forever()  # blocking
    """

    def __init__(
        self,
        scanner: ResultsScanner,
        host: str = "127.0.0.1",
        port: int = 7860,
        log_requests: bool = False,
    ):
        self.host = host
        self.port = port

        # Inject scanner into handler class
        handler = type("_Handler", (ViewerHandler,), {
            "scanner":      scanner,
            "log_requests": log_requests,
        })
        self._httpd = ThreadingHTTPServer((host, port), handler)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """Start serving in a background daemon thread."""
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()

    def serve_forever(self) -> None:
        """Block the current thread, serving requests until interrupted."""
        self._httpd.serve_forever()

    def stop(self) -> None:
        self._httpd.shutdown()
