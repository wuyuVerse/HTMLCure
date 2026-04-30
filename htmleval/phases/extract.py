"""
Phase 0: ExtractPhase — extract HTML from LLM response and write game.html.

Rejects responses where HTML cannot be extracted or is suspiciously short.
On success, injects a non-destructive keyboard/mouse helper script for
automated interaction during Phase 2 (RenderTest) and Phase 3 (AgentTest).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from htmleval.core.config import EvalConfig
from htmleval.core.context import EvalContext, PhaseResult
from htmleval.core.phase import Phase

logger = logging.getLogger("htmleval")


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

_SVG_WRAPPER_HEAD = (
    '<!DOCTYPE html><html><head><meta charset="utf-8">'
    '<style>body{margin:0;display:flex;justify-content:center;'
    'align-items:center;min-height:100vh;background:#fff}</style>'
    '</head><body>'
)
_SVG_WRAPPER_TAIL = "</body></html>"


def _wrap_svg(svg: str) -> str:
    return f"{_SVG_WRAPPER_HEAD}{svg}{_SVG_WRAPPER_TAIL}"


def _looks_like_html(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "<!doctype html",
            "<html",
            "<head",
            "<body",
            "```html",
        )
    )


def _normalize_partial_candidate(code: str) -> str:
    """Best-effort recovery for truncated HTML blocks.

    We only use this in evaluation mode, not benchmark generation validation.
    The goal is to recover the intended page instead of accidentally extracting
    an unrelated inline SVG fragment from the middle of a truncated response.
    """
    code = re.sub(r"\s*```+\s*$", "", code.strip())
    lowered = code.lower()

    if lowered.startswith("<svg"):
        return _wrap_svg(code)

    if "<body" in lowered and "</body>" not in lowered:
        code += "\n</body>"
    if "<html" in lowered and "</html>" not in lowered:
        code += "\n</html>"
    return code


def _dedupe_candidates(candidates: List[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def extract_html(response_text: str, *, allow_partial: bool = True) -> Optional[str]:
    """Extract HTML from an LLM response.

    In strict mode (allow_partial=False), only fully-closed HTML/SVG blocks are
    accepted. This is used by benchmark generation/checkpoint loading so
    truncated responses are retried instead of cached as successes.

    In recovery mode (allow_partial=True), we also salvage unclosed ```html
    fences or raw HTML tails by auto-closing </body></html>. Critically, when a
    response already contains HTML markers, we do NOT fall back to extracting an
    arbitrary inline <svg> from the middle of the page — that previously turned
    truncated full pages into tiny standalone SVG documents.
    """
    text = response_text or ""
    lowered = text.lower()
    candidates: List[str] = []

    # 1) Closed ```html ... ``` blocks
    for m in re.finditer(r"```(?:html)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE):
        code = m.group(1).strip()
        if "<html" in code.lower() or "<!doctype" in code.lower():
            candidates.append(code)

    # 1b) Truncated ```html ... EOF blocks (only in recovery mode)
    if allow_partial:
        for m in re.finditer(r"```html\s*\n", text, re.IGNORECASE):
            tail = text[m.end():]
            if "```" in tail:
                continue
            if _looks_like_html(tail):
                candidates.append(_normalize_partial_candidate(tail))

    # 2) Closed ```svg``` / ```xml``` blocks
    for m in re.finditer(r"```(?:svg|xml)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE):
        code = m.group(1).strip()
        if "<svg" in code.lower():
            candidates.append(_wrap_svg(code) if "<html" not in code.lower() else code)

    # 3) Complete raw <!DOCTYPE html> ... </html>
    for m in re.finditer(
        r"(<!DOCTYPE\s+html[^>]*>.*?</html>)",
        text,
        re.DOTALL | re.IGNORECASE,
    ):
        candidates.append(m.group(1).strip())

    # 4) Complete raw <html> ... </html>
    if not candidates:
        for m in re.finditer(
            r"(<html[^>]*>.*?</html>)",
            text,
            re.DOTALL | re.IGNORECASE,
        ):
            candidates.append(m.group(1).strip())

    # 5) Truncated raw HTML tail (only in recovery mode)
    if allow_partial:
        if not candidates:
            doctype = re.search(r"<!DOCTYPE\s+html[^>]*>", text, re.IGNORECASE)
            html_tag = re.search(r"<html[^>]*>", text, re.IGNORECASE)
            start = None
            if doctype:
                start = doctype.start()
            elif html_tag:
                start = html_tag.start()
            if start is not None:
                tail = text[start:]
                if "</html>" not in tail.lower():
                    candidates.append(_normalize_partial_candidate(tail))

    # 6) Standalone raw SVG fallback — only when the response does NOT already
    # look like HTML. This avoids corrupting truncated full pages into tiny SVGs.
    if not candidates and not _looks_like_html(text):
        for m in re.finditer(
            r"(<svg[^>]*>.*?</svg>)",
            text,
            re.DOTALL | re.IGNORECASE,
        ):
            candidates.append(_wrap_svg(m.group(1).strip()))

    candidates = _dedupe_candidates(candidates)
    if candidates:
        return max(candidates, key=len)

    # 7) Last resort: entire response itself looks like HTML/SVG
    stripped = text.strip()
    if stripped.lower().startswith(("<!doctype", "<html")):
        if allow_partial:
            return _normalize_partial_candidate(stripped)
        if "</html>" in stripped.lower():
            return stripped
    if stripped.lower().startswith("<svg") and not _looks_like_html(text):
        return _wrap_svg(stripped)

    return None


def extract_complete_html(response_text: str) -> Optional[str]:
    """Strict extraction used by benchmark generation/checkpoint validation."""
    return extract_html(response_text, allow_partial=False)


# ---------------------------------------------------------------------------
# Interaction helper script injection
# ---------------------------------------------------------------------------

_INTERACTION_HELPER = """
<script data-eval-helper="interaction">
// HTMLEval interaction helper — injected for automated testing.
// NON-DESTRUCTIVE: only adds standard focus behaviour and fallback APIs.
// Safe to leave in production HTML (does not alter game logic).
(function() {
    'use strict';

    // 1. Make canvas elements focusable
    function setupCanvas() {
        document.querySelectorAll('canvas').forEach(function(c) {
            if (!c.hasAttribute('tabindex')) c.setAttribute('tabindex', '0');
            c.style.outline = 'none';
        });
    }

    // 2. Auto-focus canvas on click
    document.addEventListener('click', function(e) {
        var canvas = e.target.closest ? e.target.closest('canvas') : null;
        if (!canvas) canvas = document.querySelector('canvas');
        if (canvas && document.activeElement !== canvas) canvas.focus();
    }, true);

    // 3. Auto-focus canvas after load
    function autoFocus() {
        var c = document.querySelector('canvas');
        if (c) c.focus();
    }
    if (document.readyState === 'complete') {
        setTimeout(autoFocus, 500);
    } else {
        window.addEventListener('load', function() { setTimeout(autoFocus, 500); });
    }

    // 4. window.__pressKey(key, durationMs) — single key press + release
    window.__pressKey = function(key, durationMs) {
        durationMs = durationMs || 100;
        var opts = {key: key, code: key, bubbles: true, cancelable: true};
        var targets = [window, document];
        var c = document.querySelector('canvas');
        if (c) targets.push(c);
        targets.forEach(function(t) { t.dispatchEvent(new KeyboardEvent('keydown', opts)); });
        setTimeout(function() {
            targets.forEach(function(t) { t.dispatchEvent(new KeyboardEvent('keyup', opts)); });
        }, durationMs);
    };

    // 5. window.__pressKeys(key, count, intervalMs) — repeated key presses
    window.__pressKeys = function(key, count, intervalMs) {
        count = count || 5; intervalMs = intervalMs || 100;
        for (var i = 0; i < count; i++) {
            (function(idx) {
                setTimeout(function() { window.__pressKey(key, intervalMs * 0.8); }, idx * intervalMs);
            })(i);
        }
    };

    // 6. window.__holdKey(key, durationMs) — hold a key for sustained effect
    window.__holdKey = function(key, durationMs) { window.__pressKey(key, durationMs || 1000); };

    // 7. window.__holdKeys(keyArray, durationMs) — hold multiple keys simultaneously
    window.__holdKeys = function(keyArray, durationMs) {
        durationMs = durationMs || 1000;
        var targets = [window, document];
        var c = document.querySelector('canvas');
        if (c) targets.push(c);
        keyArray.forEach(function(key) {
            var opts = {key: key, code: key, bubbles: true, cancelable: true};
            targets.forEach(function(t) { t.dispatchEvent(new KeyboardEvent('keydown', opts)); });
        });
        setTimeout(function() {
            keyArray.forEach(function(key) {
                var opts = {key: key, code: key, bubbles: true, cancelable: true};
                targets.forEach(function(t) { t.dispatchEvent(new KeyboardEvent('keyup', opts)); });
            });
        }, durationMs);
    };

    // 8. window.__clickAt(canvasX, canvasY) — click at canvas coordinates
    window.__clickAt = function(canvasX, canvasY) {
        var canvas = document.querySelector('canvas');
        if (!canvas) return;
        var rect = canvas.getBoundingClientRect();
        var scaleX = rect.width / (canvas.width || rect.width);
        var scaleY = rect.height / (canvas.height || rect.height);
        var cx = rect.left + canvasX * scaleX, cy = rect.top + canvasY * scaleY;
        var opts = {clientX: cx, clientY: cy, bubbles: true, cancelable: true};
        canvas.dispatchEvent(new MouseEvent('mousedown', opts));
        canvas.dispatchEvent(new MouseEvent('mouseup', opts));
        canvas.dispatchEvent(new MouseEvent('click', opts));
    };

    // 9. window.__drag(x1,y1,x2,y2,durationMs) — smooth drag in canvas coordinates
    window.__drag = function(x1, y1, x2, y2, durationMs) {
        durationMs = durationMs || 300;
        var canvas = document.querySelector('canvas');
        if (!canvas) return;
        var rect = canvas.getBoundingClientRect();
        var scaleX = rect.width / (canvas.width || rect.width);
        var scaleY = rect.height / (canvas.height || rect.height);
        function toClient(cx, cy) {
            return {clientX: rect.left + cx * scaleX, clientY: rect.top + cy * scaleY};
        }
        var start = toClient(x1, y1), end = toClient(x2, y2);
        canvas.dispatchEvent(new MouseEvent('mousedown', Object.assign({bubbles:true,cancelable:true}, start)));
        for (var i = 1; i <= 10; i++) {
            (function(step) {
                setTimeout(function() {
                    var t = step / 10;
                    window.dispatchEvent(new MouseEvent('mousemove', {
                        clientX: start.clientX + (end.clientX - start.clientX) * t,
                        clientY: start.clientY + (end.clientY - start.clientY) * t,
                        bubbles: true, cancelable: true
                    }));
                }, (durationMs / 10) * step);
            })(i);
        }
        setTimeout(function() {
            window.dispatchEvent(new MouseEvent('mouseup', Object.assign({bubbles:true,cancelable:true}, end)));
        }, durationMs + 50);
    };

    // 10. window.__probe — objective game-state evidence for automated testing
    window.__probe = (function() {
        var keysReceived = {};
        var errors = [];

        // ── Interaction latency measurement ──────────────────────────
        // Tracks time from action dispatch to first DOM/visual change.
        // MutationObserver catches DOM changes; rAF catches canvas redraws.
        var interactionTimings = [];  // [{label, actionTime, responseTime, latencyMs}]
        var _pendingAction = null;
        var _domChanged = false;

        // MutationObserver: detect DOM mutations after an action
        var _latencyObs = window.MutationObserver && new MutationObserver(function() {
            if (_pendingAction && !_domChanged) {
                _domChanged = true;
                var now = performance.now();
                var latency = now - _pendingAction.time;
                interactionTimings.push({
                    label: _pendingAction.label,
                    actionTime: Math.round(_pendingAction.time),
                    responseTime: Math.round(now),
                    latencyMs: Math.round(latency),
                    source: 'dom'
                });
            }
        });
        if (_latencyObs) {
            _latencyObs.observe(document.body || document.documentElement, {
                childList: true, subtree: true, attributes: true, characterData: true
            });
        }

        // Track every keydown (safe — read-only listener, game still receives all events)
        document.addEventListener('keydown', function(e) {
            keysReceived[e.key] = (keysReceived[e.key] || 0) + 1;
        }, true);

        // Capture uncaught JS errors
        window.addEventListener('error', function(e) {
            errors.push(e.message || String(e));
        });

        return {
            keysReceived: keysReceived,
            errors: errors,
            interactionTimings: interactionTimings,

            // Mark start of an interaction (called by test harness before click/key)
            markAction: function(label) {
                _domChanged = false;
                _pendingAction = { label: label, time: performance.now() };
                // Also try rAF-based detection for canvas changes
                // Note: For canvas games with continuous animation, rAF fires every
                // frame regardless of user input, so rAF-source timings may report
                // artificially low latency. DOM-source timings are more reliable.
                var self = this;
                var startTime = _pendingAction.time;
                var checkCount = 0;
                var hasCanvas = !!document.querySelector('canvas');
                // Skip rAF detection for continuous-animation pages (too noisy)
                if (!hasCanvas) {
                (function checkCanvas() {
                    if (checkCount++ > 30) return; // stop after ~500ms
                    requestAnimationFrame(function() {
                        // Canvas changes won't trigger MutationObserver.
                        // If we reach here and DOM hasn't changed, record rAF timing
                        // as a fallback (canvas likely repainted).
                        if (_pendingAction && _pendingAction.label === label && !_domChanged) {
                            // Only record on 2nd+ rAF (first rAF is often the same frame)
                            if (checkCount > 1) {
                                var now = performance.now();
                                interactionTimings.push({
                                    label: label,
                                    actionTime: Math.round(startTime),
                                    responseTime: Math.round(now),
                                    latencyMs: Math.round(now - startTime),
                                    source: 'raf'
                                });
                                _pendingAction = null;
                            } else {
                                checkCanvas();
                            }
                        }
                    });
                })();
                } // end skip rAF for canvas
                // Auto-expire after 2s
                setTimeout(function() {
                    if (_pendingAction && _pendingAction.label === label) {
                        interactionTimings.push({
                            label: label,
                            actionTime: Math.round(startTime),
                            responseTime: -1,
                            latencyMs: -1,
                            source: 'timeout'
                        });
                        _pendingAction = null;
                    }
                }, 2000);
            },

            snapshot: function() {
                var gameVars = {};
                var names = ['score','Score','SCORE','points','Points','lives','Lives',
                             'health','Health','level','Level','gameScore','playerScore',
                             'game','player'];
                names.forEach(function(k) {
                    if (window[k] !== undefined && typeof window[k] !== 'function') {
                        try {
                            var v = window[k];
                            if (v !== null && typeof v === 'object') {
                                var obj = {};
                                Object.keys(v).slice(0, 10).forEach(function(p) {
                                    if (typeof v[p] !== 'function') obj[p] = v[p];
                                });
                                gameVars[k] = obj;
                            } else {
                                gameVars[k] = v;
                            }
                        } catch(ignore) {}
                    }
                });
                // Canvas content analysis
                var canvasInfo = (function() {
                    var c = document.querySelector('canvas');
                    if (!c) return null;
                    try {
                        var ctx2d = c.getContext('2d');
                        if (ctx2d) {
                            var w = Math.min(c.width, 100), h = Math.min(c.height, 100);
                            if (w > 0 && h > 0) {
                                var data = ctx2d.getImageData(0, 0, w, h).data;
                                var nonZero = 0;
                                for (var ci = 0; ci < data.length; ci += 16) { // sample every 4th pixel
                                    if (data[ci] || data[ci+1] || data[ci+2]) nonZero++;
                                }
                                var total = data.length / 16;
                                return {type:'2d', w:c.width, h:c.height,
                                        hasContent: nonZero > total * 0.01,
                                        fillRatio: (nonZero / total).toFixed(3)};
                            }
                            return {type:'2d', w:c.width, h:c.height, hasContent:false, fillRatio:'0'};
                        }
                    } catch(e) {
                        // WebGL or cross-origin tainted canvas
                        try {
                            var gl = c.getContext('webgl') || c.getContext('webgl2');
                            if (gl) return {type:'webgl', w:c.width, h:c.height, hasContent:true};
                        } catch(e2) {}
                        return {type:'unknown', tainted:true};
                    }
                    return null;
                })();

                // Audio detection
                var audioInfo = (function() {
                    var elems = document.querySelectorAll('audio, video');
                    return {
                        audioElements: elems.length,
                        hasAudioContext: !!(window.AudioContext || window.webkitAudioContext),
                        mediaSources: Array.from(elems).slice(0, 3).map(function(a) {
                            return {tag: a.tagName, src: (a.src||'').substring(0,80), paused: a.paused};
                        })
                    };
                })();

                // DOM semantic inventory — objective element counts
                var domInventory = (function() {
                    var q = function(sel) { try { return document.querySelectorAll(sel).length; } catch(e) { return 0; } };
                    var vis = function(sel) {
                        try {
                            return Array.from(document.querySelectorAll(sel)).filter(function(el) {
                                var r = el.getBoundingClientRect();
                                return r.width > 0 && r.height > 0;
                            }).length;
                        } catch(e) { return 0; }
                    };
                    return {
                        buttons: vis('button, [role="button"], input[type="button"], input[type="submit"]'),
                        links: vis('a[href]'),
                        textInputs: vis('input[type="text"], input[type="email"], input[type="search"], input[type="number"], input:not([type]), textarea'),
                        selects: vis('select'),
                        images: q('img'),
                        imagesLoaded: Array.from(document.querySelectorAll('img')).filter(function(i) {
                            return i.complete && i.naturalWidth > 0;
                        }).length,
                        headings: q('h1, h2, h3, h4, h5, h6'),
                        canvas: q('canvas'),
                        tables: q('table'),
                        forms: q('form'),
                        eventHandlers: (function() {
                            var c = 0;
                            ['onclick','onchange','oninput','onsubmit','onkeydown','onmousedown'].forEach(function(a) {
                                c += q('[' + a + ']');
                            });
                            return c;
                        })()
                    };
                })();

                // Visible text extraction — structured page text for task verification
                var visibleText = (function() {
                    try {
                        var parts = [];
                        // Headings
                        document.querySelectorAll('h1,h2,h3').forEach(function(h) {
                            var t = (h.innerText || '').trim();
                            if (t) parts.push('[' + h.tagName + '] ' + t.substring(0, 100));
                        });
                        // Button labels
                        var btns = document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]');
                        var bl = Array.from(btns).map(function(b){return (b.innerText||b.value||'').trim();}).filter(Boolean).slice(0,8);
                        if (bl.length) parts.push('[Buttons] ' + bl.join(', '));
                        // Status/error messages
                        document.querySelectorAll('.error, .warning, .success, .message, [role="alert"], [role="status"]').forEach(function(m) {
                            var t = (m.innerText || '').trim();
                            if (t) parts.push('[Status] ' + t.substring(0, 100));
                        });
                        // Body text (first 500 chars, deduped from above)
                        var body = (document.body.innerText || '').substring(0, 500).replace(/\\s+/g, ' ').trim();
                        if (body) parts.push('[Body] ' + body);
                        return parts.join('\\n').substring(0, 2000);
                    } catch(e) { return ''; }
                })();

                return JSON.stringify({
                    keysReceived: keysReceived,
                    errors: errors.slice(0, 5),
                    gameVars: gameVars,
                    interactionTimings: interactionTimings.slice(0, 20),
                    canvasInfo: canvasInfo,
                    audioInfo: audioInfo,
                    domInventory: domInventory,
                    visibleText: visibleText
                });
            }
        };
    })();

    setupCanvas();
    var obs = window.MutationObserver && new MutationObserver(function() { setupCanvas(); });
    if (obs) obs.observe(document.body || document.documentElement, {childList:true, subtree:true});
})();
</script>
"""


def inject_helper(html_code: str) -> str:
    """Inject the interaction helper script before </body> (or at the end)."""
    idx = html_code.lower().rfind("</body>")
    if idx != -1:
        return html_code[:idx] + _INTERACTION_HELPER + html_code[idx:]
    return html_code + _INTERACTION_HELPER


# ---------------------------------------------------------------------------
# Phase implementation
# ---------------------------------------------------------------------------

class ExtractPhase(Phase):
    """Phase 0 — extract HTML from the LLM response, write game.html.

    Pipeline stops if extraction fails or HTML is shorter than
    config.filter.min_html_size.
    """

    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self._min_size = config.filter.min_html_size

    @property
    def name(self) -> str:
        return "extract"

    async def execute(self, ctx: EvalContext) -> PhaseResult:
        complete_html = extract_complete_html(ctx.response)
        html = complete_html or extract_html(ctx.response, allow_partial=True)

        if html is None:
            response_text = ctx.response or ""
            response_preview = re.sub(r"\s+", " ", response_text[:160]).strip()
            if not response_text.strip():
                logger.warning("[extract] %s: empty response", ctx.game_id)
                error = "HTML extraction failed — empty response"
            else:
                logger.warning(
                    "[extract] %s: no HTML found in non-empty response (len=%d, prefix=%r)",
                    ctx.game_id,
                    len(response_text),
                    response_preview,
                )
                error = "HTML extraction failed — no valid HTML found in non-empty response"
            return PhaseResult(
                phase_name=self.name,
                success=False,
                errors=[error],
                data={
                    "response_size": len(response_text),
                    "response_preview": response_preview,
                },
            )

        if len(html) < self._min_size:
            logger.warning(
                "[extract] %s: HTML too short (%d < %d chars)",
                ctx.game_id, len(html), self._min_size,
            )
            return PhaseResult(
                phase_name=self.name,
                success=False,
                errors=[f"HTML too short ({len(html)} chars, minimum {self._min_size})"],
                data={"html_size": len(html)},
            )

        if complete_html is None:
            logger.warning(
                "[extract] %s: recovered partial HTML from truncated response",
                ctx.game_id,
            )

        ctx.html_code = html

        if ctx.output_dir is not None:
            output_dir = Path(ctx.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / "game.html"
            html_path.write_text(inject_helper(html), encoding="utf-8")
            ctx.html_path = html_path
            logger.info("[extract] %s: wrote %d chars → %s", ctx.game_id, len(html), html_path)

        return PhaseResult(
            phase_name=self.name,
            success=True,
            data={
                "html_size": len(html),
                "html_recovered_partial": complete_html is None,
            },
        )

    def should_stop_pipeline(self, result: PhaseResult, ctx: EvalContext) -> bool:
        return not result.success
