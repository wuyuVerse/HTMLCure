"""
RepairStrategy ABC + shared LLM call utilities.

Each strategy:
  1. Builds a targeted prompt (build_prompt)
  2. Calls the LLM via _call_llm (3 retries, 300s timeout)
     — optionally multimodal: screenshots passed as image_url parts
  3. Extracts clean HTML from the response (_extract_html)

Supports two output modes:
  - "rewrite" (default): LLM returns complete HTML file
  - "patch": LLM returns JSON patches [{old_str, new_str}], applied surgically
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional

from openai import AsyncOpenAI
from PIL import Image

from htmlrefine.data_pipeline.repair.core.diagnosis import Diagnosis
from htmlrefine.data_pipeline.repair.patcher import PatchError, apply_patches, validate_patched_html

logger = logging.getLogger("htmlrefine.repair")

_HTML_FENCE  = re.compile(r"```(?:html)?\s*\n(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_HTML_TAG    = re.compile(r"(<!DOCTYPE.*?</html>|<html.*?</html>)", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_FENCE  = re.compile(r"```(?:json)?\s*\n(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_MIN_HTML_LEN = 200

# Global LLM concurrency semaphore (shared by all repair LLM calls).
# Set via set_llm_semaphore() before starting repairs; None = no cap.
_LLM_SEM: Optional[asyncio.Semaphore] = None


def set_llm_semaphore(n: int) -> Optional[asyncio.Semaphore]:
    """Cap total concurrent LLM calls to n. Returns the created semaphore so callers
    can share the same instance with other LLM call sites (e.g. vision evaluator)."""
    global _LLM_SEM
    _LLM_SEM = asyncio.Semaphore(n) if n > 0 else None
    return _LLM_SEM


def _select_repair_frames(
    screenshots: List[str],
    frame_annotations: Optional[List[dict]],
    max_frames: int = 5,
) -> tuple[List[str], List[dict]]:
    """Pick the most informative frames for repair context.

    Priority: stable/idle frames, interaction results, then highest visual diff.
    Preserves temporal order in the output.
    """
    if not frame_annotations or len(screenshots) <= max_frames:
        return screenshots[:max_frames], (frame_annotations or [])[:max_frames]

    # Priority label prefixes (order matters)
    priority = [
        "stable", "idle_animation", "after_click", "keyboard",
        "canvas_click", "scroll_bottom", "hover", "gameplay",
    ]
    selected_indices: List[int] = []

    for label_prefix in priority:
        for i, fa in enumerate(frame_annotations):
            if i not in selected_indices and label_prefix in fa.get("label", ""):
                selected_indices.append(i)
                if len(selected_indices) >= max_frames:
                    break
        if len(selected_indices) >= max_frames:
            break

    # Fill remaining with highest diff_from_prev (most visually different)
    if len(selected_indices) < max_frames:
        remaining = [
            (fa.get("diff_from_prev", 0), i)
            for i, fa in enumerate(frame_annotations)
            if i not in selected_indices
        ]
        remaining.sort(reverse=True)
        for _, i in remaining:
            selected_indices.append(i)
            if len(selected_indices) >= max_frames:
                break

    selected_indices.sort()  # preserve temporal order
    sel_ss = [screenshots[i] for i in selected_indices if i < len(screenshots)]
    sel_fa = [frame_annotations[i] for i in selected_indices if i < len(frame_annotations)]
    return sel_ss, sel_fa


def _encode_screenshot(path: str, max_dim: int = 960) -> str:
    """Resize and base64-encode a screenshot for multimodal repair."""
    img = Image.open(path)
    if max(img.size) > max_dim:
        r = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * r), int(img.size[1] * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)   # JPEG: smaller payload, fine for repair context
    return base64.b64encode(buf.getvalue()).decode()


def _extract_html(text: str) -> Optional[str]:
    """Extract clean HTML from an LLM response (handles fences + bare HTML).

    Reasoning models (e.g. Kimi-K2.5) wrap thinking in <think>...</think> before
    the actual answer. Strip those blocks first so the fence/tag regexes match the
    real HTML and not fragments inside the thinking text.
    """
    if not text:
        return None

    # Strip <think>...</think> blocks (reasoning models like Kimi-K2.5)
    clean = _THINK_BLOCK.sub("", text).strip()
    if not clean:
        clean = text  # fallback: use original if stripping left nothing

    # 1. Fenced code block: try ALL fences, pick first valid one
    #    (search() only finds the first match, which may be a fragment in thinking text)
    for m in _HTML_FENCE.finditer(clean):
        candidate = m.group(1).strip()
        if len(candidate) >= _MIN_HTML_LEN:
            return candidate

    # 2. Bare HTML tag anywhere in cleaned text
    m = _HTML_TAG.search(clean)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) >= _MIN_HTML_LEN:
            return candidate

    # 3. Entire cleaned response looks like HTML
    if clean.lower().startswith("<!doctype") or clean.lower().startswith("<html"):
        if len(clean) >= _MIN_HTML_LEN:
            return clean

    return None


def _validate_html(html: str) -> bool:
    """Basic sanity checks: non-empty, looks like HTML, not junk.

    Accepts truncated HTML (missing </html>) as long as it starts with
    a recognizable HTML tag and is long enough — reasoning models often
    exhaust max_tokens before closing the document.
    """
    if not html or len(html) < _MIN_HTML_LEN:
        return False
    lower = html.lower()
    # Best case: properly closed
    if "</html>" in lower or "</body>" in lower:
        return True
    # Truncated but genuine HTML: starts with doctype/html and has real content
    stripped = lower.lstrip()
    if (stripped.startswith("<!doctype") or stripped.startswith("<html")) and len(html) >= 500:
        return True
    return False


class RepairStrategy(ABC):
    """Abstract base class for all repair strategies.

    Attributes:
        name: Strategy identifier.
        mode: "rewrite" (LLM outputs full HTML) or "patch" (LLM outputs JSON patches).
    """

    name: str = "base"
    mode: str = "rewrite"  # "rewrite" | "patch"

    @abstractmethod
    def build_prompt(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        prev_iterations: Optional[List[dict]] = None,
    ) -> str:
        """Construct the repair prompt for this strategy."""
        ...

    async def repair(
        self,
        html: str,
        query: str,
        diag: Diagnosis,
        client: AsyncOpenAI,
        model: str,
        screenshots: Optional[List[str]] = None,
        frame_annotations: Optional[List[dict]] = None,
        prev_iterations: Optional[List[dict]] = None,
        visual_context: str = "",
    ) -> Optional[str]:
        """
        Run one repair attempt: build prompt → call LLM → extract result.

        For mode="patch": parses JSON patches and applies them surgically.
        For mode="rewrite": extracts complete HTML from response.
        Falls back from patch → rewrite extraction on parse/apply failure.

        Returns:
            Repaired HTML string, or None on failure.
        """
        prompt = self.build_prompt(html, query, diag, prev_iterations=prev_iterations)
        if visual_context:
            prompt = visual_context + "\n\n" + prompt
        raw    = await _call_llm(client, model, prompt,
                                 screenshots=screenshots,
                                 frame_annotations=frame_annotations)
        if not raw:
            return None

        if self.mode == "patch":
            return self._apply_patch_response(raw, html)
        else:
            return self._apply_rewrite_response(raw)

    def _apply_patch_response(self, raw: str, original_html: str) -> Optional[str]:
        """Parse patches JSON from LLM response, apply to original HTML.

        Fallback: if patch parsing/application fails, try extracting full HTML
        (the LLM may have ignored the patch instruction and returned full HTML).
        """
        # 1. Strip <think> blocks
        clean = _THINK_BLOCK.sub("", raw).strip()
        if not clean:
            clean = raw

        # 2. Try to extract JSON (fenced or bare)
        patches = None
        # Try fenced JSON first
        for m in _JSON_FENCE.finditer(clean):
            try:
                data = json.loads(m.group(1).strip())
                patches = self._normalize_patches(data)
                if patches:
                    break
            except (json.JSONDecodeError, ValueError):
                continue

        # Try bare JSON object
        if not patches:
            # Find first { ... } that looks like a patches object
            for m in re.finditer(r'\{[^{}]*"patches"[^{}]*\[.*?\]\s*\}', clean, re.DOTALL):
                try:
                    data = json.loads(m.group(0))
                    patches = self._normalize_patches(data)
                    if patches:
                        break
                except (json.JSONDecodeError, ValueError):
                    continue

        # Try the whole cleaned text as JSON
        if not patches:
            try:
                data = json.loads(clean)
                patches = self._normalize_patches(data)
            except (json.JSONDecodeError, ValueError):
                pass

        # 3. Apply patches (partial: skip failures, apply what works)
        if patches is not None:
            result, applied, total = apply_patches(original_html, patches)
            if total == 0:
                # LLM returned empty patches — "no changes needed"
                logger.info(f"[{self.name}] patch mode: LLM returned empty patches (no changes needed)")
                return original_html
            if applied > 0 and validate_patched_html(original_html, result):
                if applied < total:
                    logger.info(f"[{self.name}] patch mode: applied {applied}/{total} patches "
                                f"({total - applied} skipped)")
                else:
                    logger.info(f"[{self.name}] patch mode: applied {applied}/{total} patches successfully")
                return result
            elif applied == 0:
                logger.warning(f"[{self.name}] patch mode: 0/{total} patches matched")
            else:
                logger.warning(f"[{self.name}] patch mode: validation failed after applying {applied}/{total} patches")

        # 4. Fallback: try extracting full HTML ONLY if the LLM ignored the
        #    patch format and returned complete HTML instead.  Verify the candidate
        #    is a genuine full HTML document (must contain <html or <!DOCTYPE),
        #    not junk extracted from thinking/patch JSON fragments.
        logger.info(f"[{self.name}] patch mode failed, falling back to rewrite extraction")
        candidate = self._apply_rewrite_response(raw)
        if candidate and ("<!doctype" in candidate[:200].lower() or "<html" in candidate[:200].lower()):
            return candidate
        # Rewrite fallback produced junk — return None so engine.py can
        # trigger holistic_rewrite fallback at the strategy level.
        return None

    @staticmethod
    def _normalize_patches(data: object) -> Optional[list[dict]]:
        """Extract a list of patch dicts from parsed JSON.

        Returns [] for empty patches (LLM decided no changes needed).
        Returns None only when the data doesn't contain a valid patches structure.
        """
        if isinstance(data, dict):
            patches = data.get("patches")
            if isinstance(patches, list):
                if not patches:
                    # Empty patches list = LLM said "no changes needed"
                    return []
                # Validate each patch has old_str and new_str
                if all(isinstance(p, dict) and "old_str" in p and "new_str" in p for p in patches):
                    return patches
        elif isinstance(data, list):
            if not data:
                return []
            if all(isinstance(p, dict) and "old_str" in p and "new_str" in p for p in data):
                return data
        return None

    def _apply_rewrite_response(self, raw: str) -> Optional[str]:
        """Extract complete HTML from LLM response (existing logic)."""
        result = _extract_html(raw)
        if not result or not _validate_html(result):
            logger.warning(f"[{self.name}] LLM returned invalid/truncated HTML "
                           f"(len={len(raw)}, extracted={len(result) if result else 0})")
            return None
        return result


# ---------------------------------------------------------------------------
# Shared LLM call with retry + optional multimodal
# ---------------------------------------------------------------------------

async def _call_llm(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    screenshots: Optional[List[str]] = None,
    frame_annotations: Optional[List[dict]] = None,
) -> Optional[str]:
    """Call LLM with retries and 1200s timeout per attempt.

    Respects the global _LLM_SEM semaphore (acquired per attempt, released during
    retry sleep so other callers can proceed while this one waits to retry).

    Timeout raised from 600s→1200s: Kimi K2.5 spends significant time generating
    full HTML rewrites (holistic_rewrite outputs 5K-30K chars). With 600s timeout,
    ~80% of repair calls hit timeout on attempt 1 and retry, wasting 600s per call.
    1200s eliminates most retries while still catching genuinely stalled calls.

    When screenshots + frame_annotations are provided, uses smart frame selection
    and interleaves captions with images so the LLM understands each frame.
    """
    # Encode screenshots concurrently in thread pool (CPU-bound, would block event loop)
    content: object
    if screenshots:
        # Smart frame selection: pick up to 5 most informative frames
        sel_ss, sel_fa = _select_repair_frames(screenshots, frame_annotations, max_frames=5)

        content = [{"type": "text", "text": prompt}]
        paths = sel_ss
        if paths:
            encoded = await asyncio.gather(
                *[asyncio.to_thread(_encode_screenshot, p) for p in paths],
                return_exceptions=True,
            )
            # Section header
            content.append({"type": "text", "text": (
                "## Current Page Screenshots\n"
                "The following images show the current state captured by automated "
                "browser testing. Examine them to understand what the page actually "
                "looks like and identify visual issues."
            )})
            for idx, (path, result) in enumerate(zip(paths, encoded)):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to encode screenshot {path}: {result}")
                    continue
                # Interleave caption before each image
                if idx < len(sel_fa):
                    fa = sel_fa[idx]
                    content.append({"type": "text", "text": (
                        f"[Frame {idx+1}: {fa.get('label', 'unknown')} — "
                        f"{fa.get('description', '')}]"
                    )})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{result}"},
                })
    else:
        content = prompt

    last_err: Exception = RuntimeError("no attempts")

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            # Acquire semaphore PER ATTEMPT (not for the whole function) so it's
            # released during retry sleep, letting other callers proceed.
            if _LLM_SEM:
                async with _LLM_SEM:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": content}],
                            max_tokens=65536,
                            temperature=0.3,
                        ),
                        timeout=1800,
                    )
            else:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=65536,
                        temperature=0.3,
                    ),
                    timeout=1800,  # 30 min — user requested
                )
            msg = resp.choices[0].message
            # Reasoning models (e.g. Kimi-K2.5): content may be None during thinking
            raw = msg.content or getattr(msg, "reasoning_content", None) or ""
            return raw.strip()

        except (asyncio.TimeoutError, Exception) as e:
            last_err = e
            logger.warning(f"LLM repair attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(2 ** min(attempt, 4))  # sleep OUTSIDE semaphore, cap at 16s

    logger.error(f"LLM repair failed after {max_attempts} attempts: {last_err}")
    return None
