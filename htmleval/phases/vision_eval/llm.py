"""
Shared LLM calling infrastructure for vision_eval agents.

Extracted from evaluator.py so that ObserverAgent, ScorerAgent,
and external callers (e.g. htmlrefine contrastive.py) share one
code path for encoding, calling, and JSON-parsing.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from PIL import Image

logger = logging.getLogger("htmleval")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Global LLM concurrency semaphore (shared with repair LLM calls).
_VISION_LLM_SEM: asyncio.Semaphore | None = None


def set_vision_llm_semaphore(sem: asyncio.Semaphore | None) -> None:
    """Share a semaphore with repair LLM calls to cap total concurrent LLM usage."""
    global _VISION_LLM_SEM
    _VISION_LLM_SEM = sem


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_b64(path: str, max_dim: int = 1280) -> str:
    """Encode screenshot as JPEG (fast) for LLM vision input.

    Uses JPEG quality=85 instead of PNG+optimize: ~10x faster encoding,
    5-8x smaller payload, indistinguishable quality for visual analysis.
    """
    img = Image.open(path)
    if max(img.size) > max_dim:
        r = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * r), int(img.size[1] * r)), Image.LANCZOS)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# DOM inventory formatting
# ---------------------------------------------------------------------------

def format_dom_inventory(dom_inv: dict | None) -> str:
    """Format DOM inventory dict into a concise string for VLM prompts."""
    if not dom_inv:
        return "(not available)"
    parts = []
    parts.append(
        f"Visible: {dom_inv.get('buttons', 0)} buttons, "
        f"{dom_inv.get('links', 0)} links, "
        f"{dom_inv.get('textInputs', 0)} text inputs, "
        f"{dom_inv.get('selects', 0)} dropdowns"
    )
    imgs = dom_inv.get('images', 0)
    loaded = dom_inv.get('imagesLoaded', 0)
    if imgs > 0:
        parts.append(f"Images: {loaded}/{imgs} loaded")
    parts.append(
        f"Headings: {dom_inv.get('headings', 0)} | "
        f"Canvas: {dom_inv.get('canvas', 0)} | "
        f"Forms: {dom_inv.get('forms', 0)} | "
        f"Tables: {dom_inv.get('tables', 0)} | "
        f"Event handlers: {dom_inv.get('eventHandlers', 0)}"
    )
    return "\n".join(parts)


def _extract_balanced_json_object(text: str) -> Optional[str]:
    """Extract the first balanced top-level JSON object from mixed text."""
    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if start < 0:
            if ch == "{":
                start = i
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _json_candidates(text: str) -> List[str]:
    """Yield likely JSON payload candidates in descending reliability order."""
    stripped = text.strip()
    candidates: List[str] = []

    for pattern in (
        r"```json\s*(.*?)\s*```",
        r"```\s*(\{.*?\})\s*```",
    ):
        for match in re.finditer(pattern, stripped, re.DOTALL | re.IGNORECASE):
            candidate = match.group(1).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

    balanced = _extract_balanced_json_object(stripped)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    if stripped and stripped not in candidates:
        candidates.append(stripped)

    return candidates


# ---------------------------------------------------------------------------
# Generic VLM call with retry + JSON extraction
# ---------------------------------------------------------------------------

async def call_vlm(
    client: AsyncOpenAI,
    model: str,
    content: List[Dict[str, Any]],
    *,
    max_tokens: int = 8192,
    temperature: float = 0,
    timeout_s: float = 180,
    max_retries: int = 4,
    agent_name: str = "vlm",
) -> Dict[str, Any]:
    """Call the Vision LLM and parse JSON from its response.

    Handles:
    - Semaphore-gated concurrency
    - <think> block stripping
    - JSON extraction from ```json fences or bare {}
    - Retry with exponential backoff on timeout / parse failure

    Returns parsed JSON dict. Raises RuntimeError after all retries exhausted.
    """
    last_err: Exception = RuntimeError("no attempts made")
    json_str = ""  # Track for debug logging on failure
    for attempt in range(1, max_retries + 1):
        try:
            _sem = _VISION_LLM_SEM
            coro = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if _sem:
                async with _sem:
                    resp = await asyncio.wait_for(coro, timeout=timeout_s)
            else:
                resp = await asyncio.wait_for(coro, timeout=timeout_s)

            choice = resp.choices[0]
            finish_reason = choice.finish_reason
            msg = choice.message
            raw = (
                msg.content
                or getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )

            # Detect truncation — finish_reason="length" means max_tokens hit
            if finish_reason == "length":
                logger.warning(
                    f"[{agent_name}] response TRUNCATED (finish_reason=length, "
                    f"max_tokens={max_tokens}, raw_len={len(raw)}). "
                    f"Last 200 chars: ...{raw[-200:]}"
                )
                # Truncated JSON can't be parsed — treat as retry-able failure
                raise json.JSONDecodeError(
                    f"Response truncated (finish_reason=length, max_tokens={max_tokens})",
                    raw[-200:], 0,
                )

            text = _THINK_BLOCK.sub("", raw).strip() or raw.strip()

            parse_err: json.JSONDecodeError | None = None
            for candidate in _json_candidates(text):
                json_str = candidate
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    parse_err = e
                    continue
            if parse_err is not None:
                raise parse_err
            raise json.JSONDecodeError("No JSON candidate found in VLM response", text[:200], 0)

        except (asyncio.TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            # Log raw response context for debugging JSON parse failures
            if isinstance(e, json.JSONDecodeError):
                # Show chars around the error position
                pos = e.pos if hasattr(e, 'pos') else 0
                snippet = json_str[max(0, pos-80):pos+80] if json_str else "(empty)"
                logger.warning(
                    f"[{agent_name}] attempt {attempt}/{max_retries} failed "
                    f"({type(e).__name__}): {e}\n"
                    f"  json_str len={len(json_str)}, error near: ...{snippet}..."
                )
            else:
                logger.warning(
                    f"[{agent_name}] attempt {attempt}/{max_retries} failed "
                    f"({type(e).__name__}): {e}"
                )
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            last_err = e
            logger.warning(
                f"[{agent_name}] attempt {attempt}/{max_retries} failed: {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"{agent_name} failed after {max_retries} attempts: {last_err}")
