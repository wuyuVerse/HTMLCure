"""
Benchmark generator — call an LLM to produce HTML responses for benchmark items.

Usage:
    from htmleval.benchmark.generator import generate_responses

    items_with_responses = asyncio.run(
        generate_responses(items, base_url="http://...", api_key="...", model="...")
    )
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path

from tqdm import tqdm
from htmleval.phases.extract import extract_complete_html

logger = logging.getLogger("htmleval")

_MAX_RETRIES = 5
_DEFAULT_MAX_TOKENS = 0

_HTML_ONLY_SYSTEM_PROMPT = (
    "You generate benchmark solutions as a single self-contained HTML document. "
    "Return only the final HTML. "
    "Do not include analysis, explanations, markdown fences, or any text before/after the HTML. "
    "Start with <!DOCTYPE html> when applicable and ensure the document is complete and closed."
)

_SAFE_DEMO_RETRY_SUFFIX = (
    "\n\nProvider-safety constraints for this benchmark response: build a harmless, "
    "front-end-only demo. Use fake placeholder data only. Do not process, transmit, "
    "store, or validate real credentials or payment data; if payment fields are "
    "requested, implement UI-only demo formatting/validation with synthetic examples."
)


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After seconds from an OpenAI error response, if present."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _short_error(exc: Exception) -> str:
    """Compact provider error for logs / result metadata."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        msg = body.get("message") or body.get("error") or str(body)
    else:
        msg = str(exc)
    status = getattr(exc, "status_code", None)
    prefix = f"{status}:" if status else ""
    return f"{prefix}{type(exc).__name__}:{str(msg)[:500]}"


def _is_content_filter_error(exc: Exception) -> bool:
    """Provider blocked this item, not the whole model/channel."""
    msg = _short_error(exc).lower()
    markers = (
        "content filter",
        "content filtering",
        "output blocked",
        "blocked by content",
        "safety policy",
        "policy violation",
    )
    return any(marker in msg for marker in markers)


def _validate_response(response_text: str) -> tuple[bool, str]:
    """Return (is_valid, reason) for a generated benchmark response."""
    text = (response_text or "").strip()
    if not text:
        return False, "empty"

    if extract_complete_html(text) is not None:
        return True, "ok"

    lowered = text.lower()
    if "<!doctype" in lowered or "<html" in lowered or "```html" in lowered:
        return False, "truncated_or_unextractable_html"
    if "```svg" in lowered or "```xml" in lowered or "<svg" in lowered:
        return False, "truncated_or_unextractable_svg"
    return False, "no_extractable_html"


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """Load successful responses from a checkpoint JSONL file.

    Only returns items with non-empty responses — failed items are excluded
    so they will be retried on the next run.
    """
    checkpoint: dict[str, dict] = {}
    invalid = 0
    if not path.exists():
        return checkpoint
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = str(rec.get("id", ""))
                valid, _ = _validate_response(rec.get("response", ""))
                if rid and valid:
                    checkpoint[rid] = rec
                elif rid and rec.get("response"):
                    invalid += 1
            except json.JSONDecodeError:
                continue
    if invalid:
        logger.info(f"Checkpoint: ignored {invalid} invalid/truncated responses from {path.name}")
    return checkpoint


def _rewrite_checkpoint(path: Path, checkpoint: dict[str, dict]) -> None:
    """Rewrite checkpoint file with only successful items (compact / dedup)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in checkpoint.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def generate_responses(
    items: list[dict],
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int = 32,
    temperature: float = 0.7,
    timeout: int = 180,
    seed: int = 0,
    output_path: str = "",
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    disable_thinking: bool = False,
    checkpoint_flush_interval: int = 8,
) -> list[dict]:
    """Generate HTML responses for benchmark items via LLM.

    Args:
        items:       List of benchmark items (must have "prompt" field).
        base_url:    OpenAI-compatible API base URL.
        api_key:     API key.
        model:       Model name / path.
        concurrency: Max parallel requests.
        temperature: Sampling temperature. Negative values omit the parameter.
        timeout:     Per-request timeout in seconds.
        seed:        Seed for deterministic generation (0 = disabled).
        output_path: If set, stream completed items to this JSONL file
                     and resume from it on restart.  Only successful
                     (non-empty) responses are persisted, so re-running
                     automatically retries any previously failed items.
        max_tokens:  Max completion tokens (0 = provider default).
        disable_thinking: Best-effort disable reasoning mode on local/vLLM Qwen-style APIs.

    Returns:
        Copy of items with "response" field populated.
        Items that already have a response are returned unchanged.
    """
    from openai import (
        AsyncOpenAI,
        RateLimitError,
        InternalServerError,
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
    )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = [None] * len(items)  # type: ignore[list-item]
    abort_event = asyncio.Event()
    abort_reason = ""

    # --- Checkpoint: load previously successful items ---
    checkpoint_items: dict[str, dict] = {}
    checkpoint_lock = asyncio.Lock()
    checkpoint_file = None
    cp_path: Path | None = None
    checkpoint_buffer: list[str] = []

    if output_path:
        cp_path = Path(output_path)
        checkpoint_items = _load_checkpoint(cp_path)
        if checkpoint_items:
            logger.info(f"Checkpoint: {len(checkpoint_items)} successful items from {output_path}")
        # Rewrite checkpoint to compact (remove dups / failed entries from prior runs)
        _rewrite_checkpoint(cp_path, checkpoint_items)
        # Open for appending new successes
        checkpoint_file = open(cp_path, "a", encoding="utf-8")

    def _flush_checkpoint_buffer() -> None:
        if checkpoint_file is None or not checkpoint_buffer:
            return
        for line in checkpoint_buffer:
            checkpoint_file.write(line + "\n")
        checkpoint_file.flush()
        checkpoint_buffer.clear()

    # Count how many items actually need generation
    need_api = 0
    for it in items:
        existing = it.get("response", "")
        if existing:
            valid, _ = _validate_response(existing)
            if valid:
                continue
            # Treat inline invalid responses like missing outputs so they get regenerated.
            it["response"] = ""
        item_id = str(it.get("id", ""))
        if item_id in checkpoint_items:
            continue
        need_api += 1

    pbar = tqdm(total=len(items), desc="generate", unit="item", dynamic_ncols=True)
    # Fast-forward progress for already-done items
    pbar.update(len(items) - need_api)
    logger.info(f"Generation: {need_api} to generate, "
                f"{len(checkpoint_items)} from checkpoint, "
                f"{len(items) - need_api - len(checkpoint_items)} already have responses")

    # Build extra API kwargs
    extra_kwargs: dict = {}
    if seed:
        extra_kwargs["seed"] = seed
    if max_tokens and max_tokens > 0:
        extra_kwargs["max_tokens"] = max_tokens
    if temperature >= 0:
        extra_kwargs["temperature"] = temperature

    base_url_lower = base_url.lower()
    if disable_thinking and (
        "127.0.0.1" in base_url_lower
        or "localhost" in base_url_lower
    ):
        # Local and remote vLLM OpenAI-compatible servers support
        # chat_template_kwargs on chat completions.
        extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    async def _gen(idx: int, item: dict) -> None:
        nonlocal abort_reason
        out = copy.deepcopy(item)
        item_id = str(item.get("id", idx))

        # Skip items that already have a response
        if out.get("response"):
            valid, reason = _validate_response(out["response"])
            if valid:
                results[idx] = out
                return
            logger.info(f"[{item_id}] existing response invalid ({reason}); regenerating")
            out["response"] = ""

        # Use checkpoint if available
        if item_id in checkpoint_items:
            cp_item = checkpoint_items[item_id]
            out["response"] = cp_item["response"]
            results[idx] = out
            return

        async with sem:
            if abort_event.is_set():
                out["response"] = ""
                out["generation_error"] = f"aborted:{abort_reason}"
                results[idx] = out
                pbar.update(1)
                return

            safe_demo_retry = False
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    user_prompt = item["prompt"]
                    if safe_demo_retry:
                        user_prompt += _SAFE_DEMO_RETRY_SUFFIX
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": _HTML_ONLY_SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            **extra_kwargs,
                        ),
                        timeout=timeout,
                    )
                    message = resp.choices[0].message.content or ""
                    finish_reason = resp.choices[0].finish_reason or ""
                    valid, reason = _validate_response(message)
                    if not valid or (finish_reason == "length" and not valid):
                        if attempt < _MAX_RETRIES:
                            wait = 2 ** attempt
                            logger.warning(
                                f"[{item_id}] invalid generation "
                                f"(finish_reason={finish_reason or 'unknown'}, reason={reason}), "
                                f"retry {attempt + 1} in {wait}s"
                            )
                            await asyncio.sleep(wait)
                            continue
                        logger.warning(
                            f"[{item_id}] invalid generation after {_MAX_RETRIES + 1} attempts "
                            f"(finish_reason={finish_reason or 'unknown'}, reason={reason})"
                        )
                        out["response"] = ""
                        out["generation_error"] = f"{finish_reason or 'unknown'}:{reason}"
                        break
                    if finish_reason == "length" and valid:
                        logger.info(
                            f"[{item_id}] finish_reason=length but extracted HTML is complete; accepting response"
                        )
                    out["response"] = message
                    break
                except AuthenticationError as e:
                    # Fatal provider/config errors should stop the whole run.
                    # Retrying every remaining item just burns quota and writes
                    # hundreds of empty responses.
                    abort_reason = _short_error(e)
                    abort_event.set()
                    logger.error(f"[{item_id}] fatal error; aborting run: {abort_reason}")
                    out["response"] = ""
                    out["generation_error"] = abort_reason
                    break
                except BadRequestError as e:
                    err = _short_error(e)
                    if _is_content_filter_error(e):
                        if attempt < _MAX_RETRIES:
                            safe_demo_retry = True
                            wait = 2 ** attempt
                            logger.warning(
                                f"[{item_id}] content-filtered by provider; "
                                f"retrying with safe demo constraints in {wait}s: {err}"
                            )
                            await asyncio.sleep(wait)
                            continue
                        logger.warning(f"[{item_id}] content-filtered by provider; skipping item: {err}")
                        out["response"] = ""
                        out["generation_error"] = err
                        break
                    # Other 400s are usually bad model/channel/parameter errors,
                    # so stop the whole run instead of burning quota.
                    abort_reason = err
                    abort_event.set()
                    logger.error(f"[{item_id}] fatal bad request; aborting run: {abort_reason}")
                    out["response"] = ""
                    out["generation_error"] = abort_reason
                    break
                except RateLimitError as e:
                    if attempt < _MAX_RETRIES:
                        wait = _parse_retry_after(e) or (2 ** (attempt + 1))
                        logger.debug(f"[{item_id}] rate-limited, retry {attempt+1} in {wait:.1f}s")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"[{item_id}] rate-limited, exhausted {_MAX_RETRIES+1} attempts: {e}")
                        out["response"] = ""
                except (InternalServerError, APIConnectionError, APITimeoutError) as e:
                    if attempt < _MAX_RETRIES:
                        wait = 2 ** attempt  # 1, 2, 4, 8, 16
                        logger.debug(f"[{item_id}] server/network error, retry {attempt+1} in {wait}s: {e}")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"[{item_id}] generation failed after {_MAX_RETRIES+1} attempts: {e}")
                        out["response"] = ""
                except (asyncio.TimeoutError, TimeoutError):
                    # asyncio.wait_for() raises asyncio.TimeoutError, not openai's APITimeoutError
                    if attempt < _MAX_RETRIES:
                        # Longer backoff for timeouts — the model is busy, not erroring
                        wait = min(2 ** (attempt + 1), 30)  # 2, 4, 8, 16, 30
                        logger.warning(f"[{item_id}] timeout ({timeout}s), retry {attempt+1} in {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"[{item_id}] generation failed after {_MAX_RETRIES+1} attempts: timeout ({timeout}s)")
                        out["response"] = ""
                except Exception as e:
                    if attempt < _MAX_RETRIES:
                        wait = 2 ** attempt
                        logger.warning(f"[{item_id}] unexpected error, retry {attempt+1} in {wait}s: {type(e).__name__}: {e}")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"[{item_id}] generation failed after {_MAX_RETRIES+1} attempts: {type(e).__name__}: {e}")
                        out["response"] = ""

        # Only write successful responses to checkpoint — failed items (empty
        # response) are deliberately omitted so the next run retries them.
        if checkpoint_file is not None and out.get("response"):
            async with checkpoint_lock:
                checkpoint_buffer.append(json.dumps(out, ensure_ascii=False))
                if len(checkpoint_buffer) >= max(1, checkpoint_flush_interval):
                    _flush_checkpoint_buffer()

        results[idx] = out
        pbar.update(1)

    tasks = [asyncio.create_task(_gen(i, it)) for i, it in enumerate(items)]
    await asyncio.gather(*tasks)
    pbar.close()

    # Close checkpoint file
    if checkpoint_file is not None:
        async with checkpoint_lock:
            _flush_checkpoint_buffer()
        checkpoint_file.close()

    ok = sum(1 for r in results if r and r.get("response"))
    failed = len(items) - ok
    logger.info(f"Generated {ok}/{len(items)} responses ({failed} failed)")
    if abort_event.is_set():
        logger.error("Generation aborted early: %s", abort_reason)
    return results
