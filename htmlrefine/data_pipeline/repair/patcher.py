"""
Patch engine — apply str_replace patches to HTML.

Used by patch-mode strategies to make surgical edits instead of full rewrites.
Each patch specifies an exact substring to find and replace.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("htmlrefine.repair.patcher")


class PatchError(Exception):
    """Raised when a patch cannot be applied."""
    pass


def apply_patches(html: str, patches: list[dict], strict: bool = False) -> tuple[str, int, int]:
    """Apply a list of str_replace patches sequentially.

    Each patch: {"old_str": "...", "new_str": "..."}
    - old_str must match exactly once in the current html
    - Patches are applied in order; later patches see earlier changes

    Args:
        html: Original HTML string.
        patches: List of {old_str, new_str} dicts.
        strict: If True, raise PatchError on first failure (old behavior).
                If False (default), skip failed patches and continue.

    Returns:
        Tuple of (patched_html, applied_count, total_count).

    Raises:
        PatchError: If strict=True and any patch fails, or if patches list is empty.
    """
    if not patches:
        # Empty patches = LLM decided no changes needed (valid response)
        return html, 0, 0

    result = html
    applied = 0
    total = len(patches)

    for i, patch in enumerate(patches):
        old_str = patch.get("old_str")
        new_str = patch.get("new_str")

        if old_str is None or new_str is None:
            if strict:
                raise PatchError(f"Patch {i}: missing 'old_str' or 'new_str' key")
            logger.warning(f"Patch {i}/{total}: missing 'old_str' or 'new_str' key, skipping")
            continue

        if not old_str:
            if strict:
                raise PatchError(f"Patch {i}: empty 'old_str'")
            logger.warning(f"Patch {i}/{total}: empty 'old_str', skipping")
            continue

        count = result.count(old_str)
        if count == 1:
            result = result.replace(old_str, new_str, 1)
            applied += 1
        elif count == 0:
            # Try fuzzy match (whitespace normalization)
            fuzzy = _fuzzy_find(result, old_str)
            if fuzzy is not None:
                logger.info(f"Patch {i}/{total}: exact match failed, fuzzy match succeeded")
                result = result.replace(fuzzy, new_str, 1)
                applied += 1
            else:
                msg = (f"Patch {i}/{total}: old_str not found in HTML "
                       f"(len={len(old_str)}, first 80 chars: {old_str[:80]!r})")
                if strict:
                    raise PatchError(msg)
                logger.warning(msg + " — skipping")
        else:
            msg = (f"Patch {i}/{total}: old_str matches {count} locations "
                   f"(must be exactly 1)")
            if strict:
                raise PatchError(msg)
            logger.warning(msg + " — skipping")

    return result, applied, total


def _fuzzy_find(html: str, old_str: str) -> str | None:
    """When old_str exact match fails, try whitespace-normalized matching.

    Only handles indentation differences (tabs vs spaces, trailing spaces).
    Does NOT do substring or fuzzy text matching (too dangerous).

    Returns the actual matching substring from html, or None.
    """
    def normalize_ws(s: str) -> str:
        """Collapse each line's leading whitespace and strip trailing whitespace."""
        lines = s.split("\n")
        return "\n".join(re.sub(r"^[ \t]+", lambda m: " " * len(m.group().replace("\t", "    ")),
                                line.rstrip()) for line in lines)

    norm_old = normalize_ws(old_str)

    # Walk through html lines looking for a contiguous match
    old_lines = norm_old.split("\n")
    html_lines = html.split("\n")
    num_old = len(old_lines)

    for start in range(len(html_lines) - num_old + 1):
        candidate_lines = html_lines[start:start + num_old]
        norm_candidate = normalize_ws("\n".join(candidate_lines))
        if norm_candidate == norm_old:
            return "\n".join(candidate_lines)

    return None


def validate_patched_html(original: str, patched: str) -> bool:
    """Ensure patch didn't break HTML structure.

    Checks:
    1. Length not below 50% of original
    2. Retains core structure tags
    3. Basic HTML validity
    """
    if not patched:
        return False

    # Length check — patches shouldn't remove more than half the file
    if len(patched) < len(original) * 0.5:
        logger.warning(
            f"Patched HTML too short: {len(patched)} < {len(original) * 0.5:.0f} "
            f"(50% of original {len(original)})"
        )
        return False

    lower = patched.lower()

    # Must retain core structure tags
    for tag in ("</html>", "</body>"):
        if tag in original.lower() and tag not in lower:
            logger.warning(f"Patched HTML missing required tag: {tag}")
            return False

    # Basic non-empty check
    if len(patched) < 200:
        logger.warning(f"Patched HTML suspiciously short: {len(patched)}")
        return False

    return True
