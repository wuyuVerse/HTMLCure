"""
KeyframeSelector — pick the most informative frames from a candidate set.

All layers may produce 15-25 frames, but VLM only receives max_frames (default 12).
Selection prioritizes: diversity of label categories > high visual change > time coverage.
"""
from __future__ import annotations

from typing import List

from htmleval.phases.render_test.frame_types import AnnotatedFrame, frame_diff_score


MAX_FRAMES = 14  # Increased from 12 to accommodate responsive viewports + micro-clip bursts


def select_keyframes(
    all_frames: List[AnnotatedFrame],
    max_frames: int = MAX_FRAMES,
) -> List[AnnotatedFrame]:
    """
    Select the most informative frames.

    Strategy:
      1. Must-keep: first frame, last frame, first of each label category
      2. High-change: frames with largest visual diff from predecessor
      3. Uniform time sampling: fill remaining slots evenly
    """
    if len(all_frames) <= max_frames:
        return all_frames

    must_keep = {0, len(all_frames) - 1}

    # One frame per label category
    seen_cats = set()
    for i, f in enumerate(all_frames):
        cat = f.label.split("_")[0]
        if cat not in seen_cats:
            must_keep.add(i)
            seen_cats.add(cat)

    # Rank by visual change
    changes = []
    for i in range(1, len(all_frames)):
        changes.append((all_frames[i].diff_from_prev, i))
    changes.sort(reverse=True)

    selected = set(must_keep)
    for _, i in changes:
        if len(selected) >= max_frames:
            break
        selected.add(i)

    # Fill with uniform sampling
    if len(selected) < max_frames:
        remaining = [i for i in range(len(all_frames)) if i not in selected]
        step = max(1, len(remaining) // (max_frames - len(selected) + 1))
        for i in remaining[::step]:
            if len(selected) >= max_frames:
                break
            selected.add(i)

    return [all_frames[i] for i in sorted(selected)]
