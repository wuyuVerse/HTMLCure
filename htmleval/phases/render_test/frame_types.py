"""
Shared types for the render_test phase.

Defines AnnotatedFrame (screenshot + metadata), FPSResult (animation quality),
and ProbeResult (uniform return type for all probes).
Used by probes, evidence, keyframe_selector, renderer, structural_probe.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FPSResult:
    """FPS sampling result from observation probe."""
    visual_change_count: int
    quality: str  # "smooth" | "acceptable" | "choppy" | "frozen"


@dataclass
class AnnotatedFrame:
    """A screenshot with context annotation."""
    screenshot_path: str       # path to saved PNG file
    label: str                 # machine label: "early_load", "hover_btn_0", "gameplay_t3"
    description: str           # human-readable: "鼠标悬停在按钮「Start」上"
    timestamp: float = 0.0    # seconds since page load
    layer: str = ""           # "observation" / "interaction" / "deep"
    diff_from_prev: float = 0.0  # pixel difference score (0=same, 1=totally different)


@dataclass
class ProbeResult:
    """Uniform return type for every probe function."""
    frames: List[AnnotatedFrame] = field(default_factory=list)
    timestamp: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)


def frame_changed(data_a: bytes, data_b: bytes) -> bool:
    """Fast MD5 check: are two screenshots pixel-identical?"""
    return hashlib.md5(data_a).hexdigest() != hashlib.md5(data_b).hexdigest()


def frame_diff_score(data_a: bytes, data_b: bytes) -> float:
    """
    Compute visual difference between two screenshots (0.0=identical, 1.0=totally different).

    Uses SSIM (Structural Similarity Index) on downscaled grayscale images.
    SSIM is more perceptually meaningful than MAE — it detects structural layout
    changes (element moved, text appeared, button state changed) while ignoring
    insignificant pixel noise from compression or anti-aliasing.

    Returns 1 - SSIM so that 0=identical, 1=totally different (same API as before).
    Falls back to MAE if numpy is unavailable.
    """
    from PIL import Image

    THUMB = (160, 90)
    img_a = Image.open(io.BytesIO(data_a)).convert("L").resize(THUMB)
    img_b = Image.open(io.BytesIO(data_b)).convert("L").resize(THUMB)

    try:
        return _ssim_diff(img_a, img_b)
    except Exception:
        # Fallback to MAE if numpy unavailable
        px_a = list(img_a.getdata())
        px_b = list(img_b.getdata())
        diff = sum(abs(a - b) for a, b in zip(px_a, px_b))
        return diff / (255 * len(px_a))


def _ssim_diff(img_a, img_b) -> float:
    """Compute 1 - SSIM between two PIL grayscale images using numpy.

    SSIM formula per window:
      SSIM = (2*μx*μy + C1)(2*σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
    We compute mean SSIM over 8×8 non-overlapping blocks for speed.

    Returns value in [0, 1] where 0=identical, 1=totally different.
    """
    import numpy as np

    a = np.array(img_a, dtype=np.float64)
    b = np.array(img_b, dtype=np.float64)

    # SSIM constants (for 8-bit images, L=255)
    C1 = (0.01 * 255) ** 2   # 6.5025
    C2 = (0.03 * 255) ** 2   # 58.5225

    # Block-wise SSIM over 8×8 patches (non-overlapping for speed)
    block = 8
    h, w = a.shape
    h_blocks = h // block
    w_blocks = w // block
    if h_blocks == 0 or w_blocks == 0:
        # Image too small for block SSIM — use global
        mu_a, mu_b = a.mean(), b.mean()
        var_a, var_b = a.var(), b.var()
        cov_ab = ((a - mu_a) * (b - mu_b)).mean()
        ssim = ((2 * mu_a * mu_b + C1) * (2 * cov_ab + C2)) / \
               ((mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2))
        return float(max(0.0, min(1.0, 1.0 - ssim)))

    # Reshape into blocks
    a_crop = a[:h_blocks * block, :w_blocks * block]
    b_crop = b[:h_blocks * block, :w_blocks * block]
    a_blocks = a_crop.reshape(h_blocks, block, w_blocks, block).transpose(0, 2, 1, 3)
    b_blocks = b_crop.reshape(h_blocks, block, w_blocks, block).transpose(0, 2, 1, 3)

    mu_a = a_blocks.mean(axis=(2, 3))
    mu_b = b_blocks.mean(axis=(2, 3))
    var_a = a_blocks.var(axis=(2, 3))
    var_b = b_blocks.var(axis=(2, 3))
    cov_ab = ((a_blocks - mu_a[:, :, None, None]) *
              (b_blocks - mu_b[:, :, None, None])).mean(axis=(2, 3))

    num = (2 * mu_a * mu_b + C1) * (2 * cov_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
    ssim_map = num / den
    mean_ssim = ssim_map.mean()

    return float(max(0.0, min(1.0, 1.0 - mean_ssim)))
