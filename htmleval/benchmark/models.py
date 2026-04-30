"""Model profile loader for benchmark generation.

Reads named profiles from a models.yaml registry so users can run
``--model kimi-k2.5`` instead of passing verbose --generate-url/model/key flags.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{(\w+)\}")

_DEFAULT_SEARCH_PATHS = [
    Path("configs/models.yaml"),                          # relative to CWD
    Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml",  # repo root
]


def _expand_env(value: str) -> str:
    """Expand ``${ENV_VAR}`` references in *value*."""
    def _replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return _ENV_RE.sub(_replace, value)


def _find_config(config_path: str = "") -> Path:
    """Return the first existing models.yaml path."""
    if config_path:
        p = Path(config_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Models config not found: {config_path}")

    for candidate in _DEFAULT_SEARCH_PATHS:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No models.yaml found. Searched:\n"
        + "\n".join(f"  - {p}" for p in _DEFAULT_SEARCH_PATHS)
        + "\nProvide --models-config or create configs/models.yaml."
    )


def load_model_profiles(config_path: str = "") -> dict[str, dict[str, Any]]:
    """Load all model profiles from models.yaml.

    Returns:
        Dict mapping profile name to its settings.
    """
    path = _find_config(config_path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", {}) if data else {}


def load_model_profile(name: str, config_path: str = "") -> dict[str, Any]:
    """Load a single named model profile.

    Returns dict with keys: base_url, model, api_key, concurrency, temperature, timeout.
    Expands ``${ENV_VAR}`` in api_key.

    Raises:
        KeyError: if *name* is not found (message lists available profiles).
    """
    profiles = load_model_profiles(config_path)
    if name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none)"
        raise KeyError(
            f"Model profile '{name}' not found. Available profiles: {available}"
        )
    profile = dict(profiles[name])
    if "api_key" in profile and isinstance(profile["api_key"], str):
        profile["api_key"] = _expand_env(profile["api_key"])
    return profile
