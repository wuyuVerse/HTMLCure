from __future__ import annotations

import json
from pathlib import Path

from htmleval.benchmark.loader import load_benchmark_items
from htmleval.core.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads_without_credentials() -> None:
    config = load_config(str(ROOT / "configs" / "eval.example.yaml"))
    assert config.processing.skip_agent_phase is True
    assert config.workspace == "./outputs/eval_workspace"


def test_public_english_benchmark_count() -> None:
    items = load_benchmark_items(str(ROOT / "benchmark" / "en"))
    assert len(items) == 400
    assert {item["category"] for item in items} == {
        "apps_tools",
        "content_marketing",
        "data_visualization",
        "games_simulations",
        "three_d_webgl",
        "visual_art_animation",
    }


def test_sample_response_matches_current_prompt() -> None:
    sample_path = ROOT / "examples" / "responses" / "sample_responses.jsonl"
    sample = json.loads(sample_path.read_text(encoding="utf-8").strip())
    first_item = load_benchmark_items(str(ROOT / "benchmark" / "en" / "apps_tools.jsonl"))[0]
    assert sample["id"] == first_item["id"]
    assert sample["prompt"] == first_item["prompt"]
    assert "<!DOCTYPE html>" in sample["response"]
