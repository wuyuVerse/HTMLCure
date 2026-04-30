"""
TestCase schema — data classes for benchmark test execution.

TestStep  → one atomic browser action or assertion
TestCase  → ordered sequence of steps (with weight for scoring)
StepResult / TestCaseResult → execution outcomes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Valid actions
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset({
    # Navigation / wait
    "wait_for", "wait",
    # Interaction
    "click", "click_text", "type", "press_key", "scroll",
    "eval_js",
    "drag", "hover", "focus", "contextmenu", "select_option", "check",
    # Capture
    "screenshot", "resize",
    # Assertions
    "assert_visible", "assert_not_visible",
    "assert_text_contains", "assert_text_not_contains",
    "assert_count", "assert_attribute",
    "assert_console_clean", "assert_screenshot_changed",
    "assert_screenshot_not_blank", "assert_js_value",
    "assert_style", "assert_no_horizontal_scroll",
    "assert_semantic_html", "assert_a11y_basic",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestStep:
    """One atomic browser action or assertion."""

    action: str                         # one of VALID_ACTIONS
    selector: str = ""                  # CSS selector
    text: str = ""                      # type / assert_text_contains
    text_pattern: str = ""              # click_text regex
    key: str = ""                       # press_key
    repeat: int = 1                     # press_key repeat count
    timeout: int = 5000                 # milliseconds
    ms: int = 0                         # wait duration
    threshold: float = 0.95            # assert_screenshot_changed
    width: int = 0                      # resize
    height: int = 0                     # resize
    from_pos: Optional[List[int]] = None  # drag [x, y]
    to_pos: Optional[List[int]] = None    # drag [x, y]
    gte: Optional[int] = None           # assert_count
    lte: Optional[int] = None           # assert_count
    eq: Optional[int] = None            # assert_count
    expression: str = ""                # assert_js_value
    expected: Any = None                # assert_js_value / assert_style
    attr: str = ""                      # assert_attribute
    value: str = ""                     # assert_attribute
    property: str = ""                  # assert_style
    tags: Optional[List[str]] = None    # assert_semantic_html
    label: str = ""                     # screenshot label
    direction: str = ""                 # scroll direction
    amount: int = 300                   # scroll pixel amount


@dataclass
class TestCase:
    """Ordered sequence of steps with a weight for scoring."""

    id: str
    name: str
    weight: float = 1.0
    steps: List[TestStep] = field(default_factory=list)


@dataclass
class StepResult:
    """Execution outcome for one step."""

    action: str
    passed: bool
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class TestCaseResult:
    """Execution outcome for one test case."""

    id: str
    name: str
    weight: float
    passed: bool
    error: str = ""
    step_results: List[StepResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_test_cases(raw: list[dict]) -> list[TestCase]:
    """Convert JSON dicts to TestCase list, validating action types."""
    cases: list[TestCase] = []
    for tc_dict in raw:
        steps: list[TestStep] = []
        for s in tc_dict.get("steps", []):
            action = s.get("action", "")
            if action not in VALID_ACTIONS:
                raise ValueError(f"Unknown test action: {action!r} in test case {tc_dict.get('id', '?')}")
            steps.append(TestStep(
                action=action,
                selector=s.get("selector", ""),
                text=s.get("text", ""),
                text_pattern=s.get("text_pattern", ""),
                key=s.get("key", ""),
                repeat=s.get("repeat", 1),
                timeout=s.get("timeout", 5000),
                ms=s.get("ms", 0),
                threshold=s.get("threshold", 0.95),
                width=s.get("width", 0),
                height=s.get("height", 0),
                from_pos=s.get("from_pos"),
                to_pos=s.get("to_pos"),
                gte=s.get("gte"),
                lte=s.get("lte"),
                eq=s.get("eq"),
                expression=s.get("expression", ""),
                expected=s.get("expected"),
                attr=s.get("attr", ""),
                value=s.get("value", ""),
                property=s.get("property", ""),
                tags=s.get("tags"),
                label=s.get("label", ""),
                direction=s.get("direction", ""),
                amount=s.get("amount", 300),
            ))
        cases.append(TestCase(
            id=tc_dict.get("id", f"tc_{len(cases):03d}"),
            name=tc_dict.get("name", ""),
            weight=tc_dict.get("weight", 1.0),
            steps=steps,
        ))
    return cases
