"""
Adaptive interaction strategy for the AgentTest phase.

Generates an interaction guide appended to the agent's task prompt.
When key-scan results are available (discovered_keys), produces a structured
5-step test protocol that tells the agent exactly what to do.
When no scan data is available, falls back to the generic guide.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional


def build_interaction_guide(
    input_types: List[str],
    discovered_keys: Optional[List[str]] = None,
    game_vars: Optional[Dict[str, Any]] = None,
) -> str:
    """Return an interaction guide for the browser-use agent.

    Args:
        input_types:    detected modes from static analysis (e.g. ["keyboard"]).
        discovered_keys: keys confirmed working by pre-scan (e.g. ["ArrowLeft", "Space"]).
                        None / empty → use generic fallback.
        game_vars:      window-level variables detected at scan time (e.g. {"score": 0}).
    """
    input_types    = input_types    or []
    discovered_keys = discovered_keys or []
    game_vars       = {k: v for k, v in (game_vars or {}).items()
                       if not isinstance(v, dict) or v}  # skip empty objects

    parts: List[str] = []

    # ── Structured protocol when we have discovered keys ──────────────────
    if discovered_keys and "keyboard" in input_types:
        keys_str = ", ".join(discovered_keys)
        parts.append(_structured_keyboard_protocol(keys_str, game_vars))

    # ── Generic keyboard guide (fallback) ────────────────────────────────
    elif "keyboard" in input_types:
        parts.append(
            """## Keyboard Interaction (keyboard input detected in code):
1. **Click the page/canvas first** to give it focus before sending keys.
2. **Single key press** — use `execute_javascript("window.__pressKey('ArrowRight', 100)")`.
3. **Sustained hold** (physics/driving):
   - `execute_javascript("window.__holdKey('ArrowUp', 2000)")` — hold Up for 2 s
   - `execute_javascript("window.__holdKeys(['ArrowUp','ArrowRight'], 2000)")` — two keys
4. **Repeated taps** (platformers, menus):
   - `execute_javascript("window.__pressKeys('ArrowRight', 10, 100)")` — tap 10× every 100 ms
5. Wait 2–3 s after each action to observe the result.
6. Read game state: `execute_javascript("window.__probe.snapshot()")` — shows keysReceived counts.
Common keys to try: ArrowUp/Down/Left/Right, Space (key=" "), Enter, Escape, w/a/s/d, 1/2/3"""
        )

    # ── Mouse drag guide ─────────────────────────────────────────────────
    if "mouse_drag" in input_types:
        parts.append(
            """## Mouse / Touch Drag Interaction (drag detected in code):
- **Drag**: `execute_javascript("window.__drag(startX, startY, endX, endY, 300)")` — canvas coordinates
- **Click at position**: `execute_javascript("window.__clickAt(x, y)")` — canvas coordinates
- **Get canvas size**: `execute_javascript("document.querySelector('canvas').width + 'x' + document.querySelector('canvas').height")`
- Wait 1–2 s after each drag to observe the result."""
        )
    elif "mouse_click" in input_types and "keyboard" not in input_types:
        parts.append(
            """## Mouse Click Interaction (click detected in code):
- Click DOM elements directly (buttons, links, items).
- For canvas: `execute_javascript("window.__clickAt(x, y)")` — canvas coordinates."""
        )

    if not parts:
        parts.append(
            """## Available Interaction Methods:
- **Keyboard**: `execute_javascript("window.__pressKey('ArrowRight', 100)")` — focus + dispatch
- **Hold key**: `execute_javascript("window.__holdKey('ArrowUp', 1500)")` — sustained press
- **Drag**: `execute_javascript("window.__drag(x1, y1, x2, y2, 300)")` — canvas drag
- **Canvas click**: `execute_javascript("window.__clickAt(x, y)")` — canvas coordinates
- **Read state**: `execute_javascript("window.__probe.snapshot()")` — keysReceived + gameVars"""
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Structured protocol (used when discovered_keys is non-empty)
# ---------------------------------------------------------------------------

def _structured_keyboard_protocol(keys_str: str, game_vars: Dict[str, Any]) -> str:
    var_hint = ""
    if game_vars:
        readable = ", ".join(f"{k}={v}" for k, v in list(game_vars.items())[:5])
        var_hint = (
            f"\nPre-scan detected game variables: {readable}\n"
            "Read these again at STEP 4 to confirm they changed during play.\n"
        )

    key_list = [k.strip() for k in keys_str.split(",")]

    # Build per-key test lines for STEP 2
    step2_lines = []
    for k in key_list:
        js_key = " " if k == "Space" else k
        step2_lines.append(
            f'  - {k}: `execute_javascript("window.__pressKey(\\"{js_key}\\", 100)")`'
            f" → wait 1 s, screenshot, describe what changed"
        )
    step2_block = "\n".join(step2_lines)

    # Suggest multi-key combos for STEP 3
    combo_hint = ""
    if len(key_list) >= 2:
        combo_hint = (
            f"  Try combining keys: e.g. hold {key_list[0]} while pressing "
            f"{key_list[min(1, len(key_list)-1)]}."
        )

    return f"""## Pre-scan Results — Confirmed Responsive Keys: {keys_str}
{var_hint}
These keys were verified to trigger responses before this session started.
**Use them directly** — do NOT waste steps testing keys not in this list.

## REQUIRED Test Protocol (follow steps in order):

**STEP 1 — Focus canvas and read baseline state:**
  `execute_javascript("document.querySelector('canvas')?.focus()")`
  `execute_javascript("window.__probe.snapshot()")`
  → Note keysReceived (should be empty), any gameVars (score/lives/level).
  → Take a screenshot and describe the initial page.

**STEP 2 — Test each confirmed key individually:**
{step2_block}

**STEP 3 — Sustained play (15–20 actions over ~10 seconds):**
  Use the confirmed keys to actually play the game.
  Goal: trigger the full game loop → start → active play → score change → game over → restart.
  {combo_hint}
  Use `execute_javascript("window.__pressKeys('{key_list[0]}', 8, 150)")` for rapid taps.
  Use `execute_javascript("window.__holdKey('{key_list[0]}', 1500)")` for sustained movement.

**STEP 4 — Read final game state:**
  `execute_javascript("window.__probe.snapshot()")`
  Report the result JSON. Specifically:
  - keysReceived should show counts > 0 for every key you pressed
  - gameVars should show changed values (score increased, lives changed, etc.)

**STEP 5 — Write your report covering:**
  1. Which keys triggered visible responses (movement, animation, logic)
  2. Whether game logic worked correctly (scoring, collision, physics, win/lose)
  3. **Bug list** — exact: what happened vs. what should have happened
  4. **Missing features** — described in the task but absent or non-functional
  5. Overall assessment: Excellent / Good / Fair / Poor / Broken"""
