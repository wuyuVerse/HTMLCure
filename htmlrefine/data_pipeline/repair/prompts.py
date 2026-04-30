"""
All repair prompt templates.

Each template is evaluated by the same 5-dimension rubric as the scoring system.
Prompts explicitly show current scores, targets, and rubric so the LLM can
optimize directly for what the evaluator measures.

Key design changes (evidence-gated prompts):
  - probe_evidence block: objective keyscan results + game vars (most reliable)
  - preservation_list: probe-verified features that MUST NOT break
  - keyboard_hint: targeted guidance only when keyboard_broken is confirmed
  - Only include reliable_issues (not speculative VLM bugs) in surgical prompts

All templates use Python str.format() with named placeholders.

Patch-mode strategies use {output_instructions} at the end of each template,
injected by the strategy's build_prompt() based on self.mode.
"""

# ---------------------------------------------------------------------------
# Shared rubric (injected into every prompt)
# ---------------------------------------------------------------------------

_SCORING_RUBRIC = """\
## Scoring rubric (what you're optimizing for)
- **rendering** (0–20): 20=zero errors+animation+responsive; 18-19=zero errors+animation OR responsive; 16-17=zero errors, static only; 10-15=minor issues; 0–9=broken
- **visual_design** (0–20): 20=professional polish with rich detail; 18-19=strong design with depth (layered effects, purposeful typography, custom illustrations); 16-17=competent but generic (template-like); 14-15=basic/plain; 0–13=minimal/broken
- **functionality** (0–25): 25=all features fully working+edge cases; 22-24=all core working; 18-21=most work; 14-17=partial; 0–13=minimal. Features that EXIST but produce no observable effect count as broken.
- **interaction** (0–25): 25=all responsive+fast+accessible; 22-24=all work, minor UX gaps; 18-21=most work; 14-17=significant UX issues; 0–13=limited/broken
- **code_quality** (0–10): 10=semantic HTML+clean; 8-9=clean, minor issues; 6-7=some code smells; 0–5=poor

## Visual quality tips (for ALL strategies)
To score visual_design 18+, use:
- Google Fonts (`<link href="https://fonts.googleapis.com/css2?family=...">`) for typography
- Unsplash/picsum images for visual richness (but ONLY where thematically appropriate)
- SVG icons or emoji for UI elements instead of plain text
- CSS gradients, shadows, backdrop-filter for depth
- Smooth transitions and micro-animations for polish

## Common bugs to avoid (for ALL strategies)
- Page must start in a USABLE state — no "game over" / error on load
- Countable resources (lives, ammo, energy) must start POSITIVE, not 0
- Match input type to task: mouse tasks → mouse events, keyboard tasks → keyboard events
- Every button/clickable MUST have a working event handler
- External images must match the page's theme — don't use random photos for puzzles/icons\
"""


# ---------------------------------------------------------------------------
# Output mode instructions (patch vs rewrite)
# ---------------------------------------------------------------------------

_PATCH_INSTRUCTIONS = """\
## Output Format — str_replace patches

You MUST output a JSON object with a "patches" array. Each patch replaces an exact
substring of the HTML with a new substring.

Rules:
1. "old_str" must match EXACTLY one location in the current HTML (copy-paste precision)
2. Include 2-3 surrounding lines of context in old_str to ensure uniqueness
3. "new_str" is the replacement (can be longer, shorter, or empty to delete)
4. Patches are applied IN ORDER — later patches see earlier patches' changes
5. Do NOT output the full HTML file — ONLY the patches
6. Preserve exact indentation and whitespace from the original
7. If the code already meets all targets and no improvement is clearly beneficial,
   return: {"patches": [], "reason": "no changes needed"}

Reply ONLY with valid JSON:
```json
{
  "patches": [
    {"old_str": "exact lines from HTML", "new_str": "replacement lines"}
  ]
}
```"""

_REWRITE_CLOSING = """\
Reply with ONLY the complete HTML file. No explanation outside the code block."""


def format_patch_closing() -> str:
    """Return the patch-mode output instructions."""
    return _PATCH_INSTRUCTIONS


def format_output_instructions(mode: str) -> str:
    """Return the appropriate output instructions for the given mode."""
    if mode == "patch":
        return _PATCH_INSTRUCTIONS
    return _REWRITE_CLOSING


# ---------------------------------------------------------------------------
# Previous iterations context (injected for iter 2+)
# ---------------------------------------------------------------------------

def format_prev_iterations(prev_iterations: list) -> str:
    """Format previous repair iteration history for inclusion in prompts."""
    if not prev_iterations:
        return ""
    lines = ["## Previous repair attempts (do NOT repeat the same fixes)"]
    for it in prev_iterations:
        delta_str = f"+{it['delta']}" if it['delta'] >= 0 else str(it['delta'])
        lines.append(
            f"\nIteration {it['iteration']} — strategy: {it['strategy']} — "
            f"score: {it['score_before']} → {it['score_after']} ({delta_str})"
        )
        # Per-dimension changes: tell the LLM exactly what improved/regressed
        dim_changes = it.get("dim_changes", {})
        if dim_changes:
            parts = [f"{k}={v:+d}" for k, v in dim_changes.items()]
            lines.append(f"  Dimension changes: {', '.join(parts)}")
        if it.get("what_improved"):
            lines.append(f"  Improved: {it['what_improved']}")
        if it.get("what_remains"):
            lines.append(f"  Still failing: {it['what_remains']}")
        if it.get("new_issues"):
            lines.append(f"  New issues introduced: {it['new_issues']}")
        # Contrastive visual comparison from VLM
        contrastive = it.get("contrastive", "")
        if contrastive:
            lines.append(f"  Visual comparison: {contrastive[:500]}")
    lines.append(
        "\nFor this iteration: focus on the remaining issues above. "
        "Do NOT re-fix what was already fixed."
    )
    return "\n".join(lines)


def format_probe_evidence(diag) -> str:
    """
    Format objective probe evidence for inclusion in repair prompts.

    Only includes data from keyscan / render_test (objective facts).
    Returns empty string when no probe evidence is available.
    """
    lines = []

    if diag.discovered_keys is not None:
        if diag.keyboard_broken:
            lines.append(
                "## Probe Evidence (automated test, objective)\n"
                "⚠ KEYBOARD BROKEN: Key events were dispatched to the page "
                "but produced ZERO visual response.\n"
                f"  Tested keys: ArrowLeft, ArrowRight, ArrowUp, ArrowDown, Space, Enter, w/a/s/d\n"
                f"  Responsive keys found: NONE"
            )
        elif diag.keyboard_verified and diag.discovered_keys:
            lines.append(
                "## Probe Evidence (automated test, objective)\n"
                f"✓ Keyboard works: probe-verified responsive keys: "
                f"{', '.join(diag.discovered_keys)}"
            )

    if diag.game_vars_initial:
        vars_str = ", ".join(
            f"{k}={v}" for k, v in list(diag.game_vars_initial.items())[:8]
        )
        if lines:
            lines.append(f"  Initial game variables at load time: {vars_str}")
        else:
            lines.append(
                f"## Probe Evidence (automated test, objective)\n"
                f"  Game variables at load time: {vars_str}"
            )

    return "\n".join(lines)


def format_preservation_list(diag) -> str:
    """
    Format the preservation list — features confirmed working that MUST NOT break.

    Prioritizes probe-verified features (objective) over VLM-inferred highlights.
    """
    items = list(diag.preservation_list)
    if not items:
        items = list(diag.highlights)
    if not items:
        return "- (none confirmed yet)"
    return "\n".join(f"- {item}" for item in items)


def format_observer_evidence(diag) -> str:
    """
    Format Observer report (from 3-agent eval pipeline) for repair prompts.

    Includes page_type, visual_state, visual_elements inventory,
    working/broken lists with evidence citations.
    Returns empty string if no observer data available.
    """
    obs = getattr(diag, "observer_report", {})
    if not obs:
        return ""

    lines = ["## Observer Report (automated visual inspection, evidence-backed)"]

    page_type = obs.get("page_type", "")
    if page_type:
        lines.append(f"Page type: {page_type}")

    visual_state = obs.get("visual_state", "")
    if visual_state:
        lines.append(f"Visual state: {visual_state}")

    visual_elements = obs.get("visual_elements", [])
    if visual_elements:
        lines.append("\nVisual elements inventory (MUST ALL be preserved in repaired version):")
        for item in visual_elements:
            lines.append(f"  • {item}")

    working = obs.get("working", [])
    if working:
        lines.append("\nConfirmed WORKING (with evidence — DO NOT break):")
        for item in working:
            lines.append(f"  ✓ {item}")

    broken = obs.get("broken", [])
    if broken:
        lines.append("\nConfirmed BROKEN (with evidence — FIX these):")
        for item in broken:
            lines.append(f"  ✗ {item}")

    interaction_quality = obs.get("interaction_quality", "")
    if interaction_quality:
        lines.append(f"\nInteraction quality: {interaction_quality}")

    layout_notes = obs.get("layout_notes", "")
    if layout_notes:
        lines.append(f"Layout: {layout_notes}")

    return "\n".join(lines)


def format_requirement_checklist(diag) -> str:
    """
    Format TaskAuditor requirement checklist for repair prompts.

    Shows each requirement's status (done/broken/missing) with evidence.
    Returns empty string if no auditor data available.
    """
    audit = getattr(diag, "task_auditor_report", {})
    if not audit:
        return ""

    reqs = audit.get("requirements", [])
    if not reqs:
        return ""

    summary = audit.get("summary", {})
    total = summary.get("total", len(reqs))
    done = summary.get("done", 0)
    broken = summary.get("broken", 0)
    missing = summary.get("missing", 0)

    lines = [
        f"## Requirement Checklist ({done} done / {broken} broken / {missing} missing out of {total})"
    ]

    for req in reqs:
        status = req.get("status", "missing")
        icon = {"done": "✓", "broken": "✗", "missing": "○"}.get(status, "?")
        desc = req.get("requirement", "")
        evidence = req.get("evidence", "")
        line = f"  {icon} [{status:>7s}] {desc}"
        if evidence and status != "done":
            line += f" — {evidence}"
        lines.append(line)

    return "\n".join(lines)


def format_visual_diagnosis(diagnosis) -> str:
    """
    Format VLM visual diagnosis for inclusion in visual enrichment prompts.

    Args:
        diagnosis: VisualDiagnosis dataclass from visual_diagnosis.py
    Returns:
        Formatted string with issues, suggestions, and focus areas.
    """
    if not diagnosis or not getattr(diagnosis, "issues", None):
        return ""

    lines = ["## VLM Visual Diagnosis (automated visual analysis)"]

    if diagnosis.issues:
        lines.append("\nIdentified visual issues:")
        for issue in diagnosis.issues:
            lines.append(f"  ✗ {issue}")

    if diagnosis.suggestions:
        lines.append("\nSuggested CSS/HTML fixes:")
        for suggestion in diagnosis.suggestions:
            lines.append(f"  → {suggestion}")

    if diagnosis.css_focus_areas:
        lines.append(f"\nFocus areas: {', '.join(diagnosis.css_focus_areas)}")

    return "\n".join(lines)


def format_visual_context(evidence) -> str:
    """
    Format dynamic experience visual evidence for repair prompts.

    Includes frame annotations from the dynamic experience evaluation,
    letting the repair LLM "see" what's broken at a frame level.
    """
    annots = getattr(evidence, "frame_annotations", []) if hasattr(evidence, "frame_annotations") else []
    if not annots:
        return ""

    lines = ["## Visual Evidence (dynamic experience frames)"]
    lines.append("Automated testing captured these annotated keyframes:")
    for i, fa in enumerate(annots[:8]):
        label = fa.get("label", "")
        desc  = fa.get("description", "")
        diff  = fa.get("diff_from_prev", 0)
        marker = " ⚠️ NO CHANGE" if diff < 0.003 and i > 0 and label not in ("early_load", "first_paint") else ""
        lines.append(f"  Frame {i+1} [{label}]: {desc}{marker}")

    # Aggregate observations
    if hasattr(evidence, "dynamic_experience_ran") and evidence.dynamic_experience_ran:
        obs = []
        if not evidence.button_responsive:
            obs.append("Buttons clicked but NO visual change detected")
        if evidence.animation_detected:
            obs.append(f"Animation active (frame_change_rate={evidence.frame_change_rate:.2f})")
        elif evidence.frame_change_rate == 0:
            obs.append("No animation detected (static page or broken animation)")
        if evidence.has_below_fold:
            obs.append("Page has below-fold content (scrollable)")
        if evidence.hover_effects_count > 0:
            obs.append(f"{evidence.hover_effects_count} hover effect(s) detected")
        # Interaction latency
        avg_lat = getattr(evidence, "avg_interaction_latency_ms", None)
        if avg_lat is not None:
            if avg_lat > 500:
                obs.append(f"Interaction SLUGGISH: avg response time {avg_lat}ms (>500ms)")
            elif avg_lat > 100:
                obs.append(f"Interaction response time: {avg_lat}ms (acceptable)")
            else:
                obs.append(f"Interaction FAST: avg response time {avg_lat}ms")
        int_tout = getattr(evidence, "interactions_timed_out", 0)
        if int_tout > 0:
            obs.append(f"{int_tout} interaction(s) produced NO response (timed out)")
        # Responsive viewport
        resp_vp = getattr(evidence, "responsive_viewports_tested", 0)
        if resp_vp > 0:
            obs.append(f"Responsive: {resp_vp} extra viewport(s) tested (mobile/tablet)")
        if obs:
            lines.append("\nVisual observations:")
            for o in obs:
                lines.append(f"  - {o}")

    if annots:
        lines.append(
            "\nNote: Actual screenshots of these frames are attached to this message. "
            "Examine them carefully to see the real visual state, not just the code."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Game skeleton — verified canvas+focus+keydown+rAF pattern
# Injected into holistic_rewrite when diag.is_game to prevent re-roll failures
# ---------------------------------------------------------------------------

_GAME_SKELETON = """\
## Required Game Architecture (MUST follow this pattern)

Games need these four subsystems wired correctly to work:

```html
<!-- 1. Focusable canvas -->
<canvas id="gameCanvas" width="800" height="600" tabindex="0"></canvas>

<script>
// 2. Key state tracking on document (works even before focus)
const keys = {};
document.addEventListener('keydown', e => { keys[e.key] = true; e.preventDefault(); });
document.addEventListener('keyup', e => { keys[e.key] = false; });

// 3. Game loop with requestAnimationFrame (starts immediately)
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
canvas.focus(); // auto-focus on load

function gameLoop() {
    // Read key state: if (keys['ArrowLeft']) { ... }
    // Update game state
    // Clear and redraw canvas
    requestAnimationFrame(gameLoop);
}
gameLoop(); // start immediately, NOT after a click
</script>
```

**Critical rules:**
- Canvas MUST have `tabindex="0"` and `.focus()` on load
- keydown/keyup on `document` (NOT on canvas — document catches keys even before focus)
- `e.preventDefault()` in keydown to stop browser scrolling
- requestAnimationFrame loop MUST start immediately on page load
- Key state object (`keys`) MUST be read inside the game loop
- Do NOT gate the game loop behind a button click or user action\
"""


# ---------------------------------------------------------------------------
# FixGame — layer-specific game repair prompts (probe-driven)
# ---------------------------------------------------------------------------

FIX_GAME_INPUT = """\
You are an expert HTML/CSS/JavaScript game developer. An automated probe has \
confirmed that this game's keyboard input is completely broken. Fix ONLY the \
input handling layer — do not rewrite game logic or visuals.

## Task (what this game should do)
{query}

## Current scores
rendering={rendering}/20  visual_design={visual_design}/20
functionality={functionality}/25  interaction={interaction}/25  \
code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
{probe_evidence}
## Root-cause checklist (check in order, fix the FIRST one that applies)
1. **Canvas not focusable**: canvas needs `tabindex="0"` AND `.focus()` on load
2. **Listeners on wrong element**: keydown/keyup should be on `document`, not canvas
3. **Missing preventDefault**: browser scrolling intercepts arrow keys
4. **Key state not read**: keys object exists but game loop doesn't check it
5. **Game loop not started**: rAF loop gated behind a click/button that never fires

## Features working — DO NOT break these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Fix ONLY the input wiring — do NOT change game logic, physics, rendering, or visuals.
- After your fix, Arrow keys + WASD + Space must produce visible game responses.
- Verify: canvas has tabindex="0", document has keydown/keyup, loop reads key state.

{output_instructions}\
"""


FIX_GAME_LOOP = """\
You are an expert HTML/CSS/JavaScript game developer. An automated probe \
detected that this game's animation loop is not running — canvas exists but \
requestAnimationFrame is not being called. Fix ONLY the game loop.

## Task (what this game should do)
{query}

## Current scores
rendering={rendering}/20  visual_design={visual_design}/20
functionality={functionality}/25  interaction={interaction}/25  \
code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
{probe_evidence}
## Root-cause checklist (check in order)
1. **rAF never called**: gameLoop() or animate() function exists but is never invoked
2. **Loop gated behind event**: rAF starts only after a button click or user action
3. **Loop throws and stops**: error in update/draw kills the loop (wrap in try-catch)
4. **Missing canvas context**: getContext('2d') returns null (canvas not in DOM yet)
5. **DOMContentLoaded not firing**: script runs before canvas exists

## Features working — DO NOT break these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Fix ONLY the game loop initialization — ensure rAF starts on page load.
- Do NOT change game logic, physics, scoring, or visual design.
- Ensure the loop function calls requestAnimationFrame(itself) recursively.
- Wrap draw/update in try-catch so errors don't kill the loop.

{output_instructions}\
"""


FIX_GAME_CANVAS = """\
You are an expert HTML/CSS/JavaScript game developer. This game has a canvas \
element but it's showing nothing — the canvas is blank/empty. Fix the canvas \
rendering pipeline.

## Task (what this game should do)
{query}

## Current scores
rendering={rendering}/20  visual_design={visual_design}/20
functionality={functionality}/25  interaction={interaction}/25  \
code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
{probe_evidence}
## Root-cause checklist (check in order)
1. **Canvas size zero**: width/height not set or set to 0 — set explicit dimensions
2. **No getContext call**: canvas exists but ctx = canvas.getContext('2d') never called
3. **Drawing before DOM ready**: script tries to draw before canvas is in the DOM
4. **clearRect without redraw**: loop clears canvas but draw function is empty/broken
5. **CSS covers canvas**: overlay div with higher z-index hides the canvas
6. **Wrong canvas reference**: getElementById returns null (ID mismatch)

## Features working — DO NOT break these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Fix ONLY the canvas rendering — ensure something is drawn to the canvas on load.
- Set canvas width/height explicitly (not just via CSS).
- Ensure getContext('2d') is called and drawing operations execute.
- Do NOT change game logic or input handling if they exist.
- After your fix, the canvas should show visible game content immediately on page load.

{output_instructions}\
"""


FIX_GAME_OVERLAY = """\
You are an expert HTML/CSS/JavaScript game developer. Automated structural \
analysis detected visible overlays blocking the game content on page load. \
Fix the overlay/screen management so the game starts in a playable state.

## Task (what this game should do)
{query}

## Current scores
rendering={rendering}/20  visual_design={visual_design}/20
functionality={functionality}/25  interaction={interaction}/25  \
code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
{probe_evidence}
## Detected blocking overlays
{overlay_details}

## Root-cause checklist (check in order)
1. **Game-over screen on load**: gameState starts as 'over' or 'ended' — set to 'playing'
2. **Modal visible by default**: overlay div has display:flex/block — should be display:none
3. **Start screen never dismisses**: start button exists but has no click handler
4. **Z-index stacking**: game content behind overlay — fix z-index ordering
5. **Incorrect initial state**: score/lives start at 0 triggering game-over check

## Features working — DO NOT break these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Fix ONLY the initial state and overlay visibility.
- Ensure the game starts with overlays HIDDEN and game content VISIBLE.
- Set initial game state to 'playing' (or equivalent), NOT 'over'/'ended'.
- Ensure lives/score start with positive values (lives=3, score=0 is fine).
- Do NOT change game logic, physics, or visual design.
- If a start screen is intentional, ensure its "Play" button has a working handler.

{output_instructions}\
"""


FIX_GAME_GAMEPLAY = """\
You are an expert HTML/CSS/JavaScript game developer. This game renders and \
accepts input, but has gameplay logic bugs — the game mechanics don't work \
correctly. Fix the game logic without changing the visual design or input system.

## Task (what this game should do)
{query}

## Current scores
rendering={rendering}/20  visual_design={visual_design}/20
functionality={functionality}/25  interaction={interaction}/25  \
code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
{probe_evidence}
## Root-cause checklist — check these game logic systems in order
1. **Collision detection**: Are hitboxes/overlap checks correct? Common bugs:
   wrong coordinates, missing axis check, off-by-one in boundary math
2. **State machine**: Does the game properly transition between states
   (menu→playing→paused→game_over→restart)? Check for dead-end states.
3. **Scoring / win-lose conditions**: Does the score increment? Is there a
   win/lose check that actually triggers? Common bug: condition uses `=` not `==`.
4. **Level progression**: If multi-level, does the next level load properly?
   Check level data arrays and index bounds.
5. **Physics parameters**: Is gravity/velocity/friction reasonable? Objects
   shouldn't fly off screen or get stuck. Check boundary clamping.
6. **Timer / countdown**: Does it count correctly? Does it trigger game-over on zero?

## Features working — DO NOT break these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Fix ONLY gameplay logic bugs — collision, state transitions, scoring, physics.
- Do NOT change the visual design, CSS, layout, or color scheme.
- Do NOT add new features, game modes, levels, or enemy types.
- Do NOT rewrite the game from scratch — surgical fixes to specific functions.
- Do NOT modify the input handling or event listeners.
- After your fix, the core game loop (play→score→end→restart) should work correctly.

{output_instructions}\
"""


# ---------------------------------------------------------------------------
# BugFix
# ---------------------------------------------------------------------------

BUG_FIX = """\
You are an expert HTML/CSS/JavaScript developer. Fix the specific bugs below \
without touching anything that already works.

## Task (what this page should do)
{query}

## Current scores → targets
rendering={rendering}/20 (target 19+)  functionality={functionality}/25  \
interaction={interaction}/25  code_quality={code_quality}/10

{prev_iterations}
{probe_evidence}
{observer_evidence}
## Bugs to fix (CONFIRMED by objective tests)
{reliable_issues}

## Console errors
{console_errors}

{keyboard_hint}
## Evaluator summary
{summary}

## Features working — DO NOT break these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Fix ONLY the listed bugs and console errors.
- Do NOT rewrite working sections — minimal targeted changes only.
- If a fix requires adding missing code, add only what's needed.
- Ensure no new console errors are introduced.
- Preserve ALL items in the "Features working" list above.

{output_instructions}\
"""


# ---------------------------------------------------------------------------
# FeatureComplete
# ---------------------------------------------------------------------------

FEATURE_COMPLETE = """\
You are an expert HTML/CSS/JavaScript developer. The page partially implements \
the task. Complete all missing features to reach a production-quality result.

## Task (what this page should do)
{query}

## Current scores → targets
functionality={functionality}/25 → **target 23+**
interaction={interaction}/25    → **target 22+**
visual_design={visual_design}/20 → target 17+
rendering={rendering}/20  code_quality={code_quality}/10

{prev_iterations}
{probe_evidence}
{observer_evidence}
{requirement_checklist}
## Features missing / weak areas
{missing}

## Evaluator summary
{summary}

## Features working — preserve these exactly
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Implement ALL missing and broken features from the requirement checklist above.
- Preserve every working feature — do NOT remove or break existing code.
- Use only inline JS/CSS (no new external files unless already present).
- Ensure all user interactions are responsive and error-free.
- Aim for clean, readable code with meaningful names.

{output_instructions}\
"""


# ---------------------------------------------------------------------------
# VisualEnhance
# ---------------------------------------------------------------------------

VISUAL_ENHANCE = """\
You are an expert UI/UX designer and frontend developer. The page's functionality \
is solid but its visual quality is below standard. Elevate the visual design \
without touching any JS logic.

## Task (what this page should do)
{query}

## Current scores → targets
visual_design={visual_design}/20 → **target 18+**
rendering={rendering}/20         → target 19+
(functionality={functionality}/25 — already good, DO NOT break it)

{prev_iterations}
## Visual issues identified by evaluator
{visual_issues}

## Evaluator summary
{summary}

## What works well (preserve these)
{highlights}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Redesign the CSS for a modern, polished look: color palette, typography, spacing, layout, hierarchy.
- Add purposeful transitions/animations (entrance, hover, state changes) where they enhance UX.
- Do NOT modify any JavaScript logic or application behavior.
- Do NOT add/remove HTML elements that affect interactivity (inputs, buttons, canvas).
- All styles must remain inline (inside <style> tag or style attributes).
- Match the visual style to the content type (game → dark/vibrant; dashboard → clean/data-focused; tool → minimal/professional).

{output_instructions}\
"""


# ---------------------------------------------------------------------------
# HolisticRewrite
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Refinement prompts (Tier A: 80-94 → push toward 95+)
# ---------------------------------------------------------------------------

POLISH_VISUAL = """\
You are an expert UI/UX designer. This page already works well and scored \
{score}/100. Your job: elevate the visual design from good to excellent. \
CSS-only changes — do NOT touch JavaScript logic.

## Task
{query}

## Current scores (already good — your goal is PERFECTION)
visual_design={visual_design}/20 → **target 19-20**
rendering={rendering}/20  functionality={functionality}/25  \
interaction={interaction}/25  code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
{observer_evidence}
{probe_evidence}
## What the evaluator said
{summary}

## Features working — DO NOT break ANY of these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions — CSS polish ONLY (ADDITIVE changes)

### DO (safe visual improvements):
- Refine EXISTING color values, spacing, font-size, border-radius, shadows for a more polished look
- ADD transition/animation rules for hover states and entrance effects (keep under 300ms)
- ADD responsive media queries for mobile/tablet breakpoints
- ADD CSS custom properties (variables) for consistent theming

### DO NOT (these cause regressions):
- Do NOT add visual effects that don't already exist in the page — if there's no backdrop-filter,
  gradient background, or glassmorphism, don't invent one. Polish what's there, don't redesign.
- Do NOT modify or delete ANY existing CSS rules — only ADD new ones that override
- Do NOT modify any JavaScript, event handlers, or application behavior
- Do NOT change or remove HTML elements, classes, IDs, or structure
- Do NOT remove existing inline styles — add overriding rules in <style> instead
- Do NOT change layout fundamentals (flexbox/grid direction, position scheme)
- Do NOT change font-family to fonts that aren't already loaded

### Patch discipline:
- Make 2-4 small, focused patches. Each patch should target ONE specific aspect
  (e.g., color palette OR spacing OR hover effects — not all at once).
- Do NOT dump a giant block of 50+ new CSS rules in a single patch.
- If the page already looks polished and scores well, return empty patches.
- For game pages: avoid adding CSS that touches interactive elements (game pieces,
  canvas, SVG elements with onclick/event handlers). Only style background,
  typography, and non-interactive wrappers.

{output_instructions}\
"""


ENHANCE_INTERACTION = """\
You are an expert UX engineer. This page scored {score}/100 — it works but the \
interaction experience needs polish. Improve UX without breaking existing features.

## Task
{query}

## Current scores
interaction={interaction}/25 → **target 24-25**
functionality={functionality}/25  visual_design={visual_design}/20  \
rendering={rendering}/20  code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
{observer_evidence}
{probe_evidence}
## What the evaluator said
{summary}

## Features working — preserve ALL of these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions — interaction polish (ADDITIVE only)

### DO (safe UX improvements):
- ADD CSS hover/focus states to existing interactive elements (buttons, links, inputs)
- ADD CSS transitions for state changes (150-300ms ease)
- ADD cursor changes (pointer on clickable, not-allowed on disabled)
- ADD CSS focus-visible rings on elements that already have tabindex or are natively focusable
- ADD touch-action and min-size rules to improve mobile tap targets

### DO NOT (these cause regressions):
- Do NOT modify, remove, or rewrite ANY existing JavaScript functions or event handlers
- Do NOT add new event listeners (keydown, click, etc.) — only enhance CSS for existing ones
- Do NOT change element structure (add/remove/wrap HTML elements)
- Do NOT add tabindex to elements that don't already have it
- Do NOT add role="button" or aria-* attributes that change element semantics
- Do NOT change game logic, animation code, or state management
- Do NOT add interaction patterns that don't exist in the original (e.g., don't add drag-and-drop
  if the page doesn't have it, don't add keyboard shortcuts if there are none)
- If the page is purely visual (animation, illustration), make CSS-only micro-improvements

### Patch discipline:
- Make 2-4 small, focused patches. Each should target ONE specific UX aspect.
- If the page's interaction already works well, return empty patches.

{output_instructions}\
"""


REFINE_FUNCTIONALITY = """\
You are an expert JavaScript developer. This page scored {score}/100 — nearly \
complete but has minor functionality gaps. Fix edge cases and complete missing features.

## Task
{query}

## Current scores
functionality={functionality}/25 → **target 25**
interaction={interaction}/25  visual_design={visual_design}/20  \
rendering={rendering}/20  code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
## What the evaluator said
{summary}

{requirement_checklist}
## Missing / weak areas
{missing}

## Features working — preserve ALL of these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions — functionality completion
- Complete any missing features described in the task that aren't yet implemented.
- Fix edge cases: empty states, boundary values, rapid input, error recovery.
- Ensure all interactive elements produce the expected result.
- Add input validation where appropriate (but don't over-validate).
- Do NOT rewrite working code — make surgical additions only.
- Do NOT change the visual design or layout.
- Ensure no new console errors are introduced.

{output_instructions}\
"""


CODE_CLEANUP = """\
You are an expert frontend engineer focused on code quality. This page scored \
{score}/100 — it works and looks good, but the code quality score is low. \
Improve code quality while being EXTREMELY careful not to break anything.

## Task
{query}

## Current scores
code_quality={code_quality}/10 → **target 9-10**
rendering={rendering}/20  visual_design={visual_design}/20  \
functionality={functionality}/25  interaction={interaction}/25
**Total: {score}/100**

{prev_iterations}
## What the evaluator said
{summary}

## Features working — preserve ALL behavior and appearance
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions — SAFE code quality improvements only

### DO (safe changes):
- Replace generic `<div>` wrappers with semantic HTML (`<header>`, `<main>`, `<section>`, `<nav>`, `<footer>`, `<article>`)
- Fix inconsistent indentation and formatting
- Rename obviously unclear variables (e.g., `x` → `positionX`) but ONLY where the meaning is unambiguous
- If a CDN library is used for ONE function (e.g., jQuery just for `$(document).ready`), replace that ONE call with vanilla JS equivalent

### DO NOT (these cause regressions):
- Do NOT remove ANY JavaScript function, variable, or event listener — even if it looks unused
- Do NOT refactor callbacks, promises, or async patterns — restructuring control flow breaks timing
- Do NOT remove or reorganize CSS rules — order can matter for specificity
- Do NOT remove comments — they may document non-obvious behavior
- Do NOT replace a CDN library unless you can FULLY inline all its used features
- Do NOT change any `id`, `class`, `data-*` attribute values — other code may reference them

### Safety rule: when in doubt, DO NOT change it.

{output_instructions}\
"""


HOLISTIC_REWRITE = """\
You are an expert HTML/CSS/JavaScript developer. The existing attempt scored \
{score}/100. Write an improved, complete, high-quality implementation.

## Task
{query}

## Why the previous attempt needs improvement
{issues}

## Evaluator's diagnosis
{summary}

{observer_evidence}
{requirement_checklist}
{prev_iterations}
{visual_preservation}
{rubric}

## Requirements for your implementation
- **Rich visuals**: Use external resources generously to make the page beautiful — \
high-quality images from unsplash/picsum, Google Fonts for typography, SVG icons. \
Keep ALL external resources from the original HTML and ADD more where they improve quality.
- **Complete**: implement EVERY feature described in the task, including edge cases. \
Keep ALL form elements (inputs, selects, textareas), ALL screens/views, ALL game states.
- **Correct**: no console errors; all interactions must work as expected.
- **Code quality**: JS and CSS inline. Clear variable/function names; avoid deeply nested \
callbacks; use requestAnimationFrame for animations; handle errors gracefully.
- Target score: rendering≥19, visual_design≥19, functionality≥24, interaction≥24, \
code_quality≥9 (total ≥95).

## CRITICAL — Quality Rules (your score DEPENDS on these)

### Visual — NEVER simplify, always ELEVATE
- If the original has detailed SVG paths, illustrations, or complex shapes, your version \
must be EQUALLY or MORE detailed. Replacing detailed art with circles/rectangles = visual_design=0.
- Keep ALL gradients, filters, clip-paths, keyframes, transforms — then ADD more depth: \
layered shadows, backdrop-filter, richer color palettes, smoother easing.
- SVG `<path d="...">` data: copy original path data verbatim, then add enhancement paths. \
Never delete or simplify path data.
- Typography: keep or upgrade. Use proper font stacks and hierarchy. Never downgrade to \
plain sans-serif.
- Background: if original has gradient/textured/illustrated background, yours must too. \
Never replace with solid color.

### Functionality — NEVER reduce complexity
- If the original has multi-screen flow (menu→game→game-over→restart), yours must too. \
Do NOT collapse into a single screen.
- If the original has a state machine, level progression, scoring system, or data \
persistence — implement ALL of them with the SAME or greater depth.
- If the original has physics (gravity, velocity, bounce), collision detection, or \
particle systems — yours must have them too, at least as sophisticated.
- A "simpler version that works" is a FAILURE. The evaluator checks implementation depth, \
not just whether buttons respond.

### Self-verification (mentally check BEFORE outputting)
After writing your code, verify these common bugs:
1. **Initial state**: Does the page start in a USABLE state? Check that no "game over", \
"error", or "end" screen shows on load. The user should see the MAIN content immediately.
2. **Resources**: Any countable resource (ammo, lives, energy, attempts) must start with \
a POSITIVE value, not 0. Verify: `let lives = 3` not `let lives = 0`.
3. **Input matching**: If the task says "click/mouse/drag/tap", wire up mouse events. \
If it says "keyboard/arrow keys/WASD", wire up keyboard events. Don't mix them up.
4. **Event wiring**: Every button, clickable element, and interactive area MUST have an \
event handler (onclick, addEventListener) that actually calls a function. \
Creating a `<button>` without wiring it = broken.
5. **State transitions**: If there are multiple screens/states, verify the transition \
logic works: start→playing→end→restart must all be connected. Don't leave dead ends.
6. **Content consistency**: Images, icons, and media must match the page's theme and \
purpose. Don't use random stock photos for a card matching game — use themed icons/emoji. \
External resources should ENHANCE the theme, not clash with it.

### The test: someone comparing before/after should say "the new version is more polished, \
more detailed, and more feature-complete" — not "they gutted everything and made it basic".

{existing_html}
Reply with ONLY the complete HTML file. No explanation outside the code block.\
"""


# ---------------------------------------------------------------------------
# FixPlayability — fix keyboard / input bindings (probe-confirmed broken)
# ---------------------------------------------------------------------------

FIX_PLAYABILITY = """\
You are an expert HTML/CSS/JavaScript game developer. An automated keyboard \
probe has CONFIRMED that this game's keyboard input is completely broken — key \
events were dispatched to the page but produced ZERO visual response. Fix the \
input handling so the game is playable.

## Task (what this game should do)
{query}

## Current scores
rendering={rendering}/20  visual_design={visual_design}/20
functionality={functionality}/25  interaction={interaction}/25  \
code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
{probe_evidence}
## Automated keyboard probe results
- Keys tested: ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Space, Enter
- Keys that responded: NONE
- Visual change after key input: {keyboard_visual_change}

## Common causes of broken keyboard input (check in order)
1. **Canvas not focusable**: canvas needs `tabindex="0"` and `.focus()` on load
2. **Event listener on wrong element**: keydown/keyup on `document` but game \
expects them on canvas (or vice versa)
3. **Game loop not started**: requestAnimationFrame loop only starts after a \
button click, but key events arrive before that
4. **Key state tracking disconnected**: key state object exists but game loop \
doesn't read it, or reads a different variable
5. **Missing preventDefault**: default browser behavior (scrolling) interferes \
with game input

## Evaluator summary
{summary}

## Features working — DO NOT break these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Fix ONLY the input handling and game initialization.
- Ensure canvas has `tabindex="0"` and receives focus on page load.
- Ensure keydown/keyup listeners are attached to the correct element.
- Ensure the game loop starts immediately (not only after a click).
- Do NOT change visual design, layout, colors, or non-input game logic.
- Do NOT rewrite the entire game — surgical input fix only.
- Verify the game responds to Arrow keys, Space, and Enter after your fix.

{output_instructions}\
"""


# ---------------------------------------------------------------------------
# FixInteraction — fix general interaction responsiveness (not just keyboard)
# ---------------------------------------------------------------------------

VISUAL_ENRICHMENT = """\
You are an expert UI/UX designer and CSS specialist. This page scored \
{score}/100 — functionality is solid but visual quality is holding it back. \
A VLM has analyzed the page and identified specific visual issues. \
Your job: apply targeted CSS/HTML visual improvements to push toward 95+.

## Task
{query}

## Current scores
visual_design={visual_design}/20 → **target 19-20**
rendering={rendering}/20  functionality={functionality}/25  \
interaction={interaction}/25  code_quality={code_quality}/10
**Total: {score}/100**

{visual_diagnosis}
{prev_iterations}
{observer_evidence}
## Features working — DO NOT break ANY of these
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions — VLM-guided visual enrichment (CSS/HTML ONLY)

### Based on the VLM diagnosis above, implement these visual improvements:
- Address EVERY issue identified by the VLM analysis
- Focus on the CSS areas flagged: typography, color palette, spacing, depth, etc.
- Use Google Fonts for typography upgrades where appropriate
- Use CSS gradients, shadows, backdrop-filter for visual depth
- Add smooth transitions (150-300ms) for hover and state changes
- Consider responsive improvements if flagged

### CRITICAL CONSTRAINTS:
- Do NOT modify ANY JavaScript logic, event handlers, or application behavior
- Do NOT remove or restructure HTML elements that affect interactivity
- Do NOT change game logic, animation code, or state management
- ONLY modify/add CSS rules, HTML structure for visual elements, and external resources
- All existing features MUST continue working after your changes
- The page must still pass all interaction tests

### Game-specific safety rules (apply when page is a game):
- Do NOT add `transform` or `scale` to interactive game elements (buttons, canvas, SVG game pieces) — changes click/touch hitboxes
- Do NOT add `animation` or `transition` to elements that track game state (selected, active, correct, wrong states)
- Do NOT add `filter: drop-shadow()` or `backdrop-filter` inside game loops — causes frame drops
- Do NOT change `background-size` on multi-layered backgrounds — likely to break layout
- CSS additions should only target: background, fonts, colors, padding/margin of non-interactive wrappers

### Safety rule: if removing your CSS changes would break functionality, you've gone too far.

{output_instructions}\
"""


FIX_INTERACTION = """\
You are an expert HTML/CSS/JavaScript developer. This page's interaction \
system is broken or severely limited. The interaction score is {interaction}/25 \
— well below acceptable. Fix all interactive elements so they respond correctly.

## Task (what this page should do)
{query}

## Current scores
rendering={rendering}/20  visual_design={visual_design}/20
functionality={functionality}/25  interaction={interaction}/25  \
code_quality={code_quality}/10
**Total: {score}/100**

{prev_iterations}
{probe_evidence}
## Interaction issues identified
{interaction_issues}

## Evaluator summary
{summary}

## Features working — preserve these exactly
{preservation_list}

{rubric}

## Current HTML
```html
{html}
```

## Instructions
- Fix ALL broken interactive elements: buttons, inputs, drag handlers, \
keyboard listeners, touch events, click handlers.
- Ensure event listeners are attached to the correct elements and fire correctly.
- Ensure state updates propagate to the UI (score displays, status text, \
visual feedback on user action).
- Fix any disconnected event → state → render chains.
- Do NOT add keyboard/focus handling to decorative or display-only elements \
(SVG art, CSS animations, illustrations). Only fix elements that are ALREADY interactive.
- Do NOT change the visual design or layout significantly.
- Do NOT add new features — only fix existing broken interactions.
- Preserve all working features listed above.

{output_instructions}\
"""
