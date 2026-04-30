# Benchmark Design

HTMLCure evaluates generated single-file HTML pages across six task families:
apps/tools, content/marketing, data visualization, games/simulations,
Three.js/WebGL, and visual art/animation.

## Item Structure

Each benchmark item is a JSONL record with:

- `id`: stable item identifier.
- `category`: one of the six canonical families.
- `sub_type`: finer-grained task subtype.
- `difficulty`: `easy`, `medium`, or `hard`.
- `language`: `en` or `zh`.
- `prompt`: user-facing generation task.
- `test_cases`: browser-executable checks.
- `has_interaction`: whether user interaction is expected.

The public English split has 400 items:

| Family | Items |
|---|---:|
| `apps_tools` | 105 |
| `content_marketing` | 110 |
| `data_visualization` | 35 |
| `games_simulations` | 55 |
| `three_d_webgl` | 20 |
| `visual_art_animation` | 75 |

## Test Case Principles

Test cases are designed to be prompt-grounded rather than implementation-specific.
They should verify observable content, structure, visual surfaces, state changes,
and user interactions without requiring a particular framework, class name,
variable name, or DOM hierarchy.

Acceptable checks include:

- Required text, values, labels, charts, controls, and sections from the prompt.
- Browser rendering and console health.
- Interaction state changes tied to prompt requirements.
- Responsive layout checks.
- Accessibility and semantic HTML basics.
- Screenshot-change checks when seeded by an explicit screenshot and anchored to a prompt-specific behavior.

Avoid:

- Checking only for a generic canvas/SVG without prompt-specific evidence.
- Hidden source-code-only assumptions.
- Class-name or framework overfitting.
- Arbitrary screenshot changes without semantic/state binding.
- Tests that require real credentials, payments, network calls, or private services.

## Validation

Run schema and solidity checks before publishing benchmark changes:

```bash
python -m htmleval benchmark validate benchmark/en
python scripts/analysis/audit_benchmark_solidity.py --benchmark benchmark/en
```

The validator catches hard schema/logic errors. The audit script is a triage tool
for duplicate evidence, shallow visual checks, cross-template leftovers, and high
test-case density. Audit warnings are not automatic failures; they identify items
for human review.
