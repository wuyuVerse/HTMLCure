# HTMLCure

HTMLCure is an experience-driven evaluation and state-aware repair system for
LLM-generated interactive HTML. The core idea is that an HTML page should be
judged after it has been used in a browser, not from a first screenshot.

The evaluator renders a generated page, probes layout and interaction, records
browser evidence, selects keyframes from the executed trajectory, and then uses
a VLM only for the visual-design part of the score. The same browser trace also
drives repair: HTMLCure diagnoses the current page state, chooses a repair
family, re-executes candidates, and keeps only regression-safe improvements.

This repository contains the public evaluation/repair implementation and the
HTMLBench deterministic benchmark split used for reproducible model comparison.
It does not include private model outputs, training corpora, cluster configs, or
API credentials.

## Paper Framing

HTMLCure is organized around three claims from the paper:

- Experience-driven evaluation: evaluate interactive HTML through browser
  traces that include rendering, responsive layout, interactions, console state,
  DOM evidence, and curated visual keyframes.
- State-aware repair: route low, mid, and high quality pages to different repair
  policies instead of applying one global rewrite prompt.
- Repair as data construction: recover weak or partially usable generated pages
  under a measurable quality gate, rather than filtering them away.

HTMLBench is the deterministic benchmark view of this system. It exposes the
browser-executable part of the evaluator so model comparisons can be reproduced
without relying only on stochastic agent behavior.

## Repository Contents

- `htmleval/`: evaluation framework, benchmark CLI, browser pool,
  render/test runner, vision evaluator, metrics, and result viewer.
- `htmlrefine/`: state-aware repair/refinement pipeline built on evaluation
  evidence.
- `benchmark/en/`: public English HTMLBench-400 split across six task families.
- `benchmark/zh/`: Chinese prompt split aligned to the English metadata and
  test-case structure.
- `configs/*.example.yaml`: safe placeholder configs with no real endpoints or
  credentials.
- `scripts/`: benchmark launchers, bilingual alignment checks, audit scripts,
  and tooling helpers.

## Method Overview

```text
benchmark prompt or response JSONL
  -> extract complete HTML
  -> static analysis
  -> browser render and interaction probes
  -> deterministic test runner
  -> optional browser-use agent phase
  -> optional VLM visual scoring over curated keyframes
  -> results.jsonl and analysis.json
```

### Five-Dimensional Scoring

HTMLBench uses a frozen 6000-test-case scoring pool. Coverage/completion is
reported as execution metadata only; it is not a score dimension. For
interactive prompts, the direct 100-point rubric is:

| Dimension | Max | Primary evidence |
|---|---:|---|
| Rendering | 10 | browser load, visible page health, console/runtime state |
| Visual design | 20 | VLM judgment over experienced keyframes |
| Functionality | 55 | prompt-grounded deterministic test pass rate |
| Interaction | 10 | click, keyboard, form, animation, game, and state-change probes |
| Code quality | 5 | static HTML/CSS/JS hygiene and implementation health |

For non-interactive prompts, the interaction budget is reassigned to
functionality, yielding `rendering=10`, `visual_design=20`,
`functionality=65`, `interaction=0`, and `code_quality=5`. Functionality is
linear in the frozen weighted test-case pass rate; there is no nonlinear
normalization curve.

### State-Aware Repair

| Page state | Score band | Typical policy |
|---|---:|---|
| Low | `<40` | broad rewrite or feature completion |
| Mid | `40-79` | diagnosis-guided bug, interaction, game, or visual repair |
| High | `>=80` | conservative refinement and regression protection |

Every candidate repair is re-evaluated. HTMLCure accepts improvements only when
they survive the same browser protocol and do not damage already-working
dimensions.

## HTMLBench-400

HTMLBench evaluates single-file HTML generation across six user-intent task
families:

| Family | Items | Description |
|---|---:|---|
| `apps_tools` | 105 | Dashboards, calculators, CRUD tools, forms, editors, and productivity apps. |
| `content_marketing` | 110 | Landing pages, docs, pricing, FAQ, articles, portfolios, and marketing pages. |
| `data_visualization` | 35 | SVG/canvas charts, dashboards, maps, and interactive data views. |
| `games_simulations` | 55 | Browser games and simulations with keyboard, mouse, or time behavior. |
| `three_d_webgl` | 20 | Three.js/WebGL scenes with geometry, lights, animation, and controls. |
| `visual_art_animation` | 75 | CSS/SVG/canvas visual art, animation, and generative compositions. |

The Chinese split keeps Chinese prompts while mirroring English item ids,
metadata, test-case ids, weights, and action sequences. Use the bilingual audit
before publishing benchmark changes:

```bash
python scripts/analysis/audit_bilingual_alignment.py --strict-fields
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m playwright install chromium
```

For optional browser-use agent tests:

```bash
pip install -e ".[agent]"
```

## Offline Checks

Validate benchmark schemas:

```bash
python -m htmleval benchmark validate benchmark/en
python -m htmleval benchmark validate benchmark/zh
```

Run the built-in evaluator smoke test:

```bash
python -m htmleval test --output-dir outputs/smoke
```

Run an eval-only benchmark sample with a bundled response:

```bash
bash examples/run_eval_only.sh
```

This writes:

```text
outputs/eval_only_sample/sample/en/fast/results.jsonl
outputs/eval_only_sample/sample/en/fast/analysis.json
```

## API Model Configuration

Copy an example config and set credentials through environment variables:

```bash
cp configs/models.example.yaml configs/models.yaml
export HTMLCURE_API_KEY="your-api-key"
```

Generate and evaluate with an OpenAI-compatible provider:

```bash
python -m htmleval benchmark run benchmark/en \
  --output-dir outputs/provider_run \
  --config configs/eval.example.yaml \
  --models-config configs/models.yaml \
  --model example_api_model \
  --model-name example_api_model \
  --generate \
  --mode full \
  --limit 3
```

For a local vLLM server:

```bash
MODEL_PATH="$HOME/models/local-html-model" MODEL_NAME=local-html-model bash scripts/benchmark/run_local_vllm_benchmark.sh
```

## Output Format

Benchmark runs are organized as:

```text
<output-dir>/<model-name>/<language>/<mode>/
  responses.jsonl        # generated or staged responses at model/language level
  results.jsonl          # per-item evaluation records
  analysis.json          # aggregate scores and breakdowns
  .state/                # live run status
```

Key `analysis.json` fields:

- `overall`: count, average score, completion rate, tier distribution, confidence
  intervals, and test pass rate.
- `by_category`: metrics grouped by the six task families.
- `by_difficulty`, `by_subtype`, `by_language`: additional breakdowns.
- `failed_tests`: most frequent item-level test failures.
- `timing`: generation/evaluation wall-clock summary.
- `execution`: concurrency, mode, score version, and config fingerprint.
- `model`: normalized and raw model names.

## Reproducibility

Leaderboard-style numbers should be produced only from public benchmark files
and committed/example-compatible configs. Do not mix private prompts, private
responses, partially repaired outputs, or manually selected examples into public
claims. Report the commit hash, model profile, mode (`fast` or `full`),
generation parameters, and `analysis.json`.

`fast` mode skips the VLM phase and is intended for deterministic regression
checks. `full` mode runs the complete pipeline, including VLM visual scoring
when a compatible evaluator is configured.

## Contributing

Run these before opening a PR:

```bash
python -m htmleval benchmark validate benchmark/en
python -m htmleval benchmark validate benchmark/zh
python scripts/analysis/audit_bilingual_alignment.py --strict-fields
python scripts/analysis/audit_benchmark_solidity.py --benchmark benchmark/en
python -m compileall htmleval htmlrefine scripts
```

Do not commit API keys, provider endpoints, private model outputs, local
responses, or full evaluation artifacts.

## Citation

The paper citation will be added when the public manuscript is released. Until
then, please cite this repository as:

```bibtex
@misc{htmlcure2026,
  title = {HTMLCure: Experience-Driven Evaluation and State-Aware Repair for LLM-Generated Interactive HTML},
  year = {2026},
  howpublished = {\url{https://github.com/wuyuVerse/HTMLCure}}
}
```

## License

Code is released under Apache-2.0. Benchmark data is released for research and
reproducibility; see `NOTICE` for data-use notes.
