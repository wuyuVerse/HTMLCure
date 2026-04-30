# Evaluation Protocol

This document describes the public HTMLCure benchmark workflow.

## Modes

- `fast`: skips the vision LLM phase and uses deterministic extraction, render,
  test-runner, and fallback scoring signals. This is useful for development,
  smoke tests, and low-cost regression checks.
- `full`: runs the normal evaluation pipeline, including the vision scoring
  phase when a valid evaluator model is configured.

The browser-use agent phase is disabled in the example configs by default because
it requires optional dependencies and a compatible LLM. Enable it only when you
intend to evaluate autonomous interaction.

## Standard Public Run

1. Install the package and Playwright browser.
2. Validate `benchmark/en`.
3. Configure model profiles with environment-variable credentials.
4. Generate responses or stage existing response JSONL files.
5. Run benchmark evaluation.
6. Report `analysis.json`, model profile, mode, generation parameters, commit
   hash, and any failed/missing rows.

Example:

```bash
python -m htmleval benchmark run benchmark/en \
  --output-dir outputs/provider_run \
  --config configs/eval.example.yaml \
  --models-config configs/models.yaml \
  --model example_api_model \
  --model-name example_api_model \
  --generate \
  --mode full
```

Eval-only with an existing response file:

```bash
MODEL_PATH=path/to/responses.jsonl \
MODEL_NAME=my_model \
OUTPUT_DIR=outputs/eval_only \
BENCHMARK_PATH=benchmark/en \
EVAL_CONFIG=configs/eval.example.yaml \
bash scripts/benchmark/run_eval_only_responses_benchmark.sh
```

## Response JSONL Format

Each generated response record should contain:

```json
{
  "id": "app_dashboard_001",
  "prompt": "Create a sales dashboard...",
  "response": "<!DOCTYPE html>..."
}
```

Additional metadata fields are preserved when present. Eval-only runs reuse a
response only when the `id` matches and, when available, the prompt matches the
current benchmark item.

## Scoring Outputs

`results.jsonl` contains one serialized evaluation record per item. Important
fields include:

- `line_number`: benchmark item id.
- `eval_status`: completion/failure status.
- `score.total`: final score.
- `score.test_pass_rate`, `score.tests_passed`, `score.tests_total`: test-runner metrics.
- `render_summary`: render status, screenshot count, and probe errors.
- `test_runner_summary.results_path`: detailed per-test-case result file.
- `phase_errors`: bounded error summaries by phase.

`analysis.json` aggregates results by category, difficulty, subtype, language,
dimension, coverage, failed tests, timing, and execution metadata.

## Leaderboard Constraints

For comparable public numbers:

- Use only committed benchmark files under `benchmark/en`.
- Use a committed or clearly documented config derived from `configs/*.example.yaml`.
- Do not include private prompts, private responses, partial repair outputs, or manual cherry-picks.
- Report incomplete generations separately from evaluated failures.
- Keep generated responses and raw per-item outputs separate from the benchmark repository unless licensing permits redistribution.
