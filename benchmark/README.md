# Benchmark Layout

Current benchmark evaluation uses the six task families from the paper:

- `games_simulations`
- `apps_tools`
- `data_visualization`
- `visual_art_animation`
- `three_d_webgl`
- `content_marketing`

Active files live under:

- `benchmark/en/`: English benchmark items
- `benchmark/zh/`: Chinese benchmark items

Everything under `benchmark/archive/` is kept only for historical reference.
Those files are not part of the current benchmark entrypoint unless someone
manually points a script at them.

Archive structure:

- `benchmark/archive/snapshots/`: old benchmark snapshots before redesign rounds
- `benchmark/archive/backfill/`: one-off DeepSeek backfill splits, generated
  response dumps, and temporary repair artifacts

Each item keeps:

- `category`: canonical six-family task-family slug
- `source_category`: legacy ten-way source bucket retained for backward analysis

Common commands:

```bash
python3 -m htmleval benchmark validate benchmark/en/
python3 -m htmleval benchmark run benchmark/en/ --config configs/eval.yaml ...
```

## Scoring Overview

The current benchmark freezes a 6000-test-case score-bearing TC pool in
`tc_selection_20260430_6000.json`. Every retained TC is executed and counted in
the weighted TC pass rate. Coverage/completion is reported as execution
metadata only; it is not a score dimension.

The English and Chinese splits use the same frozen item/TC ids. The Chinese
split keeps Chinese prompts but uses the aligned frozen TC structure so both
languages exercise the same score-bearing pool.

The current benchmark uses a fixed 100-point rubric for interactive prompts:

- `rendering`: 10
- `visual_design`: 20
- `functionality`: 55
- `interaction`: 10
- `code_quality`: 5
  This compatibility key represents implementation quality of the produced
  HTML/CSS/JS artifact, not visual beauty.

For non-interactive prompts, the interaction budget is reassigned to task-fit
visual/functionality evidence rather than penalizing prompts that do not ask for
controls:

- `rendering`: 10
- `visual_design`: 20
- `functionality`: 65
- `interaction`: 0
- `code_quality`: 5

The rubric is intentionally experience-first:

- deterministic evidence totals 80 points for interactive prompts
  `rendering + functionality + interaction + code_quality`
- VLM-led visual judgment totals 20 points for interactive prompts
  `visual_design`

This keeps the benchmark aligned with the paper's core idea: the evaluator
should primarily reward what the page actually does and how it feels to use,
while treating implementation quality as a bounded constraint rather than a
dominant source-code-style preference.

Notes:

- Functionality is linear: `filtered_weighted_tc_pass_rate * functionality_budget`.
  There is no post-hoc normalization and no nonlinear pass-rate curve.
- Static-only rendering is capped only for prompts that require interaction.
  Correct non-interactive CSS-only pages can still receive full rendering
  credit.
- Final `total_score` is the direct 100-point contribution sum.
- CDN / external resources are not penalized by themselves. They only matter if
  they cause real rendering or runtime failures.
- For prompts marked `has_interaction: false`, the `interaction` dimension is
  skipped and its budget is reassigned as described above.
