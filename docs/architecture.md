# Architecture

HTMLCure has two main Python packages.

## `htmleval`

`htmleval` runs evaluation:

1. `ExtractPhase`: extracts complete HTML from model responses and writes `game.html`.
2. `StaticAnalysisPhase`: inspects structure, scripts, styles, inputs, and media.
3. `RenderTestPhase`: renders in Playwright, captures screenshots, probes layout and console health.
4. `TestRunnerPhase`: executes benchmark test cases against the rendered page.
5. `AgentTestPhase`: optional browser-use autonomous interaction test.
6. `VisionEvalPhase`: optional vision LLM scoring and reporting.

Benchmark orchestration lives in `htmleval/benchmark/`:

- `loader.py`: loads JSONL benchmark files and normalizes taxonomy.
- `generator.py`: calls OpenAI-compatible APIs to generate HTML responses.
- `runner.py`: combines generation, checkpointing, evaluation, and analysis.
- `analysis.py`: computes aggregate metrics and comparison tables.
- `metrics.py`: pass@k helpers for multi-trial runs.

## `htmlrefine`

`htmlrefine` builds on `htmleval` for repair:

- Reads failed or low-scoring HTML records.
- Uses render/test/vision evidence to diagnose problems.
- Applies repair strategies such as bug fix, feature completion, visual enhancement, and holistic rewrite.
- Re-evaluates repaired HTML and emits improved records when quality gates pass.

The public release includes the core repair implementation but no private model
outputs, training data, or cluster-specific configs.

## Data Flow

```text
benchmark JSONL or response JSONL
  -> benchmark loader
  -> optional API generation checkpoint
  -> EvalContext records
  -> extraction/static/render/test/agent/vision phases
  -> results.jsonl
  -> analysis.json
```

All credentials should enter through environment variables or explicit local
configs excluded from git.
