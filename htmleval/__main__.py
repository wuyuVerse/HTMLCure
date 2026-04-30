"""
python -m htmleval [command] [options]

Commands:
  eval                  Score a single HTML file or string (Phases 0-4, or fewer with --skip-* flags)
  test                  Run phases 0-2 on a built-in smoke-test HTML (no LLM needed)
  benchmark run         Run benchmark evaluation end-to-end
  benchmark validate    Validate benchmark items schema (fast, no browser)
  benchmark summary     Analyze existing benchmark results JSONL
  benchmark compare     Compare multiple benchmark analyses side-by-side

Examples:
  # Quick smoke test — no config or API key needed:
  python -m htmleval test

  # Evaluate from a file (skip agent for speed):
  python -m htmleval eval --query "bouncing ball" --html page.html --skip-agent

  # Use config file:
  python -m htmleval eval --config configs/eval.example.yaml --query "..." --html "..."

  # Benchmark evaluation (bilingual: en/ and zh/):
  python -m htmleval benchmark run benchmark/en/ --config configs/eval.example.yaml --generate
  python -m htmleval benchmark run benchmark/zh/ --config configs/eval.example.yaml --generate
  python -m htmleval benchmark run benchmark/ --language en --limit 10
  python -m htmleval benchmark validate benchmark/en/
  python -m htmleval benchmark summary benchmark_results/results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Built-in smoke-test HTML
# ---------------------------------------------------------------------------

_SMOKE_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>htmleval smoke test</title>
  <style>
    body { margin: 0; background: #1a1a2e; display: flex;
           align-items: center; justify-content: center; height: 100vh; }
    canvas { border: 2px solid #e94560; border-radius: 4px; }
    #score { position: fixed; top: 12px; left: 12px; color: #fff;
             font: bold 18px monospace; }
  </style>
</head>
<body>
  <div id="score">Score: 0</div>
  <canvas id="c" width="480" height="360"></canvas>
  <script>
    var canvas = document.getElementById('c');
    var ctx = canvas.getContext('2d');
    var scoreEl = document.getElementById('score');
    var x = 240, y = 180, vx = 3, vy = 2, score = 0;
    var r = 14;

    function draw() {
      ctx.fillStyle = '#0f3460';
      ctx.fillRect(0, 0, 480, 360);

      // trail
      ctx.globalAlpha = 0.15;
      ctx.fillStyle = '#1a1a2e';
      ctx.fillRect(0, 0, 480, 360);
      ctx.globalAlpha = 1;

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = '#e94560';
      ctx.shadowColor = '#e94560';
      ctx.shadowBlur = 20;
      ctx.fill();
      ctx.shadowBlur = 0;

      if (x - r < 0 || x + r > 480) { vx = -vx; score++; }
      if (y - r < 0 || y + r > 360) { vy = -vy; score++; }
      x += vx; y += vy;
      scoreEl.textContent = 'Score: ' + score;
      requestAnimationFrame(draw);
    }

    document.addEventListener('keydown', function(e) {
      if (e.key === 'ArrowUp')    vy -= 1;
      if (e.key === 'ArrowDown')  vy += 1;
      if (e.key === 'ArrowLeft')  vx -= 1;
      if (e.key === 'ArrowRight') vx += 1;
    });

    draw();
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_test(args) -> int:
    """Run phases 0–2 on the built-in smoke test (no LLM needed)."""
    from htmleval import EvalConfig, build_pipeline, EvalContext

    config = EvalConfig()
    config.processing.skip_agent_phase  = True
    config.processing.skip_vision_phase = True
    config.workspace = str(Path(args.output_dir) if args.output_dir else Path("/tmp/htmleval_smoke"))

    pipeline = build_pipeline(config)
    ctx = EvalContext(
        query="Bouncing ball that changes direction with arrow keys and counts score.",
        response=_SMOKE_HTML,
        game_id="smoke_test",
    )

    print("htmleval smoke test — running phases 0, 1, 2 (no LLM required)...")
    ctx = await pipeline.evaluate(ctx)

    ok = True
    for name, pr in ctx.phase_results.items():
        tag = "✓" if pr.success else "✗"
        ms  = f"{pr.duration_ms:.0f}ms"
        print(f"  {tag} {name:<20} {ms}")
        if pr.errors:
            print(f"    ! {pr.errors[0]}")
        if not pr.success:
            ok = False

    sa = ctx.get_phase("static_analysis")
    rt = ctx.get_phase("render_test")
    if sa:
        print(f"\n  canvas={sa.data.get('has_canvas')}  "
              f"raf={sa.data.get('has_requestanimationframe')}  "
              f"inputs={sa.data.get('input_types')}")
    if rt:
        print(f"  rendered={rt.data.get('rendered')}  "
              f"title='{rt.data.get('page_title')}'  "
              f"console_errors={len(rt.data.get('console_errors', []))}  "
              f"screenshots={len(ctx.all_screenshots)}")

    if ctx.output_dir:
        print(f"\n  Output: {ctx.output_dir}")

    status = "PASSED" if ok else "FAILED"
    print(f"\nsmoke test: {status}")
    return 0 if ok else 1


async def cmd_eval(args) -> int:
    """Evaluate a single HTML page."""
    from htmleval import EvalConfig, build_pipeline, EvalContext
    from htmleval.core.config import load_config

    config = load_config(args.config) if args.config and Path(args.config).exists() else EvalConfig()
    if args.skip_agent:
        config.processing.skip_agent_phase = True
    if args.skip_vision:
        config.processing.skip_vision_phase = True
    if args.output_dir:
        config.workspace = args.output_dir

    # Load HTML
    html_path = Path(args.html)
    if html_path.exists():
        html_code = html_path.read_text(encoding="utf-8")
    else:
        html_code = args.html  # treat as raw HTML string

    pipeline = build_pipeline(config)
    ctx = EvalContext(
        query=args.query,
        response=html_code,
        game_id=args.id or "eval_001",
    )

    print(f"Evaluating: {ctx.game_id}")
    ctx = await pipeline.evaluate(ctx)

    print(f"\nStatus : {ctx.status}")
    print(f"Score  : {ctx.total_score}/100")
    for name, pr in ctx.phase_results.items():
        tag = "✓" if pr.success else "✗"
        print(f"  {tag} {name:<20} {pr.duration_ms:.0f}ms")
        if pr.errors:
            print(f"    ! {pr.errors[0]}")

    if ctx.final_score:
        print("\nDimension scores:")
        for dim in ("rendering", "visual_design", "functionality", "interaction", "code_quality"):
            d = ctx.final_score.get(dim, {})
            if isinstance(d, dict):
                print(f"  {dim:<15} {d.get('score', 0):>3}  {d.get('reason', '')[:80]}")
        if ctx.final_score.get("summary"):
            print(f"\nSummary: {ctx.final_score['summary'][:200]}")

    if ctx.output_dir:
        print(f"\nReport : {ctx.output_dir / 'report.md'}")

    return 0 if ctx.status == "completed" else 1


# ---------------------------------------------------------------------------
# Benchmark commands
# ---------------------------------------------------------------------------

async def cmd_benchmark_run(args) -> int:
    """Run benchmark evaluation end-to-end."""
    from htmleval import EvalConfig
    from htmleval.core.config import load_config
    from htmleval.benchmark.runner import run_benchmark
    from htmleval.benchmark.analysis import print_benchmark_report

    config = load_config(args.config) if args.config and Path(args.config).exists() else EvalConfig()
    if args.skip_agent:
        config.processing.skip_agent_phase = True
    if args.skip_vision:
        config.processing.skip_vision_phase = True

    # Resolve model profile (if --model given)
    gen_url = args.generate_url
    gen_model = args.generate_model
    gen_key = args.generate_key
    gen_conc = args.gen_concurrency
    eval_conc = args.eval_concurrency
    vlm_conc = args.vlm_concurrency
    overlap_mode = args.overlap_mode
    overlap_chunk_size = args.overlap_chunk_size
    gen_temp = args.generate_temperature
    gen_timeout = args.generate_timeout
    gen_max_tokens = args.generate_max_tokens
    disable_thinking = args.disable_thinking
    m_name = args.model_name

    if args.model_profile:
        from htmleval.benchmark.models import load_model_profile
        profile = load_model_profile(args.model_profile, args.models_config)
        # Profile provides defaults; explicit --generate-* flags override
        gen_url = gen_url or profile.get("base_url", "")
        gen_model = gen_model or profile.get("model", "")
        gen_key = gen_key or profile.get("api_key", "")
        if gen_conc is None:
            gen_conc = profile.get("concurrency", gen_conc)
        gen_temp = gen_temp if args.generate_temperature != 0.7 else profile.get("temperature", gen_temp)
        gen_timeout = gen_timeout if args.generate_timeout != 180 else profile.get("timeout", gen_timeout)
        m_name = m_name or args.model_profile
        logging.getLogger("htmleval").info(
            f"Model profile '{args.model_profile}': url={gen_url} model={gen_model}"
        )

    if gen_conc is None:
        gen_conc = config.processing.generation_concurrency
    if eval_conc is None:
        eval_conc = config.processing.evaluation_concurrency
    if vlm_conc is None:
        vlm_conc = config.processing.vlm_concurrency
    if overlap_mode is None:
        overlap_mode = config.processing.overlap_mode
    if overlap_chunk_size is None:
        overlap_chunk_size = config.processing.overlap_chunk_size

    # Keep the config object aligned with the split knobs so existing code paths
    # still see the intended evaluation-side defaults.
    config.processing.generation_concurrency = gen_conc
    config.processing.evaluation_concurrency = eval_conc
    config.processing.concurrency = eval_conc
    config.processing.vlm_concurrency = vlm_conc
    config.processing.max_llm_concurrency = vlm_conc
    config.processing.overlap_mode = overlap_mode
    config.processing.overlap_chunk_size = overlap_chunk_size

    logging.getLogger("htmleval").info(
        "Benchmark concurrency: gen=%s eval=%s vlm=%s overlap=%s/%s",
        gen_conc, eval_conc, vlm_conc, overlap_mode, overlap_chunk_size,
    )

    analysis = await run_benchmark(
        benchmark_path=args.path,
        config=config,
        output_dir=args.output_dir,
        limit=args.limit,
        force=args.force,
        language=args.language,
        category=args.category,
        difficulty=args.difficulty,
        generate=args.generate,
        generate_url=gen_url,
        generate_model=gen_model,
        generate_key=gen_key,
        generate_concurrency=gen_conc,
        generate_temperature=gen_temp,
        generate_timeout=gen_timeout,
        generate_max_tokens=gen_max_tokens,
        disable_thinking=disable_thinking,
        trials=args.trials,
        fast=(args.mode == "fast"),
        model_name=m_name,
        seed=args.seed,
        strict=args.strict,
    )

    if args.json:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    else:
        print_benchmark_report(analysis)
        timing = analysis.get("timing", {})
        if timing:
            parts = []
            if timing.get("generate_s"):
                parts.append(f"generate={timing['generate_s']:.0f}s")
            parts.append(f"evaluate={timing.get('evaluate_s', 0):.0f}s")
            parts.append(f"total={timing.get('total_s', 0):.0f}s")
            print(f"\nTiming: {', '.join(parts)}")
        # Reconstruct actual output dir (model_slug/lang subdirectory)
        model_info = analysis.get("model", {})
        model_slug = model_info.get("name", "")
        # Infer lang the same way runner.py does
        bp = Path(args.path)
        if bp.name in ("en", "zh"):
            _lang = bp.name
        elif bp.parent.name in ("en", "zh"):
            _lang = bp.parent.name
        else:
            _lang = args.language or "mixed"
        mode_name = "fast" if args.mode == "fast" else "full"
        actual_out = Path(args.output_dir) / model_slug / _lang / mode_name if model_slug else Path(args.output_dir)
        print(f"Results: {actual_out}/results.jsonl")
        print(f"Analysis: {actual_out}/analysis.json")
    return 0


def cmd_benchmark_validate(args) -> int:
    """Validate benchmark items schema (fast, no browser)."""
    from htmleval.benchmark.loader import load_benchmark_items
    from htmleval.benchmark.analysis import VALIDATOR_MODE, validate_items

    items = load_benchmark_items(args.path, language=args.language)
    errors = validate_items(items)

    if errors:
        print(f"Validation FAILED [{VALIDATOR_MODE}] — {len(errors)} error(s):\n")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print(f"Validation OK [{VALIDATOR_MODE}] — {len(items)} items, no errors.")
        return 0


def cmd_benchmark_compare(args) -> int:
    """Compare multiple benchmark analyses side-by-side."""
    from htmleval.benchmark.analysis import compare_analyses, print_comparison_report

    comparison = compare_analyses(args.paths)

    if args.json:
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
    else:
        print_comparison_report(comparison)
    return 0


async def cmd_benchmark_generate(args) -> int:
    """Generate HTML responses for benchmark items via LLM."""
    from htmleval.core.config import load_config
    from htmleval.benchmark.loader import load_benchmark_items
    from htmleval.benchmark.generator import generate_responses
    from htmleval.phases.extract import extract_complete_html

    # Resolve API params: CLI flag > model profile > config yaml > error
    config = load_config(args.config) if args.config and Path(args.config).exists() else None
    base_url = args.base_url
    api_key = args.api_key
    model = args.model

    if args.model_profile:
        from htmleval.benchmark.models import load_model_profile
        profile = load_model_profile(args.model_profile, args.models_config)
        base_url = base_url or profile.get("base_url", "")
        model = model or profile.get("model", "")
        api_key = api_key or profile.get("api_key", "")
        if args.concurrency == 8:  # argparse default
            args.concurrency = profile.get("concurrency", args.concurrency)
        if args.temperature == 0.7:
            args.temperature = profile.get("temperature", args.temperature)
        if args.timeout == 120:
            args.timeout = profile.get("timeout", args.timeout)

    base_url = base_url or (config and config.evaluator.base_url) or ""
    api_key = api_key or (config and config.evaluator.api_key) or "EMPTY"
    model = model or (config and config.evaluator.model) or ""
    if not base_url or not model:
        print("Error: --base-url and --model required (or provide --config with evaluator section)")
        return 1

    items = load_benchmark_items(args.path, language=args.language, category=args.category, difficulty=args.difficulty)
    if args.limit > 0:
        items = items[:args.limit]
    if not items:
        print("No items found.")
        return 1

    print(f"Generating {len(items)} responses via {model}")
    results = await generate_responses(
        items, base_url, api_key, model,
        concurrency=args.concurrency,
        temperature=args.temperature,
        timeout=args.timeout,
        seed=args.seed,
        output_path=args.output,
        max_tokens=args.max_tokens,
        disable_thinking=args.disable_thinking,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    valid_results = [
        item for item in results
        if extract_complete_html(item.get("response", "") or "") is not None
    ]
    with open(out, "w", encoding="utf-8") as f:
        for item in valid_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    ok = len(valid_results)
    failed = len(results) - ok
    print(f"Done: {ok}/{len(results)} valid responses ({failed} missing/invalid) -> {out}")
    if failed:
        errors: dict[str, int] = {}
        for item in results:
            if extract_complete_html(item.get("response", "") or "") is None:
                err = str(item.get("generation_error") or "missing_or_invalid")
                errors[err] = errors.get(err, 0) + 1
        print("Generation incomplete; error summary:")
        for err, count in sorted(errors.items(), key=lambda x: (-x[1], x[0]))[:8]:
            print(f"  {count}x {err}")
        return 2
    return 0


def cmd_benchmark_summary(args) -> int:
    """Analyze existing benchmark results JSONL."""
    from htmleval.benchmark.analysis import analyze_results, print_benchmark_report

    analysis = analyze_results(args.results)

    if args.json:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    else:
        print_benchmark_report(analysis)
        out_path = Path(args.results).parent / "analysis.json"
        out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nAnalysis written to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m htmleval",
        description="htmleval — Universal HTML evaluation framework",
    )
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command")

    # test
    p_test = sub.add_parser("test", help="Smoke test — phases 0-2, no LLM required")
    p_test.add_argument("--output-dir", default="", help="Output directory (default: /tmp/htmleval_smoke)")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a single HTML page")
    p_eval.add_argument("--query",       required=True, help="Task description")
    p_eval.add_argument("--html",        required=True, help="HTML file path or raw HTML string")
    p_eval.add_argument("--id",          default="eval_001", help="Evaluation ID")
    p_eval.add_argument("--config",      default="", help="Path to eval.yaml (optional)")
    p_eval.add_argument("--output-dir",  default="", help="Output directory")
    p_eval.add_argument("--skip-agent",  action="store_true", help="Skip Phase 3 agent test")
    p_eval.add_argument("--skip-vision", action="store_true", help="Skip Phase 4 vision LLM scoring")

    # benchmark (with sub-subcommands)
    p_bench = sub.add_parser("benchmark", help="Benchmark evaluation commands")
    bench_sub = p_bench.add_subparsers(dest="bench_command")

    # benchmark run
    p_bench_run = bench_sub.add_parser("run", help="Run benchmark evaluation end-to-end")
    p_bench_run.add_argument("path", help="Path to benchmark dir (en/, zh/, or parent) or file")
    p_bench_run.add_argument("--output-dir",  default="./benchmark_results", help="Output directory")
    p_bench_run.add_argument("--limit",       type=int, default=0, help="Max items to evaluate (0=all)")
    p_bench_run.add_argument("--skip-agent",  action="store_true", help="Skip Phase 3 agent test")
    p_bench_run.add_argument("--skip-vision", action="store_true", help="Skip Phase 4 vision LLM scoring")
    p_bench_run.add_argument("--config",      default="", help="Path to eval.yaml (optional)")
    p_bench_run.add_argument("--force",       action="store_true", help="Re-evaluate already-scored records")
    p_bench_run.add_argument("--language",    default="", help="Filter by language (en/zh)")
    p_bench_run.add_argument(
        "--category",
        default="",
        help="Filter by task family or legacy category (e.g. games_simulations, game, apps_tools)",
    )
    p_bench_run.add_argument("--difficulty",  default="", help="Filter by difficulty (easy/medium/hard)")
    p_bench_run.add_argument("--json",        action="store_true", help="Output analysis as JSON")
    # Generation flags (used with --generate)
    p_bench_run.add_argument("--generate",    action="store_true", help="Auto-generate responses for items without them")
    p_bench_run.add_argument("--generate-url",    default="", help="LLM API URL for generation (default: config evaluator)")
    p_bench_run.add_argument("--generate-model",  default="", help="Model for generation (default: config evaluator)")
    p_bench_run.add_argument("--generate-key",    default="", help="API key for generation")
    p_bench_run.add_argument("--gen-concurrency", "--generate-concurrency",
                             dest="gen_concurrency", type=int, default=None,
                             help="Generation parallelism (default: config generation_concurrency)")
    p_bench_run.add_argument("--eval-concurrency", "--evaluation-concurrency",
                             dest="eval_concurrency", type=int, default=None,
                             help="Evaluation parallelism (default: config evaluation_concurrency)")
    p_bench_run.add_argument("--vlm-concurrency", dest="vlm_concurrency",
                             type=int, default=None,
                             help="Vision-LM parallelism (default: config vlm_concurrency)")
    p_bench_run.add_argument("--overlap-mode", choices=["off", "chunked"],
                             default=None,
                             help="Generation/evaluation overlap strategy (default: config overlap_mode)")
    p_bench_run.add_argument("--overlap-chunk-size", type=int, default=None,
                             help="Chunk size for overlap mode (default: config overlap_chunk_size)")
    p_bench_run.add_argument("--generate-temperature", type=float, default=0.7)
    p_bench_run.add_argument("--generate-timeout",     type=int, default=180, help="Per-request generation timeout (seconds)")
    p_bench_run.add_argument("--generate-max-tokens",  type=int, default=0, help="Max completion tokens for generation (0 = provider default)")
    p_bench_run.add_argument("--disable-thinking",     action="store_true", help="Best-effort disable reasoning mode on local/vLLM generation APIs")
    # Model name for output organization
    p_bench_run.add_argument("--model-name", default="",
                             help="Model name for output organization (default: auto from config)")
    # Mode and trials
    p_bench_run.add_argument("--mode", choices=["fast", "full"], default="full",
                             help="fast = skip VLM (deterministic scoring only); full = normal (default)")
    p_bench_run.add_argument("--trials", type=int, default=1,
                             help="Number of evaluation trials for pass@k (default: 1)")
    p_bench_run.add_argument("--seed", type=int, default=0,
                             help="Seed for reproducible generation (0=disabled)")
    p_bench_run.add_argument("--strict", action="store_true",
                             help="Abort on schema validation errors")
    # Model profile shortcut
    p_bench_run.add_argument("--model", default="", dest="model_profile",
                             help="Select model profile from models.yaml (overrides --generate-url/model/key)")
    p_bench_run.add_argument("--models-config", default="",
                             help="Path to models.yaml (default: configs/models.yaml)")

    # benchmark generate
    p_bench_gen = bench_sub.add_parser("generate", help="Generate HTML responses via LLM")
    p_bench_gen.add_argument("path", help="Benchmark dir (en/, zh/) or file")
    p_bench_gen.add_argument("-o", "--output", required=True, help="Output JSONL path")
    p_bench_gen.add_argument("--config",      default="", help="eval.yaml (reads evaluator section)")
    p_bench_gen.add_argument("--base-url",    default="", help="LLM API base URL (overrides config)")
    p_bench_gen.add_argument("--model",       default="", help="Model name (overrides config)")
    p_bench_gen.add_argument("--api-key",     default="", help="API key (overrides config)")
    p_bench_gen.add_argument("--concurrency", type=int, default=8, help="Parallel requests")
    p_bench_gen.add_argument("--temperature", type=float, default=0.7)
    p_bench_gen.add_argument("--timeout",     type=int, default=120, help="Per-request timeout (seconds)")
    p_bench_gen.add_argument("--max-tokens",  type=int, default=0, help="Max completion tokens (0 = provider default)")
    p_bench_gen.add_argument("--disable-thinking", action="store_true", help="Best-effort disable reasoning mode on local/vLLM APIs")
    p_bench_gen.add_argument("--limit",       type=int, default=0, help="Max items (0=all)")
    p_bench_gen.add_argument("--language",    default="", help="Filter by language (en/zh)")
    p_bench_gen.add_argument(
        "--category",
        default="",
        help="Filter by task family or legacy category",
    )
    p_bench_gen.add_argument("--difficulty",  default="", help="Filter by difficulty")
    p_bench_gen.add_argument("--seed", type=int, default=0, help="Seed for reproducible generation (0=disabled)")
    # Model profile shortcut
    p_bench_gen.add_argument("--model-profile", default="",
                             help="Select model profile from models.yaml (overrides --base-url/model/api-key)")
    p_bench_gen.add_argument("--models-config", default="",
                             help="Path to models.yaml (default: configs/models.yaml)")

    # benchmark validate
    p_bench_val = bench_sub.add_parser("validate", help="Validate benchmark items schema")
    p_bench_val.add_argument("path", help="Path to benchmark dir (en/, zh/) or file")
    p_bench_val.add_argument("--language", default="", help="Filter by language (en/zh)")

    # benchmark compare
    p_bench_cmp = bench_sub.add_parser("compare", help="Compare multiple benchmark analyses")
    p_bench_cmp.add_argument("paths", nargs="+", help="Paths to analysis.json files or result directories")
    p_bench_cmp.add_argument("--json", action="store_true", help="Output as JSON")

    # benchmark summary
    p_bench_sum = bench_sub.add_parser("summary", help="Analyze existing results JSONL")
    p_bench_sum.add_argument("results", help="Path to results JSONL file")
    p_bench_sum.add_argument("--json", action="store_true", help="Output as JSON (no file write)")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "test":
        return asyncio.run(cmd_test(args))
    elif args.command == "eval":
        return asyncio.run(cmd_eval(args))
    elif args.command == "benchmark":
        if args.bench_command == "run":
            return asyncio.run(cmd_benchmark_run(args))
        elif args.bench_command == "generate":
            return asyncio.run(cmd_benchmark_generate(args))
        elif args.bench_command == "validate":
            return cmd_benchmark_validate(args)
        elif args.bench_command == "compare":
            return cmd_benchmark_compare(args)
        elif args.bench_command == "summary":
            return cmd_benchmark_summary(args)
        else:
            p_bench.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
