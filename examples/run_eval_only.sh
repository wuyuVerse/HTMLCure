#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

export MODEL_PATH="${MODEL_PATH:-examples/responses/sample_responses.jsonl}"
export MODEL_NAME="${MODEL_NAME:-sample}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_only_sample}"
export BENCHMARK_PATH="${BENCHMARK_PATH:-benchmark/en}"
export EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.example.yaml}"
export BENCHMARK_LIMIT="${BENCHMARK_LIMIT:-1}"
export BENCHMARK_MODE="${BENCHMARK_MODE:-fast}"

bash scripts/benchmark/run_eval_only_responses_benchmark.sh
