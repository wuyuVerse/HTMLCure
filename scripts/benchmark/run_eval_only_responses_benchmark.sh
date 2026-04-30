#!/bin/bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-$(pwd)}"
cd "$ROOT_DIR"

BENCH_VENV_DIR="${BENCH_VENV_DIR:-${ROOT_DIR}/.benchmark_eval_venv}"

abspath() {
  python3 - "$1" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH must point to an existing responses.jsonl source" >&2
  exit 1
fi
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "MODEL_NAME is required" >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is required" >&2
  exit 1
fi
if [[ -z "${BENCHMARK_PATH:-}" ]]; then
  echo "BENCHMARK_PATH is required" >&2
  exit 1
fi
if [[ -z "${EVAL_CONFIG:-}" ]]; then
  echo "EVAL_CONFIG is required" >&2
  exit 1
fi

MODEL_SLUG="$(python3 - "$MODEL_NAME" <<'PY'
import sys
raw = sys.argv[1]
print(raw.rstrip("/").split("/")[-1].lower() if raw else "unknown")
PY
)"

RESP_DIR="$OUTPUT_DIR/$MODEL_SLUG/en"
STAGED_RESP="$RESP_DIR/responses.jsonl"

mkdir -p "$RESP_DIR"
MODEL_PATH_ABS="$(abspath "$MODEL_PATH")"
STAGED_RESP_ABS="$(abspath "$STAGED_RESP")"

if [[ "$MODEL_PATH_ABS" == "$STAGED_RESP_ABS" ]]; then
  if [[ -L "$STAGED_RESP" ]]; then
    echo "MODEL_PATH points to the staging destination itself: $STAGED_RESP" >&2
    echo "Refusing to create a self-referential responses.jsonl symlink. Point MODEL_PATH at the real source responses file." >&2
    exit 2
  fi
  if [[ ! -f "$STAGED_RESP" ]]; then
    echo "MODEL_PATH points to the staging destination itself, but no readable file exists there: $STAGED_RESP" >&2
    echo "Point MODEL_PATH at the real source responses file." >&2
    exit 2
  fi
else
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "responses source not found or unreadable: $MODEL_PATH" >&2
    exit 1
  fi
  ln -sfn "$MODEL_PATH_ABS" "$STAGED_RESP"
fi

BENCH_NEED_INSTALL=0
if [[ ! -x "${BENCH_VENV_DIR}/bin/python" ]]; then
  echo "Creating benchmark eval venv: ${BENCH_VENV_DIR}"
  python3 -m venv "${BENCH_VENV_DIR}"
  BENCH_NEED_INSTALL=1
fi

if ! "${BENCH_VENV_DIR}/bin/python" -c "import openai, yaml, aiohttp, tqdm, playwright" >/dev/null 2>&1; then
  BENCH_NEED_INSTALL=1
fi

if [[ "${BENCH_NEED_INSTALL}" == "1" ]]; then
  # shellcheck disable=SC1090
  source "${BENCH_VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install openai aiohttp pyyaml tqdm pillow fastapi uvicorn playwright
else
  # shellcheck disable=SC1090
  source "${BENCH_VENV_DIR}/bin/activate"
fi

if ! python -c "import openai, yaml, aiohttp, tqdm, playwright" >/dev/null 2>&1; then
  echo "Benchmark eval environment missing required modules" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PW_TEST_SCREENSHOT_NO_FONTS_READY=1

CMD=(python -m htmleval benchmark run "$BENCHMARK_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --mode "${BENCHMARK_MODE:-full}" \
  --config "$EVAL_CONFIG")

if [[ -n "${BENCHMARK_LIMIT:-}" && "$BENCHMARK_LIMIT" != "0" ]]; then
  CMD+=(--limit "$BENCHMARK_LIMIT")
fi
if [[ "${BENCHMARK_FORCE:-0}" == "1" ]]; then
  CMD+=(--force)
fi

echo "[eval-only] model=$MODEL_NAME"
echo "[eval-only] model_slug=$MODEL_SLUG"
echo "[eval-only] responses=$MODEL_PATH"
echo "[eval-only] staged_responses=$STAGED_RESP"
echo "[eval-only] output_dir=$OUTPUT_DIR"
echo "[eval-only] bench=$BENCHMARK_PATH"
"${CMD[@]}"
