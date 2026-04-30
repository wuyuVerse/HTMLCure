#!/bin/bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-$(pwd)}"
cd "$ROOT_DIR"

BENCH_VENV_DIR="${BENCH_VENV_DIR:-${ROOT_DIR}/.benchmark_eval_venv}"
MODEL_PROFILE="${MODEL_PATH:-${MODEL_NAME:-}}"
MODEL_NAME="${MODEL_NAME:-${MODEL_PROFILE}}"
MODELS_CONFIG="${MODELS_CONFIG:-${ROOT_DIR}/configs/models.yaml}"
if [[ ! -f "${MODELS_CONFIG}" && -f "${ROOT_DIR}/configs/models.example.yaml" ]]; then
  MODELS_CONFIG="${ROOT_DIR}/configs/models.example.yaml"
fi

if [[ -z "${MODEL_PROFILE}" ]]; then
  echo "MODEL_PATH or MODEL_NAME must provide the benchmark model profile alias" >&2
  exit 1
fi
if [[ -z "${MODEL_NAME}" ]]; then
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
if [[ ! -f "${MODELS_CONFIG}" ]]; then
  echo "models config not found: ${MODELS_CONFIG}" >&2
  exit 1
fi

NEED_INSTALL=0
if [[ ! -x "${BENCH_VENV_DIR}/bin/python" ]]; then
  echo "Creating benchmark eval venv: ${BENCH_VENV_DIR}"
  python3 -m venv "${BENCH_VENV_DIR}"
  NEED_INSTALL=1
fi

if ! "${BENCH_VENV_DIR}/bin/python" -c "import openai, yaml, aiohttp, tqdm, playwright" >/dev/null 2>&1; then
  NEED_INSTALL=1
fi

if [[ "${NEED_INSTALL}" == "1" ]]; then
  # shellcheck disable=SC1090
  source "${BENCH_VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install openai aiohttp pyyaml tqdm pillow fastapi uvicorn playwright
else
  # shellcheck disable=SC1090
  source "${BENCH_VENV_DIR}/bin/activate"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PW_TEST_SCREENSHOT_NO_FONTS_READY=1

CMD=(python -m htmleval benchmark run "$BENCHMARK_PATH"
  --output-dir "$OUTPUT_DIR"
  --config "$EVAL_CONFIG"
  --model "$MODEL_PROFILE"
  --models-config "$MODELS_CONFIG"
  --model-name "$MODEL_NAME"
  --generate
  --mode "${BENCHMARK_MODE:-full}")

if [[ -n "${BENCHMARK_LIMIT:-}" && "$BENCHMARK_LIMIT" != "0" ]]; then
  CMD+=(--limit "$BENCHMARK_LIMIT")
fi
if [[ "${BENCHMARK_FORCE:-0}" == "1" ]]; then
  CMD+=(--force)
fi
if [[ -n "${GENERATE_CONCURRENCY:-}" && "${GENERATE_CONCURRENCY}" != "0" ]]; then
  CMD+=(--generate-concurrency "${GENERATE_CONCURRENCY}")
fi
if [[ -n "${GENERATE_TIMEOUT:-}" && "${GENERATE_TIMEOUT}" != "0" ]]; then
  CMD+=(--generate-timeout "${GENERATE_TIMEOUT}")
fi
if [[ -n "${GENERATE_TEMPERATURE:-}" ]]; then
  CMD+=(--generate-temperature "${GENERATE_TEMPERATURE}")
fi
if [[ -n "${GENERATE_MAX_TOKENS:-}" && "${GENERATE_MAX_TOKENS}" != "0" ]]; then
  CMD+=(--generate-max-tokens "${GENERATE_MAX_TOKENS}")
fi
if [[ -n "${OVERLAP_MODE:-}" ]]; then
  CMD+=(--overlap-mode "${OVERLAP_MODE}")
fi
if [[ -n "${OVERLAP_CHUNK_SIZE:-}" && "${OVERLAP_CHUNK_SIZE}" != "0" ]]; then
  CMD+=(--overlap-chunk-size "${OVERLAP_CHUNK_SIZE}")
fi
if [[ "${DISABLE_THINKING:-0}" == "1" ]]; then
  CMD+=(--disable-thinking)
fi

echo "[api-bench] model_profile=${MODEL_PROFILE}"
echo "[api-bench] model_name=${MODEL_NAME}"
echo "[api-bench] models_config=${MODELS_CONFIG}"
echo "[api-bench] output_dir=${OUTPUT_DIR}"
echo "[api-bench] bench=${BENCHMARK_PATH}"
"${CMD[@]}"
