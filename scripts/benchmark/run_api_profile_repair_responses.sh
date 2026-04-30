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
if [[ -z "${ONLY_ID:-}" ]]; then
  echo "ONLY_ID is required" >&2
  exit 1
fi
if [[ ! -f "${MODELS_CONFIG}" ]]; then
  echo "models config not found: ${MODELS_CONFIG}" >&2
  exit 1
fi

NEED_INSTALL=0
if [[ ! -x "${BENCH_VENV_DIR}/bin/python" ]]; then
  echo "Creating benchmark repair venv: ${BENCH_VENV_DIR}"
  python3 -m venv "${BENCH_VENV_DIR}"
  NEED_INSTALL=1
fi

if ! "${BENCH_VENV_DIR}/bin/python" -c "import openai, yaml, aiohttp, tqdm" >/dev/null 2>&1; then
  NEED_INSTALL=1
fi

if [[ "${NEED_INSTALL}" == "1" ]]; then
  # shellcheck disable=SC1090
  source "${BENCH_VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install openai aiohttp pyyaml tqdm pillow
else
  # shellcheck disable=SC1090
  source "${BENCH_VENV_DIR}/bin/activate"
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CMD=(python scripts/benchmark/repair_missing_responses.py
  --benchmark-path "$BENCHMARK_PATH"
  --model-dir "$OUTPUT_DIR"
  --model-profile "$MODEL_PROFILE"
  --models-config "$MODELS_CONFIG"
  --only-id "$ONLY_ID")

if [[ -n "${GENERATE_CONCURRENCY:-}" && "${GENERATE_CONCURRENCY}" != "0" ]]; then
  CMD+=(--concurrency "${GENERATE_CONCURRENCY}")
fi
if [[ -n "${GENERATE_TIMEOUT:-}" && "${GENERATE_TIMEOUT}" != "0" ]]; then
  CMD+=(--timeout "${GENERATE_TIMEOUT}")
fi
if [[ -n "${GENERATE_TEMPERATURE:-}" ]]; then
  CMD+=(--temperature "${GENERATE_TEMPERATURE}")
fi
if [[ -n "${GENERATE_MAX_TOKENS:-}" && "${GENERATE_MAX_TOKENS}" != "0" ]]; then
  CMD+=(--max-tokens "${GENERATE_MAX_TOKENS}")
fi
if [[ "${DISABLE_THINKING:-0}" == "1" ]]; then
  CMD+=(--disable-thinking)
fi

echo "[api-repair] model_profile=${MODEL_PROFILE}"
echo "[api-repair] model_name=${MODEL_NAME}"
echo "[api-repair] models_config=${MODELS_CONFIG}"
echo "[api-repair] output=${OUTPUT_DIR}/responses.jsonl"
echo "[api-repair] only_id=${ONLY_ID}"
echo "[api-repair] bench=${BENCHMARK_PATH}"
"${CMD[@]}"

LINES="$(wc -l < "${OUTPUT_DIR}/responses.jsonl" | tr -d ' ')"
echo "[api-repair] checkpoint lines=${LINES} output=${OUTPUT_DIR}/responses.jsonl"
