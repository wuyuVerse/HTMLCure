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
if [[ ! -f "${MODELS_CONFIG}" ]]; then
  echo "models config not found: ${MODELS_CONFIG}" >&2
  exit 1
fi

NEED_INSTALL=0
if [[ ! -x "${BENCH_VENV_DIR}/bin/python" ]]; then
  echo "Creating benchmark generation venv: ${BENCH_VENV_DIR}"
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
RESPONSES_PATH="${OUTPUT_DIR}/responses.jsonl"
if [[ -n "${SEED_RESPONSES_PATH:-}" && -f "${SEED_RESPONSES_PATH}" && ! -s "${RESPONSES_PATH}" ]]; then
  cp "${SEED_RESPONSES_PATH}" "${RESPONSES_PATH}"
  echo "[api-generate] seeded responses from ${SEED_RESPONSES_PATH}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CMD=(python -m htmleval benchmark generate "$BENCHMARK_PATH"
  --output "$RESPONSES_PATH"
  --model-profile "$MODEL_PROFILE"
  --models-config "$MODELS_CONFIG")

if [[ -n "${BENCHMARK_LIMIT:-}" && "$BENCHMARK_LIMIT" != "0" ]]; then
  CMD+=(--limit "$BENCHMARK_LIMIT")
fi
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

echo "[api-generate] model_profile=${MODEL_PROFILE}"
echo "[api-generate] model_name=${MODEL_NAME}"
echo "[api-generate] models_config=${MODELS_CONFIG}"
echo "[api-generate] output=${RESPONSES_PATH}"
echo "[api-generate] bench=${BENCHMARK_PATH}"
"${CMD[@]}"

LINES="$(wc -l < "${RESPONSES_PATH}" | tr -d ' ')"
echo "[api-generate] complete lines=${LINES} output=${RESPONSES_PATH}"
