#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
BENCHMARK_PATH="${BENCHMARK_PATH:-benchmark/en}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/local_vllm_benchmark}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.example.yaml}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
INPUT_TOKEN_RESERVE="${INPUT_TOKEN_RESERVE:-1536}"
BENCHMARK_LIMIT="${BENCHMARK_LIMIT:-10}"
BENCHMARK_MODE="${BENCHMARK_MODE:-full}"
BENCHMARK_FORCE="${BENCHMARK_FORCE:-1}"
GENERATE_CONCURRENCY="${GENERATE_CONCURRENCY:-4}"
GENERATE_TIMEOUT="${GENERATE_TIMEOUT:-600}"
GENERATE_TEMPERATURE="${GENERATE_TEMPERATURE:-0.0}"
GENERATE_MAX_TOKENS="${GENERATE_MAX_TOKENS:-0}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-${REPO_ROOT}/.vllm_bench_venv}"
BENCH_VENV_DIR="${BENCH_VENV_DIR:-${REPO_ROOT}/.benchmark_eval_venv}"
WAIT_SECONDS="${WAIT_SECONDS:-1800}"

cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"
TASK_LOG="${OUTPUT_DIR}/task.log"
SERVER_LOG="${OUTPUT_DIR}/vllm_server.log"
BENCH_LOG="${OUTPUT_DIR}/benchmark.log"

exec >>"${TASK_LOG}" 2>&1
echo "[task] $(date -u '+%Y-%m-%d %H:%M:%S UTC') task start"
echo "[task] repo_root=${REPO_ROOT}"
echo "[task] model_path=${MODEL_PATH}"
echo "[task] tokenizer_path=${TOKENIZER_PATH:-<model_path>}"
echo "[task] venv_dir=${VLLM_VENV_DIR}"
echo "[task] bench_venv_dir=${BENCH_VENV_DIR}"

if (( INPUT_TOKEN_RESERVE < 0 )); then
    INPUT_TOKEN_RESERVE=0
fi
RESERVED_MAX_TOKENS="${MAX_MODEL_LEN}"
if (( MAX_MODEL_LEN > INPUT_TOKEN_RESERVE + 4096 )); then
    RESERVED_MAX_TOKENS=$((MAX_MODEL_LEN - INPUT_TOKEN_RESERVE))
fi
if [[ "${GENERATE_MAX_TOKENS}" == "0" ]]; then
    GENERATE_MAX_TOKENS="${RESERVED_MAX_TOKENS}"
fi
if (( GENERATE_MAX_TOKENS > RESERVED_MAX_TOKENS )); then
    echo "[task] clamp generate_max_tokens=${GENERATE_MAX_TOKENS} to reserved_max_tokens=${RESERVED_MAX_TOKENS}"
    GENERATE_MAX_TOKENS="${RESERVED_MAX_TOKENS}"
fi
echo "[task] max_model_len=${MAX_MODEL_LEN}"
echo "[task] input_token_reserve=${INPUT_TOKEN_RESERVE}"
echo "[task] generate_max_tokens=${GENERATE_MAX_TOKENS}"

NEED_INSTALL=0
if [[ ! -x "${VLLM_VENV_DIR}/bin/python" ]]; then
    echo "Creating vLLM venv: ${VLLM_VENV_DIR}"
    python3 -m venv "${VLLM_VENV_DIR}"
    NEED_INSTALL=1
fi

# shellcheck disable=SC1090
source "${VLLM_VENV_DIR}/bin/activate"
echo "[task] activated venv"

if ! python -c "import vllm" >/dev/null 2>&1; then
    NEED_INSTALL=1
fi

if [[ "${NEED_INSTALL}" == "1" ]]; then
    echo "Installing vLLM into ${VLLM_VENV_DIR}"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -U vllm
fi

if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "vLLM import failed after installation attempt"
    exit 1
fi
echo "[task] vllm import ok"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PW_TEST_SCREENSHOT_NO_FONTS_READY=1

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
    echo "Installing benchmark dependencies into ${BENCH_VENV_DIR}"
    # shellcheck disable=SC1090
    source "${BENCH_VENV_DIR}/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install openai aiohttp pyyaml tqdm pillow fastapi uvicorn playwright
else
    # shellcheck disable=SC1090
    source "${BENCH_VENV_DIR}/bin/activate"
fi

if ! python -c "import openai, yaml, aiohttp, tqdm, playwright" >/dev/null 2>&1; then
    echo "Benchmark eval environment missing required modules"
    exit 1
fi
echo "[task] benchmark eval env ok"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  Local vLLM Benchmark"
echo "  Model path   : ${MODEL_PATH}"
echo "  Model name   : ${MODEL_NAME}"
echo "  Benchmark    : ${BENCHMARK_PATH}"
echo "  Output dir   : ${OUTPUT_DIR}"
echo "  Eval config  : ${EVAL_CONFIG}"
echo "  TP size      : ${TENSOR_PARALLEL_SIZE}"
echo "  Port         : ${PORT}"
echo "  Limit        : ${BENCHMARK_LIMIT}"
echo "============================================================"

if [[ -n "${TOKENIZER_PATH}" ]]; then
    VLLM_CMD_TOKENIZER=(--tokenizer "${TOKENIZER_PATH}")
else
    VLLM_CMD_TOKENIZER=()
fi

"${VLLM_VENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    "${VLLM_CMD_TOKENIZER[@]}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Started vLLM server pid=${SERVER_PID} log=${SERVER_LOG}"

START_TS=$(date +%s)
while true; do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM server is ready"
        break
    fi
    NOW_TS=$(date +%s)
    if (( NOW_TS - START_TS >= WAIT_SECONDS )); then
        echo "Timed out waiting for vLLM server after ${WAIT_SECONDS}s"
        tail -n 200 "${SERVER_LOG}" || true
        exit 1
    fi
    sleep 10
done

BENCH_CMD=(
    python -m htmleval benchmark run "${BENCHMARK_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --config "${EVAL_CONFIG}"
    --generate
    --generate-url "http://127.0.0.1:${PORT}/v1"
    --generate-model "${MODEL_NAME}"
    --generate-key "EMPTY"
    --generate-concurrency "${GENERATE_CONCURRENCY}"
    --generate-timeout "${GENERATE_TIMEOUT}"
    --generate-temperature "${GENERATE_TEMPERATURE}"
    --model-name "${MODEL_NAME}"
    --limit "${BENCHMARK_LIMIT}"
    --mode "${BENCHMARK_MODE}"
)

if [[ "${BENCHMARK_FORCE}" == "1" ]]; then
    BENCH_CMD+=(--force)
fi

if [[ "${GENERATE_MAX_TOKENS}" != "0" ]]; then
    BENCH_CMD+=(--generate-max-tokens "${GENERATE_MAX_TOKENS}")
fi

if [[ "${DISABLE_THINKING}" == "1" ]]; then
    BENCH_CMD+=(--disable-thinking)
fi

echo "Running benchmark command:"
printf '  %q' "${BENCH_CMD[@]}"
printf '\n'

"${BENCH_CMD[@]}" 2>&1 | tee "${BENCH_LOG}"
