#!/bin/bash
# ============================================================
# HTMLRefine — Single-Machine Launcher
# ============================================================
# Usage:
#   bash scripts/infra/run.sh configs/refine.example.yaml
#   bash scripts/infra/run.sh configs/refine.example.yaml --shard-id 3 --num-shards 64
#   bash scripts/infra/run.sh configs/refine.example.yaml --limit 10
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${1:?Usage: bash scripts/infra/run.sh <config.yaml> [--shard-id N --num-shards M] [--limit N] [--force]}"
shift
EXTRA_ARGS="$@"

# Activate venv if present
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

# Parse config
eval "$(python3 - "$CONFIG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
pr = cfg.get("processing", {})
ws = cfg.get("workspace", "./eval_workspace")
print(f'PORT={pr.get("port", 8890)}')
print(f'WORKSPACE="{ws}"')
PYEOF
)"

# Extract shard-id for log naming
SHARD_ID=""
ARGS_ARRAY=($EXTRA_ARGS)
for i in "${!ARGS_ARRAY[@]}"; do
    if [ "${ARGS_ARRAY[$i]}" = "--shard-id" ]; then
        SHARD_ID="${ARGS_ARRAY[$((i+1))]}"
        break
    fi
done

if [ -n "$SHARD_ID" ]; then
    LOG_TAG="shard_$(printf '%04d' $SHARD_ID)"
else
    LOG_TAG="server_${PORT}"
fi

mkdir -p "$WORKSPACE/reports" "$WORKSPACE/completed" "$WORKSPACE/failed" "$WORKSPACE/logs"

SERVER_LOG="$WORKSPACE/logs/${LOG_TAG}_server.log"
BATCH_LOG="$WORKSPACE/logs/${LOG_TAG}_batch.log"

echo "============================================================"
echo "  HTMLRefine Pipeline"
echo "  Config: $CONFIG  Port: $PORT  Args: $EXTRA_ARGS"
echo "============================================================"

# Kill old process on port
fuser -k "$PORT/tcp" 2>/dev/null || true
sleep 1

# Start server
echo "Starting server on port $PORT ..."
python3 scripts/infra/run_server.py --config "$CONFIG" --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "  PID=$SERVER_PID  log=$SERVER_LOG"

cleanup() {
    echo "Stopping server (PID=$SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for server
echo "Waiting for server ..."
MAX_WAIT=120
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        echo "  Ready! (${ELAPSED}s)"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if ! curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    echo "Server failed to start. Check: $SERVER_LOG"
    tail -20 "$SERVER_LOG" 2>/dev/null
    exit 1
fi

# Run batch
echo ""
echo "Starting batch evaluation ..."
python3 scripts/infra/run_batch.py --config "$CONFIG" $EXTRA_ARGS 2>&1 | tee "$BATCH_LOG"

echo ""
echo "Done. Output: $WORKSPACE"
