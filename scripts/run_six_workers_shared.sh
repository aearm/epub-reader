#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-5100}"
WORKER_PORTS="${WORKER_PORTS:-5001 5002 5003 5004 5005 5006}"

COORDINATOR_API_URL="${COORDINATOR_API_URL:-https://api.reader.psybytes.com}"
COGNITO_REGION="${COGNITO_REGION:-eu-west-1}"
COGNITO_CLIENT_ID="${COGNITO_CLIENT_ID:-1tgsjl3qo9cbb0gvonkhfvf31n}"
TOKEN_REFRESH_INTERVAL_SECONDS="${TOKEN_REFRESH_INTERVAL_SECONDS:-2400}"

# Recommended tuned defaults.
KOKORO_SERVER_SERIALIZE="${KOKORO_SERVER_SERIALIZE:-0}"
KOKORO_SERVER_WORKERS="${KOKORO_SERVER_WORKERS:-2}"
KOKORO_SERVER_QUEUE_MAXSIZE="${KOKORO_SERVER_QUEUE_MAXSIZE:-256}"
KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC="${KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC:-2}"
KOKORO_SERVER_RESULT_TIMEOUT_SEC="${KOKORO_SERVER_RESULT_TIMEOUT_SEC:-180}"
WORKER_TTS_POOL_SIZE="${WORKER_TTS_POOL_SIZE:-2}"
WORKER_BOOK_PARALLELISM="${WORKER_BOOK_PARALLELISM:-2}"

if [[ -z "${COGNITO_USERNAME:-}" ]]; then
  read -r -p "Cognito username/email: " COGNITO_USERNAME
fi
if [[ -z "${COGNITO_PASSWORD:-}" ]]; then
  read -r -s -p "Cognito password: " COGNITO_PASSWORD
  echo
fi

if [[ -z "$COGNITO_USERNAME" || -z "$COGNITO_PASSWORD" ]]; then
  echo "Missing Cognito credentials." >&2
  exit 2
fi

for port in $WORKER_PORTS "$MODEL_SERVER_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use. Stop existing processes first." >&2
    exit 1
  fi
done

mkdir -p static/state

MODEL_SERVER_URL="http://127.0.0.1:${MODEL_SERVER_PORT}"

nohup env \
  KOKORO_SERVER_HOST=127.0.0.1 \
  KOKORO_SERVER_PORT="$MODEL_SERVER_PORT" \
  KOKORO_SERVER_SERIALIZE="$KOKORO_SERVER_SERIALIZE" \
  KOKORO_SERVER_WORKERS="$KOKORO_SERVER_WORKERS" \
  KOKORO_SERVER_QUEUE_MAXSIZE="$KOKORO_SERVER_QUEUE_MAXSIZE" \
  KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC="$KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC" \
  KOKORO_SERVER_RESULT_TIMEOUT_SEC="$KOKORO_SERVER_RESULT_TIMEOUT_SEC" \
  "$PYTHON_BIN" -u scripts/kokoro_model_server.py >/tmp/kokoro_model_server.log 2>&1 &
echo "$!" >/tmp/kokoro_model_server.pid
echo "model-server pid=$(cat /tmp/kokoro_model_server.pid)"

sleep 2
curl -fsS --max-time 12 "${MODEL_SERVER_URL}/health" >/dev/null

for port in $WORKER_PORTS; do
  nohup env \
    FLASK_ENV=production \
    WORKER_PORT="$port" \
    WORKER_TTS_POOL_SIZE="$WORKER_TTS_POOL_SIZE" \
    WORKER_BOOK_PARALLELISM="$WORKER_BOOK_PARALLELISM" \
    WORKER_TTS_SERVER_URL="$MODEL_SERVER_URL" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    COORDINATOR_API_URL="$COORDINATOR_API_URL" \
    COORDINATOR_TOKEN_STATE_PATH="static/state/coordinator_token_${port}.json" \
    python app_multithreaded.py >/tmp/worker_${port}.log 2>&1 &
  echo "$!" >/tmp/worker_${port}.pid
  echo "worker:$port pid=$(cat /tmp/worker_${port}.pid)"
done

sleep 2
for port in $WORKER_PORTS; do
  curl -fsS --max-time 8 "http://127.0.0.1:${port}/worker/health" >/dev/null
done

for port in $WORKER_PORTS; do
  nohup env \
    COGNITO_REGION="$COGNITO_REGION" \
    COGNITO_CLIENT_ID="$COGNITO_CLIENT_ID" \
    COGNITO_USERNAME="$COGNITO_USERNAME" \
    COGNITO_PASSWORD="$COGNITO_PASSWORD" \
    WORKER_TOKEN_URL="http://127.0.0.1:${port}/worker/token" \
    TOKEN_REFRESH_INTERVAL_SECONDS="$TOKEN_REFRESH_INTERVAL_SECONDS" \
    "$PYTHON_BIN" -u scripts/sync_worker_token_local.py >/tmp/worker_token_sync_${port}.log 2>&1 &
  echo "$!" >/tmp/worker_token_sync_${port}.pid
  echo "token-sync:$port pid=$(cat /tmp/worker_token_sync_${port}.pid)"
done

echo
echo "Shared-model stack started (${WORKER_PORTS})."
echo "Model server health:"
echo "  curl http://127.0.0.1:${MODEL_SERVER_PORT}/health"
echo "Per worker:"
for port in $WORKER_PORTS; do
  echo "  curl http://127.0.0.1:${port}/coordinator/status"
done
echo
echo "Stop all:"
echo "  kill \$(cat /tmp/kokoro_model_server.pid 2>/dev/null) 2>/dev/null || true"
echo "  for p in $WORKER_PORTS; do"
echo "    kill \$(cat /tmp/worker_\${p}.pid /tmp/worker_token_sync_\${p}.pid 2>/dev/null) 2>/dev/null || true"
echo "  done"
