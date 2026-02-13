#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
PORT_A="${PORT_A:-5001}"
PORT_B="${PORT_B:-5002}"
MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-5100}"
KOKORO_SERVER_SERIALIZE="${KOKORO_SERVER_SERIALIZE:-1}"
KOKORO_SERVER_WORKERS="${KOKORO_SERVER_WORKERS:-1}"
KOKORO_SERVER_QUEUE_MAXSIZE="${KOKORO_SERVER_QUEUE_MAXSIZE:-256}"
KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC="${KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC:-2}"
KOKORO_SERVER_RESULT_TIMEOUT_SEC="${KOKORO_SERVER_RESULT_TIMEOUT_SEC:-180}"
AUTOSCALE_ENABLED="${AUTOSCALE_ENABLED:-1}"
AUTOSCALE_INTERVAL_SEC="${AUTOSCALE_INTERVAL_SEC:-8}"
AUTOSCALE_SCALE_UP_QUEUE_DEPTH="${AUTOSCALE_SCALE_UP_QUEUE_DEPTH:-80}"
AUTOSCALE_SCALE_DOWN_QUEUE_DEPTH="${AUTOSCALE_SCALE_DOWN_QUEUE_DEPTH:-15}"
AUTOSCALE_SCALE_DOWN_STREAK="${AUTOSCALE_SCALE_DOWN_STREAK:-6}"
BURST_PORT="${BURST_PORT:-5003}"
BURST_WORKER_TTS_POOL_SIZE="${BURST_WORKER_TTS_POOL_SIZE:-2}"
BURST_WORKER_BOOK_PARALLELISM="${BURST_WORKER_BOOK_PARALLELISM:-2}"

COORDINATOR_API_URL="${COORDINATOR_API_URL:-https://api.reader.psybytes.com}"
COGNITO_REGION="${COGNITO_REGION:-eu-west-1}"
COGNITO_CLIENT_ID="${COGNITO_CLIENT_ID:-1tgsjl3qo9cbb0gvonkhfvf31n}"
TOKEN_REFRESH_INTERVAL_SECONDS="${TOKEN_REFRESH_INTERVAL_SECONDS:-2400}"

# With a single shared model, keep client-side parallelism modest.
WORKER_TTS_POOL_SIZE="${WORKER_TTS_POOL_SIZE:-3}"
WORKER_BOOK_PARALLELISM="${WORKER_BOOK_PARALLELISM:-3}"

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

for port in "$PORT_A" "$PORT_B" "$MODEL_SERVER_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use. Stop existing process first." >&2
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
echo "model-server pid=$(cat /tmp/kokoro_model_server.pid) log=/tmp/kokoro_model_server.log"

sleep 2
curl -fsS --max-time 12 "${MODEL_SERVER_URL}/health" >/dev/null

start_worker() {
  local port="$1"
  local log_file="/tmp/worker_${port}.log"
  local pid_file="/tmp/worker_${port}.pid"
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
    python app_multithreaded.py >"$log_file" 2>&1 &
  echo "$!" >"$pid_file"
  echo "worker:$port pid=$(cat "$pid_file") log=$log_file"
}

start_sync() {
  local port="$1"
  local log_file="/tmp/worker_token_sync_${port}.log"
  local pid_file="/tmp/worker_token_sync_${port}.pid"
  nohup env \
    COGNITO_REGION="$COGNITO_REGION" \
    COGNITO_CLIENT_ID="$COGNITO_CLIENT_ID" \
    COGNITO_USERNAME="$COGNITO_USERNAME" \
    COGNITO_PASSWORD="$COGNITO_PASSWORD" \
    WORKER_TOKEN_URL="http://127.0.0.1:${port}/worker/token" \
    TOKEN_REFRESH_INTERVAL_SECONDS="$TOKEN_REFRESH_INTERVAL_SECONDS" \
    "$PYTHON_BIN" -u scripts/sync_worker_token_local.py >"$log_file" 2>&1 &
  echo "$!" >"$pid_file"
  echo "token-sync:$port pid=$(cat "$pid_file") log=$log_file"
}

start_worker "$PORT_A"
start_worker "$PORT_B"

sleep 2
curl -fsS --max-time 8 "http://127.0.0.1:${PORT_A}/worker/health" >/dev/null
curl -fsS --max-time 8 "http://127.0.0.1:${PORT_B}/worker/health" >/dev/null

start_sync "$PORT_A"
start_sync "$PORT_B"

if [[ "$AUTOSCALE_ENABLED" != "0" && "$AUTOSCALE_ENABLED" != "false" && "$AUTOSCALE_ENABLED" != "no" ]]; then
  nohup env \
    ROOT_DIR="$ROOT_DIR" \
    MODEL_SERVER_URL="$MODEL_SERVER_URL" \
    BURST_PORT="$BURST_PORT" \
    AUTOSCALE_INTERVAL_SEC="$AUTOSCALE_INTERVAL_SEC" \
    AUTOSCALE_SCALE_UP_QUEUE_DEPTH="$AUTOSCALE_SCALE_UP_QUEUE_DEPTH" \
    AUTOSCALE_SCALE_DOWN_QUEUE_DEPTH="$AUTOSCALE_SCALE_DOWN_QUEUE_DEPTH" \
    AUTOSCALE_SCALE_DOWN_STREAK="$AUTOSCALE_SCALE_DOWN_STREAK" \
    COORDINATOR_API_URL="$COORDINATOR_API_URL" \
    TOKEN_REFRESH_INTERVAL_SECONDS="$TOKEN_REFRESH_INTERVAL_SECONDS" \
    COGNITO_REGION="$COGNITO_REGION" \
    COGNITO_CLIENT_ID="$COGNITO_CLIENT_ID" \
    COGNITO_USERNAME="$COGNITO_USERNAME" \
    COGNITO_PASSWORD="$COGNITO_PASSWORD" \
    WORKER_TTS_SERVER_URL="$MODEL_SERVER_URL" \
    BURST_WORKER_TTS_POOL_SIZE="$BURST_WORKER_TTS_POOL_SIZE" \
    BURST_WORKER_BOOK_PARALLELISM="$BURST_WORKER_BOOK_PARALLELISM" \
    "$PYTHON_BIN" -u scripts/shared_model_autoscaler.py >/tmp/shared_model_autoscaler.log 2>&1 &
  echo "$!" >/tmp/shared_model_autoscaler.pid
  echo "autoscaler pid=$(cat /tmp/shared_model_autoscaler.pid) log=/tmp/shared_model_autoscaler.log"
fi

echo
echo "Shared-model stack started successfully."
echo "Model server:"
echo "  curl ${MODEL_SERVER_URL}/health"
echo "Workers:"
echo "  curl http://127.0.0.1:${PORT_A}/coordinator/status"
echo "  curl http://127.0.0.1:${PORT_B}/coordinator/status"
echo "Autoscaler:"
echo "  tail -f /tmp/shared_model_autoscaler.log"
echo
echo "Stop:"
echo "  kill \$(cat /tmp/shared_model_autoscaler.pid /tmp/kokoro_model_server.pid /tmp/worker_${PORT_A}.pid /tmp/worker_${PORT_B}.pid /tmp/worker_${BURST_PORT}.pid /tmp/worker_token_sync_${PORT_A}.pid /tmp/worker_token_sync_${PORT_B}.pid /tmp/worker_token_sync_${BURST_PORT}.pid 2>/dev/null)"
