#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT_A="${PORT_A:-5001}"
PORT_B="${PORT_B:-5002}"

# Tuned defaults for Apple M3 Pro + ~19 GB RAM.
WORKER_TTS_POOL_SIZE="${WORKER_TTS_POOL_SIZE:-6}"
WORKER_BOOK_PARALLELISM="${WORKER_BOOK_PARALLELISM:-6}"
COORDINATOR_API_URL="${COORDINATOR_API_URL:-https://api.reader.psybytes.com}"
COGNITO_REGION="${COGNITO_REGION:-eu-west-1}"
COGNITO_CLIENT_ID="${COGNITO_CLIENT_ID:-1tgsjl3qo9cbb0gvonkhfvf31n}"
TOKEN_REFRESH_INTERVAL_SECONDS="${TOKEN_REFRESH_INTERVAL_SECONDS:-2400}"

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

if lsof -nP -iTCP:"$PORT_A" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT_A is already in use. Stop existing worker first." >&2
  exit 1
fi
if lsof -nP -iTCP:"$PORT_B" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT_B is already in use. Stop existing worker first." >&2
  exit 1
fi

mkdir -p static/state

start_worker() {
  local port="$1"
  local log_file="/tmp/worker_${port}.log"
  local pid_file="/tmp/worker_${port}.pid"
  nohup env \
    FLASK_ENV=production \
    WORKER_PORT="$port" \
    WORKER_TTS_POOL_SIZE="$WORKER_TTS_POOL_SIZE" \
    WORKER_BOOK_PARALLELISM="$WORKER_BOOK_PARALLELISM" \
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
    python3 -u scripts/sync_worker_token_local.py >"$log_file" 2>&1 &
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

echo
echo "Started two workers successfully."
echo "Check status:"
echo "  curl http://127.0.0.1:${PORT_A}/coordinator/status"
echo "  curl http://127.0.0.1:${PORT_B}/coordinator/status"
echo "  python3 scripts/watch_audio_jobs.py"
echo
echo "Stop:"
echo "  kill \$(cat /tmp/worker_${PORT_A}.pid /tmp/worker_${PORT_B}.pid /tmp/worker_token_sync_${PORT_A}.pid /tmp/worker_token_sync_${PORT_B}.pid)"
