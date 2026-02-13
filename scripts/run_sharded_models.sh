#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-2}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-3}"
BASE_MODEL_PORT="${BASE_MODEL_PORT:-5100}"
BASE_WORKER_PORT="${BASE_WORKER_PORT:-5001}"

COORDINATOR_API_URL="${COORDINATOR_API_URL:-https://api.reader.psybytes.com}"
COGNITO_REGION="${COGNITO_REGION:-eu-west-1}"
COGNITO_CLIENT_ID="${COGNITO_CLIENT_ID:-1tgsjl3qo9cbb0gvonkhfvf31n}"
TOKEN_REFRESH_INTERVAL_SECONDS="${TOKEN_REFRESH_INTERVAL_SECONDS:-2400}"

KOKORO_SERVER_SERIALIZE="${KOKORO_SERVER_SERIALIZE:-0}"
KOKORO_SERVER_WORKERS="${KOKORO_SERVER_WORKERS:-2}"
KOKORO_SERVER_QUEUE_MAXSIZE="${KOKORO_SERVER_QUEUE_MAXSIZE:-256}"
KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC="${KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC:-2}"
KOKORO_SERVER_RESULT_TIMEOUT_SEC="${KOKORO_SERVER_RESULT_TIMEOUT_SEC:-180}"

WORKER_TTS_POOL_SIZE="${WORKER_TTS_POOL_SIZE:-2}"
WORKER_BOOK_PARALLELISM="${WORKER_BOOK_PARALLELISM:-2}"
MODEL_STARTUP_TIMEOUT_SEC="${MODEL_STARTUP_TIMEOUT_SEC:-90}"
WORKER_STARTUP_TIMEOUT_SEC="${WORKER_STARTUP_TIMEOUT_SEC:-60}"

usage() {
  cat <<EOF
Usage:
  $0 --models N --workers-per-model M [options]

Options:
  --models N                 Number of shared model servers (N)
  --workers-per-model M      Worker processes attached to each model server (M)
  --base-model-port P        First model server port (default: ${BASE_MODEL_PORT})
  --base-worker-port P       First worker port (default: ${BASE_WORKER_PORT})
  --python-bin BIN           Python executable (default: ${PYTHON_BIN})
  --help                     Show this help

Environment knobs:
  KOKORO_SERVER_SERIALIZE, KOKORO_SERVER_WORKERS, KOKORO_SERVER_QUEUE_MAXSIZE,
  KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC, KOKORO_SERVER_RESULT_TIMEOUT_SEC,
  WORKER_TTS_POOL_SIZE, WORKER_BOOK_PARALLELISM, TOKEN_REFRESH_INTERVAL_SECONDS,
  MODEL_STARTUP_TIMEOUT_SEC, WORKER_STARTUP_TIMEOUT_SEC
EOF
}

wait_http_json() {
  local url="$1"
  local timeout_sec="$2"
  local start_ts now_ts
  start_ts="$(date +%s)"
  while true; do
    if curl -fsS --max-time 8 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout_sec )); then
      return 1
    fi
    sleep 1
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS="$2"
      shift 2
      ;;
    --workers-per-model)
      WORKERS_PER_MODEL="$2"
      shift 2
      ;;
    --base-model-port)
      BASE_MODEL_PORT="$2"
      shift 2
      ;;
    --base-worker-port)
      BASE_WORKER_PORT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! [[ "$MODELS" =~ ^[0-9]+$ ]] || ! [[ "$WORKERS_PER_MODEL" =~ ^[0-9]+$ ]]; then
  echo "--models and --workers-per-model must be integers." >&2
  exit 2
fi

if (( MODELS < 1 || WORKERS_PER_MODEL < 1 )); then
  echo "--models and --workers-per-model must be >= 1." >&2
  exit 2
fi

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

declare -a MODEL_PORTS=()
declare -a WORKER_PORTS=()

for ((i=0; i<MODELS; i++)); do
  MODEL_PORTS+=($((BASE_MODEL_PORT + i)))
done

for ((i=0; i<MODELS*WORKERS_PER_MODEL; i++)); do
  WORKER_PORTS+=($((BASE_WORKER_PORT + i)))
done

for port in "${MODEL_PORTS[@]}" "${WORKER_PORTS[@]}"; do
  if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port ${port} is already in use. Stop existing processes first." >&2
    exit 1
  fi
done

mkdir -p static/state

declare -a STARTED_PID_FILES=()

cleanup_partial() {
  for pid_file in "${STARTED_PID_FILES[@]}"; do
    if [[ -f "$pid_file" ]]; then
      pid="$(cat "$pid_file" 2>/dev/null || true)"
      if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
      fi
      rm -f "$pid_file"
    fi
  done
}

on_exit() {
  code=$?
  if (( code != 0 )); then
    echo "Startup failed; cleaning partial processes..." >&2
    cleanup_partial
  fi
}
trap on_exit EXIT

echo "Starting ${MODELS} model server(s) with ${WORKERS_PER_MODEL} worker(s) each..."

for idx in "${!MODEL_PORTS[@]}"; do
  port="${MODEL_PORTS[$idx]}"
  pid_file="/tmp/kokoro_model_server_${port}.pid"
  log_file="/tmp/kokoro_model_server_${port}.log"

  nohup env \
    KOKORO_SERVER_HOST=127.0.0.1 \
    KOKORO_SERVER_PORT="${port}" \
    KOKORO_SERVER_SERIALIZE="${KOKORO_SERVER_SERIALIZE}" \
    KOKORO_SERVER_WORKERS="${KOKORO_SERVER_WORKERS}" \
    KOKORO_SERVER_QUEUE_MAXSIZE="${KOKORO_SERVER_QUEUE_MAXSIZE}" \
    KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC="${KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC}" \
    KOKORO_SERVER_RESULT_TIMEOUT_SEC="${KOKORO_SERVER_RESULT_TIMEOUT_SEC}" \
    "${PYTHON_BIN}" -u scripts/kokoro_model_server.py >"${log_file}" 2>&1 &
  echo "$!" > "${pid_file}"
  STARTED_PID_FILES+=("${pid_file}")
  echo "model-server:${port} pid=$(cat "${pid_file}") log=${log_file}"
done

sleep 2
for port in "${MODEL_PORTS[@]}"; do
  if ! wait_http_json "http://127.0.0.1:${port}/health" "${MODEL_STARTUP_TIMEOUT_SEC}"; then
    echo "Model server on port ${port} did not become healthy within ${MODEL_STARTUP_TIMEOUT_SEC}s." >&2
    exit 1
  fi
done

for idx in "${!WORKER_PORTS[@]}"; do
  worker_port="${WORKER_PORTS[$idx]}"
  model_idx=$((idx / WORKERS_PER_MODEL))
  model_port="${MODEL_PORTS[$model_idx]}"
  model_url="http://127.0.0.1:${model_port}"

  worker_pid_file="/tmp/worker_${worker_port}.pid"
  worker_log="/tmp/worker_${worker_port}.log"
  nohup env \
    FLASK_ENV=production \
    WORKER_PORT="${worker_port}" \
    WORKER_TTS_POOL_SIZE="${WORKER_TTS_POOL_SIZE}" \
    WORKER_BOOK_PARALLELISM="${WORKER_BOOK_PARALLELISM}" \
    WORKER_TTS_SERVER_URL="${model_url}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    COORDINATOR_API_URL="${COORDINATOR_API_URL}" \
    COORDINATOR_TOKEN_STATE_PATH="static/state/coordinator_token_${worker_port}.json" \
    "${PYTHON_BIN}" app_multithreaded.py >"${worker_log}" 2>&1 &
  echo "$!" > "${worker_pid_file}"
  STARTED_PID_FILES+=("${worker_pid_file}")
  echo "worker:${worker_port} -> model:${model_port} pid=$(cat "${worker_pid_file}") log=${worker_log}"
done

sleep 2
for worker_port in "${WORKER_PORTS[@]}"; do
  if ! wait_http_json "http://127.0.0.1:${worker_port}/worker/health" "${WORKER_STARTUP_TIMEOUT_SEC}"; then
    echo "Worker on port ${worker_port} did not become healthy within ${WORKER_STARTUP_TIMEOUT_SEC}s." >&2
    exit 1
  fi
done

for worker_port in "${WORKER_PORTS[@]}"; do
  sync_pid_file="/tmp/worker_token_sync_${worker_port}.pid"
  sync_log="/tmp/worker_token_sync_${worker_port}.log"
  nohup env \
    COGNITO_REGION="${COGNITO_REGION}" \
    COGNITO_CLIENT_ID="${COGNITO_CLIENT_ID}" \
    COGNITO_USERNAME="${COGNITO_USERNAME}" \
    COGNITO_PASSWORD="${COGNITO_PASSWORD}" \
    WORKER_TOKEN_URL="http://127.0.0.1:${worker_port}/worker/token" \
    TOKEN_REFRESH_INTERVAL_SECONDS="${TOKEN_REFRESH_INTERVAL_SECONDS}" \
    "${PYTHON_BIN}" -u scripts/sync_worker_token_local.py >"${sync_log}" 2>&1 &
  echo "$!" > "${sync_pid_file}"
  STARTED_PID_FILES+=("${sync_pid_file}")
  echo "token-sync:${worker_port} pid=$(cat "${sync_pid_file}") log=${sync_log}"
done

trap - EXIT

echo
echo "Sharded shared-model stack started."
echo "Topology: N=${MODELS}, M=${WORKERS_PER_MODEL}, total_workers=$((MODELS * WORKERS_PER_MODEL))"
echo "Model servers:"
for port in "${MODEL_PORTS[@]}"; do
  echo "  curl http://127.0.0.1:${port}/health"
done
echo "Workers:"
for worker_port in "${WORKER_PORTS[@]}"; do
  echo "  curl http://127.0.0.1:${worker_port}/coordinator/status"
done
echo
echo "Stop all:"
echo "  for p in ${MODEL_PORTS[*]}; do kill \$(cat /tmp/kokoro_model_server_\${p}.pid 2>/dev/null) 2>/dev/null || true; done"
echo "  for p in ${WORKER_PORTS[*]}; do kill \$(cat /tmp/worker_\${p}.pid /tmp/worker_token_sync_\${p}.pid 2>/dev/null) 2>/dev/null || true; done"
