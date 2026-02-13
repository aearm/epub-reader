#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(os.environ.get("ROOT_DIR") or Path(__file__).resolve().parents[1]).resolve()
MODEL_SERVER_URL = (os.environ.get("MODEL_SERVER_URL") or "http://127.0.0.1:5100").strip().rstrip("/")

BURST_PORT = int(os.environ.get("BURST_PORT") or "5003")
INTERVAL_SEC = max(1.0, float(os.environ.get("AUTOSCALE_INTERVAL_SEC") or "8"))
SCALE_UP_QUEUE_DEPTH = max(1, int(os.environ.get("AUTOSCALE_SCALE_UP_QUEUE_DEPTH") or "80"))
SCALE_DOWN_QUEUE_DEPTH = max(0, int(os.environ.get("AUTOSCALE_SCALE_DOWN_QUEUE_DEPTH") or "15"))
SCALE_DOWN_STREAK = max(1, int(os.environ.get("AUTOSCALE_SCALE_DOWN_STREAK") or "6"))

COORDINATOR_API_URL = (os.environ.get("COORDINATOR_API_URL") or "https://api.reader.psybytes.com").strip()
TOKEN_REFRESH_INTERVAL_SECONDS = max(
    60, int(os.environ.get("TOKEN_REFRESH_INTERVAL_SECONDS") or str(40 * 60))
)
COGNITO_REGION = (os.environ.get("COGNITO_REGION") or "eu-west-1").strip()
COGNITO_CLIENT_ID = (os.environ.get("COGNITO_CLIENT_ID") or "").strip()
COGNITO_USERNAME = (os.environ.get("COGNITO_USERNAME") or "").strip()
COGNITO_PASSWORD = os.environ.get("COGNITO_PASSWORD") or ""

BURST_WORKER_TTS_POOL_SIZE = max(1, int(os.environ.get("BURST_WORKER_TTS_POOL_SIZE") or "2"))
BURST_WORKER_BOOK_PARALLELISM = max(1, int(os.environ.get("BURST_WORKER_BOOK_PARALLELISM") or "2"))
WORKER_TTS_SERVER_URL = (os.environ.get("WORKER_TTS_SERVER_URL") or MODEL_SERVER_URL).strip()

WORKER_PID_FILE = Path(f"/tmp/worker_{BURST_PORT}.pid")
SYNC_PID_FILE = Path(f"/tmp/worker_token_sync_{BURST_PORT}.pid")
WORKER_LOG = Path(f"/tmp/worker_{BURST_PORT}.log")
SYNC_LOG = Path(f"/tmp/worker_token_sync_{BURST_PORT}.log")

STOP = False


def _on_signal(signum, _frame):
    global STOP
    STOP = True
    print(f"[autoscaler] Received signal {signum}; stopping.", flush=True)


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)


def fetch_json(url: str, timeout: float = 6.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except Exception as exc:
        return None, str(exc)


def read_pid(pid_file: Path):
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def is_pid_alive(pid: int):
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def is_port_busy(port: int):
    proc = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def start_process(cmd, env, cwd: Path, log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "ab", buffering=0) as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def start_burst_worker():
    if is_port_busy(BURST_PORT):
        print(f"[autoscaler] Burst port {BURST_PORT} already in use; skip start.", flush=True)
        return False

    worker_env = os.environ.copy()
    worker_env.update(
        {
            "FLASK_ENV": "production",
            "WORKER_PORT": str(BURST_PORT),
            "WORKER_TTS_POOL_SIZE": str(BURST_WORKER_TTS_POOL_SIZE),
            "WORKER_BOOK_PARALLELISM": str(BURST_WORKER_BOOK_PARALLELISM),
            "WORKER_TTS_SERVER_URL": WORKER_TTS_SERVER_URL,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "COORDINATOR_API_URL": COORDINATOR_API_URL,
            "COORDINATOR_TOKEN_STATE_PATH": f"static/state/coordinator_token_{BURST_PORT}.json",
        }
    )
    worker_pid = start_process(["python", "app_multithreaded.py"], worker_env, ROOT_DIR, WORKER_LOG)
    WORKER_PID_FILE.write_text(str(worker_pid), encoding="utf-8")

    sync_env = os.environ.copy()
    sync_env.update(
        {
            "COGNITO_REGION": COGNITO_REGION,
            "COGNITO_CLIENT_ID": COGNITO_CLIENT_ID,
            "COGNITO_USERNAME": COGNITO_USERNAME,
            "COGNITO_PASSWORD": COGNITO_PASSWORD,
            "WORKER_TOKEN_URL": f"http://127.0.0.1:{BURST_PORT}/worker/token",
            "TOKEN_REFRESH_INTERVAL_SECONDS": str(TOKEN_REFRESH_INTERVAL_SECONDS),
        }
    )
    sync_pid = start_process(
        ["python3", "-u", "scripts/sync_worker_token_local.py"], sync_env, ROOT_DIR, SYNC_LOG
    )
    SYNC_PID_FILE.write_text(str(sync_pid), encoding="utf-8")
    print(f"[autoscaler] Started burst worker:{BURST_PORT} pid={worker_pid} sync_pid={sync_pid}", flush=True)
    return True


def stop_pidfile(pid_file: Path, label: str):
    pid = read_pid(pid_file)
    if not pid:
        pid_file.unlink(missing_ok=True)
        return False
    if is_pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    pid_file.unlink(missing_ok=True)
    print(f"[autoscaler] Stopped {label} pid={pid}", flush=True)
    return True


def stop_burst_worker():
    stopped_sync = stop_pidfile(SYNC_PID_FILE, f"token-sync:{BURST_PORT}")
    stopped_worker = stop_pidfile(WORKER_PID_FILE, f"worker:{BURST_PORT}")
    return stopped_sync or stopped_worker


def burst_running():
    worker_pid = read_pid(WORKER_PID_FILE)
    sync_pid = read_pid(SYNC_PID_FILE)
    return is_pid_alive(worker_pid or 0) and is_pid_alive(sync_pid or 0)


def validate_credentials():
    missing = []
    if not COGNITO_CLIENT_ID:
        missing.append("COGNITO_CLIENT_ID")
    if not COGNITO_USERNAME:
        missing.append("COGNITO_USERNAME")
    if not COGNITO_PASSWORD:
        missing.append("COGNITO_PASSWORD")
    if missing:
        raise RuntimeError(f"Missing required env for autoscaler burst sync: {', '.join(missing)}")


def main():
    validate_credentials()
    print(
        f"[autoscaler] Started (up>={SCALE_UP_QUEUE_DEPTH}, down<={SCALE_DOWN_QUEUE_DEPTH} "
        f"for {SCALE_DOWN_STREAK} intervals, every {INTERVAL_SEC:.1f}s).",
        flush=True,
    )

    cool_down_streak = 0
    last_rejected = 0
    last_timed_out = 0

    while not STOP:
        health, err = fetch_json(f"{MODEL_SERVER_URL}/health")
        if err:
            print(f"[autoscaler] model health error: {err}", flush=True)
            time.sleep(INTERVAL_SEC)
            continue

        queue_depth = int(health.get("queue_depth") or 0)
        rejected = int(health.get("requests_rejected") or 0)
        timed_out = int(health.get("requests_timed_out") or 0)
        pressure = (
            queue_depth >= SCALE_UP_QUEUE_DEPTH
            or rejected > last_rejected
            or timed_out > last_timed_out
        )
        last_rejected = rejected
        last_timed_out = timed_out

        if pressure:
            cool_down_streak = 0
            if not burst_running():
                start_burst_worker()
        else:
            if queue_depth <= SCALE_DOWN_QUEUE_DEPTH:
                cool_down_streak += 1
            else:
                cool_down_streak = 0

            if cool_down_streak >= SCALE_DOWN_STREAK and burst_running():
                stop_burst_worker()
                cool_down_streak = 0

        time.sleep(INTERVAL_SEC)

    stop_burst_worker()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[autoscaler] fatal: {exc}", file=sys.stderr, flush=True)
        raise
