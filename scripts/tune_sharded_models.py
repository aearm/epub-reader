#!/usr/bin/env python3
import argparse
import glob
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple


DEFAULT_CONFIGS = [
    "3x3x2",
    "3x4x2",
    "4x3x2",
    "4x4x2",
    "3x3x3",
    "2x5x2",
]
PID_FILE_PATTERNS = [
    "/tmp/kokoro_model_server_*.pid",
    "/tmp/worker_*.pid",
    "/tmp/worker_token_sync_*.pid",
]


@dataclass
class Totals:
    completed: int = 0
    failed: int = 0
    claimed: int = 0
    requests_failed: int = 0
    requests_ok: int = 0


@dataclass
class Result:
    config: str
    models: int
    workers_per_model: int
    model_workers: int
    total_workers: int
    completed_delta: int
    failed_delta: int
    claimed_delta: int
    requests_failed_delta: int
    requests_ok_delta: int
    seconds: float
    completed_per_min: float
    failed_per_min: float


def parse_config(text: str) -> Tuple[int, int, int]:
    parts = text.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid config '{text}' (expected NxMxK)")
    n, m, k = (int(parts[0]), int(parts[1]), int(parts[2]))
    if n < 1 or m < 1 or k < 1:
        raise ValueError(f"Invalid config '{text}' (all values must be >= 1)")
    return n, m, k


def fetch_json(url: str, timeout: float = 4.0):
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_pid(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int((f.read() or "").strip())
    except Exception:
        return 0


def try_kill(pid: int, sig: int):
    if pid <= 0:
        return
    try:
        os.kill(pid, sig)
    except Exception:
        pass


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def stop_stack():
    pid_files = []
    for pattern in PID_FILE_PATTERNS:
        pid_files.extend(glob.glob(pattern))
    pids = []
    for path in pid_files:
        pid = read_pid(path)
        if pid > 0:
            pids.append(pid)
    for pid in pids:
        try_kill(pid, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < 6.0:
        if not any(pid_alive(pid) for pid in pids):
            break
        time.sleep(0.2)
    for pid in pids:
        if pid_alive(pid):
            try_kill(pid, signal.SIGKILL)
    for path in pid_files:
        try:
            os.remove(path)
        except Exception:
            pass


def wait_http_ok(url: str, timeout_sec: float):
    t0 = time.time()
    last_err = ""
    while time.time() - t0 < timeout_sec:
        try:
            fetch_json(url, timeout=4.0)
            return
        except Exception as e:
            last_err = str(e)
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_err}")


def collect_totals(worker_ports: List[int]) -> Totals:
    totals = Totals()
    for port in worker_ports:
        data = fetch_json(f"http://127.0.0.1:{port}/coordinator/status", timeout=6.0)
        metrics = (data or {}).get("metrics") or {}
        totals.completed += int(metrics.get("idle_tasks_completed") or 0)
        totals.failed += int(metrics.get("idle_tasks_failed") or 0)
        totals.claimed += int(metrics.get("idle_tasks_claimed") or 0)
        totals.requests_failed += int(metrics.get("requests_failed") or 0)
        totals.requests_ok += int(metrics.get("requests_ok") or 0)
    return totals


def run_cmd(cmd: List[str], env: dict):
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc


def benchmark_config(
    root_dir: str,
    username: str,
    password: str,
    python_bin: str,
    coordinator_url: str,
    config_text: str,
    warmup_sec: float,
    sample_sec: float,
    queue_maxsize: int,
    tts_pool: int,
    book_parallelism: int,
) -> Result:
    models, workers_per_model, model_workers = parse_config(config_text)
    total_workers = models * workers_per_model
    worker_ports = [5001 + i for i in range(total_workers)]

    env = os.environ.copy()
    env.update(
        {
            "COGNITO_USERNAME": username,
            "COGNITO_PASSWORD": password,
            "PYTHON_BIN": python_bin,
            "COORDINATOR_API_URL": coordinator_url,
            "KOKORO_SERVER_WORKERS": str(model_workers),
            "KOKORO_SERVER_QUEUE_MAXSIZE": str(queue_maxsize),
            "WORKER_TTS_POOL_SIZE": str(tts_pool),
            "WORKER_BOOK_PARALLELISM": str(book_parallelism),
        }
    )

    run_cmd(
        [
            "./scripts/run_sharded_models.sh",
            "--models",
            str(models),
            "--workers-per-model",
            str(workers_per_model),
        ],
        env=env,
    )

    wait_http_ok(f"http://127.0.0.1:{worker_ports[0]}/worker/health", timeout_sec=30.0)
    for port in worker_ports:
        wait_http_ok(f"http://127.0.0.1:{port}/coordinator/status", timeout_sec=30.0)

    print(f"  warmup {warmup_sec:.0f}s ...", flush=True)
    time.sleep(warmup_sec)
    start = collect_totals(worker_ports)
    t0 = time.time()
    print(f"  sample {sample_sec:.0f}s ...", flush=True)
    time.sleep(sample_sec)
    end = collect_totals(worker_ports)
    elapsed = max(1e-6, time.time() - t0)

    completed_delta = max(0, end.completed - start.completed)
    failed_delta = max(0, end.failed - start.failed)
    claimed_delta = max(0, end.claimed - start.claimed)
    requests_failed_delta = max(0, end.requests_failed - start.requests_failed)
    requests_ok_delta = max(0, end.requests_ok - start.requests_ok)

    return Result(
        config=config_text,
        models=models,
        workers_per_model=workers_per_model,
        model_workers=model_workers,
        total_workers=total_workers,
        completed_delta=completed_delta,
        failed_delta=failed_delta,
        claimed_delta=claimed_delta,
        requests_failed_delta=requests_failed_delta,
        requests_ok_delta=requests_ok_delta,
        seconds=elapsed,
        completed_per_min=(completed_delta * 60.0 / elapsed),
        failed_per_min=(failed_delta * 60.0 / elapsed),
    )


def print_summary(results: List[Result]):
    ordered = sorted(results, key=lambda r: r.completed_per_min, reverse=True)
    print("\n=== Ranking (higher completed/min is better) ===")
    print(
        "config   total_workers  completed/min  completed  failed  claimed  req_ok  req_failed"
    )
    for r in ordered:
        print(
            f"{r.config:7s} {r.total_workers:13d} "
            f"{r.completed_per_min:13.2f} {r.completed_delta:10d} "
            f"{r.failed_delta:7d} {r.claimed_delta:8d} "
            f"{r.requests_ok_delta:7d} {r.requests_failed_delta:11d}"
        )
    best = ordered[0]
    print(
        f"\nBEST={best.config} "
        f"(N={best.models}, M={best.workers_per_model}, KOKORO_SERVER_WORKERS={best.model_workers}) "
        f"completed/min={best.completed_per_min:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Tune sharded shared-model topology by local throughput."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="List of configs as NxMxK, where N=models, M=workers/model, K=model workers.",
    )
    parser.add_argument("--warmup-sec", type=float, default=30.0)
    parser.add_argument("--sample-sec", type=float, default=120.0)
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python"))
    parser.add_argument(
        "--coordinator-url",
        default=os.environ.get("COORDINATOR_API_URL", "https://api.reader.psybytes.com"),
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("COGNITO_USERNAME", ""),
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("COGNITO_PASSWORD", ""),
    )
    parser.add_argument(
        "--queue-maxsize",
        type=int,
        default=int(os.environ.get("KOKORO_SERVER_QUEUE_MAXSIZE", "256")),
    )
    parser.add_argument(
        "--worker-tts-pool-size",
        type=int,
        default=int(os.environ.get("WORKER_TTS_POOL_SIZE", "2")),
    )
    parser.add_argument(
        "--worker-book-parallelism",
        type=int,
        default=int(os.environ.get("WORKER_BOOK_PARALLELISM", "2")),
    )
    args = parser.parse_args()

    if not args.username or not args.password:
        raise SystemExit(
            "Missing credentials. Set COGNITO_USERNAME and COGNITO_PASSWORD, or pass --username/--password."
        )

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)

    results: List[Result] = []
    try:
        for cfg in args.configs:
            print(f"\n=== Benchmark {cfg} ===", flush=True)
            stop_stack()
            try:
                result = benchmark_config(
                    root_dir=root_dir,
                    username=args.username,
                    password=args.password,
                    python_bin=args.python_bin,
                    coordinator_url=args.coordinator_url,
                    config_text=cfg,
                    warmup_sec=args.warmup_sec,
                    sample_sec=args.sample_sec,
                    queue_maxsize=args.queue_maxsize,
                    tts_pool=args.worker_tts_pool_size,
                    book_parallelism=args.worker_book_parallelism,
                )
                results.append(result)
                print(
                    f"  done: completed={result.completed_delta} "
                    f"failed={result.failed_delta} "
                    f"completed/min={result.completed_per_min:.2f}",
                    flush=True,
                )
            except Exception as e:
                print(f"  FAILED {cfg}: {e}", flush=True)
    finally:
        stop_stack()

    if not results:
        raise SystemExit("No successful benchmark runs.")
    print_summary(results)


if __name__ == "__main__":
    main()
