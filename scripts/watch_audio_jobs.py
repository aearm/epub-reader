#!/usr/bin/env python3
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request


COORDINATOR_STATS_URL = os.environ.get(
    "COORDINATOR_STATS_URL",
    "https://api.reader.psybytes.com/stats",
).strip()
WORKER_JOBS_URL = os.environ.get(
    "WORKER_JOBS_URL",
    "http://127.0.0.1:5001/worker/jobs",
).strip()
WORKER_COORD_STATUS_URL = os.environ.get(
    "WORKER_COORD_STATUS_URL",
    "http://127.0.0.1:5001/coordinator/status",
).strip()
INTERVAL_SEC = max(0.5, float(os.environ.get("WATCH_INTERVAL_SEC", "2")))


def fetch_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            body = response.read().decode("utf-8")
            return json.loads(body), None
    except Exception as exc:
        return None, str(exc)


def clear_screen():
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def fmt_int(value):
    try:
        return int(value)
    except Exception:
        return 0


def print_block():
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    coord, coord_err = fetch_json(COORDINATOR_STATS_URL)
    worker, worker_err = fetch_json(WORKER_JOBS_URL)
    worker_coord, worker_coord_err = fetch_json(WORKER_COORD_STATUS_URL)

    clear_screen()
    print(f"[{now}] Audio Jobs Monitor")
    print(f"Coordinator: {COORDINATOR_STATS_URL}")
    print(f"Worker:      {WORKER_JOBS_URL}")
    print(f"Coord Link:  {WORKER_COORD_STATUS_URL}")
    print("-" * 72)

    if coord_err:
        print(f"Coordinator status: ERROR ({coord_err})")
    else:
        pending = fmt_int(coord.get("pending"))
        generating = fmt_int(coord.get("generating"))
        ready = fmt_int(coord.get("ready"))
        total = fmt_int(coord.get("total"))
        left = pending + generating
        print(
            "Coordinator jobs: "
            f"left={left} (pending={pending}, generating={generating}) "
            f"ready={ready} total={total}"
        )

    if worker_err:
        print(f"Worker status: ERROR ({worker_err})")
    else:
        summary = worker.get("summary") or {}
        queue_size = worker.get("queue_size")
        active_jobs = worker.get("active_jobs") or []
        idle_backfill = bool(worker.get("idle_backfill_active"))
        print(
            "Worker queue: "
            f"generation_active={bool(worker.get('generation_active'))} "
            f"queue_size={queue_size} "
            f"alive_workers={fmt_int(worker.get('alive_workers'))} "
            f"idle_backfill_active={idle_backfill}"
        )
        print(
            "Worker jobs: "
            f"running={fmt_int(summary.get('running_jobs'))} "
            f"queued={fmt_int(summary.get('queued_jobs'))} "
            f"completed={fmt_int(summary.get('completed_jobs'))} "
            f"failed={fmt_int(summary.get('failed_jobs'))} "
            f"left_items={fmt_int(summary.get('left_items'))}"
        )

        if active_jobs:
            print("Active jobs:")
            for job in active_jobs[:8]:
                print(
                    f"  - {job.get('job_id')} "
                    f"book={job.get('book_id')} "
                    f"status={job.get('status')} "
                    f"ready={fmt_int(job.get('ready'))}/{fmt_int(job.get('total'))} "
                    f"left={fmt_int(job.get('left'))}"
                )
        else:
            print("Active jobs: none")

    if worker_coord_err:
        print(f"Worker->Coordinator: ERROR ({worker_coord_err})")
    else:
        metrics = (worker_coord or {}).get("metrics") or {}
        print(
            "Worker->Coordinator: "
            f"enabled={bool(worker_coord.get('enabled'))} "
            f"has_token={bool(worker_coord.get('has_bearer_token'))} "
            f"requests_ok={fmt_int(metrics.get('requests_ok'))} "
            f"requests_failed={fmt_int(metrics.get('requests_failed'))} "
            f"idle_claimed={fmt_int(metrics.get('idle_tasks_claimed'))} "
            f"idle_completed={fmt_int(metrics.get('idle_tasks_completed'))}"
        )

    print("-" * 72)
    print(f"Refreshing every {INTERVAL_SEC:.1f}s. Press Ctrl+C to stop.")


def main():
    try:
        while True:
            print_block()
            time.sleep(INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
