#!/usr/bin/env python3
import datetime as dt
import json
import os
import subprocess
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
TERRAFORM_DIR = os.environ.get("TERRAFORM_DIR", "terraform").strip()
SQS_QUEUE_URL = os.environ.get("AUDIO_SQS_QUEUE_URL", "").strip()
SQS_DLQ_URL = os.environ.get("AUDIO_SQS_DLQ_URL", "").strip()
SQS_REGION = os.environ.get("AUDIO_SQS_REGION", os.environ.get("AWS_REGION", "")).strip()
EWMA_ALPHA = max(0.05, min(0.95, float(os.environ.get("WATCH_ETA_EWMA_ALPHA", "0.35"))))

SESSION_STATE = {
    "started_at": time.time(),
    "initialized": False,
    "last_ts": 0.0,
    "last_ready": 0,
    "ewma_rate": 0.0,
    "processed_in_session": 0,
}


def command_exists(name: str) -> bool:
    from shutil import which
    return which(name) is not None


def run_command(args, cwd=None, timeout=8):
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return None, str(exc)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        return None, stderr or stdout or f"exit {proc.returncode}"
    return (proc.stdout or "").strip(), None


def terraform_output(name: str):
    if not command_exists("terraform"):
        return ""
    if not TERRAFORM_DIR or not os.path.isdir(TERRAFORM_DIR):
        return ""
    out, err = run_command(["terraform", "output", "-raw", name], cwd=TERRAFORM_DIR, timeout=10)
    if err:
        return ""
    return (out or "").strip()


def resolve_sqs_targets():
    queue_url = SQS_QUEUE_URL or terraform_output("audio_sqs_queue_url")
    dlq_url = SQS_DLQ_URL or terraform_output("audio_sqs_dlq_url")
    region = SQS_REGION or terraform_output("audio_sqs_region") or terraform_output("aws_region")
    enabled = bool(queue_url and region and command_exists("aws"))
    return {
        "enabled": enabled,
        "queue_url": queue_url,
        "dlq_url": dlq_url,
        "region": region,
        "error": None if enabled else "missing aws cli or queue/region",
    }


SQS_TARGETS = resolve_sqs_targets()


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


def fmt_duration(seconds):
    if seconds is None:
        return "n/a"
    try:
        total = max(0, int(seconds))
    except Exception:
        return "n/a"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def update_session_eta(ready_count: int, left_count: int):
    now = time.time()
    state = SESSION_STATE
    ready_count = max(0, int(ready_count))
    left_count = max(0, int(left_count))

    if not state["initialized"]:
        state["initialized"] = True
        state["last_ts"] = now
        state["last_ready"] = ready_count
        return {
            "rate": 0.0,
            "per_message_sec": None,
            "all_left_sec": None,
            "elapsed_sec": now - state["started_at"],
            "projected_total_sec": None,
            "processed": state["processed_in_session"],
        }

    dt_sec = max(0.001, now - float(state["last_ts"]))
    delta_ready = max(0, ready_count - int(state["last_ready"]))
    inst_rate = delta_ready / dt_sec
    if inst_rate > 0:
        ewma_rate = float(state["ewma_rate"])
        state["ewma_rate"] = inst_rate if ewma_rate <= 0 else (EWMA_ALPHA * inst_rate + (1.0 - EWMA_ALPHA) * ewma_rate)
    state["processed_in_session"] += delta_ready
    state["last_ts"] = now
    state["last_ready"] = ready_count

    rate = float(state["ewma_rate"])
    per_message_sec = (1.0 / rate) if rate > 0 else None
    all_left_sec = (left_count / rate) if rate > 0 else None
    elapsed_sec = now - float(state["started_at"])
    projected_total_sec = (elapsed_sec + all_left_sec) if all_left_sec is not None else None
    return {
        "rate": rate,
        "per_message_sec": per_message_sec,
        "all_left_sec": all_left_sec,
        "elapsed_sec": elapsed_sec,
        "projected_total_sec": projected_total_sec,
        "processed": state["processed_in_session"],
    }


def fetch_sqs_attributes(queue_url: str, region: str):
    out, err = run_command(
        [
            "aws",
            "sqs",
            "get-queue-attributes",
            "--queue-url",
            queue_url,
            "--region",
            region,
            "--attribute-names",
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
            "ApproximateNumberOfMessagesDelayed",
        ],
        timeout=8,
    )
    if err:
        return None, err
    try:
        data = json.loads(out or "{}")
    except Exception as exc:
        return None, str(exc)
    attrs = data.get("Attributes") or {}
    return {
        "visible": fmt_int(attrs.get("ApproximateNumberOfMessages")),
        "in_flight": fmt_int(attrs.get("ApproximateNumberOfMessagesNotVisible")),
        "delayed": fmt_int(attrs.get("ApproximateNumberOfMessagesDelayed")),
    }, None


def print_block():
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    coord, coord_err = fetch_json(COORDINATOR_STATS_URL)
    worker, worker_err = fetch_json(WORKER_JOBS_URL)
    worker_coord, worker_coord_err = fetch_json(WORKER_COORD_STATUS_URL)
    sqs_main = sqs_main_err = sqs_dlq = sqs_dlq_err = None
    if SQS_TARGETS["enabled"]:
        sqs_main, sqs_main_err = fetch_sqs_attributes(SQS_TARGETS["queue_url"], SQS_TARGETS["region"])
        if SQS_TARGETS["dlq_url"]:
            sqs_dlq, sqs_dlq_err = fetch_sqs_attributes(SQS_TARGETS["dlq_url"], SQS_TARGETS["region"])

    clear_screen()
    print(f"[{now}] Audio Jobs Monitor")
    print(f"Coordinator: {COORDINATOR_STATS_URL}")
    print(f"Worker:      {WORKER_JOBS_URL}")
    print(f"Coord Link:  {WORKER_COORD_STATUS_URL}")
    print(
        "SQS:         "
        f"enabled={SQS_TARGETS['enabled']} "
        f"region={SQS_TARGETS['region'] or '-'}"
    )
    print("-" * 72)

    if coord_err:
        print(f"Coordinator status: ERROR ({coord_err})")
    else:
        pending = fmt_int(coord.get("pending"))
        generating = fmt_int(coord.get("generating"))
        ready = fmt_int(coord.get("ready"))
        total = fmt_int(coord.get("total"))
        left = pending + generating
        eta = update_session_eta(ready, left)
        print(
            "Coordinator jobs: "
            f"left={left} (pending={pending}, generating={generating}) "
            f"ready={ready} total={total}"
        )
        rate = eta["rate"]
        rate_min = rate * 60.0 if rate > 0 else 0.0
        print(
            "Session timing: "
            f"elapsed={fmt_duration(eta['elapsed_sec'])} "
            f"processed={fmt_int(eta['processed'])} "
            f"avg_per_msg={fmt_duration(eta['per_message_sec'])} "
            f"eta_all_left={fmt_duration(eta['all_left_sec'])} "
            f"projected_total={fmt_duration(eta['projected_total_sec'])} "
            f"rate={rate:.3f}/s ({rate_min:.1f}/min)"
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

    if not SQS_TARGETS["enabled"]:
        print(f"SQS monitor: disabled ({SQS_TARGETS['error']})")
    elif sqs_main_err:
        print(f"SQS main queue: ERROR ({sqs_main_err})")
    else:
        print(
            "SQS main: "
            f"visible={fmt_int(sqs_main.get('visible'))} "
            f"in_flight={fmt_int(sqs_main.get('in_flight'))} "
            f"delayed={fmt_int(sqs_main.get('delayed'))}"
        )
        if SQS_TARGETS["dlq_url"]:
            if sqs_dlq_err:
                print(f"SQS DLQ: ERROR ({sqs_dlq_err})")
            elif sqs_dlq:
                print(
                    "SQS DLQ: "
                    f"visible={fmt_int(sqs_dlq.get('visible'))} "
                    f"in_flight={fmt_int(sqs_dlq.get('in_flight'))} "
                    f"delayed={fmt_int(sqs_dlq.get('delayed'))}"
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
