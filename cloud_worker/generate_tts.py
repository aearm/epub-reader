#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlparse

import boto3
import requests
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


DEFAULT_INSTRUCT = (
    "Professional audiobook narration. Warm, clear female voice. "
    "Natural pacing with brief pauses at commas and sentence ends. "
    "Smooth phrasing, no exaggeration. Consistent volume and tone. "
    "Crisp consonants, gentle sibilance, minimal breath noise. "
    "Slightly slower than conversational speech. "
    "Maintain a calm, engaging storyteller tone."
)


def env_int(name: str, default: int, minimum: int = None, maximum: int = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def infer_region_from_queue_url(queue_url: str) -> str:
    parsed = urlparse(queue_url or "")
    host = (parsed.netloc or "").strip().lower()
    if host.startswith("sqs.") and ".amazonaws.com" in host:
        parts = host.split(".")
        if len(parts) >= 3:
            return parts[1]
    return ""


COORDINATOR_API_URL = (os.environ.get("COORDINATOR_API_URL") or "").strip().rstrip("/")
WORKER_SHARED_SECRET = (os.environ.get("WORKER_SHARED_SECRET") or "").strip()
AUDIO_SQS_QUEUE_URL = (os.environ.get("AUDIO_SQS_QUEUE_URL") or "").strip()
AUDIO_SQS_REGION = (
    (os.environ.get("AUDIO_SQS_REGION") or "").strip()
    or infer_region_from_queue_url(AUDIO_SQS_QUEUE_URL)
    or "us-east-2"
)

SQS_WAIT_SECONDS = env_int("AUDIO_SQS_WAIT_SECONDS", 20, minimum=0, maximum=20)
SQS_VISIBILITY_TIMEOUT = env_int("AUDIO_SQS_VISIBILITY_TIMEOUT", 900, minimum=30)
IDLE_MAX_EMPTY_RECEIVES = env_int("WORKER_IDLE_MAX_EMPTY_RECEIVES", 10, minimum=1)
UPLOAD_TIMEOUT_SECONDS = env_int("WORKER_UPLOAD_TIMEOUT_SECONDS", 240, minimum=30)
COORDINATOR_TIMEOUT_SECONDS = env_int("COORDINATOR_TIMEOUT_SECONDS", 45, minimum=5)
FFMPEG_AUDIO_BITRATE = (os.environ.get("WORKER_M4B_BITRATE") or "64k").strip() or "64k"

QWEN_MODEL_PATH = (
    os.environ.get("QWEN_MODEL_PATH")
    or "/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
).strip()
QWEN_DEVICE = (os.environ.get("QWEN_DEVICE") or "cuda:0").strip()
QWEN_ATTN_IMPLEMENTATION = (os.environ.get("QWEN_ATTN_IMPLEMENTATION") or "flash_attention_2").strip()
QWEN_SPEAKER = (os.environ.get("QWEN_SPEAKER") or "Ryan").strip()
QWEN_LANGUAGE = (os.environ.get("QWEN_LANGUAGE") or "Auto").strip()
QWEN_INSTRUCT = (os.environ.get("QWEN_INSTRUCT") or DEFAULT_INSTRUCT).strip()

QWEN_DTYPE_NAME = (os.environ.get("QWEN_DTYPE") or "bfloat16").strip().lower()
if QWEN_DTYPE_NAME in ("float16", "fp16"):
    QWEN_DTYPE = torch.float16
else:
    QWEN_DTYPE = torch.bfloat16

STOP_REQUESTED = False
MODEL = None
SESSION = requests.Session()


def log(message: str):
    print(f"[cloud-worker] {message}", flush=True)


def fail_fast_if_missing_env():
    missing = []
    if not COORDINATOR_API_URL:
        missing.append("COORDINATOR_API_URL")
    if not WORKER_SHARED_SECRET:
        missing.append("WORKER_SHARED_SECRET")
    if not AUDIO_SQS_QUEUE_URL:
        missing.append("AUDIO_SQS_QUEUE_URL")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def signal_handler(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(f"Received signal {signum}, stopping after current message.")


def auth_headers(json_body: bool = True) -> dict:
    headers = {"X-Worker-Secret": WORKER_SHARED_SECRET}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def coordinator_request(method: str, path: str, payload: dict = None, timeout: int = None) -> dict:
    url = urljoin(f"{COORDINATOR_API_URL}/", path.lstrip("/"))
    timeout = timeout or COORDINATOR_TIMEOUT_SECONDS
    response = SESSION.request(
        method=method.upper(),
        url=url,
        headers=auth_headers(json_body=True),
        json=payload,
        timeout=timeout
    )
    if response.status_code >= 400:
        body = (response.text or "")[:800]
        raise RuntimeError(f"{method.upper()} {path} failed ({response.status_code}): {body}")
    if not response.content:
        return {}
    return response.json()


def ensure_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    log(
        "Loading Qwen model "
        f"(path={QWEN_MODEL_PATH}, device={QWEN_DEVICE}, dtype={QWEN_DTYPE_NAME}, attn={QWEN_ATTN_IMPLEMENTATION})"
    )
    MODEL = Qwen3TTSModel.from_pretrained(
        QWEN_MODEL_PATH,
        device_map=QWEN_DEVICE,
        dtype=QWEN_DTYPE,
        attn_implementation=QWEN_ATTN_IMPLEMENTATION,
    )
    return MODEL


def extract_task_body(message: dict) -> dict:
    raw_body = message.get("Body")
    if not raw_body:
        return {}
    try:
        body = json.loads(raw_body)
    except Exception:
        return {}
    if isinstance(body, dict) and isinstance(body.get("Message"), str):
        try:
            nested = json.loads(body["Message"])
            if isinstance(nested, dict):
                body = nested
        except Exception:
            pass
    return body if isinstance(body, dict) else {}


def delete_message(sqs_client, receipt_handle: str):
    if not receipt_handle:
        return
    sqs_client.delete_message(
        QueueUrl=AUDIO_SQS_QUEUE_URL,
        ReceiptHandle=receipt_handle
    )


def synthesize_to_m4b(sentence_hash: str, text: str, work_dir: Path) -> Path:
    model = ensure_model()
    wavs, sample_rate = model.generate_custom_voice(
        text=text,
        language=QWEN_LANGUAGE,
        speaker=QWEN_SPEAKER,
        instruct=QWEN_INSTRUCT,
    )
    if not wavs or len(wavs) == 0:
        raise RuntimeError("Qwen model returned empty audio list")

    wav_path = work_dir / f"{sentence_hash}_{uuid.uuid4().hex[:8]}.wav"
    m4b_path = wav_path.with_suffix(".m4b")
    sf.write(str(wav_path), wavs[0], sample_rate)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-c:a",
        "aac",
        "-b:a",
        FFMPEG_AUDIO_BITRATE,
        str(m4b_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"ffmpeg conversion failed: {stderr[:800]}")

    try:
        wav_path.unlink(missing_ok=True)
    except Exception:
        pass

    return m4b_path


def upload_audio_file(upload_url: str, m4b_path: Path):
    with m4b_path.open("rb") as audio_file:
        put = SESSION.put(
            upload_url,
            data=audio_file,
            headers={"Content-Type": "audio/mp4"},
            timeout=(10, UPLOAD_TIMEOUT_SECONDS)
        )
    if put.status_code >= 400:
        body = (put.text or "")[:800]
        raise RuntimeError(f"S3 upload failed ({put.status_code}): {body}")


def maybe_skip_if_ready(sentence_hash: str, text: str) -> bool:
    check = coordinator_request("POST", "/check", {"hash": sentence_hash, "text": text})
    return check.get("status") == "ready" and bool(check.get("url"))


def process_message(sqs_client, message: dict, work_dir: Path) -> bool:
    payload = extract_task_body(message)
    receipt_handle = (message.get("ReceiptHandle") or "").strip()
    sentence_hash = (payload.get("hash") or "").strip()
    text = (payload.get("text") or "").strip()

    if not receipt_handle:
        return True

    if not sentence_hash or not text:
        log("Dropping malformed SQS message (missing hash/text).")
        delete_message(sqs_client, receipt_handle)
        return True

    try:
        if maybe_skip_if_ready(sentence_hash, text):
            delete_message(sqs_client, receipt_handle)
            log(f"{sentence_hash}: already ready, acked.")
            return True

        m4b_path = synthesize_to_m4b(sentence_hash, text, work_dir)
        upload = coordinator_request(
            "POST",
            "/upload_url",
            {"hash": sentence_hash, "format": "m4b"}
        )
        upload_url = (upload.get("upload_url") or "").strip()
        final_url = (upload.get("final_url") or "").strip()
        if not upload_url or not final_url:
            raise RuntimeError("Coordinator returned invalid upload URL payload")

        upload_audio_file(upload_url, m4b_path)
        coordinator_request(
            "POST",
            "/complete",
            {"hash": sentence_hash, "s3_url": final_url}
        )
        delete_message(sqs_client, receipt_handle)
        log(f"{sentence_hash}: uploaded and completed.")
        return True
    except Exception as e:
        log(f"{sentence_hash}: failed ({e})")
        # Do not delete message on failure: let SQS retry policy handle it.
        return False
    finally:
        for suffix in (".wav", ".m4b"):
            try:
                for stale in work_dir.glob(f"{sentence_hash}_*{suffix}"):
                    stale.unlink(missing_ok=True)
            except Exception:
                pass


def main():
    fail_fast_if_missing_env()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    log(
        "Starting "
        f"(queue_region={AUDIO_SQS_REGION}, wait={SQS_WAIT_SECONDS}s, "
        f"idle_polls={IDLE_MAX_EMPTY_RECEIVES}, visibility={SQS_VISIBILITY_TIMEOUT}s)"
    )
    sqs_client = boto3.client("sqs", region_name=AUDIO_SQS_REGION)

    empty_receives = 0
    with tempfile.TemporaryDirectory(prefix="qwen-worker-") as temp_dir:
        work_dir = Path(temp_dir)
        while not STOP_REQUESTED:
            response = sqs_client.receive_message(
                QueueUrl=AUDIO_SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=SQS_WAIT_SECONDS,
                VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
                AttributeNames=["All"],
            )
            messages = response.get("Messages") or []
            if not messages:
                empty_receives += 1
                if empty_receives >= IDLE_MAX_EMPTY_RECEIVES:
                    log("Queue idle window reached. Exiting.")
                    break
                continue

            empty_receives = 0
            for message in messages:
                if STOP_REQUESTED:
                    break
                process_message(sqs_client, message, work_dir)

    log("Stopped.")


if __name__ == "__main__":
    main()
