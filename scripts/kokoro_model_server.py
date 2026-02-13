#!/usr/bin/env python3
import io
import os
import threading
import time
import queue
import sys
from concurrent.futures import Future, TimeoutError
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request

# Ensure repo root is on sys.path when running as scripts/kokoro_model_server.py.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.kokoro_engine import KokoroEngine


HOST = (os.environ.get('KOKORO_SERVER_HOST') or '127.0.0.1').strip()
PORT = int(os.environ.get('KOKORO_SERVER_PORT') or '5100')
MAX_TEXT_CHARS = max(1, int(os.environ.get('KOKORO_SERVER_MAX_TEXT_CHARS') or '5000'))
SERIALIZE_REQUESTS = (os.environ.get('KOKORO_SERVER_SERIALIZE') or '1').strip().lower() not in (
    '0', 'false', 'no', 'off'
)
QUEUE_MAXSIZE = max(1, int(os.environ.get('KOKORO_SERVER_QUEUE_MAXSIZE') or '256'))
ENQUEUE_TIMEOUT_SEC = max(0.1, float(os.environ.get('KOKORO_SERVER_ENQUEUE_TIMEOUT_SEC') or '2'))
RESULT_TIMEOUT_SEC = max(1.0, float(os.environ.get('KOKORO_SERVER_RESULT_TIMEOUT_SEC') or '180'))
WORKER_THREADS = max(1, int(os.environ.get('KOKORO_SERVER_WORKERS') or '1'))
if SERIALIZE_REQUESTS:
    WORKER_THREADS = 1

app = Flask(__name__)
engine_lock = threading.Lock()
engine = KokoroEngine()
started_at = time.time()
requests_total = 0
requests_failed = 0
requests_rejected = 0
requests_timed_out = 0
processed_ok = 0
processed_failed = 0
total_inference_ms = 0.0
metrics_lock = threading.Lock()
job_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)


class SynthesisJob:
    def __init__(self, text: str):
        self.text = text
        self.future: Future = Future()
        self.enqueued_at = time.time()


def _engine_generate(text: str):
    if SERIALIZE_REQUESTS:
        with engine_lock:
            audio = engine.generate_audio(text)
            sample_rate = int(engine.sample_rate)
            return audio, sample_rate
    audio = engine.generate_audio(text)
    sample_rate = int(engine.sample_rate)
    return audio, sample_rate


def _worker_loop(worker_id: int):
    global processed_ok, processed_failed, total_inference_ms
    while True:
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            break

        start = time.time()
        try:
            audio, sample_rate = _engine_generate(job.text)
            elapsed_ms = (time.time() - start) * 1000.0
            with metrics_lock:
                total_inference_ms += elapsed_ms
            if audio is None:
                with metrics_lock:
                    processed_failed += 1
                if not job.future.done():
                    job.future.set_exception(RuntimeError('TTS returned no audio'))
            else:
                with metrics_lock:
                    processed_ok += 1
                if not job.future.done():
                    job.future.set_result((audio, sample_rate))
        except Exception as e:
            with metrics_lock:
                processed_failed += 1
            if not job.future.done():
                job.future.set_exception(e)
        finally:
            job_queue.task_done()


for idx in range(WORKER_THREADS):
    t = threading.Thread(target=_worker_loop, args=(idx,), daemon=True)
    t.start()


@app.route('/health', methods=['GET'])
def health():
    with metrics_lock:
        avg_inference_ms = (
            round(total_inference_ms / processed_ok, 2)
            if processed_ok > 0 else 0.0
        )
    return jsonify({
        'status': 'ok',
        'backend': engine.backend,
        'voice': engine.voice,
        'sample_rate': engine.sample_rate,
        'serialize_requests': SERIALIZE_REQUESTS,
        'worker_threads': WORKER_THREADS,
        'queue_maxsize': QUEUE_MAXSIZE,
        'queue_depth': job_queue.qsize(),
        'uptime_sec': int(time.time() - started_at),
        'requests_total': requests_total,
        'requests_failed': requests_failed,
        'requests_rejected': requests_rejected,
        'requests_timed_out': requests_timed_out,
        'processed_ok': processed_ok,
        'processed_failed': processed_failed,
        'avg_inference_ms': avg_inference_ms,
    })


@app.route('/generate', methods=['POST'])
def generate():
    global requests_total, requests_failed, requests_rejected, requests_timed_out
    requests_total += 1
    payload = request.get_json(silent=True) or {}
    text = str(payload.get('text') or '').strip()
    if not text:
        requests_failed += 1
        return jsonify({'error': 'Missing text'}), 400
    if len(text) > MAX_TEXT_CHARS:
        requests_failed += 1
        return jsonify({'error': f'Text too long ({len(text)} > {MAX_TEXT_CHARS})'}), 400

    job = SynthesisJob(text)
    try:
        job_queue.put(job, timeout=ENQUEUE_TIMEOUT_SEC)
    except queue.Full:
        requests_rejected += 1
        requests_failed += 1
        return jsonify({
            'error': 'Model server queue is full',
            'queue_depth': job_queue.qsize(),
            'queue_maxsize': QUEUE_MAXSIZE
        }), 429

    try:
        audio, sample_rate = job.future.result(timeout=RESULT_TIMEOUT_SEC)
    except TimeoutError:
        requests_timed_out += 1
        requests_failed += 1
        return jsonify({
            'error': f'Model server timeout after {RESULT_TIMEOUT_SEC:.1f}s',
            'queue_depth': job_queue.qsize(),
        }), 504
    except Exception as e:
        requests_failed += 1
        return jsonify({'error': f'TTS failure: {e}'}), 500

    try:
        if audio is None:
            requests_failed += 1
            return jsonify({'error': 'TTS returned no audio'}), 502

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            requests_failed += 1
            return jsonify({'error': 'TTS returned empty audio'}), 502

        buf = io.BytesIO()
        sf.write(buf, arr, sample_rate, format='WAV')
        body = buf.getvalue()
        headers = {
            'X-Sample-Rate': str(sample_rate),
            'X-Backend': str(engine.backend),
            'X-Queue-Depth': str(job_queue.qsize()),
            'Cache-Control': 'no-store',
        }
        return Response(body, status=200, mimetype='audio/wav', headers=headers)
    except Exception as e:
        requests_failed += 1
        return jsonify({'error': f'TTS failure: {e}'}), 500


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
