# app_multithreaded.py
from flask import Flask, request, jsonify, send_file, redirect
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import json
import hashlib
import base64
from pathlib import Path
from werkzeug.utils import secure_filename
from utils.epub_processor_v2 import EPUBProcessorV2 as EPUBProcessor
from utils.reading_state import get_state_manager
from utils.translation_service import translate_text, get_languages
from utils.library_manager import get_library_manager
from utils.worker_audio_backend import WorkerAudioBackend
import threading
from queue import PriorityQueue, Empty
import time
import itertools
import shutil
import requests
from urllib.parse import urljoin
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.config import Config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)


def _detect_memory_limit_mb():
    """Best-effort container memory limit detection."""
    candidates = (
        '/sys/fs/cgroup/memory.max',
        '/sys/fs/cgroup/memory/memory.limit_in_bytes',
    )
    for path in candidates:
        try:
            raw = Path(path).read_text(encoding='utf-8').strip()
        except Exception:
            continue

        if not raw or raw == 'max':
            continue
        try:
            limit_bytes = int(raw)
        except ValueError:
            continue

        # Ignore nonsense/unbounded values.
        if limit_bytes <= 0 or limit_bytes > (1 << 60):
            continue
        return limit_bytes / (1024 * 1024)
    return None

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'
ALLOWED_EXTENSIONS = {'epub'}

COORDINATOR_API_URL = (
    os.environ.get('COORDINATOR_API_URL') or 'https://api.reader.psybytes.com'
).strip().rstrip('/')


def _extract_worker_shared_secret(value):
    if not isinstance(value, str):
        return ''
    text = value.strip()
    if not text:
        return ''
    if text.startswith('{') and text.endswith('}'):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                for key in ('worker_shared_secret', 'WORKER_SHARED_SECRET', 'secret'):
                    candidate = payload.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
        except Exception:
            return text
    return text


def load_worker_shared_secret():
    env_secret = _extract_worker_shared_secret(os.environ.get('WORKER_SHARED_SECRET') or '')
    if env_secret:
        return env_secret, 'env'

    aws_region = (os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or '').strip() or None
    secret_ids = [
        (os.environ.get('WORKER_SHARED_SECRET_SECRET_ID') or '').strip(),
    ]
    parameter_names = [
        (os.environ.get('WORKER_SHARED_SECRET_PARAMETER_NAME') or '').strip(),
        '/epub-reader/worker-shared-secret',
    ]

    # Try Secrets Manager first, then SSM Parameter Store.
    try:
        session = boto3.session.Session(region_name=aws_region)
    except Exception:
        return '', None
    boto_config = Config(connect_timeout=2, read_timeout=2, retries={'max_attempts': 1})

    for secret_id in [s for s in secret_ids if s]:
        try:
            sm = session.client('secretsmanager', config=boto_config)
            response = sm.get_secret_value(SecretId=secret_id)
            secret = _extract_worker_shared_secret(response.get('SecretString', ''))
            if secret:
                return secret, f'secretsmanager:{secret_id}'
        except Exception:
            continue

    for name in [n for n in parameter_names if n]:
        try:
            ssm = session.client('ssm', config=boto_config)
            response = ssm.get_parameter(Name=name, WithDecryption=True)
            secret = _extract_worker_shared_secret(response.get('Parameter', {}).get('Value', ''))
            if secret:
                return secret, f'ssm:{name}'
        except Exception:
            continue

    return '', None


WORKER_SHARED_SECRET, WORKER_SHARED_SECRET_SOURCE = load_worker_shared_secret()
COORDINATOR_BEARER_TOKEN = os.environ.get('COORDINATOR_BEARER_TOKEN', '').strip()
COORDINATOR_TIMEOUT = float(os.environ.get('COORDINATOR_TIMEOUT', '20'))
COORDINATOR_BATCH_SIZE = 100
AUDIO_UPLOAD_FORMAT = os.environ.get('AUDIO_UPLOAD_FORMAT', 'm4b').strip().lower()
DEFAULT_CPU = os.cpu_count() or 4
WORKER_TTS_POOL_REQUESTED = max(
    2,
    min(32, int(os.environ.get('WORKER_TTS_POOL_SIZE', str(min(32, max(8, DEFAULT_CPU))))))
)
WORKER_TTS_HARD_CAP = max(2, int(os.environ.get('WORKER_TTS_HARD_CAP', '12')))
WORKER_TTS_INSTANCE_MEM_MB = max(180, int(os.environ.get('WORKER_TTS_INSTANCE_MEM_MB', '300')))
WORKER_MEMORY_LIMIT_MB = _detect_memory_limit_mb()
if WORKER_MEMORY_LIMIT_MB:
    worker_tts_pool_by_mem = max(2, int((WORKER_MEMORY_LIMIT_MB * 0.55) / WORKER_TTS_INSTANCE_MEM_MB))
else:
    worker_tts_pool_by_mem = WORKER_TTS_POOL_REQUESTED
WORKER_TTS_POOL_SIZE = max(2, min(WORKER_TTS_POOL_REQUESTED, worker_tts_pool_by_mem, WORKER_TTS_HARD_CAP))
if WORKER_TTS_POOL_SIZE < WORKER_TTS_POOL_REQUESTED:
    print(
        f"INFO: Capping WORKER_TTS_POOL_SIZE from {WORKER_TTS_POOL_REQUESTED} to {WORKER_TTS_POOL_SIZE} "
        f"(hard_cap={WORKER_TTS_HARD_CAP}, memory_limit_mb={WORKER_MEMORY_LIMIT_MB}, "
        f"est_mem_per_instance_mb={WORKER_TTS_INSTANCE_MEM_MB})"
    )

WORKER_BOOK_PARALLELISM = max(
    1,
    min(32, int(os.environ.get('WORKER_BOOK_PARALLELISM', str(min(32, max(16, DEFAULT_CPU))))))
)
WORKER_UPLOAD_URL_BATCH_SIZE = max(1, int(os.environ.get('WORKER_UPLOAD_URL_BATCH_SIZE', '80')))
WORKER_COMPLETE_BATCH_SIZE = max(1, int(os.environ.get('WORKER_COMPLETE_BATCH_SIZE', '50')))
WORKER_MP3_BITRATE = os.environ.get('WORKER_MP3_BITRATE', '64k').strip() or '64k'
WORKER_MP3_SAMPLE_RATE = str(int(os.environ.get('WORKER_MP3_SAMPLE_RATE', '24000')))
COORDINATOR_IDLE_POLL_INTERVAL = max(0.5, float(os.environ.get('COORDINATOR_IDLE_POLL_INTERVAL', '2.0')))
COORDINATOR_IDLE_TASK_BATCH_SIZE = max(
    1,
    min(10, int(os.environ.get('COORDINATOR_IDLE_TASK_BATCH_SIZE', '10')))
)
WORKER_IDLE_BACKFILL_ENABLED = os.environ.get('WORKER_IDLE_BACKFILL_ENABLED', '1').strip().lower() not in ('0', 'false', 'no')
COORDINATOR_TOKEN_STATE_PATH = (
    os.environ.get('COORDINATOR_TOKEN_STATE_PATH') or 'static/state/coordinator_token.json'
).strip()
COORDINATOR_TOKEN_SYNC_POLL_SECONDS = max(
    0.5,
    float(os.environ.get('COORDINATOR_TOKEN_SYNC_POLL_SECONDS', '2.0'))
)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
os.makedirs('static/state', exist_ok=True)  # For reading position persistence
os.makedirs('static/library', exist_ok=True)  # For library data
os.makedirs('static/library/covers', exist_ok=True)  # For book covers
os.makedirs('static/models', exist_ok=True)  # For Kokoro ONNX model assets

# -----------------------------------------------------------------------------
# Global state (scoped to process, not per-user)
# -----------------------------------------------------------------------------
current_book = None          # Parsed book structure + sentences
epub_processor = None
audio_backend = WorkerAudioBackend(
    default_upload_format=AUDIO_UPLOAD_FORMAT,
    mp3_bitrate=WORKER_MP3_BITRATE,
    sample_rate=WORKER_MP3_SAMPLE_RATE,
    tts_pool_size=WORKER_TTS_POOL_SIZE
)
audio_generation_queue = PriorityQueue()
audio_generation_threads = []
audio_status = {}            # sentence_id -> 'pending' | 'generating' | 'ready' | 'failed'
generation_active = False
priority_counter = itertools.count()  # Unique counter to keep queue stable
current_chapter_index = 0
current_page_sentences = []  # list of sentence_ids for currently visible page
sentence_hashes = {}          # sentence_id -> coordinator hash
sentence_remote_urls = {}     # sentence_id -> S3 URL
coordinator_enabled = bool(COORDINATOR_API_URL)
coordinator_sync_lock = threading.Lock()
coordinator_metrics = {
    'requests_ok': 0,
    'requests_failed': 0,
    'uploads_ok': 0,
    'uploads_failed': 0,
    'idle_tasks_claimed': 0,
    'idle_tasks_completed': 0,
    'idle_tasks_failed': 0,
    'last_error': None
}
runtime_coordinator_token = None
runtime_token_lock = threading.Lock()
runtime_token_last_sync_at = 0.0
runtime_token_last_mtime = None
book_jobs = {}
book_jobs_lock = threading.Lock()
idle_backfill_lock = threading.Lock()
idle_task_poll_lock = threading.Lock()
idle_task_backoff_until = 0.0
idle_backfill_thread = None

if coordinator_enabled and not COORDINATOR_BEARER_TOKEN and not WORKER_SHARED_SECRET:
    print("WARNING: Coordinator is enabled without startup token. "
          "Expect bearer token from logged-in web app or set COORDINATOR_BEARER_TOKEN/WORKER_SHARED_SECRET.")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_book_hash(filepath: str) -> str:
    """Generate a unique hash for a book file."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def hash_text_for_coordinator(text: str) -> str:
    """Match coordinator hash format (sha256, first 16 chars)."""
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()[:16]


def _extract_bearer_token(value: str):
    if not value:
        return None
    value = value.strip()
    if not value.lower().startswith('bearer '):
        return None
    token = value[7:].strip()
    return token or None


def _decode_jwt_payload_unverified(token: str):
    if not isinstance(token, str):
        return None
    parts = token.split('.')
    if len(parts) != 3:
        return None
    segment = parts[1]
    padding = '=' * (-len(segment) % 4)
    try:
        raw = base64.urlsafe_b64decode(segment + padding)
        payload = json.loads(raw.decode('utf-8'))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _is_jwt_expired(token: str, leeway_seconds: int = 30) -> bool:
    payload = _decode_jwt_payload_unverified(token)
    if not payload:
        return False
    exp = payload.get('exp')
    if exp is None:
        return False
    try:
        return time.time() >= float(exp) - leeway_seconds
    except (TypeError, ValueError):
        return False


def set_runtime_coordinator_token(token: str):
    """Store runtime Cognito token for background coordinator calls."""
    global runtime_coordinator_token, runtime_token_last_sync_at
    if not token:
        return
    with runtime_token_lock:
        runtime_coordinator_token = token
    runtime_token_last_sync_at = time.time()
    persist_runtime_coordinator_token(token)


def clear_runtime_coordinator_token():
    """Clear runtime token and remove persisted copy."""
    global runtime_coordinator_token, runtime_token_last_mtime, runtime_token_last_sync_at
    with runtime_token_lock:
        runtime_coordinator_token = None
    runtime_token_last_mtime = None
    runtime_token_last_sync_at = time.time()
    try:
        path = Path(COORDINATOR_TOKEN_STATE_PATH)
        if path.exists():
            path.unlink()
    except Exception as e:
        print(f"WARNING: Failed to clear persisted coordinator token: {e}")


def persist_runtime_coordinator_token(token: str):
    """Persist runtime token so worker reconnects after container restart."""
    global runtime_token_last_mtime, runtime_token_last_sync_at
    if not token or token == COORDINATOR_BEARER_TOKEN:
        return
    try:
        path = Path(COORDINATOR_TOKEN_STATE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token.strip(), encoding='utf-8')
        runtime_token_last_mtime = path.stat().st_mtime
        runtime_token_last_sync_at = time.time()
    except Exception as e:
        print(f"WARNING: Failed to persist coordinator token: {e}")


def load_runtime_coordinator_token(force=False):
    """Load persisted runtime token if available."""
    global runtime_coordinator_token, runtime_token_last_sync_at, runtime_token_last_mtime
    now = time.time()
    if not force and (now - runtime_token_last_sync_at) < COORDINATOR_TOKEN_SYNC_POLL_SECONDS:
        return
    runtime_token_last_sync_at = now
    try:
        path = Path(COORDINATOR_TOKEN_STATE_PATH)
        if not path.exists():
            runtime_token_last_mtime = None
            with runtime_token_lock:
                runtime_coordinator_token = None
            return
        stat = path.stat()
        if not force and runtime_token_last_mtime is not None and stat.st_mtime <= runtime_token_last_mtime:
            return
        token = path.read_text(encoding='utf-8').strip()
        runtime_token_last_mtime = stat.st_mtime
        if token and token.count('.') == 2:
            with runtime_token_lock:
                runtime_coordinator_token = token
        else:
            with runtime_token_lock:
                runtime_coordinator_token = None
    except Exception as e:
        print(f"WARNING: Failed to load persisted coordinator token: {e}")


def capture_coordinator_token_from_request(data=None):
    """Capture token from Authorization header or request body."""
    token = _extract_bearer_token(request.headers.get('Authorization', ''))
    if not token and data and isinstance(data, dict):
        provided = data.get('coordinator_access_token') or data.get('access_token')
        if isinstance(provided, str) and provided.count('.') == 2:
            token = provided.strip()

    if token:
        set_runtime_coordinator_token(token)
        return True
    return False


def get_effective_coordinator_token():
    """Return runtime token first, then static env token."""
    load_runtime_coordinator_token()
    runtime_token = None
    with runtime_token_lock:
        if runtime_coordinator_token:
            runtime_token = runtime_coordinator_token

    if runtime_token and _is_jwt_expired(runtime_token):
        clear_runtime_coordinator_token()
        runtime_token = None

    if runtime_token:
        return runtime_token

    env_token = COORDINATOR_BEARER_TOKEN or None
    if env_token and _is_jwt_expired(env_token):
        return None
    return env_token


def apply_worker_token_request(data=None):
    """
    Set/clear runtime coordinator token from worker API calls.
    Accepts:
    - Authorization: Bearer <token>
    - JSON body { "access_token": "<token>" }
    - JSON body { "clear": true } to clear runtime token
    """
    payload = data if isinstance(data, dict) else (request.get_json(silent=True) or {})
    if payload.get('clear') is True:
        clear_runtime_coordinator_token()
        return {'success': True, 'cleared': True}

    captured = capture_coordinator_token_from_request(payload)
    if not captured:
        return {'success': False, 'error': 'Missing bearer token'}

    return {
        'success': True,
        'cleared': False,
        'has_bearer_token': bool(get_effective_coordinator_token()),
        'token_source': 'runtime' if bool(runtime_coordinator_token) else ('env' if bool(COORDINATOR_BEARER_TOKEN) else None),
    }


def coordinator_headers(include_json=True):
    """Headers for coordinator API requests."""
    headers = {}
    if include_json:
        headers['Content-Type'] = 'application/json'
    token = get_effective_coordinator_token()
    if token:
        headers['Authorization'] = f'Bearer {token}'
    if WORKER_SHARED_SECRET:
        headers['X-Worker-Secret'] = WORKER_SHARED_SECRET
    return headers


def coordinator_request(method: str, path: str, json_data=None, timeout=None):
    """Perform a coordinator API request with graceful fallback."""
    if not coordinator_enabled:
        return None

    url = urljoin(f"{COORDINATOR_API_URL}/", path.lstrip('/'))

    try:
        response = None
        for attempt in range(2):
            before = get_effective_coordinator_token()
            response = requests.request(
                method=method,
                url=url,
                json=json_data,
                headers=coordinator_headers(include_json=True),
                timeout=timeout or COORDINATOR_TIMEOUT
            )
            if response.status_code < 400:
                coordinator_metrics['requests_ok'] += 1
                return response.json()
            if response.status_code == 401 and attempt == 0:
                load_runtime_coordinator_token(force=True)
                after = get_effective_coordinator_token()
                if after and after != before:
                    continue
                with runtime_token_lock:
                    had_runtime = bool(runtime_coordinator_token)
                if had_runtime:
                    clear_runtime_coordinator_token()
            break

        print(f"Coordinator error {response.status_code} for {path}: {response.text[:300]}")
        coordinator_metrics['requests_failed'] += 1
        coordinator_metrics['last_error'] = f"{path}: HTTP {response.status_code}"
        return None
    except Exception as e:
        print(f"Coordinator request failed ({path}): {e}")
        coordinator_metrics['requests_failed'] += 1
        coordinator_metrics['last_error'] = f"{path}: {str(e)}"
        return None


def cache_remote_audio(remote_url: str, target_path: str) -> bool:
    """Download remote audio once and store locally for future playback."""
    if os.path.exists(target_path):
        return True

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        tmp_path = f"{target_path}.part"
        with requests.get(remote_url, stream=True, timeout=COORDINATOR_TIMEOUT) as response:
            if not response.ok:
                return False
            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, target_path)
        return True
    except Exception as e:
        print(f"Failed to cache remote audio {remote_url}: {e}")
        return False


def upload_audio_to_coordinator(sentence_hash: str, audio_path: str):
    """Upload generated audio to S3 via coordinator and mark task complete."""
    if not coordinator_enabled:
        return None

    asset = audio_backend.build_upload_asset(audio_path, AUDIO_UPLOAD_FORMAT)

    upload = coordinator_request(
        'POST',
        '/upload_url',
        json_data={'hash': sentence_hash, 'format': asset.audio_format}
    )
    if not upload:
        coordinator_metrics['uploads_failed'] += 1
        return None

    upload_url = upload.get('upload_url')
    final_url = upload.get('final_url')
    if not upload_url or not final_url:
        coordinator_metrics['uploads_failed'] += 1
        return None

    try:
        put_response = upload_to_presigned_url(upload_url, asset.upload_path, asset.content_type)
        if put_response.status_code >= 400:
            print(f"S3 upload failed ({put_response.status_code}) for {sentence_hash}")
            coordinator_metrics['uploads_failed'] += 1
            coordinator_metrics['last_error'] = f"s3 PUT failed: {put_response.status_code}"
            return None
        # Uploaded chunks are no longer needed on worker disk after S3 confirms PUT.
        audio_backend.cleanup_paths(asset.cleanup_paths)
    except Exception as e:
        print(f"Failed uploading audio to S3 for {sentence_hash}: {e}")
        coordinator_metrics['uploads_failed'] += 1
        coordinator_metrics['last_error'] = f"s3 PUT failed: {str(e)}"
        return None

    complete = coordinator_request(
        'POST',
        '/complete',
        json_data={'hash': sentence_hash, 's3_url': final_url}
    )
    if not complete:
        coordinator_metrics['uploads_failed'] += 1
        return None

    coordinator_metrics['uploads_ok'] += 1
    return final_url

def upload_to_presigned_url(upload_url: str, upload_path: str, content_type: str):
    with open(upload_path, 'rb') as f:
        return requests.put(
            upload_url,
            data=f,
            headers={'Content-Type': content_type},
            timeout=COORDINATOR_TIMEOUT
        )


def complete_batch_with_coordinator(items, book_id=None):
    """Complete multiple uploaded items in one coordinator call."""
    if not items:
        return True
    payload = {'items': items}
    if book_id:
        payload['book_id'] = book_id
    result = coordinator_request('POST', '/complete_batch', json_data=payload, timeout=max(COORDINATOR_TIMEOUT, 30))
    return bool(result)


def request_upload_urls_batch(items):
    """Get a batch of presigned upload URLs for hash items."""
    if not items:
        return {}
    result = coordinator_request(
        'POST',
        '/upload_urls_batch',
        json_data={'items': items},
        timeout=max(COORDINATOR_TIMEOUT, 30)
    )
    if not result:
        return {}
    mapped = {}
    for entry in result.get('items', []):
        sentence_hash = (entry.get('hash') or '').strip()
        if sentence_hash:
            mapped[sentence_hash] = entry
    return mapped


def generate_sentence_locally(sentence_hash: str, text: str, output_dir: str):
    """Generate one sentence WAV in an isolated backend module."""
    return audio_backend.generate_sentence_wav(
        sentence_id=sentence_hash,
        text=text,
        output_dir=output_dir,
        sentence_index=0
    )


def process_idle_coordinator_task(task):
    """Generate and upload one coordinator task payload."""
    task = task or {}
    sentence_hash = (task.get('hash') or '').strip()
    text = (task.get('text') or '').strip()
    if not sentence_hash or not text:
        coordinator_metrics['idle_tasks_failed'] += 1
        coordinator_metrics['last_error'] = 'idle task missing hash/text'
        return False

    coordinator_metrics['idle_tasks_claimed'] += 1

    # Double-check before generation in case another node already completed.
    existing = coordinator_request(
        'POST',
        '/check',
        json_data={'hash': sentence_hash, 'text': text}
    )
    if existing and existing.get('status') == 'ready' and existing.get('url'):
        coordinator_metrics['idle_tasks_completed'] += 1
        return True

    output_dir = os.path.join(app.config['AUDIO_FOLDER'], 'worker_cache')
    os.makedirs(output_dir, exist_ok=True)

    audio_path = generate_sentence_locally(sentence_hash, text, output_dir)
    if not audio_path:
        # One retry for transient generation failures.
        time.sleep(0.15)
        audio_path = generate_sentence_locally(sentence_hash, text, output_dir)
    if not audio_path:
        coordinator_metrics['idle_tasks_failed'] += 1
        coordinator_metrics['last_error'] = f'idle generation failed for {sentence_hash}'
        return False

    uploaded_url = upload_audio_to_coordinator(sentence_hash, audio_path)
    if uploaded_url:
        coordinator_metrics['idle_tasks_completed'] += 1
        return True

    # Last check: another worker may have completed between our upload/complete attempts.
    existing = coordinator_request(
        'POST',
        '/check',
        json_data={'hash': sentence_hash, 'text': text}
    )
    if existing and existing.get('status') == 'ready' and existing.get('url'):
        coordinator_metrics['idle_tasks_completed'] += 1
        return True

    coordinator_metrics['idle_tasks_failed'] += 1
    coordinator_metrics['last_error'] = f'idle upload/complete failed for {sentence_hash}'
    return False


def process_idle_coordinator_task_once():
    """
    Generate one coordinator-pending sentence when the local priority queue is empty.
    This keeps current/nearby-page work highest priority and uses only spare capacity
    for other books.
    """
    global idle_task_backoff_until

    if not coordinator_enabled or not get_effective_coordinator_token():
        return False

    # Only one worker performs idle backfill at a time.
    if not idle_backfill_lock.acquire(blocking=False):
        return False

    try:
        now = time.time()
        with idle_task_poll_lock:
            if now < idle_task_backoff_until:
                return False

        result = coordinator_request('GET', f'/tasks?limit={COORDINATOR_IDLE_TASK_BATCH_SIZE}')
        tasks = (result or {}).get('tasks') or []
        if not tasks:
            with idle_task_poll_lock:
                idle_task_backoff_until = time.time() + COORDINATOR_IDLE_POLL_INTERVAL
            return False

        with idle_task_poll_lock:
            idle_task_backoff_until = 0.0

        did_work = False
        for task in tasks:
            if process_idle_coordinator_task(task):
                did_work = True

        return did_work
    except Exception as e:
        coordinator_metrics['idle_tasks_failed'] += 1
        coordinator_metrics['last_error'] = f'idle task error: {str(e)}'
        print(f"Idle coordinator task failed: {e}")
        print(traceback.format_exc())
        return False
    finally:
        idle_backfill_lock.release()


def initialize_sentence_hashes(book_data):
    """Compute and attach coordinator hash for each sentence."""
    global sentence_hashes
    sentence_hashes = {}

    for sentence in book_data.get('sentences', []):
        sentence_hash = hash_text_for_coordinator(sentence.get('text', ''))
        sentence['hash'] = sentence_hash
        sentence_hashes[sentence['id']] = sentence_hash

    # Ensure chapter sentence objects carry the same hash
    for chapter in book_data.get('chapters', []):
        for sentence in chapter.get('sentences', []):
            sentence_id = sentence.get('id')
            if sentence_id in sentence_hashes:
                sentence['hash'] = sentence_hashes[sentence_id]


def sync_book_with_coordinator(book_id: str, sentences: list, book_audio_dir: str):
    """
    Query coordinator for existing S3 audio so local generation only handles
    truly missing items.
    """
    global sentence_remote_urls

    if not coordinator_enabled:
        return

    with coordinator_sync_lock:
        if not current_book or current_book.get('book_id') != book_id:
            return

        # Build unique hash->text lookup for coordinator checks.
        missing_payload = {}
        sentence_ids_by_hash = {}

        for sentence in sentences:
            sentence_id = sentence['id']
            sentence_hash = sentence_hashes.get(sentence_id) or hash_text_for_coordinator(sentence.get('text', ''))
            sentence_hashes[sentence_id] = sentence_hash
            sentence_ids_by_hash.setdefault(sentence_hash, []).append(sentence_id)

            if sentence_hash not in missing_payload:
                missing_payload[sentence_hash] = {
                    'hash': sentence_hash,
                    'text': sentence.get('text', '')
                }

        payload_values = list(missing_payload.values())
        if not payload_values:
            return

        for batch_start in range(0, len(payload_values), COORDINATOR_BATCH_SIZE):
            batch = payload_values[batch_start:batch_start + COORDINATOR_BATCH_SIZE]
            result = coordinator_request('POST', '/check_batch', json_data={'sentences': batch})
            if not result:
                continue

            for item in result.get('results', []):
                sentence_hash = item.get('hash')
                if not sentence_hash:
                    continue

                status = item.get('status')
                url = item.get('url')
                sentence_ids = sentence_ids_by_hash.get(sentence_hash, [])
                local_paths = [os.path.join(book_audio_dir, f"{sid}.wav") for sid in sentence_ids]

                for sentence_id in sentence_ids:
                    if status == 'ready' and url:
                        sentence_remote_urls[sentence_id] = url
                        # Keep worker disk clean once cloud audio is confirmed ready.
                        audio_backend.cleanup_paths([os.path.join(book_audio_dir, f"{sentence_id}.wav")])
                        if audio_status.get(sentence_id) != 'ready':
                            audio_status[sentence_id] = 'ready'
                            socketio.emit('audio_ready', {
                                'sentence_id': sentence_id,
                                'status': 'ready'
                            })
                if status == 'ready' and url:
                    continue

                # Local legacy WAV exists but coordinator is missing this hash.
                # Upload once and fan out the resulting URL to all same-hash sentences.
                local_existing = next((p for p in local_paths if os.path.exists(p)), None)
                if local_existing:
                    uploaded_url = upload_audio_to_coordinator(sentence_hash, local_existing)
                    if uploaded_url:
                        for sentence_id in sentence_ids:
                            sentence_remote_urls[sentence_id] = uploaded_url
                            audio_backend.cleanup_paths([os.path.join(book_audio_dir, f"{sentence_id}.wav")])
                            if audio_status.get(sentence_id) != 'ready':
                                audio_status[sentence_id] = 'ready'
                                socketio.emit('audio_ready', {
                                    'sentence_id': sentence_id,
                                    'status': 'ready'
                                })
                        continue

                for sentence_id in sentence_ids:
                    if audio_status.get(sentence_id) not in ('ready', 'generating'):
                        audio_status[sentence_id] = 'pending'

def calculate_priority(sentence_index: int, sentence_id: str, chapter_index: int) -> int:
    """
    Calculate priority for a sentence based on reading position.
    Lower number = higher priority.

    Priority bands:
      0 -  99: current page sentences
    100 - 199: current chapter, near page (by index)
    200 - 499: rest of current chapter
    500 - 699: next chapter
    700 - 899: previous chapter
    900+:      other chapters (further away = lower priority)
    """
    global current_chapter_index, current_page_sentences

    # Highest priority: visible page sentences
    if sentence_id in current_page_sentences:
        return 0

    chapter_distance = chapter_index - current_chapter_index

    if chapter_index == current_chapter_index:
        # Same chapter: closer to beginning gets higher priority (but below current page)
        base = 100
        return base + min(sentence_index, 399)
    elif chapter_index == current_chapter_index + 1:
        # Next chapter
        return 500 + (sentence_index % 100)
    elif chapter_index == current_chapter_index - 1:
        # Previous chapter
        return 700 + (sentence_index % 100)
    else:
        # Farther chapters
        distance_penalty = abs(chapter_distance) * 100
        return 900 + distance_penalty + (sentence_index % 100)


def clear_audio_queue():
    """Remove all queued jobs without touching existing files."""
    while not audio_generation_queue.empty():
        try:
            audio_generation_queue.get_nowait()
        except Exception:
            break


def now_ts():
    return time.time()


def normalize_upload_format(value: str):
    return audio_backend.normalize_upload_format(value, AUDIO_UPLOAD_FORMAT)


def infer_upload_format_from_url(url: str):
    return audio_backend.infer_format_from_url(url)


def build_job_snapshot(job):
    total = max(0, int(job.get('total', 0)))
    ready = max(0, int(job.get('ready', 0)))
    failed = max(0, int(job.get('failed', 0)))
    processed = min(total, ready + failed)
    generating = max(0, total - processed)
    percentage = (processed / total * 100.0) if total else 100.0
    return {
        'job_id': job['job_id'],
        'book_id': job['book_id'],
        'status': job['status'],
        'total': total,
        'ready': ready,
        'failed': failed,
        'processed': processed,
        'left': max(0, total - ready),
        'generating': generating,
        'percentage': round(max(0.0, min(100.0, percentage)), 2),
        'started_at': job.get('started_at'),
        'updated_at': job.get('updated_at'),
        'finished_at': job.get('finished_at'),
        'parallelism': job.get('parallelism'),
        'upload_format': job.get('upload_format'),
        'last_error': job.get('last_error')
    }


def set_job_state(job_id, **updates):
    with book_jobs_lock:
        job = book_jobs.get(job_id)
        if not job:
            return None
        job.update(updates)
        job['updated_at'] = now_ts()
        return build_job_snapshot(job)


def get_job_snapshot(job_id):
    with book_jobs_lock:
        job = book_jobs.get(job_id)
        if not job:
            return None
        return build_job_snapshot(job)


def sanitize_book_sentences(sentences):
    cleaned = []
    for idx, item in enumerate(sentences or []):
        if not isinstance(item, dict):
            continue
        text = (item.get('text') or '').strip()
        if not text:
            continue
        sentence_id = str(item.get('id') or f"s_{idx}").strip()
        try:
            sentence_index = max(0, int(item.get('sentence_index', idx) or idx))
        except (TypeError, ValueError):
            sentence_index = idx
        sentence_hash = (item.get('hash') or hash_text_for_coordinator(text)).strip()
        cleaned.append({
            'id': sentence_id,
            'sentence_index': sentence_index,
            'text': text,
            'hash': sentence_hash
        })
    return cleaned


def _upload_one_hash_entry(entry, ticket, upload_format, worker_output_dir):
    """Generate and upload one hash entry. Returns dict with ok/url/format/error."""
    sentence_hash = entry['hash']
    text = entry['text']

    # Guard against race conditions: if coordinator already has this hash, skip local generation.
    existing = coordinator_request(
        'POST',
        '/check',
        json_data={'hash': sentence_hash, 'text': text}
    )
    if existing and existing.get('status') == 'ready' and existing.get('url'):
        ready_url = existing['url']
        return {
            'ok': True,
            'url': ready_url,
            'format': infer_upload_format_from_url(ready_url)
        }

    audio_path = generate_sentence_locally(sentence_hash, text, worker_output_dir)
    if not audio_path:
        return {'ok': False, 'error': 'Local audio generation failed'}

    asset = audio_backend.build_upload_asset(audio_path, upload_format)
    effective_format = asset.audio_format
    effective_ticket = ticket
    if effective_ticket:
        ticket_format = normalize_upload_format(effective_ticket.get('format') or effective_format)
        if ticket_format != effective_format:
            effective_ticket = None

    if not effective_ticket:
        single_upload = coordinator_request(
            'POST',
            '/upload_url',
            json_data={'hash': sentence_hash, 'format': effective_format}
        )
        if not single_upload:
            return {'ok': False, 'error': 'Failed to request upload URL'}
        effective_ticket = {
            'upload_url': single_upload.get('upload_url'),
            'final_url': single_upload.get('final_url'),
            'format': single_upload.get('format') or effective_format,
            'content_type': single_upload.get('content_type') or asset.content_type
        }

    upload_url = effective_ticket.get('upload_url')
    final_url = effective_ticket.get('final_url')
    if not upload_url or not final_url:
        return {'ok': False, 'error': 'Invalid upload URL payload'}

    ticket_content_type = effective_ticket.get('content_type') or asset.content_type
    put_response = upload_to_presigned_url(upload_url, asset.upload_path, ticket_content_type)
    if put_response.status_code >= 400:
        return {'ok': False, 'error': f'S3 upload failed ({put_response.status_code})'}
    audio_backend.cleanup_paths(asset.cleanup_paths)

    return {
        'ok': True,
        'url': final_url,
        'format': effective_ticket.get('format') or effective_format
    }


def run_book_generation_job(job_id):
    """Background job: parallel local generation + S3 upload + batched coordinator completion."""
    snapshot = set_job_state(job_id, status='running', started_at=now_ts(), last_error=None)
    if not snapshot:
        return

    with book_jobs_lock:
        job = book_jobs.get(job_id)
        if not job:
            return
        book_id = job['book_id']
        upload_format = job['upload_format']
        parallelism = job['parallelism']
        sentences = list(job['sentences'])
        access_token = job.get('access_token')

    if access_token:
        set_runtime_coordinator_token(access_token)

    # Hash-dedupe within the book: generate once per unique text hash.
    hash_entries = {}
    for sentence in sentences:
        sentence_hash = sentence['hash']
        entry = hash_entries.get(sentence_hash)
        if not entry:
            entry = {
                'hash': sentence_hash,
                'text': sentence['text'],
                'refs': []
            }
            hash_entries[sentence_hash] = entry
        entry['refs'].append({
            'id': sentence['id'],
            'sentence_index': sentence['sentence_index'],
            'text': sentence['text']
        })

    # Check existing cloud audio in batches first.
    ready_hash_urls = {}
    hash_list = list(hash_entries.values())
    for i in range(0, len(hash_list), COORDINATOR_BATCH_SIZE):
        chunk = hash_list[i:i + COORDINATOR_BATCH_SIZE]
        result = coordinator_request(
            'POST',
            '/check_batch',
            json_data={'sentences': [{'hash': h['hash'], 'text': h['text']} for h in chunk]},
            timeout=max(COORDINATOR_TIMEOUT, 30)
        )
        if not result:
            continue
        for item in result.get('results', []):
            if item.get('status') == 'ready' and item.get('url') and item.get('hash'):
                ready_hash_urls[item['hash']] = item['url']

    completion_buffer = []
    buffer_lock = threading.Lock()
    worker_output_dir = os.path.join(app.config['AUDIO_FOLDER'], 'worker_cache')
    os.makedirs(worker_output_dir, exist_ok=True)

    def flush_completions(force=False):
        payload = []
        with buffer_lock:
            if not completion_buffer:
                return True
            if not force and len(completion_buffer) < WORKER_COMPLETE_BATCH_SIZE:
                return True
            payload = completion_buffer[:]
            completion_buffer.clear()
        ok = complete_batch_with_coordinator(payload, book_id=book_id)
        if not ok:
            with buffer_lock:
                completion_buffer.extend(payload)
        return ok

    # Count already-ready hashes as ready sentences and persist them in book_audio.
    initial_ready = 0
    for sentence_hash, url in ready_hash_urls.items():
        entry = hash_entries.get(sentence_hash)
        if not entry:
            continue
        for ref in entry['refs']:
            completion_buffer.append({
                'hash': sentence_hash,
                's3_url': url,
                'sentence_id': ref['id'],
                'sentence_index': ref['sentence_index'],
                'text': ref['text'],
                'audio_format': infer_upload_format_from_url(url)
            })
            initial_ready += 1
    if initial_ready:
        set_job_state(job_id, ready=initial_ready)
        flush_completions(force=True)

    # Generate/upload only missing hashes.
    pending_hash_entries = [h for h in hash_list if h['hash'] not in ready_hash_urls]

    try:
        for start in range(0, len(pending_hash_entries), WORKER_UPLOAD_URL_BATCH_SIZE):
            batch = pending_hash_entries[start:start + WORKER_UPLOAD_URL_BATCH_SIZE]
            if not batch:
                continue

            upload_tickets = request_upload_urls_batch([
                {'hash': h['hash'], 'format': upload_format}
                for h in batch
            ])

            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                future_map = {
                    executor.submit(
                        _upload_one_hash_entry,
                        entry,
                        upload_tickets.get(entry['hash']),
                        upload_format,
                        worker_output_dir
                    ): entry
                    for entry in batch
                }

                for future in as_completed(future_map):
                    entry = future_map[future]
                    refs = entry['refs']
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {'ok': False, 'error': str(e)}

                    if result.get('ok') and result.get('url'):
                        for ref in refs:
                            completion_buffer.append({
                                'hash': entry['hash'],
                                's3_url': result['url'],
                                'sentence_id': ref['id'],
                                'sentence_index': ref['sentence_index'],
                                'text': ref['text'],
                                'audio_format': result.get('format') or upload_format
                            })

                        with book_jobs_lock:
                            active = book_jobs.get(job_id)
                            if active:
                                active['ready'] += len(refs)
                                active['updated_at'] = now_ts()
                        flush_completions(force=False)
                    else:
                        with book_jobs_lock:
                            active = book_jobs.get(job_id)
                            if active:
                                active['failed'] += len(refs)
                                active['last_error'] = result.get('error') or 'Unknown upload error'
                                active['updated_at'] = now_ts()

            flush_completions(force=True)

        flush_completions(force=True)
        final = get_job_snapshot(job_id)
        status = 'completed'
        if final and final.get('ready', 0) <= 0 and final.get('failed', 0) > 0:
            status = 'failed'
        set_job_state(job_id, status=status, finished_at=now_ts())
    except Exception as e:
        print(f"Book generation job failed ({job_id}): {e}")
        print(traceback.format_exc())
        set_job_state(job_id, status='failed', finished_at=now_ts(), last_error=str(e))


# -----------------------------------------------------------------------------
# Audio generation worker
# -----------------------------------------------------------------------------
def audio_generation_worker():
    """Background worker thread that generates audio files from queue."""
    global audio_status, generation_active, sentence_remote_urls

    while generation_active:
        try:
            # (priority, counter, payload)
            priority, counter, (book_id, sentence_data, output_dir) = audio_generation_queue.get(timeout=1)
            sentence_id = sentence_data['id']
            sentence_hash = sentence_hashes.get(sentence_id) or hash_text_for_coordinator(sentence_data.get('text', ''))
            sentence_hashes[sentence_id] = sentence_hash

            # If status was updated by coordinator while queued, skip redundant work.
            if audio_status.get(sentence_id) == 'ready':
                audio_generation_queue.task_done()
                continue

            # If legacy local WAV exists, promote it to coordinator/S3 and cleanup.
            audio_path = os.path.join(output_dir, f"{sentence_id}.wav")
            if os.path.exists(audio_path):
                remote_url = sentence_remote_urls.get(sentence_id)
                if not remote_url and coordinator_enabled:
                    check = coordinator_request(
                        'POST',
                        '/check',
                        json_data={'hash': sentence_hash, 'text': sentence_data.get('text', '')}
                    )
                    if check and check.get('status') == 'ready' and check.get('url'):
                        remote_url = check['url']
                        sentence_remote_urls[sentence_id] = remote_url

                if not remote_url:
                    remote_url = upload_audio_to_coordinator(sentence_hash, audio_path)
                    if remote_url:
                        sentence_remote_urls[sentence_id] = remote_url

                if remote_url:
                    # Ensure any leftover local WAV is removed after cloud success.
                    audio_backend.cleanup_paths([audio_path])
                    audio_status[sentence_id] = 'ready'
                    socketio.emit('audio_ready', {
                        'sentence_id': sentence_id,
                        'status': 'ready'
                    })
                else:
                    audio_status[sentence_id] = 'failed'
                    socketio.emit('audio_failed', {
                        'sentence_id': sentence_id,
                        'status': 'failed'
                    })
                audio_generation_queue.task_done()
                continue

            # If coordinator already has this sentence, use remote URL and skip generation.
            remote_url = sentence_remote_urls.get(sentence_id)
            if not remote_url and coordinator_enabled:
                check = coordinator_request(
                    'POST',
                    '/check',
                    json_data={'hash': sentence_hash, 'text': sentence_data.get('text', '')}
                )
                if check and check.get('status') == 'ready' and check.get('url'):
                    remote_url = check['url']
                    sentence_remote_urls[sentence_id] = remote_url

            if remote_url:
                audio_status[sentence_id] = 'ready'
                socketio.emit('audio_ready', {
                    'sentence_id': sentence_id,
                    'status': 'ready'
                })
                audio_generation_queue.task_done()
                continue

            audio_status[sentence_id] = 'generating'
            socketio.emit('audio_generating', {
                'sentence_id': sentence_id,
                'status': 'generating'
            })

            generated_wav = audio_backend.generate_sentence_wav(
                sentence_id=sentence_id,
                text=sentence_data.get('text', ''),
                output_dir=output_dir,
                sentence_index=sentence_data.get('sentence_index', 0)
            )
            success = bool(generated_wav and os.path.exists(generated_wav))

            if success:
                uploaded_url = upload_audio_to_coordinator(sentence_hash, generated_wav)
                if uploaded_url:
                    sentence_remote_urls[sentence_id] = uploaded_url
                audio_status[sentence_id] = 'ready'
                socketio.emit('audio_ready', {
                    'sentence_id': sentence_id,
                    'status': 'ready'
                })
            else:
                audio_status[sentence_id] = 'failed'
                socketio.emit('audio_failed', {
                    'sentence_id': sentence_id,
                    'status': 'failed'
                })

            audio_generation_queue.task_done()

        except Empty:
            # Queue is empty: use spare worker time for coordinator backlog.
            # Current/nearby sentence work always wins because it is queued first.
            did_idle_work = process_idle_coordinator_task_once()
            if not did_idle_work:
                time.sleep(0.1)
            continue
        except Exception as e:
            if generation_active:
                print(f"Worker error: {e}")
            # Don't crash the worker on a single bad sentence
            continue


def start_workers_if_needed():
    """Ensure worker threads are running, matching pool size."""
    global audio_generation_threads, generation_active

    if not generation_active:
        return

    # Clean dead threads
    audio_generation_threads = [t for t in audio_generation_threads if t.is_alive()]

    desired = audio_backend.pool_size or 4
    missing = desired - len(audio_generation_threads)

    for _ in range(max(missing, 0)):
        t = threading.Thread(target=audio_generation_worker, daemon=True)
        t.start()
        audio_generation_threads.append(t)


def should_run_idle_backfill():
    if not WORKER_IDLE_BACKFILL_ENABLED:
        return False
    if not coordinator_enabled or not get_effective_coordinator_token():
        return False
    if generation_active:
        return False
    try:
        if not audio_generation_queue.empty():
            return False
    except Exception:
        return False

    with book_jobs_lock:
        for job in book_jobs.values():
            if job.get('status') in ('queued', 'running'):
                return False
    return True


def idle_backfill_worker_loop():
    """Background coordinator backfill when local worker is otherwise idle."""
    while True:
        try:
            if not should_run_idle_backfill():
                time.sleep(0.3)
                continue

            did_work = process_idle_coordinator_task_once()
            if not did_work:
                time.sleep(COORDINATOR_IDLE_POLL_INTERVAL)
                continue
            time.sleep(0.05)
        except Exception as e:
            coordinator_metrics['idle_tasks_failed'] += 1
            coordinator_metrics['last_error'] = str(e)
            time.sleep(1.0)


def ensure_idle_backfill_thread():
    global idle_backfill_thread

    if not WORKER_IDLE_BACKFILL_ENABLED:
        return
    if idle_backfill_thread and idle_backfill_thread.is_alive():
        return

    idle_backfill_thread = threading.Thread(target=idle_backfill_worker_loop, daemon=True)
    idle_backfill_thread.start()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    """Legacy UI endpoint (deprecated)."""
    return jsonify({
        'status': 'deprecated',
        'message': 'Legacy template frontend is deprecated. Use the calm-reader frontend deployment.'
    }), 410


@app.route('/upload', methods=['POST'])
def upload_epub():
    """Upload an EPUB, parse it, and start background audio generation."""
    global current_book, epub_processor, audio_status, generation_active, current_chapter_index
    global sentence_remote_urls, current_page_sentences
    capture_coordinator_token_from_request()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save upload to temporary path first so we can compute content hash.
    filename = secure_filename(file.filename) or 'upload.epub'
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f".tmp_{time.time_ns()}_{filename}")
    file.save(tmp_path)

    # Identify book and move to canonical cache path (prevents name collisions).
    book_id = get_book_hash(tmp_path)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{book_id}.epub")
    if os.path.exists(filepath):
        os.remove(tmp_path)
    else:
        shutil.move(tmp_path, filepath)

    # Audio directory per book hash
    book_audio_dir = os.path.join(app.config['AUDIO_FOLDER'], book_id)
    os.makedirs(book_audio_dir, exist_ok=True)

    # Process EPUB
    epub_processor = EPUBProcessor(filepath)
    book_data = epub_processor.process()

    # Reset state
    audio_status = {}
    current_chapter_index = 0
    current_page_sentences = []
    sentence_remote_urls = {}
    initialize_sentence_hashes(book_data)

    current_book = {
        'id': book_id,
        'title': book_data['title'],
        'chapters': book_data['chapters'],
        'sentences': book_data['sentences'],
        'book_id': book_id
    }

    # Initialize audio status for all sentences.
    for sentence in book_data['sentences']:
        if coordinator_enabled:
            audio_status[sentence['id']] = 'pending'
        else:
            audio_file = os.path.join(book_audio_dir, f"{sentence['id']}.wav")
            audio_status[sentence['id']] = 'ready' if os.path.exists(audio_file) else 'pending'

    # Discover existing remote audio in coordinator so we avoid duplicate generation.
    if coordinator_enabled:
        threading.Thread(
            target=sync_book_with_coordinator,
            args=(book_id, current_book['sentences'], book_audio_dir),
            daemon=True
        ).start()

    # Save book metadata for recent books list
    state_manager = get_state_manager()
    state_manager.save_book_metadata(book_id, current_book['title'], len(current_book['chapters']))

    # Add to library with cover image and file path
    library_manager = get_library_manager()
    cover_data = None
    if epub_processor.cover_image:
        import base64
        cover_data = f"data:image/jpeg;base64,{base64.b64encode(epub_processor.cover_image).decode()}"

    library_manager.add_book(
        book_id=book_id,
        title=current_book['title'],
        author=book_data.get('author', ''),
        cover_data=cover_data,
        total_chapters=len(current_book['chapters']),
        total_sentences=len(current_book['sentences']),
        file_path=filepath
    )

    # Audio generation will be triggered when user opens the book content
    # Not automatically on upload

    return jsonify({
        'success': True,
        'book': {
            'id': book_id,
            'title': current_book['title'],
            'total_chapters': len(current_book['chapters']),
            'total_sentences': len(current_book['sentences'])
        }
    })


@app.route('/book/content')
def get_book_content():
    """Return complete book structure for the reader UI."""
    if not current_book:
        return jsonify({'error': 'No book loaded'}), 404

    return jsonify({
        'title': current_book['title'],
        'chapters': current_book['chapters'],   # each has id, title, html, sentences
        'sentences': current_book['sentences'], # flat list for audio mapping
        'book_id': current_book['book_id'],
        'audio_status': audio_status
    })


@app.route('/load_book/<book_id>', methods=['POST'])
def load_book_from_library(book_id):
    """Load a book from library by its ID."""
    global current_book, epub_processor, audio_status, current_chapter_index
    global sentence_remote_urls, current_page_sentences
    capture_coordinator_token_from_request()

    # Check if book exists in library
    library_manager = get_library_manager()
    book_info = library_manager.get_book(book_id)

    if not book_info:
        return jsonify({'success': False, 'error': 'Book not found in library'}), 404

    # Get the epub file path from library
    epub_path = book_info.get('file_path')

    # Verify the file exists, or try to find it (fallback for old entries)
    if not epub_path or not os.path.exists(epub_path):
        upload_folder = app.config['UPLOAD_FOLDER']
        canonical_path = os.path.join(upload_folder, f"{book_id}.epub")
        if os.path.exists(canonical_path):
            epub_path = canonical_path
        else:
            # Fallback for legacy file names.
            epub_path = None
            for filename in os.listdir(upload_folder):
                if filename.endswith('.epub'):
                    filepath = os.path.join(upload_folder, filename)
                    if get_book_hash(filepath) == book_id:
                        epub_path = filepath
                        break

    if not epub_path or not os.path.exists(epub_path):
        return jsonify({'success': False, 'error': 'Book file not found. Please re-upload.'}), 404

    # Process the EPUB
    book_audio_dir = os.path.join(app.config['AUDIO_FOLDER'], book_id)
    os.makedirs(book_audio_dir, exist_ok=True)

    epub_processor = EPUBProcessor(epub_path)
    book_data = epub_processor.process()

    # Reset state
    audio_status = {}
    current_chapter_index = 0
    current_page_sentences = []
    sentence_remote_urls = {}
    initialize_sentence_hashes(book_data)

    current_book = {
        'id': book_id,
        'title': book_data['title'],
        'chapters': book_data['chapters'],
        'sentences': book_data['sentences'],
        'book_id': book_id
    }

    # Initialize audio status.
    for sentence in book_data['sentences']:
        if coordinator_enabled:
            audio_status[sentence['id']] = 'pending'
        else:
            audio_file = os.path.join(book_audio_dir, f"{sentence['id']}.wav")
            audio_status[sentence['id']] = 'ready' if os.path.exists(audio_file) else 'pending'

    # Sync with coordinator in background for already-generated S3 audio.
    if coordinator_enabled:
        threading.Thread(
            target=sync_book_with_coordinator,
            args=(book_id, current_book['sentences'], book_audio_dir),
            daemon=True
        ).start()

    # Update library open count
    library_manager.add_book(
        book_id=book_id,
        title=current_book['title'],
        author=book_data.get('author', ''),
        total_chapters=len(current_book['chapters']),
        total_sentences=len(current_book['sentences']),
        file_path=epub_path
    )

    return jsonify({'success': True, 'book_id': book_id})


@app.route('/audio/<book_id>/<sentence_id>')
def get_audio(book_id, sentence_id):
    """Return generated audio for a sentence if available."""
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], book_id, f'{sentence_id}.wav')
    if os.path.exists(audio_path) and not coordinator_enabled:
        return send_file(audio_path, mimetype='audio/wav')

    remote_url = sentence_remote_urls.get(sentence_id)
    if remote_url:
        audio_backend.cleanup_paths([audio_path])
        return redirect(remote_url, code=302)

    # Legacy fallback: local WAV exists but remote URL is unknown.
    if os.path.exists(audio_path):
        sentence_hash = sentence_hashes.get(sentence_id)
        if sentence_hash:
            uploaded_url = upload_audio_to_coordinator(sentence_hash, audio_path)
            if uploaded_url:
                sentence_remote_urls[sentence_id] = uploaded_url
                return redirect(uploaded_url, code=302)
        return send_file(audio_path, mimetype='audio/wav')

    return jsonify({'error': 'Audio not found'}), 404


@app.route('/audio/status/<sentence_id>')
def get_audio_status(sentence_id):
    """Check if audio is ready for a specific sentence."""
    status = audio_status.get(sentence_id, 'unknown')
    return jsonify({'status': status})


@app.route('/worker/health')
def worker_health():
    """Health check for local Docker worker integration."""
    capture_coordinator_token_from_request()
    return jsonify({
        'status': 'ok',
        'service': 'local-audio-worker',
        'coordinator_enabled': coordinator_enabled,
        'coordinator_api_url': COORDINATOR_API_URL or None,
        'tts_pool_requested': WORKER_TTS_POOL_REQUESTED,
        'tts_pool_size': WORKER_TTS_POOL_SIZE,
        'memory_limit_mb': WORKER_MEMORY_LIMIT_MB,
        'book_parallelism': WORKER_BOOK_PARALLELISM,
        'upload_format': normalize_upload_format(AUDIO_UPLOAD_FORMAT)
    })


@app.route('/worker/jobs')
def worker_jobs():
    """Current worker job summary for terminal monitoring."""
    with book_jobs_lock:
        snapshots = [build_job_snapshot(job) for job in book_jobs.values()]

    snapshots.sort(key=lambda x: x.get('updated_at') or 0, reverse=True)

    status_counts = {
        'queued': 0,
        'running': 0,
        'completed': 0,
        'failed': 0,
    }
    total_left = 0
    total_ready = 0
    total_failed_items = 0
    active_jobs = []

    for snap in snapshots:
        status = snap.get('status') or ''
        if status in status_counts:
            status_counts[status] += 1
        total_left += max(0, int(snap.get('left') or 0))
        total_ready += max(0, int(snap.get('ready') or 0))
        total_failed_items += max(0, int(snap.get('failed') or 0))
        if status in ('queued', 'running'):
            active_jobs.append(snap)

    try:
        queue_size = audio_generation_queue.qsize()
    except Exception:
        queue_size = None

    queue_workers_alive = sum(1 for t in audio_generation_threads if t.is_alive())
    idle_backfill_alive = bool(idle_backfill_thread and idle_backfill_thread.is_alive())
    alive_workers = queue_workers_alive + (1 if idle_backfill_alive else 0)

    return jsonify({
        'status': 'ok',
        'generation_active': generation_active,
        'current_book_id': current_book.get('book_id') if isinstance(current_book, dict) else None,
        'queue_size': queue_size,
        'alive_workers': alive_workers,
        'queue_workers_alive': queue_workers_alive,
        'idle_backfill_active': idle_backfill_alive,
        'summary': {
            'total_jobs': len(snapshots),
            'queued_jobs': status_counts['queued'],
            'running_jobs': status_counts['running'],
            'completed_jobs': status_counts['completed'],
            'failed_jobs': status_counts['failed'],
            'left_items': total_left,
            'ready_items': total_ready,
            'failed_items': total_failed_items,
        },
        'active_jobs': active_jobs,
        'updated_at': now_ts(),
    })


@app.route('/worker/book/start', methods=['POST'])
def worker_start_book_generation():
    """
    Start background generation for a full book using local worker resources.
    """
    data = request.get_json(silent=True) or {}
    capture_coordinator_token_from_request(data)

    if not coordinator_enabled:
        return jsonify({'success': False, 'error': 'Coordinator is not configured'}), 503
    if not get_effective_coordinator_token():
        return jsonify({'success': False, 'error': 'Missing Cognito access token'}), 401

    book_id = str(data.get('book_id') or '').strip().lower()
    if not book_id:
        return jsonify({'success': False, 'error': 'Missing book_id'}), 400

    sentences = sanitize_book_sentences(data.get('sentences') or [])
    if not sentences:
        return jsonify({'success': False, 'error': 'Missing or empty sentences'}), 400

    # Reuse active job for the same book to avoid duplicate heavy processing.
    with book_jobs_lock:
        for existing in book_jobs.values():
            if existing.get('book_id') == book_id and existing.get('status') in ('queued', 'running'):
                return jsonify({
                    'success': True,
                    'reused': True,
                    'job': build_job_snapshot(existing)
                })

    requested_parallelism = data.get('parallelism')
    try:
        parallelism = int(requested_parallelism) if requested_parallelism is not None else WORKER_BOOK_PARALLELISM
    except (TypeError, ValueError):
        parallelism = WORKER_BOOK_PARALLELISM
    parallelism = max(1, min(parallelism, 32))

    upload_format = normalize_upload_format(data.get('upload_format') or data.get('format') or AUDIO_UPLOAD_FORMAT)
    job_id = uuid.uuid4().hex[:16]
    created_at = now_ts()
    job = {
        'job_id': job_id,
        'book_id': book_id,
        'status': 'queued',
        'total': len(sentences),
        'ready': 0,
        'failed': 0,
        'parallelism': parallelism,
        'upload_format': upload_format,
        'created_at': created_at,
        'started_at': None,
        'updated_at': created_at,
        'finished_at': None,
        'last_error': None,
        'sentences': sentences,
        'access_token': get_effective_coordinator_token()
    }

    with book_jobs_lock:
        book_jobs[job_id] = job

        # Keep memory bounded: drop oldest finished jobs if map grows.
        if len(book_jobs) > 40:
            finished = [j for j in book_jobs.values() if j.get('status') in ('completed', 'failed')]
            finished.sort(key=lambda x: x.get('updated_at') or 0)
            for stale in finished[:max(0, len(book_jobs) - 30)]:
                book_jobs.pop(stale['job_id'], None)

    threading.Thread(target=run_book_generation_job, args=(job_id,), daemon=True).start()
    return jsonify({'success': True, 'job': build_job_snapshot(job)})


@app.route('/worker/book/status/<job_id>', methods=['GET'])
def worker_book_generation_status(job_id):
    snapshot = get_job_snapshot((job_id or '').strip())
    if not snapshot:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    return jsonify({'success': True, 'job': snapshot})


@app.route('/worker/book/status', methods=['GET'])
def worker_book_generation_status_by_book():
    book_id = str(request.args.get('book_id') or '').strip().lower()
    if not book_id:
        return jsonify({'success': False, 'error': 'Missing book_id'}), 400

    with book_jobs_lock:
        matches = [j for j in book_jobs.values() if j.get('book_id') == book_id]
    if not matches:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    matches.sort(key=lambda x: x.get('updated_at') or 0, reverse=True)
    return jsonify({'success': True, 'job': build_job_snapshot(matches[0])})


@app.route('/worker/generate', methods=['POST'])
def worker_generate_audio():
    """
    Generate audio locally on this machine and upload to S3 via coordinator.
    Expects Cognito bearer token in Authorization header.
    """
    try:
        data = request.get_json(silent=True) or {}
        capture_coordinator_token_from_request(data)

        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({'success': False, 'error': 'Missing text'}), 400

        if not coordinator_enabled:
            return jsonify({'success': False, 'error': 'Coordinator is not configured'}), 503

        if not get_effective_coordinator_token():
            return jsonify({'success': False, 'error': 'Missing Cognito access token'}), 401

        sentence_hash = (data.get('hash') or hash_text_for_coordinator(text)).strip()

        # If S3 audio already exists, return immediately.
        check = coordinator_request('POST', '/check', json_data={
            'hash': sentence_hash,
            'text': text
        })
        if check and check.get('status') == 'ready' and check.get('url'):
            return jsonify({
                'success': True,
                'hash': sentence_hash,
                'url': check['url'],
                'source': 'coordinator'
            })

        output_dir = os.path.join(app.config['AUDIO_FOLDER'], 'worker_cache')
        audio_path = generate_sentence_locally(sentence_hash, text, output_dir)
        if not audio_path:
            # One quick retry to smooth over transient runtime/model failures.
            time.sleep(0.15)
            audio_path = generate_sentence_locally(sentence_hash, text, output_dir)
        if not audio_path:
            return jsonify({'success': False, 'error': 'Local audio generation failed'}), 503

        final_url = upload_audio_to_coordinator(sentence_hash, audio_path)
        if not final_url:
            # One last check in case another client completed meanwhile.
            check = coordinator_request('POST', '/check', json_data={
                'hash': sentence_hash,
                'text': text
            })
            if check and check.get('status') == 'ready' and check.get('url'):
                final_url = check['url']

        if not final_url:
            return jsonify({'success': False, 'error': 'Failed to upload audio to coordinator/S3'}), 502

        return jsonify({
            'success': True,
            'hash': sentence_hash,
            'url': final_url,
            'source': 'local-worker'
        })
    except Exception as e:
        print(f"Unhandled /worker/generate failure: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': 'Internal worker error'}), 503


@app.route('/coordinator/status')
def coordinator_status():
    """Expose coordinator connectivity/auth status for diagnostics."""
    capture_coordinator_token_from_request()
    has_runtime = bool(get_effective_coordinator_token()) and not bool(COORDINATOR_BEARER_TOKEN)
    has_env_token = bool(COORDINATOR_BEARER_TOKEN)
    has_worker_secret = bool(WORKER_SHARED_SECRET)
    return jsonify({
        'enabled': coordinator_enabled,
        'api_url': COORDINATOR_API_URL or None,
        'has_bearer_token': bool(get_effective_coordinator_token()),
        'has_worker_secret': has_worker_secret,
        'worker_secret_source': WORKER_SHARED_SECRET_SOURCE,
        'token_source': 'runtime' if has_runtime else ('env' if has_env_token else None),
        'metrics': coordinator_metrics
    })


@app.route('/worker/token', methods=['POST'])
def set_worker_token():
    """Canonical endpoint for browser -> local worker token sync."""
    result = apply_worker_token_request()
    code = 200 if result.get('success') else 400
    return jsonify(result), code


@app.route('/coordinator/token', methods=['POST'])
def set_coordinator_token():
    """
    Backward-compatible alias for /worker/token.
    """
    result = apply_worker_token_request()
    code = 200 if result.get('success') else 400
    return jsonify(result), code


@app.route('/prioritize', methods=['POST'])
def prioritize_sentences(data=None):
    """
    Prioritize audio generation for specific sentences.
    Body: { "sentence_ids": [id1, id2, ...] }
    """
    global current_book

    if data is None:
        data = request.json or {}
        capture_coordinator_token_from_request(data)
    sentence_ids = data.get('sentence_ids', [])

    if not current_book:
        return jsonify({'error': 'No book loaded'}), 404

    book_audio_dir = os.path.join(app.config['AUDIO_FOLDER'], current_book['book_id'])

    for sentence_id in sentence_ids:
        if audio_status.get(sentence_id) != 'pending':
            continue

        sentence = next((s for s in current_book['sentences'] if s['id'] == sentence_id), None)
        if not sentence:
            continue

        counter = next(priority_counter)
        # Absolute highest priority = 0
        audio_generation_queue.put((0, counter, (current_book['book_id'], sentence, book_audio_dir)))

    start_workers_if_needed()
    return jsonify({'success': True})


@app.route('/update_position', methods=['POST'])
def update_reading_position():
    """
    Update current reading position for smarter prioritization.
    Body: { "chapter_index": int, "page_sentences": [sentence_id, ...] }
    """
    global current_chapter_index, current_page_sentences, current_book

    data = request.json or {}
    capture_coordinator_token_from_request(data)
    current_chapter_index = data.get('chapter_index', 0)
    current_page_sentences = data.get('page_sentences', []) or []

    if current_book and current_page_sentences:
        book_audio_dir = os.path.join(app.config['AUDIO_FOLDER'], current_book['book_id'])

        # Ensure visible sentences are in the queue with top priority
        for sent_id in current_page_sentences:
            if audio_status.get(sent_id) == 'pending':
                sentence = next((s for s in current_book['sentences'] if s['id'] == sent_id), None)
                if not sentence:
                    continue
                counter = next(priority_counter)
                audio_generation_queue.put((0, counter, (current_book['book_id'], sentence, book_audio_dir)))

        start_workers_if_needed()

    return jsonify({'success': True, 'chapter': current_chapter_index})


@app.route('/clear')
def clear_book():
    """
    Reset backend state. This does NOT delete audio files from disk,
    just clears in-memory state and queue.
    """
    global current_book, epub_processor, generation_active
    global audio_status, current_chapter_index, current_page_sentences
    global sentence_hashes, sentence_remote_urls

    generation_active = False
    current_book = None
    epub_processor = None
    audio_status = {}
    sentence_hashes = {}
    sentence_remote_urls = {}
    current_chapter_index = 0
    current_page_sentences = []

    clear_audio_queue()
    return jsonify({'success': True})


# -----------------------------------------------------------------------------
# Reading Position Persistence
# -----------------------------------------------------------------------------
@app.route('/position/save', methods=['POST'])
def save_reading_position():
    """Save current reading position for a book."""
    data = request.json or {}
    capture_coordinator_token_from_request(data)
    book_id = data.get('book_id')
    position = data.get('position')

    if not book_id or not position:
        return jsonify({'error': 'Missing book_id or position'}), 400

    state_manager = get_state_manager()
    success = state_manager.save_position(book_id, position)

    # Also save audio progress if available
    if current_book and current_book.get('book_id') == book_id and audio_status:
        state_manager.save_audio_status(book_id, audio_status)

    return jsonify({'success': success})


@app.route('/position/load/<book_id>')
def load_reading_position(book_id):
    """Load saved reading position for a book."""
    state_manager = get_state_manager()
    position = state_manager.load_position(book_id)

    if position:
        return jsonify({'success': True, 'position': position})
    return jsonify({'success': False, 'position': None})


@app.route('/books/recent')
def get_recent_books():
    """Get list of recently read books."""
    state_manager = get_state_manager()
    books = state_manager.get_recent_books()
    return jsonify({'books': books})


@app.route('/start_generation', methods=['POST'])
def start_audio_generation():
    """
    Trigger audio generation when book content is displayed.
    Called when user navigates to a book (not just uploads).
    """
    global generation_active, current_book, audio_status

    data = request.json or {}
    capture_coordinator_token_from_request(data)
    book_id = data.get('book_id')

    if not current_book or current_book.get('book_id') != book_id:
        return jsonify({'error': 'Book not loaded'}), 404

    book_audio_dir = os.path.join(app.config['AUDIO_FOLDER'], book_id)

    # Keep local status in sync with coordinator before queueing generation.
    if coordinator_enabled:
        sync_book_with_coordinator(book_id, current_book['sentences'], book_audio_dir)

    # If generation is already active, just ensure workers are running
    if generation_active:
        start_workers_if_needed()
        return jsonify({'success': True, 'message': 'Generation already active'})

    # Initialize generation backend pool if needed
    audio_backend.preload()

    # Start generation

    def queue_pending_sentences():
        global generation_active

        generation_active = True
        clear_audio_queue()

        # Map sentence_id -> chapter_index for priority
        chapter_sentence_map = {}
        for ch_idx, chapter in enumerate(current_book['chapters']):
            for sent in chapter.get('sentences', []):
                chapter_sentence_map[sent['id']] = ch_idx

        for idx, sentence in enumerate(current_book['sentences']):
            if audio_status.get(sentence['id']) == 'ready':
                continue

            ch_idx = chapter_sentence_map.get(sentence['id'], 0)
            priority = calculate_priority(idx, sentence['id'], ch_idx)
            counter = next(priority_counter)
            audio_generation_queue.put((priority, counter, (book_id, sentence, book_audio_dir)))

        start_workers_if_needed()

    threading.Thread(target=queue_pending_sentences, daemon=True).start()

    return jsonify({'success': True})


# -----------------------------------------------------------------------------
# Ollama Proxy (to avoid CORS issues)
# -----------------------------------------------------------------------------
@app.route('/ollama/status')
def ollama_status():
    """Check if Ollama is running and get available models."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.ok:
            data = response.json()
            models = [m.get('name', '') for m in data.get('models', [])]
            return jsonify({'connected': True, 'models': models})
    except Exception as e:
        print(f"Ollama connection error: {e}")
    return jsonify({'connected': False, 'models': []})


@app.route('/ollama/chat', methods=['POST'])
def ollama_chat():
    """Proxy chat requests to Ollama."""
    try:
        data = request.json or {}
        prompt = data.get('prompt', '')
        model = data.get('model', 'llama3.2')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Make request to Ollama
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            },
            timeout=120  # 2 minute timeout for generation
        )

        if response.ok:
            result = response.json()
            return jsonify({
                'success': True,
                'response': result.get('response', ''),
                'model': model
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Ollama returned status {response.status_code}'
            }), 500

    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'Request timed out'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'error': 'Cannot connect to Ollama. Is it running?'}), 503
    except Exception as e:
        print(f"Ollama chat error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# -----------------------------------------------------------------------------
# Translation API
# -----------------------------------------------------------------------------
@app.route('/translate', methods=['POST'])
def translate():
    """
    Translate text to target language.
    POST body: { "text": "hello", "target": "es", "source": "auto" }
    """
    try:
        data = request.json
        text = data.get('text', '').strip()
        target_lang = data.get('target', 'es')
        source_lang = data.get('source', 'auto')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = translate_text(text, target_lang, source_lang)
        return jsonify(result)

    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/languages')
def languages():
    """Get list of supported languages for translation."""
    return jsonify(get_languages())


# -----------------------------------------------------------------------------
# Library Management
# -----------------------------------------------------------------------------
@app.route('/library')
def get_library():
    """Get all books in the library with progress."""
    library_manager = get_library_manager()
    books = library_manager.get_all_books()

    # Enrich with cover URLs
    for book in books:
        book['cover_url'] = library_manager.get_cover_url(book['book_id'])

    return jsonify({'books': books})


@app.route('/library/cover/<book_id>')
def get_library_cover(book_id):
    """Serve a book's cover image."""
    library_manager = get_library_manager()
    cover_path = library_manager.covers_path / f"{book_id}.jpg"

    if cover_path.exists():
        return send_file(cover_path, mimetype='image/jpeg')

    # Return a placeholder if no cover
    return '', 404


@app.route('/library/<book_id>', methods=['DELETE'])
def remove_from_library(book_id):
    """Remove a book from the library."""
    library_manager = get_library_manager()
    library_manager.remove_book(book_id)
    return jsonify({'success': True})


@app.route('/library/book/<book_id>')
def get_library_book(book_id):
    """Get a specific book's details from the library."""
    library_manager = get_library_manager()
    book = library_manager.get_book(book_id)

    if book:
        book['cover_url'] = library_manager.get_cover_url(book_id)
        return jsonify({'success': True, 'book': book})

    return jsonify({'success': False, 'error': 'Book not found'}), 404


@app.route('/library/update_progress', methods=['POST'])
def update_library_progress():
    """Update reading/audio progress for a book in library."""
    global audio_status

    data = request.json or {}
    book_id = data.get('book_id')

    if not book_id:
        return jsonify({'error': 'Missing book_id'}), 400

    library_manager = get_library_manager()

    # Update reading progress
    if 'chapter_index' in data:
        library_manager.update_reading_progress(
            book_id=book_id,
            current_chapter=data.get('chapter_index', 0),
            current_sentence_index=data.get('sentence_index', 0),
            total_chapters=data.get('total_chapters')
        )

    # Update audio progress
    if audio_status:
        ready_count = sum(1 for status in audio_status.values() if status == 'ready')
        total = len(audio_status)
        library_manager.update_audio_progress(book_id, ready_count, total)

    return jsonify({'success': True})


# -----------------------------------------------------------------------------
# Socket.IO events
# -----------------------------------------------------------------------------
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('request_chapter_audio')
def handle_chapter_audio_request(data):
    """
    Prioritize audio generation for a specific chapter.
    Payload: { "chapter_index": idx }
    """
    global current_book

    if not current_book:
        return

    chapter_index = data.get('chapter_index', 0)
    if chapter_index < 0 or chapter_index >= len(current_book['chapters']):
        return

    chapter = current_book['chapters'][chapter_index]
    sentence_ids = [s['id'] for s in chapter.get('sentences', []) if audio_status.get(s['id']) == 'pending']

    # Limit how many we push at once to keep queue responsive
    prioritize_sentences({'sentence_ids': sentence_ids[:50]})


def preload_models():
    """Preload ML models in background at startup."""
    # Preload SpaCy model
    print("Preloading SpaCy model...")
    from utils.epub_processor import get_spacy_model
    get_spacy_model()

    # Preload audio generation backend pool
    print("Preloading audio generation backend...")
    audio_backend.preload()
    print("All models preloaded!")


if __name__ == '__main__':
    # Preload models in background thread at startup
    load_runtime_coordinator_token()
    threading.Thread(target=preload_models, daemon=True).start()
    ensure_idle_backfill_thread()

    # Bind all interfaces so Docker port mapping can reach the service.
    is_production = os.environ.get('FLASK_ENV', '').lower() == 'production'
    run_kwargs = {
        'debug': not is_production,
        'host': '0.0.0.0',
        'port': int(os.environ.get('WORKER_PORT') or os.environ.get('PORT') or '5001'),
    }
    if is_production:
        try:
            socketio.run(app, allow_unsafe_werkzeug=True, **run_kwargs)
        except TypeError:
            # Older Werkzeug/Flask combinations do not accept allow_unsafe_werkzeug.
            socketio.run(app, **run_kwargs)
    else:
        socketio.run(app, **run_kwargs)
