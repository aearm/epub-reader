"""
EPUB Reader Coordinator API

Manages distributed audio generation:
- Tracks audio status (pending/ready)
- Assigns tasks to volunteer nodes
- Stores index in SQLite
- Audio files stored in S3
"""

import os
import sqlite3
import hashlib
import time
import json
import boto3
import threading
import io
import wave
from array import array
from flask import Flask, request, jsonify, g, Response, stream_with_context
from flask_cors import CORS
from functools import wraps
from jose import jwt, JWTError
import requests
from urllib.parse import unquote, urlparse

app = Flask(__name__)
CORS(app, origins=[
    "https://reader.psybytes.com",
    "http://localhost:5001",
    "http://localhost:3000"
])

# Configuration from environment
AUDIO_BUCKET = os.environ.get('AUDIO_BUCKET', 'epub-reader-audio')
AWS_REGION = os.environ.get('AWS_REGION', 'eu-west-1')
COGNITO_POOL_ID = os.environ.get('COGNITO_POOL_ID', '')
COGNITO_CLIENT_ID = os.environ.get('COGNITO_CLIENT_ID', '')
DATABASE_PATH = os.environ.get('DATABASE_PATH', '/opt/epub-reader/coordinator.db')
SQLITE_TIMEOUT_SECONDS = max(1.0, float(os.environ.get('SQLITE_TIMEOUT_SECONDS', '30')))
SQLITE_BUSY_TIMEOUT_MS = max(
    1000,
    int(os.environ.get('SQLITE_BUSY_TIMEOUT_MS', str(int(SQLITE_TIMEOUT_SECONDS * 1000))))
)
KOKORO_MODEL_DIR = os.environ.get('KOKORO_MODEL_DIR', '/opt/epub-reader/models')
KOKORO_MODEL_PATH = os.environ.get('KOKORO_MODEL_PATH', os.path.join(KOKORO_MODEL_DIR, 'kokoro-v1.0.onnx'))
KOKORO_VOICES_PATH = os.environ.get('KOKORO_VOICES_PATH', os.path.join(KOKORO_MODEL_DIR, 'voices-v1.0.bin'))
KOKORO_MODEL_URL = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx'
KOKORO_VOICES_URL = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin'
KOKORO_VOICE = os.environ.get('KOKORO_VOICE', 'af_sarah')
KOKORO_LANG = os.environ.get('KOKORO_LANG', 'en-us')
MAX_GENERATE_SENTENCES = int(os.environ.get('MAX_GENERATE_SENTENCES', '6000'))
MAX_MANIFEST_ITEMS = int(os.environ.get('MAX_MANIFEST_ITEMS', '10000'))
AUDIO_SQS_QUEUE_URL = (os.environ.get('AUDIO_SQS_QUEUE_URL') or '').strip()
AUDIO_SQS_DLQ_URL = (os.environ.get('AUDIO_SQS_DLQ_URL') or '').strip()
AUDIO_SQS_WAIT_SECONDS = max(0, min(int(os.environ.get('AUDIO_SQS_WAIT_SECONDS', '2')), 20))
AUDIO_SQS_MAX_MESSAGES = max(1, min(int(os.environ.get('AUDIO_SQS_MAX_MESSAGES', '10')), 10))
AUDIO_SQS_VISIBILITY_TIMEOUT = max(30, int(os.environ.get('AUDIO_SQS_VISIBILITY_TIMEOUT', '180')))
AUDIO_SQS_MAX_RECEIVE_COUNT = max(1, int(os.environ.get('AUDIO_SQS_MAX_RECEIVE_COUNT', '8')))
AUDIO_GENERATING_STALE_SECONDS = max(
    AUDIO_SQS_VISIBILITY_TIMEOUT,
    int(os.environ.get('AUDIO_GENERATING_STALE_SECONDS', str(AUDIO_SQS_VISIBILITY_TIMEOUT)))
)
GENERATE_AUDIO_DB_BATCH_SIZE = max(50, int(os.environ.get('GENERATE_AUDIO_DB_BATCH_SIZE', '250')))

# S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)
sqs_client = boto3.client('sqs', region_name=AWS_REGION) if AUDIO_SQS_QUEUE_URL else None

# Cognito JWKS URL
COGNITO_JWKS_URL = f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{COGNITO_POOL_ID}/.well-known/jwks.json"
jwks_cache = None
jwks_cache_time = 0
active_generation_jobs = set()
active_generation_lock = threading.Lock()
kokoro_lock = threading.Lock()
kokoro_engine = None
kokoro_voice = KOKORO_VOICE


def get_db():
    """Get database connection for current request."""
    if 'db' not in g:
        g.db = open_db_connection()
    return g.db


@app.teardown_appcontext
def close_db(exception):
    """Close database connection at end of request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize database tables."""
    db = open_db_connection()
    db.execute('''
        CREATE TABLE IF NOT EXISTS audio_index (
            hash TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            s3_url TEXT,
            created_at REAL,
            completed_at REAL,
            claimed_by TEXT,
            claimed_at REAL,
            updated_at REAL,
            queue_message_id TEXT,
            queue_receipt_handle TEXT,
            attempt_count INTEGER DEFAULT 0,
            error TEXT
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT,
            tasks_completed INTEGER DEFAULT 0,
            last_active REAL
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS user_books (
            user_id TEXT NOT NULL,
            book_id TEXT NOT NULL,
            title TEXT NOT NULL,
            author TEXT DEFAULT '',
            epub_key TEXT NOT NULL,
            epub_url TEXT NOT NULL,
            cover_key TEXT,
            cover_url TEXT,
            total_chapters INTEGER DEFAULT 0,
            total_sentences INTEGER DEFAULT 0,
            chapter_index INTEGER DEFAULT 0,
            sentence_index INTEGER DEFAULT 0,
            sentence_id TEXT,
            reading_percentage REAL DEFAULT 0,
            ready_audio_count INTEGER DEFAULT 0,
            audio_percentage REAL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_opened REAL NOT NULL,
            PRIMARY KEY(user_id, book_id)
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS book_audio (
            user_id TEXT NOT NULL,
            book_id TEXT NOT NULL,
            sentence_id TEXT NOT NULL,
            sentence_index INTEGER DEFAULT 0,
            hash TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            s3_key TEXT,
            s3_url TEXT,
            audio_format TEXT DEFAULT 'wav',
            error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            completed_at REAL,
            PRIMARY KEY(user_id, book_id, sentence_id)
        )
    ''')
    db.execute('CREATE INDEX IF NOT EXISTS idx_status ON audio_index(status)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_audio_index_claimed_at ON audio_index(claimed_at)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_user_books_user_last_opened ON user_books(user_id, last_opened DESC)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_book_audio_owner_status ON book_audio(user_id, book_id, status)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_book_audio_hash ON book_audio(hash)')

    # Migrate older databases in place.
    existing_audio_cols = {
        row[1] for row in db.execute('PRAGMA table_info(audio_index)').fetchall()
    }
    if 'updated_at' not in existing_audio_cols:
        db.execute("ALTER TABLE audio_index ADD COLUMN updated_at REAL")
    if 'queue_message_id' not in existing_audio_cols:
        db.execute("ALTER TABLE audio_index ADD COLUMN queue_message_id TEXT")
    if 'queue_receipt_handle' not in existing_audio_cols:
        db.execute("ALTER TABLE audio_index ADD COLUMN queue_receipt_handle TEXT")
    if 'attempt_count' not in existing_audio_cols:
        db.execute("ALTER TABLE audio_index ADD COLUMN attempt_count INTEGER DEFAULT 0")
    if 'error' not in existing_audio_cols:
        db.execute("ALTER TABLE audio_index ADD COLUMN error TEXT")

    db.commit()
    db.close()


def open_db_connection():
    db = sqlite3.connect(DATABASE_PATH, timeout=SQLITE_TIMEOUT_SECONDS)
    db.row_factory = sqlite3.Row
    db.execute(f'PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}')
    db.execute('PRAGMA journal_mode = WAL')
    db.execute('PRAGMA synchronous = NORMAL')
    return db


def get_jwks():
    """Fetch and cache Cognito JWKS."""
    global jwks_cache, jwks_cache_time

    # Cache for 1 hour
    if jwks_cache and (time.time() - jwks_cache_time) < 3600:
        return jwks_cache

    try:
        response = requests.get(COGNITO_JWKS_URL, timeout=5)
        jwks_cache = response.json()
        jwks_cache_time = time.time()
        return jwks_cache
    except Exception as e:
        print(f"Error fetching JWKS: {e}")
        return jwks_cache  # Return stale cache if available


def verify_token(token):
    """Verify Cognito JWT token."""
    try:
        jwks = get_jwks()
        if not jwks:
            return None

        # Decode header to get key ID
        headers = jwt.get_unverified_header(token)
        kid = headers['kid']

        # Find matching key
        key = None
        for k in jwks.get('keys', []):
            if k['kid'] == kid:
                key = k
                break

        if not key:
            return None

        # Verify token
        payload = jwt.decode(
            token,
            key,
            algorithms=['RS256'],
            audience=COGNITO_CLIENT_ID,
            issuer=f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{COGNITO_POOL_ID}"
        )

        return payload
    except JWTError as e:
        print(f"JWT verification failed: {e}")
        return None


def require_auth(f):
    """Decorator to require valid Cognito token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')

        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        token = auth_header[7:]  # Remove 'Bearer ' prefix
        payload = verify_token(token)

        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401

        g.user = payload
        return f(*args, **kwargs)

    return decorated


def hash_text(text):
    """Generate hash for sentence text."""
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()[:16]


def normalize_book_id(raw_book_id):
    """Normalize external book IDs to a safe path/database key."""
    if not isinstance(raw_book_id, str):
        return None
    cleaned = ''.join(ch for ch in raw_book_id.strip().lower() if ch.isalnum() or ch in ('-', '_'))
    if not cleaned:
        return None
    return cleaned[:128]


def get_current_user_id():
    return g.user.get('sub', 'anonymous')


def build_s3_public_url(key):
    return f"https://{AUDIO_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def normalize_audio_format(value, default='wav'):
    fmt = (value or default).strip().lower()
    if fmt in ('mp3', 'mpeg'):
        return 'mp3', 'audio/mpeg'
    if fmt in ('m4b', 'm4a', 'mp4'):
        return 'm4b', 'audio/mp4'
    return 'wav', 'audio/wav'


def infer_audio_meta_from_url(s3_url):
    if not s3_url:
        return None, 'wav'
    parsed = urlparse(s3_url)
    key = unquote((parsed.path or '').lstrip('/'))
    if key.lower().endswith('.mp3'):
        return key, 'mp3'
    if key.lower().endswith('.m4b') or key.lower().endswith('.m4a'):
        return key, 'm4b'
    return key, 'wav'


def parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sqs_enabled():
    return bool(AUDIO_SQS_QUEUE_URL and sqs_client)


def sqs_is_fifo_queue():
    return AUDIO_SQS_QUEUE_URL.lower().endswith('.fifo')


def enqueue_audio_task(sentence_hash, text):
    """
    Push one sentence task to SQS. Returns True when enqueue request succeeds.
    """
    if not sqs_enabled():
        return False

    payload = {
        'hash': (sentence_hash or '').strip(),
        'text': (text or '').strip(),
        'enqueued_at': time.time(),
    }
    if not payload['hash'] or not payload['text']:
        return False

    args = {
        'QueueUrl': AUDIO_SQS_QUEUE_URL,
        'MessageBody': json.dumps(payload, separators=(',', ':')),
    }
    if sqs_is_fifo_queue():
        args['MessageGroupId'] = 'audio-generation'
        args['MessageDeduplicationId'] = payload['hash']

    try:
        sqs_client.send_message(**args)
        return True
    except Exception as e:
        print(f"Failed to enqueue SQS task {payload['hash']}: {e}")
        return False


def _extract_task_body(message):
    raw = message.get('Body')
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except Exception:
        return None
    if not isinstance(decoded, dict):
        return None

    # Support SNS-to-SQS fanout envelopes by unwrapping nested JSON "Message".
    nested = decoded.get('Message')
    if isinstance(nested, str):
        try:
            nested_obj = json.loads(nested)
        except Exception:
            nested_obj = None
        if isinstance(nested_obj, dict):
            decoded = nested_obj
    return decoded


def send_to_dlq(task_payload, reason):
    if not AUDIO_SQS_DLQ_URL or not sqs_client:
        return False
    body = {
        'task': task_payload,
        'reason': reason,
        'failed_at': time.time(),
    }
    try:
        sqs_client.send_message(
            QueueUrl=AUDIO_SQS_DLQ_URL,
            MessageBody=json.dumps(body, separators=(',', ':'))
        )
        return True
    except Exception as e:
        print(f"Failed sending task to DLQ: {e}")
        return False


def delete_sqs_message(receipt_handle):
    if not sqs_enabled() or not receipt_handle:
        return False
    try:
        sqs_client.delete_message(
            QueueUrl=AUDIO_SQS_QUEUE_URL,
            ReceiptHandle=receipt_handle
        )
        return True
    except Exception as e:
        print(f"Failed deleting SQS message: {e}")
        return False


def reclaim_stale_generating_rows(db):
    stale_before = time.time() - AUDIO_GENERATING_STALE_SECONDS
    db.execute(
        '''
        UPDATE audio_index
        SET status = 'pending',
            claimed_by = NULL,
            claimed_at = NULL,
            queue_message_id = NULL,
            queue_receipt_handle = NULL,
            updated_at = ?
        WHERE status = 'generating' AND claimed_at IS NOT NULL AND claimed_at < ?
        ''',
        (time.time(), stale_before)
    )


def backfill_pending_rows_to_sqs(db, limit=100):
    """
    Ensure pending rows are represented in SQS.
    Useful after coordinator restarts or transient enqueue failures.
    """
    if not sqs_enabled():
        return 0

    stale_queued_before = time.time() - max(AUDIO_SQS_VISIBILITY_TIMEOUT * 2, 300)
    rows = db.execute(
        '''
        SELECT hash, text
        FROM audio_index
        WHERE status = 'pending'
          AND (
                queue_message_id IS NULL
                OR queue_message_id = ''
                OR (
                    queue_message_id = 'queued'
                    AND (updated_at IS NULL OR updated_at < ?)
                )
          )
        ORDER BY created_at ASC
        LIMIT ?
        ''',
        (stale_queued_before, max(1, min(limit, 500)))
    ).fetchall()

    pushed = 0
    queued_hashes = []
    for row in rows:
        sentence_hash = (row['hash'] or '').strip()
        text = (row['text'] or '').strip()
        if not sentence_hash or not text:
            continue
        if not enqueue_audio_task(sentence_hash, text):
            continue
        queued_hashes.append(sentence_hash)
        pushed += 1
    if queued_hashes:
        now = time.time()
        db.executemany(
            '''
            UPDATE audio_index
            SET queue_message_id = COALESCE(queue_message_id, 'queued'),
                updated_at = ?
            WHERE hash = ? AND status = 'pending'
            ''',
            [(now, sentence_hash) for sentence_hash in queued_hashes]
        )
    return pushed


def count_book_audio(db, user_id, book_id):
    row = db.execute(
        '''
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END) AS ready,
            SUM(CASE WHEN status = 'generating' THEN 1 ELSE 0 END) AS generating,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
        FROM book_audio
        WHERE user_id = ? AND book_id = ?
        ''',
        (user_id, book_id)
    ).fetchone()

    if row is None:
        total = ready = generating = failed = 0
    elif isinstance(row, sqlite3.Row):
        total = int(row['total'] or 0)
        ready = int(row['ready'] or 0)
        generating = int(row['generating'] or 0)
        failed = int(row['failed'] or 0)
    else:
        # Defensive fallback for connections without sqlite Row factory.
        total = int((row[0] if len(row) > 0 else 0) or 0)
        ready = int((row[1] if len(row) > 1 else 0) or 0)
        generating = int((row[2] if len(row) > 2 else 0) or 0)
        failed = int((row[3] if len(row) > 3 else 0) or 0)
    pending = max(0, total - ready - generating - failed)
    percentage = (ready / total * 100.0) if total else 0.0
    return {
        'total': total,
        'ready': ready,
        'generating': generating,
        'pending': pending,
        'failed': failed,
        'percentage': max(0.0, min(100.0, percentage))
    }


def refresh_user_book_audio_progress(db, user_id, book_id):
    summary = count_book_audio(db, user_id, book_id)
    now = time.time()
    db.execute(
        '''
        UPDATE user_books
        SET ready_audio_count = ?,
            audio_percentage = ?,
            total_sentences = CASE
                WHEN total_sentences > 0 THEN total_sentences
                ELSE ?
            END,
            updated_at = ?
        WHERE user_id = ? AND book_id = ?
        ''',
        (
            summary['ready'],
            summary['percentage'],
            summary['total'],
            now,
            user_id,
            book_id
        )
    )
    return summary


def claim_next_book_audio_job(user_id, book_id):
    db = open_db_connection()
    now = time.time()
    try:
        db.execute('BEGIN IMMEDIATE')
        # Reclaim stale in-progress rows if a previous worker crashed.
        db.execute(
            '''
            UPDATE book_audio
            SET status = 'pending', updated_at = ?
            WHERE user_id = ? AND book_id = ? AND status = 'generating' AND updated_at < ?
            ''',
            (now, user_id, book_id, now - 300)
        )
        row = db.execute(
            '''
            SELECT user_id, book_id, sentence_id, sentence_index, hash, text
            FROM book_audio
            WHERE user_id = ? AND book_id = ? AND status = 'pending'
            ORDER BY sentence_index ASC, created_at ASC
            LIMIT 1
            ''',
            (user_id, book_id)
        ).fetchone()
        if not row:
            db.commit()
            return None

        updated = db.execute(
            '''
            UPDATE book_audio
            SET status = 'generating', updated_at = ?, error = NULL
            WHERE user_id = ? AND book_id = ? AND sentence_id = ? AND status = 'pending'
            ''',
            (now, row['user_id'], row['book_id'], row['sentence_id'])
        ).rowcount

        if updated != 1:
            db.rollback()
            return None

        db.commit()
        return dict(row)
    finally:
        db.close()


def mark_book_audio_failed(user_id, book_id, sentence_id, error_message):
    db = open_db_connection()
    now = time.time()
    try:
        db.execute(
            '''
            UPDATE book_audio
            SET status = 'failed',
                error = ?,
                updated_at = ?
            WHERE user_id = ? AND book_id = ? AND sentence_id = ?
            ''',
            (str(error_message)[:500], now, user_id, book_id, sentence_id)
        )
        refresh_user_book_audio_progress(db, user_id, book_id)
        db.commit()
    finally:
        db.close()


def mark_book_audio_ready(user_id, book_id, sentence_id, sentence_hash, s3_key, s3_url):
    db = open_db_connection()
    now = time.time()
    _, inferred_format = infer_audio_meta_from_url(s3_url)
    try:
        db.execute(
            '''
            UPDATE book_audio
            SET status = 'ready',
                s3_key = ?,
                s3_url = ?,
                audio_format = ?,
                completed_at = ?,
                updated_at = ?,
                error = NULL
            WHERE user_id = ? AND book_id = ? AND sentence_id = ?
            ''',
            (s3_key, s3_url, inferred_format, now, now, user_id, book_id, sentence_id)
        )

        db.execute(
            '''
            INSERT INTO audio_index (hash, text, status, s3_url, created_at, completed_at)
            SELECT hash, text, 'ready', ?, ?, ?
            FROM book_audio
            WHERE user_id = ? AND book_id = ? AND sentence_id = ?
            ON CONFLICT(hash) DO UPDATE SET
                status = 'ready',
                s3_url = excluded.s3_url,
                completed_at = excluded.completed_at
            ''',
            (s3_url, now, now, user_id, book_id, sentence_id)
        )

        refresh_user_book_audio_progress(db, user_id, book_id)
        db.commit()
    finally:
        db.close()


def download_file(url, output_path):
    tmp_path = f'{output_path}.part'
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp_path, 'wb') as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    os.replace(tmp_path, output_path)


def ensure_kokoro_assets():
    os.makedirs(KOKORO_MODEL_DIR, exist_ok=True)
    if not os.path.exists(KOKORO_MODEL_PATH):
        print(f'Downloading Kokoro model to {KOKORO_MODEL_PATH}')
        download_file(KOKORO_MODEL_URL, KOKORO_MODEL_PATH)
    if not os.path.exists(KOKORO_VOICES_PATH):
        print(f'Downloading Kokoro voices to {KOKORO_VOICES_PATH}')
        download_file(KOKORO_VOICES_URL, KOKORO_VOICES_PATH)


def get_kokoro_engine():
    global kokoro_engine, kokoro_voice

    with kokoro_lock:
        if kokoro_engine is None:
            ensure_kokoro_assets()
            from kokoro_onnx import Kokoro  # type: ignore

            kokoro_engine = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
            available_voices = set(kokoro_engine.get_voices())
            if kokoro_voice not in available_voices and available_voices:
                if 'af_sarah' in available_voices:
                    kokoro_voice = 'af_sarah'
                else:
                    kokoro_voice = sorted(available_voices)[0]
                print(f"Kokoro voice '{KOKORO_VOICE}' unavailable, using '{kokoro_voice}'")

    return kokoro_engine, kokoro_voice


def float_audio_to_wav_bytes(samples, sample_rate):
    pcm = array('h')
    for sample in samples:
        value = float(sample)
        if value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        pcm.append(int(value * 32767))

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def synthesize_audio_wav(text):
    payload = text.strip()
    if not payload:
        raise ValueError('Empty sentence text')

    # Keep sentence payload bounded for stable runtime.
    payload = payload[:2000]

    engine, voice = get_kokoro_engine()
    audio, sample_rate = engine.create(
        payload,
        voice=voice,
        speed=1.0,
        lang=KOKORO_LANG
    )

    if audio is None:
        raise RuntimeError('Kokoro did not return audio')

    return float_audio_to_wav_bytes(audio, sample_rate)


def generate_one_book_audio_job(job):
    user_id = job['user_id']
    book_id = job['book_id']
    sentence_id = job['sentence_id']
    sentence_hash = job['hash']
    text = (job['text'] or '').strip()

    if not text:
        mark_book_audio_failed(user_id, book_id, sentence_id, 'Empty sentence text')
        return

    db = open_db_connection()
    try:
        existing = db.execute(
            'SELECT status, s3_url FROM audio_index WHERE hash = ?',
            (sentence_hash,)
        ).fetchone()
        if existing and existing['status'] == 'ready' and existing['s3_url']:
            mark_book_audio_ready(
                user_id=user_id,
                book_id=book_id,
                sentence_id=sentence_id,
                sentence_hash=sentence_hash,
                s3_key='',
                s3_url=existing['s3_url']
            )
            return
    finally:
        db.close()

    try:
        audio_bytes = synthesize_audio_wav(text)
        key = f'audio/{sentence_hash}.wav'
        s3_client.put_object(
            Bucket=AUDIO_BUCKET,
            Key=key,
            Body=audio_bytes,
            ContentType='audio/wav',
            CacheControl='public, max-age=31536000'
        )
        final_url = build_s3_public_url(key)
        mark_book_audio_ready(
            user_id=user_id,
            book_id=book_id,
            sentence_id=sentence_id,
            sentence_hash=sentence_hash,
            s3_key=key,
            s3_url=final_url
        )
    except Exception as e:
        mark_book_audio_failed(user_id, book_id, sentence_id, str(e))


def process_book_audio_jobs(user_id, book_id):
    job_key = f'{user_id}:{book_id}'
    with active_generation_lock:
        if job_key in active_generation_jobs:
            return
        active_generation_jobs.add(job_key)

    try:
        while True:
            job = claim_next_book_audio_job(user_id, book_id)
            if not job:
                break
            try:
                generate_one_book_audio_job(job)
            except Exception as e:
                print(f'book audio worker job failed unexpectedly: {e}')
    finally:
        with active_generation_lock:
            active_generation_jobs.discard(job_key)


def start_book_audio_generation(user_id, book_id):
    thread = threading.Thread(
        target=process_book_audio_jobs,
        args=(user_id, book_id),
        daemon=True
    )
    thread.start()


def serialize_book_row(row):
    if not row:
        return None

    total_sentences = max(0, row['total_sentences'] or 0)
    ready_audio_count = max(0, row['ready_audio_count'] or 0)
    if total_sentences:
        ready_audio_count = min(ready_audio_count, total_sentences)

    audio_left_count = max(0, total_sentences - ready_audio_count)
    reading_percentage = max(0.0, min(100.0, row['reading_percentage'] or 0.0))
    audio_percentage = max(0.0, min(100.0, row['audio_percentage'] or 0.0))

    return {
        'book_id': row['book_id'],
        'title': row['title'],
        'author': row['author'] or '',
        'cover_url': row['cover_url'],
        'epub_url': row['epub_url'],
        'total_chapters': row['total_chapters'] or 0,
        'total_sentences': total_sentences,
        'chapter_index': row['chapter_index'] or 0,
        'sentence_index': row['sentence_index'] or 0,
        'sentence_id': row['sentence_id'],
        'reading_progress': {
            'percentage': reading_percentage,
            'left_percentage': max(0.0, 100.0 - reading_percentage)
        },
        'audio_progress': {
            'ready_count': ready_audio_count,
            'total_sentences': total_sentences,
            'percentage': audio_percentage,
            'left_count': audio_left_count,
            'left_percentage': max(0.0, 100.0 - audio_percentage)
        },
        'created_at': row['created_at'],
        'updated_at': row['updated_at'],
        'last_opened': row['last_opened']
    }


# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'time': time.time()})


@app.route('/check', methods=['POST'])
@require_auth
def check_audio():
    """
    Check if audio exists for a sentence.

    Request: { "hash": "abc123", "text": "Hello world." }
    Response: { "status": "ready|pending|scheduled", "url": "s3://..." }
    """
    data = request.json or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Missing text'}), 400

    sentence_hash = (data.get('hash') or hash_text(text)).strip()

    db = get_db()
    row = db.execute(
        'SELECT status, s3_url, queue_message_id FROM audio_index WHERE hash = ?',
        (sentence_hash,)
    ).fetchone()

    if row:
        # If SQS mode is on and this row is pending without queue marker, enqueue now.
        if row['status'] == 'pending' and sqs_enabled() and not (row['queue_message_id'] or '').strip():
            if enqueue_audio_task(sentence_hash, text):
                db.execute(
                    '''
                    UPDATE audio_index
                    SET queue_message_id = 'queued',
                        updated_at = ?
                    WHERE hash = ? AND status = 'pending'
                    ''',
                    (time.time(), sentence_hash)
                )
                db.commit()
        return jsonify({
            'hash': sentence_hash,
            'status': row['status'],
            'url': row['s3_url']
        })

    # Not found - add to queue as pending
    now = time.time()
    db.execute(
        '''
        INSERT INTO audio_index (hash, text, status, created_at, updated_at)
        VALUES (?, ?, 'pending', ?, ?)
        ''',
        (sentence_hash, text, now, now)
    )
    if sqs_enabled() and enqueue_audio_task(sentence_hash, text):
        db.execute(
            '''
            UPDATE audio_index
            SET queue_message_id = 'queued',
                updated_at = ?
            WHERE hash = ?
            ''',
            (time.time(), sentence_hash)
        )
    db.commit()

    return jsonify({
        'hash': sentence_hash,
        'status': 'scheduled',
        'url': None
    })


@app.route('/check_batch', methods=['POST'])
@require_auth
def check_batch():
    """
    Check multiple sentences at once.

    Request: { "sentences": [{"hash": "...", "text": "..."}, ...] }
    Response: { "results": [...] }
    """
    data = request.json or {}
    sentences = data.get('sentences', [])

    if not sentences:
        return jsonify({'error': 'Missing sentences'}), 400

    db = get_db()
    results = []

    now = time.time()
    for item in sentences[:100]:  # Limit to 100 per request
        text = item.get('text', '').strip()
        if not text:
            continue

        sentence_hash = (item.get('hash') or hash_text(text)).strip()

        row = db.execute(
            'SELECT status, s3_url, queue_message_id FROM audio_index WHERE hash = ?',
            (sentence_hash,)
        ).fetchone()

        if row:
            if row['status'] == 'pending' and sqs_enabled() and not (row['queue_message_id'] or '').strip():
                if enqueue_audio_task(sentence_hash, text):
                    db.execute(
                        '''
                        UPDATE audio_index
                        SET queue_message_id = 'queued',
                            updated_at = ?
                        WHERE hash = ? AND status = 'pending'
                        ''',
                        (time.time(), sentence_hash)
                    )
            results.append({
                'hash': sentence_hash,
                'status': row['status'],
                'url': row['s3_url']
            })
        else:
            # Add to queue
            db.execute(
                '''
                INSERT OR IGNORE INTO audio_index (hash, text, status, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?)
                ''',
                (sentence_hash, text, now, now)
            )
            if sqs_enabled() and enqueue_audio_task(sentence_hash, text):
                db.execute(
                    '''
                    UPDATE audio_index
                    SET queue_message_id = 'queued',
                        updated_at = ?
                    WHERE hash = ? AND status = 'pending'
                    ''',
                    (time.time(), sentence_hash)
                )
            results.append({
                'hash': sentence_hash,
                'status': 'scheduled',
                'url': None
            })

    db.commit()
    return jsonify({'results': results})


@app.route('/tasks', methods=['GET'])
@require_auth
def get_tasks():
    """
    Get pending tasks for volunteer nodes.

    Response: { "tasks": [{"hash": "...", "text": "..."}, ...] }
    """
    limit = min(int(request.args.get('limit', 10)), 50)

    db = get_db()
    user_id = g.user.get('sub', 'anonymous')

    # Single-queue SQS mode (recommended).
    if sqs_enabled():
        reclaim_stale_generating_rows(db)
        backfill_pending_rows_to_sqs(db, limit=max(100, limit * 20))
        db.commit()

        tasks = []
        receive_limit = min(limit, AUDIO_SQS_MAX_MESSAGES)

        try:
            response = sqs_client.receive_message(
                QueueUrl=AUDIO_SQS_QUEUE_URL,
                MaxNumberOfMessages=receive_limit,
                WaitTimeSeconds=AUDIO_SQS_WAIT_SECONDS,
                VisibilityTimeout=AUDIO_SQS_VISIBILITY_TIMEOUT,
                MessageAttributeNames=['All'],
                AttributeNames=['All']
            )
        except Exception as e:
            print(f"SQS receive_message failed: {e}")
            return jsonify({'tasks': []})

        now = time.time()
        messages = response.get('Messages') or []
        for message in messages:
            payload = _extract_task_body(message) or {}
            sentence_hash = (payload.get('hash') or '').strip()
            text = (payload.get('text') or '').strip()
            message_id = message.get('MessageId')
            receipt_handle = message.get('ReceiptHandle')
            receive_count = parse_int((message.get('Attributes') or {}).get('ApproximateReceiveCount'), 1)

            if not sentence_hash or not text or not receipt_handle:
                delete_sqs_message(receipt_handle)
                continue

            row = db.execute(
                'SELECT status, s3_url FROM audio_index WHERE hash = ?',
                (sentence_hash,)
            ).fetchone()

            if row and row['status'] == 'ready' and row['s3_url']:
                # Duplicate/stale queue message; ready already exists.
                delete_sqs_message(receipt_handle)
                continue

            if receive_count > AUDIO_SQS_MAX_RECEIVE_COUNT:
                # Poison message guard: cap retries and mark as failed.
                send_to_dlq(payload, f'max_receive_exceeded:{receive_count}')
                delete_sqs_message(receipt_handle)
                db.execute(
                    '''
                    INSERT INTO audio_index (
                        hash, text, status, created_at, updated_at, completed_at,
                        claimed_by, claimed_at, queue_message_id, queue_receipt_handle,
                        attempt_count, error
                    )
                    VALUES (?, ?, 'failed', ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                    ON CONFLICT(hash) DO UPDATE SET
                        status = 'failed',
                        claimed_by = excluded.claimed_by,
                        claimed_at = excluded.claimed_at,
                        queue_message_id = NULL,
                        queue_receipt_handle = NULL,
                        attempt_count = excluded.attempt_count,
                        error = excluded.error,
                        updated_at = excluded.updated_at
                    ''',
                    (
                        sentence_hash,
                        text,
                        now,
                        now,
                        now,
                        user_id,
                        now,
                        receive_count,
                        f'max_receive_exceeded:{receive_count}',
                    )
                )
                continue

            db.execute(
                '''
                INSERT INTO audio_index (
                    hash, text, status, created_at, updated_at,
                    claimed_by, claimed_at, queue_message_id, queue_receipt_handle,
                    attempt_count, error
                )
                VALUES (?, ?, 'generating', ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(hash) DO UPDATE SET
                    text = COALESCE(NULLIF(excluded.text, ''), audio_index.text),
                    status = CASE
                        WHEN audio_index.status = 'ready' THEN 'ready'
                        ELSE 'generating'
                    END,
                    claimed_by = excluded.claimed_by,
                    claimed_at = excluded.claimed_at,
                    queue_message_id = excluded.queue_message_id,
                    queue_receipt_handle = excluded.queue_receipt_handle,
                    attempt_count = MAX(COALESCE(audio_index.attempt_count, 0), excluded.attempt_count),
                    error = NULL,
                    updated_at = excluded.updated_at
                ''',
                (
                    sentence_hash,
                    text,
                    now,
                    now,
                    user_id,
                    now,
                    message_id,
                    receipt_handle,
                    receive_count
                )
            )

            fresh = db.execute(
                'SELECT status FROM audio_index WHERE hash = ?',
                (sentence_hash,)
            ).fetchone()
            if fresh and fresh['status'] == 'ready':
                # Another worker completed between receive + upsert.
                delete_sqs_message(receipt_handle)
                db.execute(
                    '''
                    UPDATE audio_index
                    SET queue_message_id = NULL,
                        queue_receipt_handle = NULL,
                        updated_at = ?
                    WHERE hash = ?
                    ''',
                    (time.time(), sentence_hash)
                )
                continue

            tasks.append({
                'hash': sentence_hash,
                'text': text
            })

        db.commit()
        return jsonify({'tasks': tasks})

    # Legacy DB polling mode (no SQS configured).
    rows = db.execute(
        '''SELECT hash, text FROM audio_index
           WHERE status = 'pending' AND (claimed_by IS NULL OR claimed_at < ?)
           ORDER BY created_at ASC
           LIMIT ?''',
        (time.time() - 300, limit)  # Reclaim tasks older than 5 minutes
    ).fetchall()

    tasks = []
    now = time.time()
    for row in rows:
        db.execute(
            '''
            UPDATE audio_index
            SET claimed_by = ?, claimed_at = ?, status = 'generating', updated_at = ?
            WHERE hash = ?
            ''',
            (user_id, now, now, row['hash'])
        )
        tasks.append({
            'hash': row['hash'],
            'text': row['text']
        })

    db.commit()
    return jsonify({'tasks': tasks})


@app.route('/complete', methods=['POST'])
@require_auth
def complete_task():
    """
    Mark task as complete and store S3 URL.

    Request: { "hash": "abc123", "s3_url": "https://..." }
    """
    data = request.json or {}
    sentence_hash = data.get('hash')
    s3_url = data.get('s3_url')

    if not sentence_hash or not s3_url:
        return jsonify({'error': 'Missing hash or s3_url'}), 400

    db = get_db()
    user_id = g.user.get('sub', 'anonymous')
    now = time.time()
    existing = db.execute(
        '''
        SELECT status, queue_receipt_handle
        FROM audio_index
        WHERE hash = ?
        ''',
        (sentence_hash,)
    ).fetchone()
    receipt_handle = (existing['queue_receipt_handle'] if existing else None)

    # Idempotent upsert to ready.
    db.execute(
        '''
        INSERT INTO audio_index (
            hash, text, status, s3_url, created_at, completed_at, updated_at,
            claimed_by, claimed_at, queue_message_id, queue_receipt_handle,
            attempt_count, error
        )
        VALUES (?, '', 'ready', ?, ?, ?, ?, ?, ?, NULL, NULL, 0, NULL)
        ON CONFLICT(hash) DO UPDATE SET
            status = 'ready',
            s3_url = excluded.s3_url,
            completed_at = excluded.completed_at,
            updated_at = excluded.updated_at,
            claimed_by = NULL,
            claimed_at = NULL,
            queue_message_id = NULL,
            queue_receipt_handle = NULL,
            error = NULL
        ''',
        (sentence_hash, s3_url, now, now, now, user_id, now)
    )

    # Count completion only when this wasn't already ready.
    if not existing or existing['status'] != 'ready':
        db.execute(
            '''
            INSERT INTO users (user_id, tasks_completed, last_active)
            VALUES (?, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                tasks_completed = tasks_completed + 1,
                last_active = ?
            ''',
            (user_id, now, now)
        )

    db.commit()

    sqs_deleted = delete_sqs_message(receipt_handle)
    return jsonify({'status': 'ok', 'idempotent': True, 'queue_deleted': bool(sqs_deleted)})


@app.route('/complete_batch', methods=['POST'])
@require_auth
def complete_batch():
    """
    Mark multiple uploads complete.

    Request:
    {
      "book_id": "optional",
      "items": [
        {
          "hash": "...",
          "s3_url": "https://...",
          "sentence_id": "optional",
          "sentence_index": 0,
          "text": "optional",
          "audio_format": "mp3|wav|m4b"
        }
      ]
    }
    """
    data = request.json or {}
    items = data.get('items') or []
    if not isinstance(items, list) or not items:
        return jsonify({'error': 'Missing items'}), 400

    items = items[:500]
    default_book_id = normalize_book_id(data.get('book_id'))
    user_id = g.user.get('sub', 'anonymous')
    db = get_db()
    now = time.time()
    processed = 0
    updated_books = set()
    queue_receipts = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        sentence_hash = (item.get('hash') or '').strip()
        s3_url = (item.get('s3_url') or '').strip()
        if not sentence_hash or not s3_url:
            continue

        prior = db.execute(
            'SELECT queue_receipt_handle FROM audio_index WHERE hash = ?',
            (sentence_hash,)
        ).fetchone()
        if prior and prior['queue_receipt_handle']:
            queue_receipts.add(prior['queue_receipt_handle'])

        item_text = (item.get('text') or '').strip()
        _, inferred_format = infer_audio_meta_from_url(s3_url)
        audio_format, _ = normalize_audio_format(item.get('audio_format') or item.get('format') or inferred_format)

        db.execute(
            '''
            INSERT INTO audio_index (hash, text, status, s3_url, created_at, completed_at)
            VALUES (
                ?,
                COALESCE(NULLIF(?, ''), COALESCE((SELECT text FROM audio_index WHERE hash = ?), '')),
                'ready',
                ?,
                ?,
                ?
            )
            ON CONFLICT(hash) DO UPDATE SET
                status = 'ready',
                s3_url = excluded.s3_url,
                completed_at = excluded.completed_at,
                updated_at = excluded.completed_at,
                claimed_by = NULL,
                claimed_at = NULL,
                queue_message_id = NULL,
                queue_receipt_handle = NULL,
                error = NULL
            ''',
            (sentence_hash, item_text, sentence_hash, s3_url, now, now)
        )

        book_id = normalize_book_id(item.get('book_id')) or default_book_id
        sentence_id = (item.get('sentence_id') or '').strip()
        if book_id and sentence_id:
            sentence_index = max(0, parse_int(item.get('sentence_index'), 0))
            s3_key, _ = infer_audio_meta_from_url(s3_url)
            db.execute(
                '''
                INSERT INTO book_audio (
                    user_id, book_id, sentence_id, sentence_index, hash, text,
                    status, s3_key, s3_url, audio_format, error,
                    created_at, updated_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(user_id, book_id, sentence_id) DO UPDATE SET
                    sentence_index = excluded.sentence_index,
                    hash = excluded.hash,
                    text = CASE
                        WHEN excluded.text IS NOT NULL AND excluded.text != '' THEN excluded.text
                        ELSE book_audio.text
                    END,
                    status = 'ready',
                    s3_key = excluded.s3_key,
                    s3_url = excluded.s3_url,
                    audio_format = excluded.audio_format,
                    error = NULL,
                    updated_at = excluded.updated_at,
                    completed_at = excluded.completed_at
                ''',
                (
                    user_id,
                    book_id,
                    sentence_id,
                    sentence_index,
                    sentence_hash,
                    item_text,
                    s3_key,
                    s3_url,
                    audio_format,
                    now,
                    now,
                    now
                )
            )
            updated_books.add(book_id)

        processed += 1

    if processed > 0:
        db.execute(
            '''
            INSERT INTO users (user_id, tasks_completed, last_active)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                tasks_completed = tasks_completed + excluded.tasks_completed,
                last_active = excluded.last_active
            ''',
            (user_id, processed, now)
        )

    summaries = {}
    for book_id in updated_books:
        summaries[book_id] = refresh_user_book_audio_progress(db, user_id, book_id)

    db.commit()
    queue_deleted = 0
    for handle in queue_receipts:
        if delete_sqs_message(handle):
            queue_deleted += 1
    return jsonify({
        'status': 'ok',
        'processed': processed,
        'updated_books': len(updated_books),
        'books': summaries,
        'queue_deleted': queue_deleted
    })


@app.route('/upload_url', methods=['POST'])
@require_auth
def get_upload_url():
    """
    Get presigned URL for uploading audio to S3.

    Request: { "hash": "abc123" }
    Response: { "upload_url": "...", "final_url": "..." }
    """
    data = request.json or {}
    sentence_hash = data.get('hash')

    if not sentence_hash:
        return jsonify({'error': 'Missing hash'}), 400

    audio_ext, content_type = normalize_audio_format(data.get('format') or data.get('audio_format') or 'wav')

    key = f"audio/{sentence_hash}.{audio_ext}"

    # Generate presigned upload URL
    upload_url = s3_client.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': AUDIO_BUCKET,
            'Key': key,
            'ContentType': content_type
        },
        ExpiresIn=300  # 5 minutes
    )

    final_url = f"https://{AUDIO_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

    return jsonify({
        'upload_url': upload_url,
        'final_url': final_url,
        'format': audio_ext,
        'content_type': content_type
    })


@app.route('/upload_urls_batch', methods=['POST'])
@require_auth
def get_upload_urls_batch():
    """
    Get presigned URLs for multiple sentence hashes.

    Request:
    {
      "items": [
        {"hash": "...", "format": "mp3|wav|m4b"},
        ...
      ]
    }
    """
    data = request.json or {}
    items = data.get('items') or []
    if not isinstance(items, list) or not items:
        return jsonify({'error': 'Missing items'}), 400

    results = []
    seen = set()
    for item in items[:500]:
        if not isinstance(item, dict):
            continue
        sentence_hash = (item.get('hash') or '').strip()
        if not sentence_hash or sentence_hash in seen:
            continue
        seen.add(sentence_hash)

        audio_ext, content_type = normalize_audio_format(item.get('format') or item.get('audio_format') or 'wav')
        key = f"audio/{sentence_hash}.{audio_ext}"

        upload_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': AUDIO_BUCKET,
                'Key': key,
                'ContentType': content_type
            },
            ExpiresIn=300
        )

        results.append({
            'hash': sentence_hash,
            'upload_url': upload_url,
            'final_url': build_s3_public_url(key),
            'format': audio_ext,
            'content_type': content_type
        })

    return jsonify({'items': results})


# =============================================================================
# Per-User Cloud Library
# =============================================================================

@app.route('/books/upload_urls', methods=['POST'])
@require_auth
def get_book_upload_urls():
    """
    Get presigned URLs for book and cover upload.

    Request: { "book_id": "...", "cover_content_type": "image/jpeg" }
    """
    data = request.json or {}
    book_id = normalize_book_id(data.get('book_id'))
    if not book_id:
        return jsonify({'error': 'Missing or invalid book_id'}), 400

    user_id = get_current_user_id()
    cover_content_type = (data.get('cover_content_type') or 'image/jpeg').strip().lower()

    if cover_content_type == 'image/png':
        cover_ext = 'png'
    elif cover_content_type == 'image/webp':
        cover_ext = 'webp'
    else:
        cover_content_type = 'image/jpeg'
        cover_ext = 'jpg'

    epub_key = f"users/{user_id}/books/{book_id}.epub"
    cover_key = f"users/{user_id}/covers/{book_id}.{cover_ext}"

    epub_upload_url = s3_client.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': AUDIO_BUCKET,
            'Key': epub_key,
            'ContentType': 'application/epub+zip'
        },
        ExpiresIn=900
    )
    cover_upload_url = s3_client.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': AUDIO_BUCKET,
            'Key': cover_key,
            'ContentType': cover_content_type
        },
        ExpiresIn=900
    )

    return jsonify({
        'book_id': book_id,
        'epub': {
            'key': epub_key,
            'upload_url': epub_upload_url,
            'final_url': build_s3_public_url(epub_key)
        },
        'cover': {
            'key': cover_key,
            'upload_url': cover_upload_url,
            'final_url': build_s3_public_url(cover_key),
            'content_type': cover_content_type
        }
    })


@app.route('/books', methods=['POST'])
@require_auth
def upsert_user_book():
    """
    Create or update a user book record after upload.
    """
    data = request.json or {}
    user_id = get_current_user_id()
    book_id = normalize_book_id(data.get('book_id'))
    title = (data.get('title') or '').strip()

    if not book_id or not title:
        return jsonify({'error': 'Missing book_id or title'}), 400

    epub_key = (data.get('epub_key') or f"users/{user_id}/books/{book_id}.epub").strip()
    epub_url = (data.get('epub_url') or build_s3_public_url(epub_key)).strip()
    cover_key = (data.get('cover_key') or '').strip() or None
    cover_url = (data.get('cover_url') or '').strip() or None

    now = time.time()
    db = get_db()
    db.execute(
        '''
        INSERT INTO user_books (
            user_id, book_id, title, author, epub_key, epub_url, cover_key, cover_url,
            total_chapters, total_sentences, chapter_index, sentence_index, sentence_id,
            reading_percentage, ready_audio_count, audio_percentage, created_at, updated_at, last_opened
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, NULL, 0, 0, 0, ?, ?, ?)
        ON CONFLICT(user_id, book_id) DO UPDATE SET
            title = excluded.title,
            author = excluded.author,
            epub_key = excluded.epub_key,
            epub_url = excluded.epub_url,
            cover_key = COALESCE(excluded.cover_key, user_books.cover_key),
            cover_url = COALESCE(excluded.cover_url, user_books.cover_url),
            total_chapters = CASE
                WHEN excluded.total_chapters > 0 THEN excluded.total_chapters
                ELSE user_books.total_chapters
            END,
            total_sentences = CASE
                WHEN excluded.total_sentences > 0 THEN excluded.total_sentences
                ELSE user_books.total_sentences
            END,
            updated_at = excluded.updated_at,
            last_opened = excluded.last_opened
        ''',
        (
            user_id,
            book_id,
            title,
            (data.get('author') or '').strip(),
            epub_key,
            epub_url,
            cover_key,
            cover_url,
            max(0, parse_int(data.get('total_chapters'), 0)),
            max(0, parse_int(data.get('total_sentences'), 0)),
            now,
            now,
            now
        )
    )
    db.commit()

    row = db.execute(
        'SELECT * FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    ).fetchone()
    return jsonify({'status': 'ok', 'book': serialize_book_row(row)})


@app.route('/books', methods=['GET'])
@require_auth
def list_user_books():
    """List current user's books for homepage library."""
    db = get_db()
    user_id = get_current_user_id()
    rows = db.execute(
        'SELECT * FROM user_books WHERE user_id = ? ORDER BY last_opened DESC',
        (user_id,)
    ).fetchall()
    return jsonify({'books': [serialize_book_row(r) for r in rows]})


@app.route('/books/<book_id>', methods=['GET'])
@require_auth
def get_user_book(book_id):
    """Get one cloud-stored user book."""
    user_id = get_current_user_id()
    book_id = normalize_book_id(book_id)
    if not book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    db = get_db()
    row = db.execute(
        'SELECT * FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Book not found'}), 404
    return jsonify({'book': serialize_book_row(row)})


@app.route('/books/<book_id>/download_url', methods=['GET'])
@require_auth
def get_user_book_download_url(book_id):
    """Get a short-lived download URL for user's EPUB file."""
    user_id = get_current_user_id()
    book_id = normalize_book_id(book_id)
    if not book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    db = get_db()
    row = db.execute(
        'SELECT epub_key, epub_url FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Book not found'}), 404

    public_url = (row['epub_url'] or build_s3_public_url(row['epub_key'])).strip()
    signed_url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': AUDIO_BUCKET,
            'Key': row['epub_key']
        },
        ExpiresIn=900
    )
    return jsonify({
        'book_id': book_id,
        'url': signed_url,
        'download_url': public_url,
        'public_url': public_url,
        'signed_url': signed_url
    })


@app.route('/books/<book_id>/file', methods=['GET'])
@require_auth
def download_user_book_file(book_id):
    """
    Stream user's EPUB file via coordinator domain to avoid browser/S3 fetch issues.
    """
    user_id = get_current_user_id()
    book_id = normalize_book_id(book_id)
    if not book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    db = get_db()
    row = db.execute(
        'SELECT title, epub_key FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Book not found'}), 404

    try:
        obj = s3_client.get_object(Bucket=AUDIO_BUCKET, Key=row['epub_key'])
        body = obj['Body']
        content_length = int(obj.get('ContentLength') or 0)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch EPUB from storage: {str(e)}'}), 502

    def stream_epub():
        try:
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                body.close()
            except Exception:
                pass

    safe_name = (row['title'] or book_id).strip().replace('"', '')
    headers = {
        'Content-Disposition': f'inline; filename="{safe_name}.epub"',
        'Cache-Control': 'no-store',
        'Accept-Ranges': 'bytes'
    }
    if content_length > 0:
        headers['Content-Length'] = str(content_length)

    return Response(
        stream_with_context(stream_epub()),
        mimetype='application/epub+zip',
        headers=headers,
        direct_passthrough=True
    )


@app.route('/books/<book_id>/progress', methods=['POST'])
@require_auth
def update_user_book_progress(book_id):
    """
    Update reading/audio progress and resume position for a user book.
    """
    user_id = get_current_user_id()
    book_id = normalize_book_id(book_id)
    if not book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    db = get_db()
    current = db.execute(
        'SELECT * FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    ).fetchone()
    if not current:
        return jsonify({'error': 'Book not found'}), 404

    data = request.json or {}
    total_chapters = max(0, parse_int(data.get('total_chapters'), current['total_chapters'] or 0))
    total_sentences = max(0, parse_int(data.get('total_sentences'), current['total_sentences'] or 0))
    chapter_index = max(0, parse_int(data.get('chapter_index'), current['chapter_index'] or 0))
    sentence_index = max(0, parse_int(data.get('sentence_index'), current['sentence_index'] or 0))
    sentence_id = (data.get('sentence_id') or current['sentence_id'] or '').strip() or None

    reading_percentage = parse_float(data.get('reading_percentage'), -1.0)
    if reading_percentage < 0:
        denom = max(1, total_chapters)
        reading_percentage = min(100.0, ((chapter_index + 1) / denom) * 100.0)
    reading_percentage = max(0.0, min(100.0, reading_percentage))

    ready_audio_count = max(0, parse_int(data.get('ready_audio_count'), current['ready_audio_count'] or 0))
    if total_sentences:
        ready_audio_count = min(ready_audio_count, total_sentences)

    audio_percentage = parse_float(data.get('audio_percentage'), -1.0)
    if audio_percentage < 0:
        audio_percentage = (ready_audio_count / total_sentences * 100.0) if total_sentences else 0.0
    audio_percentage = max(0.0, min(100.0, audio_percentage))

    now = time.time()
    db.execute(
        '''
        UPDATE user_books
        SET total_chapters = ?,
            total_sentences = ?,
            chapter_index = ?,
            sentence_index = ?,
            sentence_id = ?,
            reading_percentage = ?,
            ready_audio_count = ?,
            audio_percentage = ?,
            updated_at = ?,
            last_opened = ?
        WHERE user_id = ? AND book_id = ?
        ''',
        (
            total_chapters,
            total_sentences,
            chapter_index,
            sentence_index,
            sentence_id,
            reading_percentage,
            ready_audio_count,
            audio_percentage,
            now,
            now,
            user_id,
            book_id
        )
    )
    db.commit()

    row = db.execute(
        'SELECT * FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    ).fetchone()
    return jsonify({'status': 'ok', 'book': serialize_book_row(row)})


@app.route('/books/<book_id>', methods=['DELETE'])
@require_auth
def delete_user_book(book_id):
    """Remove one book from the current user's cloud library metadata."""
    user_id = get_current_user_id()
    book_id = normalize_book_id(book_id)
    if not book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    db = get_db()
    db.execute(
        'DELETE FROM book_audio WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    )
    db.execute(
        'DELETE FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    )
    db.commit()
    return jsonify({'status': 'ok'})


@app.route('/books/<book_id>/generate_audio', methods=['POST'])
@require_auth
def generate_book_audio(book_id):
    """
    Schedule per-book audio generation tasks.
    Coordinator never synthesizes audio itself; workers consume queued tasks.
    """
    user_id = get_current_user_id()
    book_id = normalize_book_id(book_id)
    if not book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    data = request.json or {}
    sentences = data.get('sentences') or []
    if not isinstance(sentences, list):
        return jsonify({'error': 'Invalid sentences payload'}), 400

    db = get_db()
    exists = db.execute(
        'SELECT 1 FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    ).fetchone()
    if not exists:
        return jsonify({'error': 'Book not found'}), 404

    now = time.time()
    accepted = 0
    scheduled = 0

    for idx, item in enumerate(sentences[:MAX_GENERATE_SENTENCES]):
        if not isinstance(item, dict):
            continue
        text = (item.get('text') or '').strip()
        if not text:
            continue
        sentence_id = (item.get('id') or f"s_{idx}").strip()
        if not sentence_id:
            continue
        sentence_index = max(0, parse_int(item.get('sentence_index'), idx))
        sentence_hash = (item.get('hash') or hash_text(text)).strip()
        if not sentence_hash:
            continue

        existing = db.execute(
            'SELECT status, s3_url FROM audio_index WHERE hash = ?',
            (sentence_hash,)
        ).fetchone()
        existing_ready = bool(existing and existing['status'] == 'ready' and existing['s3_url'])

        if existing_ready:
            s3_url = existing['s3_url']
            s3_key, inferred = infer_audio_meta_from_url(s3_url)
            audio_format, _ = normalize_audio_format(item.get('audio_format') or item.get('format') or inferred)
            db.execute(
                '''
                INSERT INTO book_audio (
                    user_id, book_id, sentence_id, sentence_index, hash, text,
                    status, s3_key, s3_url, audio_format, error,
                    created_at, updated_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(user_id, book_id, sentence_id) DO UPDATE SET
                    sentence_index = excluded.sentence_index,
                    hash = excluded.hash,
                    text = excluded.text,
                    status = 'ready',
                    s3_key = excluded.s3_key,
                    s3_url = excluded.s3_url,
                    audio_format = excluded.audio_format,
                    error = NULL,
                    updated_at = excluded.updated_at,
                    completed_at = excluded.completed_at
                ''',
                (
                    user_id,
                    book_id,
                    sentence_id,
                    sentence_index,
                    sentence_hash,
                    text,
                    s3_key,
                    s3_url,
                    audio_format,
                    now,
                    now,
                    now
                )
            )
            accepted += 1
            continue

        db.execute(
            '''
            INSERT INTO audio_index (
                hash, text, status, created_at, completed_at,
                claimed_by, claimed_at, updated_at,
                queue_message_id, queue_receipt_handle, attempt_count, error
            )
            VALUES (?, ?, 'pending', ?, NULL, NULL, NULL, ?, NULL, NULL, 0, NULL)
            ON CONFLICT(hash) DO UPDATE SET
                text = COALESCE(NULLIF(excluded.text, ''), audio_index.text),
                status = CASE
                    WHEN audio_index.status = 'ready' THEN 'ready'
                    ELSE 'pending'
                END,
                claimed_by = NULL,
                claimed_at = NULL,
                queue_message_id = NULL,
                queue_receipt_handle = NULL,
                updated_at = excluded.updated_at,
                error = NULL
            ''',
            (sentence_hash, text, now, now)
        )

        db.execute(
            '''
            INSERT INTO book_audio (
                user_id, book_id, sentence_id, sentence_index, hash, text,
                status, s3_key, s3_url, audio_format, error,
                created_at, updated_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, 'wav', NULL, ?, ?, NULL)
            ON CONFLICT(user_id, book_id, sentence_id) DO UPDATE SET
                sentence_index = excluded.sentence_index,
                hash = excluded.hash,
                text = excluded.text,
                status = CASE
                    WHEN book_audio.status = 'ready' THEN 'ready'
                    ELSE 'pending'
                END,
                error = NULL,
                updated_at = excluded.updated_at
            ''',
            (
                user_id,
                book_id,
                sentence_id,
                sentence_index,
                sentence_hash,
                text,
                now,
                now
            )
        )

        accepted += 1

        if accepted % GENERATE_AUDIO_DB_BATCH_SIZE == 0:
            db.commit()

    # SQS enqueue is deferred to /tasks backfill to keep this user-facing endpoint fast.
    # This avoids long DB write locks and worker timeouts on very large book imports.
    if sqs_enabled():
        scheduled = 0

    summary = refresh_user_book_audio_progress(db, user_id, book_id)
    db.commit()

    return jsonify({
        'status': 'queued',
        'accepted': accepted,
        'scheduled': scheduled,
        'sqs_enabled': sqs_enabled(),
        'audio_progress': summary
    })


@app.route('/books/<book_id>/audio_progress', methods=['GET'])
@require_auth
def get_book_audio_progress(book_id):
    user_id = get_current_user_id()
    book_id = normalize_book_id(book_id)
    if not book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    db = get_db()
    exists = db.execute(
        'SELECT 1 FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    ).fetchone()
    if not exists:
        return jsonify({'error': 'Book not found'}), 404

    summary = refresh_user_book_audio_progress(db, user_id, book_id)
    db.commit()
    return jsonify({
        'book_id': book_id,
        'audio_progress': summary
    })


@app.route('/books/<book_id>/audio_manifest', methods=['GET'])
@require_auth
def get_book_audio_manifest(book_id):
    user_id = get_current_user_id()
    book_id = normalize_book_id(book_id)
    if not book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    limit = max(1, min(parse_int(request.args.get('limit'), MAX_MANIFEST_ITEMS), MAX_MANIFEST_ITEMS))

    db = get_db()
    exists = db.execute(
        'SELECT 1 FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, book_id)
    ).fetchone()
    if not exists:
        return jsonify({'error': 'Book not found'}), 404

    rows = db.execute(
        '''
        SELECT sentence_id, sentence_index, hash, s3_url, audio_format
        FROM book_audio
        WHERE user_id = ? AND book_id = ? AND status = 'ready' AND s3_url IS NOT NULL
        ORDER BY sentence_index ASC
        LIMIT ?
        ''',
        (user_id, book_id, limit)
    ).fetchall()

    summary = refresh_user_book_audio_progress(db, user_id, book_id)
    db.commit()

    items = [{
        'sentence_id': row['sentence_id'],
        'sentence_index': row['sentence_index'] or 0,
        'hash': row['hash'],
        'url': row['s3_url'],
        'format': row['audio_format'] or 'wav'
    } for row in rows]

    return jsonify({
        'book_id': book_id,
        'count': len(items),
        'items': items,
        'audio_progress': summary
    })


@app.route('/audio/<sentence_hash>', methods=['GET'])
def proxy_audio_by_hash(sentence_hash):
    """
    Stream audio by hash via API domain to avoid client-side S3 CORS issues.
    """
    sentence_hash = (sentence_hash or '').strip()
    if not sentence_hash:
        return jsonify({'error': 'Missing hash'}), 400

    db = get_db()
    row = db.execute(
        'SELECT s3_url FROM audio_index WHERE hash = ? AND status = "ready" LIMIT 1',
        (sentence_hash,)
    ).fetchone()
    if not row or not row['s3_url']:
        return jsonify({'error': 'Audio not found'}), 404

    s3_key, inferred_format = infer_audio_meta_from_url(row['s3_url'])
    if not s3_key:
        return jsonify({'error': 'Invalid audio URL mapping'}), 500

    try:
        obj = s3_client.get_object(Bucket=AUDIO_BUCKET, Key=s3_key)
        body = obj['Body']
        content_length = int(obj.get('ContentLength') or 0)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch audio from storage: {str(e)}'}), 502

    def stream_audio():
        try:
            while True:
                chunk = body.read(256 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                body.close()
            except Exception:
                pass

    if inferred_format == 'mp3':
        content_type = 'audio/mpeg'
    elif inferred_format == 'm4b':
        content_type = 'audio/mp4'
    else:
        content_type = 'audio/wav'
    headers = {
        'Cache-Control': 'public, max-age=31536000',
        'Accept-Ranges': 'bytes'
    }
    if content_length > 0:
        headers['Content-Length'] = str(content_length)

    return Response(
        stream_with_context(stream_audio()),
        mimetype=content_type,
        headers=headers,
        direct_passthrough=True
    )


@app.route('/stats')
def stats():
    """Get coordinator statistics."""
    db = get_db()

    total = db.execute('SELECT COUNT(*) as c FROM audio_index').fetchone()['c']
    ready = db.execute('SELECT COUNT(*) as c FROM audio_index WHERE status = "ready"').fetchone()['c']
    pending = db.execute('SELECT COUNT(*) as c FROM audio_index WHERE status = "pending"').fetchone()['c']
    generating = db.execute('SELECT COUNT(*) as c FROM audio_index WHERE status = "generating"').fetchone()['c']

    return jsonify({
        'total': total,
        'ready': ready,
        'pending': pending,
        'generating': generating
    })


# =============================================================================
# Main
# =============================================================================

# Initialize database on startup
init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
