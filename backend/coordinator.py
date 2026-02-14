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
import hmac
import time
import json
import random
import string
import uuid
import boto3
import threading
import io
import wave
from array import array
from datetime import datetime, timezone
from botocore.config import Config
from flask import Flask, request, jsonify, g, Response, stream_with_context
from flask_cors import CORS
from functools import wraps
from jose import jwt, JWTError
import requests
from urllib.parse import unquote, urlparse
from openai import OpenAI

app = Flask(__name__)
ALLOWED_CORS_ORIGINS = {
    "https://reader.psybytes.com",
    "http://localhost:5001",
    "http://localhost:3000",
}
CORS(
    app,
    origins=list(ALLOWED_CORS_ORIGINS),
    allow_headers=['Authorization', 'Content-Type'],
    methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS']
)

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
OPENAI_API_KEY = (os.environ.get('OPENAI_API_KEY') or '').strip()
OPENAI_API_BASE = (os.environ.get('OPENAI_API_BASE') or 'https://api.openai.com/v1').strip()
OPENAI_CHAT_MODEL = (
    (os.environ.get('OPENAI_CHAT_MODEL') or '').strip()
    or (os.environ.get('OPENAI_MODEL') or '').strip()
    or 'gpt-4o-mini'
)
OPENAI_TRANSLATION_MODEL = (
    (os.environ.get('OPENAI_TRANSLATION_MODEL') or '').strip()
    or (os.environ.get('OPENAI_MODEL') or '').strip()
    or OPENAI_CHAT_MODEL
)
OPENAI_TIMEOUT_SECONDS = max(5.0, float(os.environ.get('OPENAI_TIMEOUT_SECONDS', '25')))
OPENAI_CHAT_HISTORY_LIMIT = max(6, int(os.environ.get('OPENAI_CHAT_HISTORY_LIMIT', '24')))
OPENAI_MAX_REPLY_CHARS = max(400, int(os.environ.get('OPENAI_MAX_REPLY_CHARS', '6000')))


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

    secret_ids = [
        (os.environ.get('WORKER_SHARED_SECRET_SECRET_ID') or '').strip(),
    ]
    parameter_names = [
        (os.environ.get('WORKER_SHARED_SECRET_PARAMETER_NAME') or '').strip(),
        '/epub-reader/worker-shared-secret',
    ]

    try:
        session = boto3.session.Session(region_name=AWS_REGION or None)
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
if not WORKER_SHARED_SECRET:
    print("WARNING: WORKER_SHARED_SECRET is empty. Worker service-auth will be disabled until set.")

# S3 client
# Force regional, virtual-host style presigned URLs with SigV4 to avoid
# browser CORS/preflight failures caused by global S3 endpoint redirects.
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    config=Config(signature_version='s3v4', s3={'addressing_style': 'virtual'})
)
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
openai_client = None

if OPENAI_API_KEY:
    try:
        if OPENAI_API_BASE:
            openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, timeout=OPENAI_TIMEOUT_SECONDS)
        else:
            openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECONDS)
    except Exception as e:
        openai_client = None
        print(f"WARNING: OpenAI client init failed: {e}")
else:
    print("INFO: OPENAI_API_KEY is not set; chat and translation APIs will return 503 until configured.")


@app.after_request
def ensure_cors_headers(response):
    origin = (request.headers.get('Origin') or '').strip()
    if origin in ALLOWED_CORS_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
        response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
    return response


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
    db.execute('''
        CREATE TABLE IF NOT EXISTS clubs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_user_id TEXT NOT NULL,
            invite_code TEXT NOT NULL UNIQUE,
            join_policy TEXT NOT NULL DEFAULT 'link',
            active_book_id TEXT,
            plan_json TEXT,
            pinned_post_id TEXT
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS club_members (
            club_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            display_name TEXT NOT NULL,
            avatar_color TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'member',
            joined_at REAL NOT NULL,
            hide_progress INTEGER NOT NULL DEFAULT 0,
            progress_by_book_json TEXT NOT NULL DEFAULT '{}',
            notifications_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(club_id, user_id)
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS club_posts (
            id TEXT PRIMARY KEY,
            club_id TEXT NOT NULL,
            author_user_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            text TEXT NOT NULL,
            spoiler_boundary_json TEXT,
            reactions_json TEXT NOT NULL DEFAULT '{}',
            reply_to_post_id TEXT,
            mentions_json TEXT,
            attached_passage_thread_id TEXT
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS club_threads (
            id TEXT PRIMARY KEY,
            club_id TEXT NOT NULL,
            book_id TEXT NOT NULL,
            chapter_id TEXT NOT NULL,
            sentence_start INTEGER NOT NULL DEFAULT 0,
            sentence_end INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            created_by_user_id TEXT NOT NULL,
            title TEXT,
            spoiler_boundary_json TEXT
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS club_thread_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            author_user_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            text TEXT NOT NULL,
            reactions_json TEXT NOT NULL DEFAULT '{}'
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS club_join_requests (
            club_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            requested_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            PRIMARY KEY(club_id, user_id)
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS club_spoiler_preferences (
            club_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            preference TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY(club_id, user_id)
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS book_chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            book_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            chapter_id TEXT,
            sentence_index INTEGER,
            quote TEXT,
            created_at REAL NOT NULL
        )
    ''')
    db.execute('CREATE INDEX IF NOT EXISTS idx_status ON audio_index(status)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_audio_index_claimed_at ON audio_index(claimed_at)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_user_books_user_last_opened ON user_books(user_id, last_opened DESC)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_book_audio_owner_status ON book_audio(user_id, book_id, status)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_book_audio_hash ON book_audio(hash)')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_clubs_invite_code ON clubs(invite_code)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_club_members_user ON club_members(user_id)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_club_posts_club_created ON club_posts(club_id, created_at DESC)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_club_threads_club_book ON club_threads(club_id, book_id)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_club_thread_messages_thread_created ON club_thread_messages(thread_id, created_at ASC)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_book_chat_messages_book_created ON book_chat_messages(user_id, book_id, created_at ASC)')

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

        # Verify signature/issuer first, then validate client binding based on token_use.
        payload = jwt.decode(
            token,
            key,
            algorithms=['RS256'],
            issuer=f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{COGNITO_POOL_ID}",
            options={'verify_aud': False}
        )

        if COGNITO_CLIENT_ID:
            token_use = (payload.get('token_use') or '').strip().lower()
            aud = (payload.get('aud') or '').strip()
            client_id = (payload.get('client_id') or '').strip()

            if token_use == 'access':
                if client_id != COGNITO_CLIENT_ID:
                    return None
            elif token_use == 'id':
                if aud != COGNITO_CLIENT_ID:
                    return None
            elif aud != COGNITO_CLIENT_ID and client_id != COGNITO_CLIENT_ID:
                # Defensive fallback for tokens without token_use claim.
                return None

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


def require_worker_or_auth(f):
    """
    Allow either:
    - user JWT (Authorization: Bearer ...)
    - trusted worker shared secret (X-Worker-Secret)
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        worker_secret = (request.headers.get('X-Worker-Secret') or '').strip()
        if WORKER_SHARED_SECRET and worker_secret and hmac.compare_digest(worker_secret, WORKER_SHARED_SECRET):
            g.user = {'sub': 'worker-service', 'role': 'worker'}
            return f(*args, **kwargs)

        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        token = auth_header[7:]
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


def claim_pending_tasks_from_db(db, user_id: str, limit: int):
    """Claim pending rows directly from DB (fallback when SQS is unavailable/empty)."""
    rows = db.execute(
        '''SELECT hash, text FROM audio_index
           WHERE status = 'pending' AND (claimed_by IS NULL OR claimed_at < ?)
           ORDER BY created_at ASC
           LIMIT ?''',
        (time.time() - 300, limit)
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

    return tasks


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


def now_iso(ts=None):
    value = time.time() if ts is None else parse_float(ts, time.time())
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def safe_json_load(value, default):
    if value is None or value == '':
        return default
    try:
        parsed = json.loads(value)
    except Exception:
        return default
    return parsed if parsed is not None else default


LANGUAGE_NAMES = {
    'auto': 'auto-detect',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'ru': 'Russian',
    'hi': 'Hindi',
    'tr': 'Turkish',
}


def normalize_language_code(value, default='auto'):
    code = str(value or '').strip().lower()
    if not code:
        return default
    return code


def language_name(code):
    key = normalize_language_code(code, 'auto')
    return LANGUAGE_NAMES.get(key, key or 'auto-detect')


def require_openai():
    if not OPENAI_API_KEY or openai_client is None:
        raise RuntimeError('OpenAI is not configured on backend. Missing OPENAI_API_KEY.')
    return openai_client


def extract_openai_text(response):
    try:
        choices = getattr(response, 'choices', None) or []
        if not choices:
            return ''
        message = getattr(choices[0], 'message', None)
        if not message:
            return ''
        content = getattr(message, 'content', '')
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, 'text', None) if not isinstance(item, dict) else item.get('text')
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return '\n'.join(parts).strip()
        return str(content or '').strip()
    except Exception:
        return ''


def openai_chat_completion(messages, model, temperature=0.3):
    client = require_openai()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    text = extract_openai_text(response)
    used_model = getattr(response, 'model', None) or model
    if not text:
        raise RuntimeError('OpenAI returned an empty response.')
    return text, used_model


def serialize_book_chat_row(row):
    return {
        'id': int(row['id']),
        'role': row['role'],
        'content': row['content'] or '',
        'created_at': parse_float(row['created_at'], time.time()),
    }


def fetch_book_chat_messages(db, user_id, book_id, limit=80):
    safe_limit = max(1, min(parse_int(limit, 80), 500))
    rows = db.execute(
        '''
        SELECT id, role, content, created_at
        FROM book_chat_messages
        WHERE user_id = ? AND book_id = ?
        ORDER BY id DESC
        LIMIT ?
        ''',
        (user_id, book_id, safe_limit)
    ).fetchall()
    ordered = list(reversed(rows))
    return [serialize_book_chat_row(row) for row in ordered]


def build_reader_assistant_system_prompt():
    return (
        "You are a careful reading assistant. Prioritize clarity, correctness, and concise explanations. "
        "When useful, simplify difficult text into plain language without losing meaning. "
        "If the user asks for summary, produce structured bullets. "
        "If asked for flashcards, return concise Q/A pairs. "
        "If asked for counter-argument, provide balanced critique. "
        "Never invent facts from outside the provided text context."
    )


def build_chat_messages_for_model(db, user_id, book_id, user_message, context):
    history_rows = db.execute(
        '''
        SELECT role, content
        FROM book_chat_messages
        WHERE user_id = ? AND book_id = ? AND role IN ('user', 'assistant')
        ORDER BY id DESC
        LIMIT ?
        ''',
        (user_id, book_id, OPENAI_CHAT_HISTORY_LIMIT)
    ).fetchall()
    history = list(reversed(history_rows))
    messages = [{'role': 'system', 'content': build_reader_assistant_system_prompt()}]

    if isinstance(context, dict):
        chapter_id = str(context.get('chapter_id') or context.get('chapterId') or '').strip()
        sentence_index = parse_int(context.get('sentence_index') or context.get('sentenceIndex'), -1)
        quote = str(context.get('quote') or '').strip()
        if chapter_id or sentence_index >= 0 or quote:
            context_lines = ['Reading context:']
            if chapter_id:
                context_lines.append(f"- chapter_id: {chapter_id}")
            if sentence_index >= 0:
                context_lines.append(f"- sentence_index: {sentence_index}")
            if quote:
                context_lines.append(f"- passage: {quote}")
            messages.append({'role': 'system', 'content': '\n'.join(context_lines)})

    for row in history:
        role = row['role']
        content = (row['content'] or '').strip()
        if role not in ('user', 'assistant') or not content:
            continue
        messages.append({'role': role, 'content': content})

    messages.append({'role': 'user', 'content': user_message})
    return messages


def translate_with_openai(text, target_lang, source_lang='auto'):
    cleaned = (text or '').strip()
    if not cleaned:
        raise ValueError('Missing text')
    target_code = normalize_language_code(target_lang, 'en')
    source_code = normalize_language_code(source_lang, 'auto')
    source_name = language_name(source_code)
    target_name = language_name(target_code)

    prompt = (
        f"Translate the text from {source_name} to {target_name}. "
        "Return only the translated sentence, no explanation.\n\n"
        f"Text:\n{cleaned}"
    )
    translated, used_model = openai_chat_completion(
        messages=[
            {'role': 'system', 'content': 'You are a precise translator.'},
            {'role': 'user', 'content': prompt},
        ],
        model=OPENAI_TRANSLATION_MODEL,
        temperature=0.1,
    )
    return {
        'translated_text': translated.strip(),
        'target_language': target_code,
        'source_language': source_code,
        'model': used_model,
    }


def _default_member_notifications():
    return {
        'replies': True,
        'mentions': True,
        'milestonePosts': True,
        'digest': 'daily',
    }


def generate_invite_code(length=8):
    alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    return ''.join(random.choice(alphabet) for _ in range(max(4, length)))


def build_avatar_color(seed):
    digest = hashlib.sha256((seed or 'reader').encode('utf-8')).hexdigest()
    hue = int(digest[:8], 16) % 360
    return f'hsl({hue} 45% 38%)'


def current_user_profile():
    user_id = get_current_user_id()
    email = (g.user.get('email') or '').strip()
    display_name = (g.user.get('name') or email or 'Reader').strip()
    avatar_color = build_avatar_color(email or user_id)
    return {
        'user_id': user_id,
        'email': email,
        'display_name': display_name,
        'avatar_color': avatar_color,
    }


def _normalize_progress_map(value):
    progress = value if isinstance(value, dict) else {}
    normalized = {}
    for book_id, item in progress.items():
        if not isinstance(item, dict):
            continue
        chapter_id = str(item.get('chapterId') or item.get('chapter_id') or '').strip()
        sentence_index = max(0, parse_int(item.get('sentenceIndex') or item.get('sentence_index'), 0))
        if not chapter_id:
            continue
        normalized[str(book_id)] = {
            'chapterId': chapter_id,
            'sentenceIndex': sentence_index,
        }
    return normalized


def _normalize_notifications(value):
    defaults = _default_member_notifications()
    incoming = value if isinstance(value, dict) else {}
    digest = str(incoming.get('digest') or defaults['digest']).strip().lower()
    if digest not in ('off', 'daily', 'weekly'):
        digest = defaults['digest']
    return {
        'replies': bool(incoming.get('replies', defaults['replies'])),
        'mentions': bool(incoming.get('mentions', defaults['mentions'])),
        'milestonePosts': bool(incoming.get('milestonePosts', defaults['milestonePosts'])),
        'digest': digest,
    }


def serialize_club_member(row):
    progress = _normalize_progress_map(safe_json_load(row['progress_by_book_json'], {}))
    notifications = _normalize_notifications(safe_json_load(row['notifications_json'], {}))
    return {
        'userId': row['user_id'],
        'displayName': row['display_name'],
        'avatarColor': row['avatar_color'],
        'role': row['role'] if row['role'] in ('owner', 'member') else 'member',
        'joinedAt': now_iso(row['joined_at']),
        'progressByBook': progress,
        'notifications': notifications,
        'hideProgress': bool(row['hide_progress']),
    }


def serialize_club_row(db, row, include_members=True):
    members = []
    if include_members:
        member_rows = db.execute(
            '''
            SELECT *
            FROM club_members
            WHERE club_id = ?
            ORDER BY CASE role WHEN 'owner' THEN 0 ELSE 1 END, joined_at ASC
            ''',
            (row['id'],)
        ).fetchall()
        members = [serialize_club_member(member) for member in member_rows]

    plan = safe_json_load(row['plan_json'], None)
    if plan is not None and not isinstance(plan, dict):
        plan = None

    return {
        'id': row['id'],
        'name': row['name'],
        'description': row['description'],
        'createdAt': now_iso(row['created_at']),
        'ownerUserId': row['owner_user_id'],
        'inviteCode': row['invite_code'],
        'joinPolicy': row['join_policy'] if row['join_policy'] in ('link', 'approval') else 'link',
        'members': members,
        'activeBookId': row['active_book_id'],
        'plan': plan,
        'pinnedPostId': row['pinned_post_id'],
    }


def _default_spoiler_boundary(thread_or_post):
    if not isinstance(thread_or_post, dict):
        return None
    chapter_id = str(thread_or_post.get('chapterId') or thread_or_post.get('chapter_id') or '').strip()
    if not chapter_id:
        return None
    sentence_index = max(0, parse_int(thread_or_post.get('sentenceIndex') or thread_or_post.get('sentence_index'), 0))
    return {'chapterId': chapter_id, 'sentenceIndex': sentence_index}


def serialize_club_post_row(row):
    spoiler_boundary = _default_spoiler_boundary(safe_json_load(row['spoiler_boundary_json'], {}))
    reactions_raw = safe_json_load(row['reactions_json'], {})
    reactions = {}
    if isinstance(reactions_raw, dict):
        for emoji, user_ids in reactions_raw.items():
            if not isinstance(emoji, str):
                continue
            if isinstance(user_ids, list):
                reactions[emoji] = [str(uid) for uid in user_ids if uid]
    mentions_raw = safe_json_load(row['mentions_json'], None)
    mentions = [str(x) for x in mentions_raw if x] if isinstance(mentions_raw, list) else None
    return {
        'id': row['id'],
        'clubId': row['club_id'],
        'authorUserId': row['author_user_id'],
        'createdAt': now_iso(row['created_at']),
        'text': row['text'] or '',
        'spoilerBoundary': spoiler_boundary,
        'reactions': reactions,
        'replyToPostId': row['reply_to_post_id'],
        'mentions': mentions,
        'attachedPassageThreadId': row['attached_passage_thread_id'],
    }


def serialize_club_message_row(row):
    reactions_raw = safe_json_load(row['reactions_json'], {})
    reactions = {}
    if isinstance(reactions_raw, dict):
        for emoji, user_ids in reactions_raw.items():
            if not isinstance(emoji, str):
                continue
            if isinstance(user_ids, list):
                reactions[emoji] = [str(uid) for uid in user_ids if uid]
    return {
        'id': row['id'],
        'authorUserId': row['author_user_id'],
        'createdAt': now_iso(row['created_at']),
        'text': row['text'] or '',
        'reactions': reactions,
    }


def serialize_club_thread_row(db, row, include_messages=True):
    spoiler_boundary = _default_spoiler_boundary(safe_json_load(row['spoiler_boundary_json'], {})) or {
        'chapterId': row['chapter_id'],
        'sentenceIndex': max(0, parse_int(row['sentence_start'], 0)),
    }
    messages = []
    if include_messages:
        message_rows = db.execute(
            '''
            SELECT *
            FROM club_thread_messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
            ''',
            (row['id'],)
        ).fetchall()
        messages = [serialize_club_message_row(message_row) for message_row in message_rows]

    return {
        'id': row['id'],
        'clubId': row['club_id'],
        'bookId': row['book_id'],
        'chapterId': row['chapter_id'],
        'sentenceRange': {
            'start': max(0, parse_int(row['sentence_start'], 0)),
            'end': max(0, parse_int(row['sentence_end'], 0)),
        },
        'createdAt': now_iso(row['created_at']),
        'createdByUserId': row['created_by_user_id'],
        'title': row['title'],
        'spoilerBoundary': spoiler_boundary,
        'messages': messages,
    }


def get_club_with_role(db, club_id, user_id):
    return db.execute(
        '''
        SELECT c.*, m.role AS member_role
        FROM clubs c
        LEFT JOIN club_members m ON m.club_id = c.id AND m.user_id = ?
        WHERE c.id = ?
        ''',
        (user_id, club_id)
    ).fetchone()


def ensure_member_access(db, club_id, user_id):
    row = get_club_with_role(db, club_id, user_id)
    if not row:
        return None, (jsonify({'error': 'Club not found'}), 404)
    if not row['member_role']:
        return None, (jsonify({'error': 'Forbidden'}), 403)
    return row, None


def ensure_owner_access(db, club_id, user_id):
    row, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return None, error_response
    if row['owner_user_id'] != user_id:
        return None, (jsonify({'error': 'Owner access required'}), 403)
    return row, None


def sqs_enabled():
    return bool(AUDIO_SQS_QUEUE_URL and sqs_client)


def sqs_is_fifo_queue():
    return AUDIO_SQS_QUEUE_URL.lower().endswith('.fifo')


def _build_sqs_payload(sentence_hash, text):
    payload = {
        'hash': (sentence_hash or '').strip(),
        'text': (text or '').strip(),
        'enqueued_at': time.time(),
    }
    if not payload['hash'] or not payload['text']:
        return None
    return payload


def enqueue_audio_tasks_batch(items):
    """
    Push many sentence tasks to SQS using send_message_batch (10 per request).
    Returns a set of successfully enqueued hashes.
    """
    if not sqs_enabled():
        return set()

    # Keep first text for each hash to avoid duplicate enqueue calls.
    deduped = {}
    for sentence_hash, text in items:
        payload = _build_sqs_payload(sentence_hash, text)
        if not payload:
            continue
        if payload['hash'] in deduped:
            continue
        deduped[payload['hash']] = payload['text']

    if not deduped:
        return set()

    pairs = list(deduped.items())
    success_hashes = set()

    for start in range(0, len(pairs), 10):
        chunk = pairs[start:start + 10]
        entries = []
        id_to_hash = {}
        for idx, (sentence_hash, text) in enumerate(chunk):
            payload = _build_sqs_payload(sentence_hash, text)
            if not payload:
                continue
            entry_id = f"m{idx}"
            id_to_hash[entry_id] = payload['hash']
            entry = {
                'Id': entry_id,
                'MessageBody': json.dumps(payload, separators=(',', ':')),
            }
            if sqs_is_fifo_queue():
                entry['MessageGroupId'] = 'audio-generation'
                entry['MessageDeduplicationId'] = payload['hash']
            entries.append(entry)

        if not entries:
            continue

        try:
            response = sqs_client.send_message_batch(
                QueueUrl=AUDIO_SQS_QUEUE_URL,
                Entries=entries
            )
        except Exception as e:
            print(f"Failed batch enqueue to SQS: {e}")
            continue

        for ok in (response.get('Successful') or []):
            message_id = ok.get('Id')
            sentence_hash = id_to_hash.get(message_id)
            if sentence_hash:
                success_hashes.add(sentence_hash)

        for failed in (response.get('Failed') or []):
            message_id = failed.get('Id')
            sentence_hash = id_to_hash.get(message_id)
            if sentence_hash:
                print(f"Failed to enqueue SQS task {sentence_hash}: {failed.get('Code')} {failed.get('Message')}")

    return success_hashes


def enqueue_audio_task(sentence_hash, text):
    """
    Push one sentence task to SQS. Returns True when enqueue request succeeds.
    """
    return bool(enqueue_audio_tasks_batch([(sentence_hash, text)]))


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

    batch = []
    for row in rows:
        sentence_hash = (row['hash'] or '').strip()
        text = (row['text'] or '').strip()
        if not sentence_hash or not text:
            continue
        batch.append((sentence_hash, text))

    queued_hashes = list(enqueue_audio_tasks_batch(batch))
    pushed = len(queued_hashes)
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


def mark_book_audio_ready_for_hash(db, sentence_hash, s3_url, now=None):
    """
    When a hash becomes ready, mark matching pending/generating/failed book rows ready too.
    Returns affected (user_id, book_id) pairs for progress refresh.
    """
    sentence_hash = (sentence_hash or '').strip()
    s3_url = (s3_url or '').strip()
    if not sentence_hash or not s3_url:
        return set()

    now = now or time.time()
    s3_key, inferred = infer_audio_meta_from_url(s3_url)
    audio_format, _ = normalize_audio_format(inferred)

    rows = db.execute(
        '''
        SELECT DISTINCT user_id, book_id
        FROM book_audio
        WHERE hash = ? AND status != 'ready'
        ''',
        (sentence_hash,)
    ).fetchall()
    affected = {(r['user_id'], r['book_id']) for r in rows}
    if not affected:
        return set()

    db.execute(
        '''
        UPDATE book_audio
        SET status = 'ready',
            s3_key = ?,
            s3_url = ?,
            audio_format = ?,
            error = NULL,
            completed_at = COALESCE(completed_at, ?),
            updated_at = ?
        WHERE hash = ? AND status != 'ready'
        ''',
        (s3_key, s3_url, audio_format, now, now, sentence_hash)
    )
    return affected


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
@require_worker_or_auth
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
@require_worker_or_auth
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
@require_worker_or_auth
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
            tasks = claim_pending_tasks_from_db(db, user_id, limit)
            db.commit()
            return jsonify({'tasks': tasks})

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

        if not tasks:
            # Fallback keeps workers productive when queue delivery is empty/misconfigured.
            tasks = claim_pending_tasks_from_db(db, user_id, limit)

        db.commit()
        return jsonify({'tasks': tasks})

    # Legacy DB polling mode (no SQS configured).
    tasks = claim_pending_tasks_from_db(db, user_id, limit)

    db.commit()
    return jsonify({'tasks': tasks})


@app.route('/complete', methods=['POST'])
@require_worker_or_auth
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
    affected_books = mark_book_audio_ready_for_hash(db, sentence_hash, s3_url, now=now)
    for owner_id, owner_book_id in affected_books:
        refresh_user_book_audio_progress(db, owner_id, owner_book_id)

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
@require_worker_or_auth
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
    hashes_to_propagate = {}
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
        hashes_to_propagate[sentence_hash] = s3_url

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
            updated_books.add((user_id, book_id))

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

    for sentence_hash, s3_url in hashes_to_propagate.items():
        updated_books.update(mark_book_audio_ready_for_hash(db, sentence_hash, s3_url, now=now))

    summaries = {}
    for owner_id, owner_book_id in updated_books:
        summaries[owner_book_id] = refresh_user_book_audio_progress(db, owner_id, owner_book_id)

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
@require_worker_or_auth
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
@require_worker_or_auth
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


@app.route('/translate', methods=['POST'])
@require_auth
def translate_text():
    payload = request.get_json(silent=True)
    data = payload if isinstance(payload, dict) else {}
    text = str(data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'Missing text'}), 400

    if len(text) > 12000:
        return jsonify({'error': 'Text is too long'}), 400

    target_language = normalize_language_code(
        data.get('target_language') or data.get('targetLanguage'),
        'en'
    )
    source_language = normalize_language_code(
        data.get('source_language') or data.get('sourceLanguage'),
        'auto'
    )

    try:
        result = translate_with_openai(text, target_language, source_language)
        return jsonify({
            'translated_text': result.get('translated_text', ''),
            'target_language': result.get('target_language', target_language),
            'source_language': result.get('source_language', source_language),
            'model': result.get('model'),
        })
    except RuntimeError as e:
        message = str(e)
        code = 503 if 'not configured' in message.lower() else 502
        return jsonify({'error': message}), code
    except Exception as e:
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500


@app.route('/books/<book_id>/chat_history', methods=['GET'])
@require_auth
def get_book_chat_history(book_id):
    user_id = get_current_user_id()
    normalized_book_id = normalize_book_id(book_id)
    if not normalized_book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    db = get_db()
    exists = db.execute(
        'SELECT 1 FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, normalized_book_id)
    ).fetchone()
    if not exists:
        return jsonify({'error': 'Book not found'}), 404

    limit = parse_int(request.args.get('limit'), 80)
    messages = fetch_book_chat_messages(db, user_id, normalized_book_id, limit=limit)
    return jsonify({
        'book_id': normalized_book_id,
        'messages': messages,
    })


@app.route('/books/<book_id>/chat', methods=['POST'])
@require_auth
def send_book_chat_message(book_id):
    user_id = get_current_user_id()
    normalized_book_id = normalize_book_id(book_id)
    if not normalized_book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    db = get_db()
    exists = db.execute(
        'SELECT 1 FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, normalized_book_id)
    ).fetchone()
    if not exists:
        return jsonify({'error': 'Book not found'}), 404

    payload = request.get_json(silent=True)
    data = payload if isinstance(payload, dict) else {}
    message = str(data.get('message') or '').strip()
    if not message:
        return jsonify({'error': 'Missing message'}), 400
    if len(message) > 12000:
        return jsonify({'error': 'Message is too long'}), 400

    raw_context = data.get('context') if isinstance(data.get('context'), dict) else {}
    chapter_id = str(raw_context.get('chapter_id') or raw_context.get('chapterId') or '').strip() or None
    sentence_index = parse_int(raw_context.get('sentence_index') or raw_context.get('sentenceIndex'), -1)
    if sentence_index < 0:
        sentence_index = None
    quote = str(raw_context.get('quote') or '').strip()
    if len(quote) > 12000:
        quote = quote[:12000]
    if not quote:
        quote = None

    context_payload = {
        'chapter_id': chapter_id,
        'sentence_index': sentence_index,
        'quote': quote,
    }
    model_messages = build_chat_messages_for_model(db, user_id, normalized_book_id, message, context_payload)

    now = time.time()
    db.execute(
        '''
        INSERT INTO book_chat_messages (user_id, book_id, role, content, chapter_id, sentence_index, quote, created_at)
        VALUES (?, ?, 'user', ?, ?, ?, ?, ?)
        ''',
        (user_id, normalized_book_id, message, chapter_id, sentence_index, quote, now)
    )
    db.commit()

    try:
        assistant_reply, used_model = openai_chat_completion(
            messages=model_messages,
            model=OPENAI_CHAT_MODEL,
            temperature=0.4,
        )
    except RuntimeError as e:
        message_text = str(e)
        code = 503 if 'not configured' in message_text.lower() else 502
        return jsonify({'error': message_text}), code
    except Exception as e:
        return jsonify({'error': f'AI request failed: {str(e)}'}), 500

    cleaned_reply = (assistant_reply or '').strip()
    if len(cleaned_reply) > OPENAI_MAX_REPLY_CHARS:
        cleaned_reply = cleaned_reply[:OPENAI_MAX_REPLY_CHARS].rstrip()

    db.execute(
        '''
        INSERT INTO book_chat_messages (user_id, book_id, role, content, chapter_id, sentence_index, quote, created_at)
        VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?)
        ''',
        (user_id, normalized_book_id, cleaned_reply, chapter_id, sentence_index, quote, time.time())
    )
    db.commit()

    messages = fetch_book_chat_messages(db, user_id, normalized_book_id, limit=80)
    return jsonify({
        'book_id': normalized_book_id,
        'reply': cleaned_reply,
        'model': used_model,
        'messages': messages,
    })


@app.route('/books/<book_id>/chat_history', methods=['DELETE'])
@require_auth
def clear_book_chat_history(book_id):
    user_id = get_current_user_id()
    normalized_book_id = normalize_book_id(book_id)
    if not normalized_book_id:
        return jsonify({'error': 'Invalid book_id'}), 400

    db = get_db()
    exists = db.execute(
        'SELECT 1 FROM user_books WHERE user_id = ? AND book_id = ?',
        (user_id, normalized_book_id)
    ).fetchone()
    if not exists:
        return jsonify({'error': 'Book not found'}), 404

    cursor = db.execute(
        'DELETE FROM book_chat_messages WHERE user_id = ? AND book_id = ?',
        (user_id, normalized_book_id)
    )
    db.commit()
    return jsonify({
        'ok': True,
        'cleared': int(cursor.rowcount or 0),
    })


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
    pending_hash_to_text = {}

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
        pending_hash_to_text[sentence_hash] = text

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

    if sqs_enabled() and pending_hash_to_text:
        now = time.time()
        scheduled_hashes = enqueue_audio_tasks_batch(list(pending_hash_to_text.items()))
        scheduled = len(scheduled_hashes)
        if scheduled_hashes:
            db.executemany(
                '''
                UPDATE audio_index
                SET queue_message_id = 'queued',
                    updated_at = ?
                WHERE hash = ? AND status = 'pending'
                ''',
                [(now, sentence_hash) for sentence_hash in scheduled_hashes]
            )

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


# =============================================================================
# Book Clubs
# =============================================================================

@app.route('/clubs', methods=['GET'])
@require_auth
def list_clubs():
    db = get_db()
    user_id = get_current_user_id()
    rows = db.execute(
        '''
        SELECT c.*
        FROM clubs c
        JOIN club_members m ON m.club_id = c.id
        WHERE m.user_id = ?
        ORDER BY c.updated_at DESC, c.created_at DESC
        ''',
        (user_id,)
    ).fetchall()
    return jsonify({'clubs': [serialize_club_row(db, row, include_members=True) for row in rows]})


@app.route('/clubs', methods=['POST'])
@require_auth
def create_club():
    data = request.json or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'Missing club name'}), 400

    description = (data.get('description') or '').strip() or None
    join_policy = (data.get('joinPolicy') or data.get('join_policy') or 'link').strip().lower()
    if join_policy not in ('link', 'approval'):
        join_policy = 'link'

    profile = current_user_profile()
    now = time.time()
    club_id = f"club-{uuid.uuid4().hex[:12]}"

    db = get_db()
    invite_code = None
    for _ in range(8):
        candidate = generate_invite_code(8)
        existing = db.execute('SELECT 1 FROM clubs WHERE invite_code = ?', (candidate,)).fetchone()
        if not existing:
            invite_code = candidate
            break
    if not invite_code:
        return jsonify({'error': 'Failed generating invite code'}), 500

    db.execute(
        '''
        INSERT INTO clubs (
            id, name, description, created_at, updated_at,
            owner_user_id, invite_code, join_policy, active_book_id,
            plan_json, pinned_post_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
        ''',
        (club_id, name, description, now, now, profile['user_id'], invite_code, join_policy)
    )
    db.execute(
        '''
        INSERT INTO club_members (
            club_id, user_id, display_name, avatar_color, role, joined_at,
            hide_progress, progress_by_book_json, notifications_json
        )
        VALUES (?, ?, ?, ?, 'owner', ?, 0, '{}', ?)
        ''',
        (club_id, profile['user_id'], profile['display_name'], profile['avatar_color'], now, json.dumps(_default_member_notifications()))
    )
    db.commit()

    row = db.execute('SELECT * FROM clubs WHERE id = ?', (club_id,)).fetchone()
    return jsonify({'club': serialize_club_row(db, row, include_members=True)})


@app.route('/clubs/join', methods=['POST'])
@require_auth
def join_club():
    data = request.json or {}
    invite_code = (data.get('inviteCode') or data.get('invite_code') or '').strip().upper()
    if not invite_code:
        return jsonify({'error': 'Missing invite code'}), 400

    db = get_db()
    profile = current_user_profile()
    row = db.execute(
        'SELECT * FROM clubs WHERE invite_code = ?',
        (invite_code,)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Invalid invite code'}), 404

    membership = db.execute(
        'SELECT role FROM club_members WHERE club_id = ? AND user_id = ?',
        (row['id'], profile['user_id'])
    ).fetchone()
    if membership:
        return jsonify({'status': 'joined', 'club': serialize_club_row(db, row, include_members=True)})

    if row['join_policy'] == 'approval':
        db.execute(
            '''
            INSERT INTO club_join_requests (club_id, user_id, requested_at, status)
            VALUES (?, ?, ?, 'pending')
            ON CONFLICT(club_id, user_id) DO UPDATE SET
                requested_at = excluded.requested_at,
                status = 'pending'
            ''',
            (row['id'], profile['user_id'], time.time())
        )
        db.commit()
        return jsonify({'status': 'pending', 'clubId': row['id']}), 202

    now = time.time()
    db.execute(
        '''
        INSERT INTO club_members (
            club_id, user_id, display_name, avatar_color, role, joined_at,
            hide_progress, progress_by_book_json, notifications_json
        )
        VALUES (?, ?, ?, ?, 'member', ?, 0, '{}', ?)
        ''',
        (row['id'], profile['user_id'], profile['display_name'], profile['avatar_color'], now, json.dumps(_default_member_notifications()))
    )
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (now, row['id']))
    db.commit()
    refreshed = db.execute('SELECT * FROM clubs WHERE id = ?', (row['id'],)).fetchone()
    return jsonify({'status': 'joined', 'club': serialize_club_row(db, refreshed, include_members=True)})


@app.route('/clubs/<club_id>', methods=['GET'])
@require_auth
def get_club(club_id):
    db = get_db()
    user_id = get_current_user_id()
    row, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response
    return jsonify({'club': serialize_club_row(db, row, include_members=True)})


@app.route('/clubs/<club_id>', methods=['PATCH'])
@require_auth
def update_club(club_id):
    db = get_db()
    user_id = get_current_user_id()
    row, error_response = ensure_owner_access(db, club_id, user_id)
    if error_response:
        return error_response

    data = request.json or {}
    name = (data.get('name') if 'name' in data else row['name']) or row['name']
    name = str(name).strip()
    if not name:
        return jsonify({'error': 'Club name cannot be empty'}), 400

    description = data.get('description') if 'description' in data else row['description']
    if description is not None:
        description = str(description).strip() or None

    join_policy = data.get('joinPolicy') if 'joinPolicy' in data else data.get('join_policy')
    if join_policy is None:
        join_policy = row['join_policy']
    join_policy = str(join_policy).strip().lower()
    if join_policy not in ('link', 'approval'):
        return jsonify({'error': 'Invalid joinPolicy'}), 400

    active_book_id = data.get('activeBookId') if 'activeBookId' in data else data.get('active_book_id')
    if active_book_id is None:
        active_book_id = row['active_book_id']
    active_book_id = str(active_book_id).strip() or None

    plan = data.get('plan') if 'plan' in data else safe_json_load(row['plan_json'], None)
    if plan is not None and not isinstance(plan, dict):
        return jsonify({'error': 'Invalid plan payload'}), 400

    pinned_post_id = data.get('pinnedPostId') if 'pinnedPostId' in data else data.get('pinned_post_id')
    if pinned_post_id is None:
        pinned_post_id = row['pinned_post_id']
    pinned_post_id = str(pinned_post_id).strip() or None

    now = time.time()
    db.execute(
        '''
        UPDATE clubs
        SET name = ?,
            description = ?,
            join_policy = ?,
            active_book_id = ?,
            plan_json = ?,
            pinned_post_id = ?,
            updated_at = ?
        WHERE id = ?
        ''',
        (name, description, join_policy, active_book_id, json.dumps(plan) if plan is not None else None, pinned_post_id, now, club_id)
    )
    db.commit()

    refreshed = db.execute('SELECT * FROM clubs WHERE id = ?', (club_id,)).fetchone()
    return jsonify({'club': serialize_club_row(db, refreshed, include_members=True)})


@app.route('/clubs/<club_id>', methods=['DELETE'])
@require_auth
def delete_club(club_id):
    db = get_db()
    user_id = get_current_user_id()
    row, error_response = ensure_owner_access(db, club_id, user_id)
    if error_response:
        return error_response

    db.execute('DELETE FROM club_thread_messages WHERE thread_id IN (SELECT id FROM club_threads WHERE club_id = ?)', (club_id,))
    db.execute('DELETE FROM club_threads WHERE club_id = ?', (club_id,))
    db.execute('DELETE FROM club_posts WHERE club_id = ?', (club_id,))
    db.execute('DELETE FROM club_members WHERE club_id = ?', (club_id,))
    db.execute('DELETE FROM club_join_requests WHERE club_id = ?', (club_id,))
    db.execute('DELETE FROM club_spoiler_preferences WHERE club_id = ?', (club_id,))
    db.execute('DELETE FROM clubs WHERE id = ?', (club_id,))
    db.commit()
    return jsonify({'status': 'deleted', 'clubId': club_id})


@app.route('/clubs/<club_id>/leave', methods=['POST'])
@require_auth
def leave_club(club_id):
    db = get_db()
    user_id = get_current_user_id()
    row, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    if row['owner_user_id'] == user_id:
        db.execute('DELETE FROM club_thread_messages WHERE thread_id IN (SELECT id FROM club_threads WHERE club_id = ?)', (club_id,))
        db.execute('DELETE FROM club_threads WHERE club_id = ?', (club_id,))
        db.execute('DELETE FROM club_posts WHERE club_id = ?', (club_id,))
        db.execute('DELETE FROM club_members WHERE club_id = ?', (club_id,))
        db.execute('DELETE FROM club_join_requests WHERE club_id = ?', (club_id,))
        db.execute('DELETE FROM club_spoiler_preferences WHERE club_id = ?', (club_id,))
        db.execute('DELETE FROM clubs WHERE id = ?', (club_id,))
        db.commit()
        return jsonify({'status': 'deleted', 'clubId': club_id})

    db.execute('DELETE FROM club_members WHERE club_id = ? AND user_id = ?', (club_id, user_id))
    db.execute('DELETE FROM club_spoiler_preferences WHERE club_id = ? AND user_id = ?', (club_id, user_id))
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (time.time(), club_id))
    db.commit()
    return jsonify({'status': 'left', 'clubId': club_id})


@app.route('/clubs/<club_id>/invite_code/regenerate', methods=['POST'])
@require_auth
def regenerate_club_invite_code(club_id):
    db = get_db()
    user_id = get_current_user_id()
    _, error_response = ensure_owner_access(db, club_id, user_id)
    if error_response:
        return error_response

    invite_code = None
    for _ in range(8):
        candidate = generate_invite_code(8)
        existing = db.execute('SELECT 1 FROM clubs WHERE invite_code = ? AND id != ?', (candidate, club_id)).fetchone()
        if not existing:
            invite_code = candidate
            break
    if not invite_code:
        return jsonify({'error': 'Failed generating invite code'}), 500

    now = time.time()
    db.execute('UPDATE clubs SET invite_code = ?, updated_at = ? WHERE id = ?', (invite_code, now, club_id))
    db.commit()
    row = db.execute('SELECT * FROM clubs WHERE id = ?', (club_id,)).fetchone()
    return jsonify({'club': serialize_club_row(db, row, include_members=True)})


@app.route('/clubs/<club_id>/members', methods=['GET'])
@require_auth
def list_club_members(club_id):
    db = get_db()
    user_id = get_current_user_id()
    _, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    rows = db.execute(
        '''
        SELECT *
        FROM club_members
        WHERE club_id = ?
        ORDER BY CASE role WHEN 'owner' THEN 0 ELSE 1 END, joined_at ASC
        ''',
        (club_id,)
    ).fetchall()
    return jsonify({'members': [serialize_club_member(row) for row in rows]})


@app.route('/clubs/<club_id>/members/<member_user_id>', methods=['DELETE'])
@require_auth
def remove_club_member(club_id, member_user_id):
    db = get_db()
    user_id = get_current_user_id()
    row, error_response = ensure_owner_access(db, club_id, user_id)
    if error_response:
        return error_response
    if member_user_id == row['owner_user_id']:
        return jsonify({'error': 'Cannot remove owner'}), 400

    db.execute('DELETE FROM club_members WHERE club_id = ? AND user_id = ?', (club_id, member_user_id))
    db.execute('DELETE FROM club_spoiler_preferences WHERE club_id = ? AND user_id = ?', (club_id, member_user_id))
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (time.time(), club_id))
    db.commit()
    return jsonify({'status': 'ok', 'clubId': club_id, 'userId': member_user_id})


@app.route('/clubs/<club_id>/members/me/progress', methods=['POST'])
@require_auth
def update_member_progress(club_id):
    db = get_db()
    user_id = get_current_user_id()
    _, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    data = request.json or {}
    book_id = str(data.get('bookId') or data.get('book_id') or '').strip()
    chapter_id = str(data.get('chapterId') or data.get('chapter_id') or '').strip()
    sentence_index = max(0, parse_int(data.get('sentenceIndex') or data.get('sentence_index'), 0))
    if not book_id or not chapter_id:
        return jsonify({'error': 'Missing bookId/chapterId'}), 400

    current = db.execute(
        'SELECT progress_by_book_json FROM club_members WHERE club_id = ? AND user_id = ?',
        (club_id, user_id)
    ).fetchone()
    progress_map = _normalize_progress_map(safe_json_load(current['progress_by_book_json'] if current else '{}', {}))
    progress_map[book_id] = {
        'chapterId': chapter_id,
        'sentenceIndex': sentence_index,
    }

    db.execute(
        '''
        UPDATE club_members
        SET progress_by_book_json = ?
        WHERE club_id = ? AND user_id = ?
        ''',
        (json.dumps(progress_map), club_id, user_id)
    )
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (time.time(), club_id))
    db.commit()
    return jsonify({'status': 'ok', 'progressByBook': progress_map})


@app.route('/clubs/<club_id>/posts', methods=['GET'])
@require_auth
def list_club_posts(club_id):
    db = get_db()
    user_id = get_current_user_id()
    _, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    rows = db.execute(
        '''
        SELECT *
        FROM club_posts
        WHERE club_id = ?
        ORDER BY created_at DESC
        ''',
        (club_id,)
    ).fetchall()
    return jsonify({'posts': [serialize_club_post_row(row) for row in rows]})


@app.route('/clubs/<club_id>/posts', methods=['POST'])
@require_auth
def create_club_post(club_id):
    db = get_db()
    profile = current_user_profile()
    _, error_response = ensure_member_access(db, club_id, profile['user_id'])
    if error_response:
        return error_response

    data = request.json or {}
    text = str(data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'Missing text'}), 400

    post_id = str(data.get('id') or f"post-{uuid.uuid4().hex[:12]}").strip()
    spoiler_boundary = _default_spoiler_boundary(data.get('spoilerBoundary') or data.get('spoiler_boundary'))
    reply_to_post_id = str(data.get('replyToPostId') or data.get('reply_to_post_id') or '').strip() or None
    attached_thread_id = str(data.get('attachedPassageThreadId') or data.get('attached_passage_thread_id') or '').strip() or None
    mentions = data.get('mentions')
    mentions_json = json.dumps([str(x) for x in mentions if x]) if isinstance(mentions, list) else None
    now = time.time()

    db.execute(
        '''
        INSERT INTO club_posts (
            id, club_id, author_user_id, created_at, text,
            spoiler_boundary_json, reactions_json, reply_to_post_id,
            mentions_json, attached_passage_thread_id
        )
        VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)
        ''',
        (
            post_id,
            club_id,
            profile['user_id'],
            now,
            text,
            json.dumps(spoiler_boundary) if spoiler_boundary else None,
            reply_to_post_id,
            mentions_json,
            attached_thread_id
        )
    )
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (now, club_id))
    db.commit()
    row = db.execute('SELECT * FROM club_posts WHERE id = ?', (post_id,)).fetchone()
    return jsonify({'post': serialize_club_post_row(row)})


@app.route('/clubs/<club_id>/posts/<post_id>', methods=['DELETE'])
@require_auth
def delete_club_post(club_id, post_id):
    db = get_db()
    user_id = get_current_user_id()
    club_row, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    row = db.execute(
        'SELECT author_user_id FROM club_posts WHERE id = ? AND club_id = ?',
        (post_id, club_id)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Post not found'}), 404

    if row['author_user_id'] != user_id and club_row['owner_user_id'] != user_id:
        return jsonify({'error': 'Forbidden'}), 403

    db.execute('DELETE FROM club_posts WHERE id = ? AND club_id = ?', (post_id, club_id))
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (time.time(), club_id))
    db.commit()
    return jsonify({'status': 'ok', 'postId': post_id})


@app.route('/clubs/<club_id>/posts/<post_id>/reactions', methods=['PUT'])
@require_auth
def toggle_post_reaction(club_id, post_id):
    db = get_db()
    user_id = get_current_user_id()
    _, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    data = request.json or {}
    emoji = str(data.get('emoji') or '').strip()
    if not emoji:
        return jsonify({'error': 'Missing emoji'}), 400

    row = db.execute(
        'SELECT * FROM club_posts WHERE id = ? AND club_id = ?',
        (post_id, club_id)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Post not found'}), 404

    reactions = safe_json_load(row['reactions_json'], {})
    if not isinstance(reactions, dict):
        reactions = {}
    users = [str(uid) for uid in reactions.get(emoji, []) if uid]
    if user_id in users:
        users = [uid for uid in users if uid != user_id]
    else:
        users.append(user_id)
    if users:
        reactions[emoji] = users
    elif emoji in reactions:
        del reactions[emoji]

    db.execute(
        'UPDATE club_posts SET reactions_json = ? WHERE id = ?',
        (json.dumps(reactions), post_id)
    )
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (time.time(), club_id))
    db.commit()
    refreshed = db.execute('SELECT * FROM club_posts WHERE id = ?', (post_id,)).fetchone()
    return jsonify({'post': serialize_club_post_row(refreshed)})


@app.route('/clubs/<club_id>/threads', methods=['GET'])
@require_auth
def list_club_threads(club_id):
    db = get_db()
    user_id = get_current_user_id()
    _, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    book_id = str(request.args.get('bookId') or request.args.get('book_id') or '').strip()
    if book_id:
        rows = db.execute(
            '''
            SELECT *
            FROM club_threads
            WHERE club_id = ? AND book_id = ?
            ORDER BY created_at DESC
            ''',
            (club_id, book_id)
        ).fetchall()
    else:
        rows = db.execute(
            '''
            SELECT *
            FROM club_threads
            WHERE club_id = ?
            ORDER BY created_at DESC
            ''',
            (club_id,)
        ).fetchall()

    threads = [serialize_club_thread_row(db, row, include_messages=True) for row in rows]
    return jsonify({'threads': threads})


@app.route('/clubs/<club_id>/threads', methods=['POST'])
@require_auth
def create_club_thread(club_id):
    db = get_db()
    profile = current_user_profile()
    _, error_response = ensure_member_access(db, club_id, profile['user_id'])
    if error_response:
        return error_response

    data = request.json or {}
    book_id = str(data.get('bookId') or data.get('book_id') or '').strip()
    chapter_id = str(data.get('chapterId') or data.get('chapter_id') or '').strip()
    sentence_range = data.get('sentenceRange') if isinstance(data.get('sentenceRange'), dict) else {}
    if not book_id or not chapter_id:
        return jsonify({'error': 'Missing bookId/chapterId'}), 400

    sentence_start = max(0, parse_int(sentence_range.get('start'), parse_int(data.get('sentence_start'), 0)))
    sentence_end = max(sentence_start, parse_int(sentence_range.get('end'), parse_int(data.get('sentence_end'), sentence_start)))
    title = str(data.get('title') or '').strip() or None
    spoiler_boundary = _default_spoiler_boundary(data.get('spoilerBoundary') or data.get('spoiler_boundary')) or {
        'chapterId': chapter_id,
        'sentenceIndex': sentence_start,
    }
    initial_message = str(data.get('initialMessage') or data.get('initial_message') or '').strip()

    now = time.time()
    thread_id = str(data.get('id') or f"thread-{uuid.uuid4().hex[:12]}").strip()
    db.execute(
        '''
        INSERT INTO club_threads (
            id, club_id, book_id, chapter_id, sentence_start, sentence_end,
            created_at, created_by_user_id, title, spoiler_boundary_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            thread_id,
            club_id,
            book_id,
            chapter_id,
            sentence_start,
            sentence_end,
            now,
            profile['user_id'],
            title,
            json.dumps(spoiler_boundary)
        )
    )

    if initial_message:
        message_id = f"msg-{uuid.uuid4().hex[:12]}"
        db.execute(
            '''
            INSERT INTO club_thread_messages (
                id, thread_id, author_user_id, created_at, text, reactions_json
            )
            VALUES (?, ?, ?, ?, ?, '{}')
            ''',
            (message_id, thread_id, profile['user_id'], now, initial_message)
        )

    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (now, club_id))
    db.commit()
    row = db.execute('SELECT * FROM club_threads WHERE id = ? AND club_id = ?', (thread_id, club_id)).fetchone()
    return jsonify({'thread': serialize_club_thread_row(db, row, include_messages=True)})


@app.route('/clubs/<club_id>/threads/<thread_id>/messages', methods=['POST'])
@require_auth
def add_thread_message(club_id, thread_id):
    db = get_db()
    profile = current_user_profile()
    _, error_response = ensure_member_access(db, club_id, profile['user_id'])
    if error_response:
        return error_response

    thread = db.execute(
        'SELECT id FROM club_threads WHERE id = ? AND club_id = ?',
        (thread_id, club_id)
    ).fetchone()
    if not thread:
        return jsonify({'error': 'Thread not found'}), 404

    data = request.json or {}
    text = str(data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'Missing text'}), 400

    now = time.time()
    message_id = str(data.get('id') or f"msg-{uuid.uuid4().hex[:12]}").strip()
    db.execute(
        '''
        INSERT INTO club_thread_messages (
            id, thread_id, author_user_id, created_at, text, reactions_json
        )
        VALUES (?, ?, ?, ?, ?, '{}')
        ''',
        (message_id, thread_id, profile['user_id'], now, text)
    )
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (now, club_id))
    db.commit()

    row = db.execute(
        'SELECT * FROM club_thread_messages WHERE id = ? AND thread_id = ?',
        (message_id, thread_id)
    ).fetchone()
    return jsonify({'message': serialize_club_message_row(row)})


@app.route('/clubs/<club_id>/threads/<thread_id>/messages/<message_id>', methods=['DELETE'])
@require_auth
def delete_thread_message(club_id, thread_id, message_id):
    db = get_db()
    user_id = get_current_user_id()
    club_row, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    message = db.execute(
        '''
        SELECT m.author_user_id
        FROM club_thread_messages m
        JOIN club_threads t ON t.id = m.thread_id
        WHERE m.id = ? AND m.thread_id = ? AND t.club_id = ?
        ''',
        (message_id, thread_id, club_id)
    ).fetchone()
    if not message:
        return jsonify({'error': 'Message not found'}), 404

    if message['author_user_id'] != user_id and club_row['owner_user_id'] != user_id:
        return jsonify({'error': 'Forbidden'}), 403

    db.execute('DELETE FROM club_thread_messages WHERE id = ? AND thread_id = ?', (message_id, thread_id))
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (time.time(), club_id))
    db.commit()
    return jsonify({'status': 'ok', 'messageId': message_id})


@app.route('/clubs/<club_id>/threads/<thread_id>/messages/<message_id>/reactions', methods=['PUT'])
@require_auth
def toggle_thread_message_reaction(club_id, thread_id, message_id):
    db = get_db()
    user_id = get_current_user_id()
    _, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    data = request.json or {}
    emoji = str(data.get('emoji') or '').strip()
    if not emoji:
        return jsonify({'error': 'Missing emoji'}), 400

    row = db.execute(
        '''
        SELECT m.*
        FROM club_thread_messages m
        JOIN club_threads t ON t.id = m.thread_id
        WHERE m.id = ? AND m.thread_id = ? AND t.club_id = ?
        ''',
        (message_id, thread_id, club_id)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Message not found'}), 404

    reactions = safe_json_load(row['reactions_json'], {})
    if not isinstance(reactions, dict):
        reactions = {}
    users = [str(uid) for uid in reactions.get(emoji, []) if uid]
    if user_id in users:
        users = [uid for uid in users if uid != user_id]
    else:
        users.append(user_id)
    if users:
        reactions[emoji] = users
    elif emoji in reactions:
        del reactions[emoji]

    db.execute(
        'UPDATE club_thread_messages SET reactions_json = ? WHERE id = ? AND thread_id = ?',
        (json.dumps(reactions), message_id, thread_id)
    )
    db.execute('UPDATE clubs SET updated_at = ? WHERE id = ?', (time.time(), club_id))
    db.commit()
    refreshed = db.execute(
        'SELECT * FROM club_thread_messages WHERE id = ? AND thread_id = ?',
        (message_id, thread_id)
    ).fetchone()
    return jsonify({'message': serialize_club_message_row(refreshed)})


@app.route('/clubs/<club_id>/spoiler_preference', methods=['GET'])
@require_auth
def get_club_spoiler_preference(club_id):
    db = get_db()
    user_id = get_current_user_id()
    _, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    row = db.execute(
        '''
        SELECT preference
        FROM club_spoiler_preferences
        WHERE club_id = ? AND user_id = ?
        ''',
        (club_id, user_id)
    ).fetchone()
    return jsonify({'clubId': club_id, 'preference': row['preference'] if row else None})


@app.route('/clubs/<club_id>/spoiler_preference', methods=['PUT'])
@require_auth
def set_club_spoiler_preference(club_id):
    db = get_db()
    user_id = get_current_user_id()
    _, error_response = ensure_member_access(db, club_id, user_id)
    if error_response:
        return error_response

    data = request.json or {}
    preference = str(data.get('preference') or '').strip()
    if preference not in ('reveal_all',):
        return jsonify({'error': 'Invalid preference'}), 400

    now = time.time()
    db.execute(
        '''
        INSERT INTO club_spoiler_preferences (club_id, user_id, preference, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(club_id, user_id) DO UPDATE SET
            preference = excluded.preference,
            updated_at = excluded.updated_at
        ''',
        (club_id, user_id, preference, now)
    )
    db.commit()
    return jsonify({'clubId': club_id, 'preference': preference})


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
