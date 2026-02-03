# app_multithreaded.py
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import json
import hashlib
from pathlib import Path
from werkzeug.utils import secure_filename
from utils.epub_processor_v2 import EPUBProcessorV2 as EPUBProcessor
from utils.tts_generator_pool import TTSPool
from utils.reading_state import get_state_manager
from utils.translation_service import translate_text, get_languages
from utils.library_manager import get_library_manager
import threading
from queue import PriorityQueue, Empty
import time
import itertools
import shutil
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'
ALLOWED_EXTENSIONS = {'epub'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
os.makedirs('static/state', exist_ok=True)  # For reading position persistence
os.makedirs('static/library', exist_ok=True)  # For library data
os.makedirs('static/library/covers', exist_ok=True)  # For book covers

# -----------------------------------------------------------------------------
# Global state (scoped to process, not per-user)
# -----------------------------------------------------------------------------
current_book = None          # Parsed book structure + sentences
epub_processor = None
tts_pool = None              # TTSPool instance
audio_generation_queue = PriorityQueue()
audio_generation_threads = []
audio_status = {}            # sentence_id -> 'pending' | 'generating' | 'ready' | 'failed'
generation_active = False
priority_counter = itertools.count()  # Unique counter to keep queue stable
current_chapter_index = 0
current_page_sentences = []  # list of sentence_ids for currently visible page


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_book_hash(filepath: str) -> str:
    """Generate a unique hash for a book file."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


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


# -----------------------------------------------------------------------------
# Audio generation worker
# -----------------------------------------------------------------------------
def audio_generation_worker():
    """Background worker thread that generates audio files from queue."""
    global audio_status, generation_active, tts_pool

    while generation_active:
        try:
            # (priority, counter, payload)
            priority, counter, (book_id, sentence_data, output_dir) = audio_generation_queue.get(timeout=1)
            sentence_id = sentence_data['id']

            # If file already exists, just mark as ready
            audio_path = os.path.join(output_dir, f"{sentence_id}.wav")
            if os.path.exists(audio_path):
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

            if not tts_pool:
                print("TTS pool is not initialized; cannot generate audio.")
                audio_status[sentence_id] = 'failed'
                audio_generation_queue.task_done()
                continue

            success = tts_pool.generate_single_sentence(sentence_data, output_dir)

            if success:
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
            # Nothing to do, loop again (and check generation_active)
            continue
        except Exception as e:
            if generation_active:
                print(f"Worker error: {e}")
            # Don't crash the worker on a single bad sentence
            continue


def start_workers_if_needed():
    """Ensure worker threads are running, matching pool size."""
    global audio_generation_threads, generation_active, tts_pool

    if not generation_active:
        return

    # Clean dead threads
    audio_generation_threads = [t for t in audio_generation_threads if t.is_alive()]

    desired = tts_pool.pool_size if tts_pool else 4
    missing = desired - len(audio_generation_threads)

    for _ in range(max(missing, 0)):
        t = threading.Thread(target=audio_generation_worker, daemon=True)
        t.start()
        audio_generation_threads.append(t)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    """Serve main reader UI."""
    return render_template('index_async.html')


@app.route('/upload', methods=['POST'])
def upload_epub():
    """Upload an EPUB, parse it, and start background audio generation."""
    global current_book, epub_processor, audio_status, generation_active, current_chapter_index

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Identify book and audio directory
    book_id = get_book_hash(filepath)
    book_audio_dir = os.path.join(app.config['AUDIO_FOLDER'], book_id)
    os.makedirs(book_audio_dir, exist_ok=True)

    # Process EPUB
    epub_processor = EPUBProcessor(filepath)
    book_data = epub_processor.process()

    # Reset state
    audio_status = {}
    current_chapter_index = 0
    current_page_sentences = []

    current_book = {
        'id': book_id,
        'title': book_data['title'],
        'chapters': book_data['chapters'],
        'sentences': book_data['sentences'],
        'book_id': book_id
    }

    # Initialize audio status for all sentences (check for existing files)
    for sentence in book_data['sentences']:
        audio_file = os.path.join(book_audio_dir, f"{sentence['id']}.wav")
        audio_status[sentence['id']] = 'ready' if os.path.exists(audio_file) else 'pending'

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

    # Check if book exists in library
    library_manager = get_library_manager()
    book_info = library_manager.get_book(book_id)

    if not book_info:
        return jsonify({'success': False, 'error': 'Book not found in library'}), 404

    # Get the epub file path from library
    epub_path = book_info.get('file_path')

    # Verify the file exists, or try to find it (fallback for old entries)
    if not epub_path or not os.path.exists(epub_path):
        # Fallback: search in uploads folder
        upload_folder = app.config['UPLOAD_FOLDER']
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

    current_book = {
        'id': book_id,
        'title': book_data['title'],
        'chapters': book_data['chapters'],
        'sentences': book_data['sentences'],
        'book_id': book_id
    }

    # Initialize audio status (check for existing files)
    for sentence in book_data['sentences']:
        audio_file = os.path.join(book_audio_dir, f"{sentence['id']}.wav")
        audio_status[sentence['id']] = 'ready' if os.path.exists(audio_file) else 'pending'

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
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    return jsonify({'error': 'Audio not found'}), 404


@app.route('/audio/status/<sentence_id>')
def get_audio_status(sentence_id):
    """Check if audio is ready for a specific sentence."""
    status = audio_status.get(sentence_id, 'unknown')
    return jsonify({'status': status})


@app.route('/prioritize', methods=['POST'])
def prioritize_sentences(data=None):
    """
    Prioritize audio generation for specific sentences.
    Body: { "sentence_ids": [id1, id2, ...] }
    """
    global current_book

    if data is None:
        data = request.json or {}
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

    generation_active = False
    current_book = None
    epub_processor = None
    audio_status = {}
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
    global generation_active, current_book, audio_status, tts_pool

    data = request.json or {}
    book_id = data.get('book_id')

    if not current_book or current_book.get('book_id') != book_id:
        return jsonify({'error': 'Book not loaded'}), 404

    # If generation is already active, just ensure workers are running
    if generation_active:
        start_workers_if_needed()
        return jsonify({'success': True, 'message': 'Generation already active'})

    # Initialize TTS pool if needed
    if not tts_pool:
        tts_pool = TTSPool()

    # Start generation
    book_audio_dir = os.path.join(app.config['AUDIO_FOLDER'], book_id)

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
    global tts_pool

    # Preload SpaCy model
    print("Preloading SpaCy model...")
    from utils.epub_processor import get_spacy_model
    get_spacy_model()

    # Preload TTS pool
    print("Preloading TTS pool...")
    if not tts_pool:
        tts_pool = TTSPool()
    print("All models preloaded!")


if __name__ == '__main__':
    # Preload models in background thread at startup
    threading.Thread(target=preload_models, daemon=True).start()

    # Use 0.0.0.0 if you want to access from other devices on LAN
    socketio.run(app, debug=True, port=5001)
