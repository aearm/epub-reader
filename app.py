from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import hashlib
from pathlib import Path
from werkzeug.utils import secure_filename
from utils.epub_processor import EPUBProcessor
from utils.tts_generator import TTSGenerator
import shutil

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'
ALLOWED_EXTENSIONS = {'epub'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Global variables to store current book data
current_book = None
epub_processor = None
tts_generator = TTSGenerator()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_book_hash(filepath):
    """Generate unique hash for book."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_epub():
    global current_book, epub_processor

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Generate unique book ID
        book_id = get_book_hash(filepath)
        book_audio_dir = os.path.join(app.config['AUDIO_FOLDER'], book_id)

        # Process EPUB
        epub_processor = EPUBProcessor(filepath)
        book_data = epub_processor.process()

        # Check if audio already exists
        audio_exists = os.path.exists(book_audio_dir) and \
                      os.path.exists(os.path.join(book_audio_dir, 'sentences.json'))

        if not audio_exists:
            # Create audio directory
            os.makedirs(book_audio_dir, exist_ok=True)

            # Generate TTS for all sentences
            print(f"Generating audio for {len(book_data['sentences'])} sentences...")
            audio_mapping = tts_generator.generate_book_audio(
                book_data['sentences'],
                book_audio_dir
            )

            # Save sentence mapping
            with open(os.path.join(book_audio_dir, 'sentences.json'), 'w') as f:
                json.dump(audio_mapping, f)
        else:
            # Load existing mapping
            with open(os.path.join(book_audio_dir, 'sentences.json'), 'r') as f:
                audio_mapping = json.load(f)
            print(f"Using existing audio for book {book_id}")

        current_book = {
            'id': book_id,
            'title': book_data['title'],
            'chapters': book_data['chapters'],
            'sentences': book_data['sentences'],
            'audio_mapping': audio_mapping,
            'book_id': book_id  # Add explicit book_id for frontend
        }

        return jsonify({
            'success': True,
            'book': {
                'id': book_id,
                'title': current_book['title'],
                'total_chapters': len(current_book['chapters']),
                'total_sentences': len(current_book['sentences'])
            }
        })

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/book/content')
def get_book_content():
    if not current_book:
        return jsonify({'error': 'No book loaded'}), 404

    return jsonify({
        'title': current_book['title'],
        'chapters': current_book['chapters'],
        'sentences': current_book['sentences'],
        'audio_mapping': current_book['audio_mapping'],
        'book_id': current_book['book_id']  # Include book_id
    })

@app.route('/audio/<book_id>/<sentence_id>')
def get_audio(book_id, sentence_id):
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], book_id, f'{sentence_id}.wav')
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    return jsonify({'error': 'Audio not found'}), 404

@app.route('/clear')
def clear_book():
    global current_book, epub_processor
    current_book = None
    epub_processor = None
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)