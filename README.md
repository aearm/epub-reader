# EPUB Reader with Text-to-Speech

A web-based EPUB reader that converts text to speech using the Kokoro TTS model. Click any sentence to hear it read aloud with support for auto-play and playback controls.

## Features

- **EPUB Rendering**: Display EPUB files with original styling and formatting
- **Sentence-Level TTS**: Click any sentence to hear it read aloud
- **Audio Pre-generation**: Automatically generates and caches audio for all sentences on first upload
- **Auto-play Mode**: Continuous reading with automatic progression through sentences
- **Playback Controls**: Play/pause, stop, next/previous sentence, speed adjustment
- **Visual Feedback**: Highlights currently playing sentence
- **Chapter Navigation**: Easy navigation through book chapters

## Installation

1. Clone the repository:
```bash
cd epub_reader
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the Kokoro model properly set up (it should work if your voice.py file works)

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload an EPUB file using the "Upload EPUB" button

4. Wait for the initial processing (this generates audio for all sentences and caches them)

5. Once loaded:
   - Click any sentence to hear it read
   - Use playback controls at the bottom
   - Enable auto-play for continuous reading
   - Navigate chapters using the sidebar
   - Adjust playback speed as needed

## How It Works

1. **EPUB Processing**: When you upload an EPUB file, the system:
   - Extracts all chapters and content
   - Splits text into individual sentences
   - Preserves original HTML structure and styling

2. **Audio Generation**: On first upload:
   - Creates a unique folder for the book (based on file hash)
   - Generates audio for each sentence using Kokoro TTS
   - Saves audio files with unique IDs
   - Creates a mapping file linking sentences to audio

3. **Playback**: During reading:
   - Clicking a sentence triggers its corresponding audio file
   - Auto-play mode automatically progresses through sentences
   - Visual highlighting shows the current sentence
   - Playback speed can be adjusted in real-time

## File Structure

```
epub_reader/
├── app.py                 # Flask backend
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   ├── js/
│   │   └── app.js        # Frontend logic
│   ├── audio/            # Generated audio files (organized by book)
│   └── uploads/          # Uploaded EPUB files
├── templates/
│   └── index.html        # Main HTML template
├── utils/
│   ├── epub_processor.py # EPUB parsing and processing
│   └── tts_generator.py  # Kokoro TTS integration
└── requirements.txt      # Python dependencies
```

## Audio Caching

- Audio files are generated once per book and cached
- Each book gets a unique ID based on its content hash
- Re-uploading the same book uses existing audio files
- Audio files are stored in `static/audio/{book_id}/`

## Supported Features

- [x] EPUB file upload and parsing
- [x] Sentence-level text extraction
- [x] Audio generation with Kokoro TTS
- [x] Click-to-play sentences
- [x] Auto-play mode
- [x] Playback controls (play/pause, stop, next/prev)
- [x] Speed adjustment (0.5x to 2.0x)
- [x] Sentence highlighting
- [x] Chapter navigation
- [x] Audio caching and reuse

## Notes

- Large EPUB files may take time to process initially (audio generation)
- Audio files are WAV format for best quality
- The system preserves the original EPUB styling as much as possible
- Sentence detection uses basic punctuation rules (can be improved)

## Troubleshooting

1. **Kokoro not working**: Ensure the Kokoro model is properly installed and your voice.py works
2. **Audio not playing**: Check browser console for errors, ensure audio files are generated
3. **Slow initial load**: Audio generation takes time, especially for large books
4. **Sentence detection issues**: The current implementation uses simple regex; complex formatting may cause issues