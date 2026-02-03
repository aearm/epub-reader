# SpaCy Setup for EPUB Reader

## Installation

1. **Install SpaCy and download language model:**

```bash
# Install SpaCy (already in requirements.txt)
pip install spacy

# Download the English language model (choose one):

# Option 1: Small model (12MB) - Fast, good accuracy
python -m spacy download en_core_web_sm

# Option 2: Medium model (40MB) - Better accuracy
python -m spacy download en_core_web_md

# Option 3: Transformer model (500MB) - Best accuracy
python -m spacy download en_core_web_trf
```

2. **Or use the setup script:**

```bash
python setup_spacy.py
```

## Benefits of SpaCy

SpaCy provides state-of-the-art sentence segmentation that handles:

- ✅ Abbreviations (Dr., Mr., U.S.A., etc.)
- ✅ Decimal numbers (3.14, $19.99)
- ✅ Ellipses (...)
- ✅ Dialogue and quotes ("Hello!" she said.)
- ✅ Complex punctuation (em-dashes, smart quotes)
- ✅ Multi-sentence paragraphs
- ✅ Sentences with embedded HTML tags

## How It Works

1. **Automatic Model Selection**: The EPUB processor tries to load models in order:
   - `en_core_web_trf` (best accuracy)
   - `en_core_web_md` (balanced)
   - `en_core_web_sm` (fastest)

2. **Fallback**: If no SpaCy model is available, it falls back to regex-based splitting

3. **Performance**: SpaCy only loads when needed (lazy loading)

## Troubleshooting

If you see "No SpaCy model found":

```bash
# Manually install the small model
python -m spacy download en_core_web_sm

# Or install directly with pip
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

## Debug Output

When processing problematic sentences, you'll see debug output like:

```
============================================================
DEBUG: Processing paragraph with text: The single word for vertical...
Full text length: 329 chars
SpaCy/Regex split into 4 sentences:
  [0] The single word for vertical, 0 to 1 progress is technology.
  [1] The rapid progress of information technology...
  [2] But there is no reason why technology should be limited to computers.
  [3] Properly understood, any new and better way of doing things is technology.
============================================================
```

This helps identify sentence detection issues.