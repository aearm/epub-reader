#!/usr/bin/env python3
"""
Setup script to download SpaCy language models.
Run this after installing requirements.txt
"""

import subprocess
import sys

def download_spacy_models():
    """Download SpaCy language models."""
    # Only download the small model by default
    # Users can manually download larger models if needed
    models = [
        "en_core_web_sm",  # Small model (12MB) - fastest, good accuracy
    ]

    print("Note: By default, we'll install the small English model (en_core_web_sm)")
    print("For better accuracy, you can manually install:")
    print("  - en_core_web_md (40MB): python -m spacy download en_core_web_md")
    print("  - en_core_web_trf (500MB): python -m spacy download en_core_web_trf")
    print()

    for model in models:
        print(f"Downloading SpaCy model: {model}")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            print(f"✅ Successfully downloaded {model}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {model}: {e}")
            print("You can manually install it with:")
            print(f"  python -m spacy download {model}")

if __name__ == "__main__":
    print("Setting up SpaCy language models...")
    print("This may take a few minutes on first run.")
    download_spacy_models()
    print("\nSetup complete! The EPUB reader is ready to use.")