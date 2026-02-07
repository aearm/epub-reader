"""
Kokoro engine adapter.

Supports either:
- `kokoro` (KPipeline API), or
- `kokoro_onnx` (ONNX API)

If ONNX is used and model assets are missing, this module can auto-download
`kokoro-v1.0.onnx` and `voices-v1.0.bin`.
"""

import os
import threading
from pathlib import Path

import requests

from utils.text_normalizer import normalize_text

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

DEFAULT_MODEL_DIR = os.environ.get("KOKORO_MODEL_DIR", "static/models")
DEFAULT_MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", str(Path(DEFAULT_MODEL_DIR) / "kokoro-v1.0.onnx"))
DEFAULT_VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", str(Path(DEFAULT_MODEL_DIR) / "voices-v1.0.bin"))
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_sarah")
DEFAULT_LANG = os.environ.get("KOKORO_LANG", "en-us")
DEFAULT_BACKEND = os.environ.get("KOKORO_BACKEND", "onnx").strip().lower()

_download_lock = threading.Lock()


def _download_file(url: str, output_path: str):
    """Download a file atomically."""
    tmp_path = f"{output_path}.part"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, output_path)


def ensure_onnx_assets(model_path: str = DEFAULT_MODEL_PATH, voices_path: str = DEFAULT_VOICES_PATH):
    """Ensure ONNX model/voice assets exist; download if missing."""
    model_path = str(model_path)
    voices_path = str(voices_path)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(voices_path).parent.mkdir(parents=True, exist_ok=True)

    with _download_lock:
        if not os.path.exists(model_path):
            print(f"Downloading Kokoro ONNX model to {model_path} ...")
            _download_file(MODEL_URL, model_path)
        if not os.path.exists(voices_path):
            print(f"Downloading Kokoro voices to {voices_path} ...")
            _download_file(VOICES_URL, voices_path)

    return model_path, voices_path


class KokoroEngine:
    """Unified API for generating audio with Kokoro."""

    def __init__(self, lang_code: str = "a", voice: str = DEFAULT_VOICE):
        self.voice = voice or DEFAULT_VOICE
        self.sample_rate = 24000
        self.backend = None
        self._engine = None
        self._lang_code = lang_code
        self._onnx_lang = DEFAULT_LANG
        self._available_voices = None

        onnx_first = DEFAULT_BACKEND in ("onnx", "auto")
        attempts = [self._init_onnx_backend, self._init_pipeline_backend] if onnx_first else [
            self._init_pipeline_backend, self._init_onnx_backend
        ]

        last_error = None
        for initializer in attempts:
            try:
                initializer()
                return
            except Exception as e:
                last_error = e
                print(f"Kokoro backend init failed ({initializer.__name__}): {e}")

        raise RuntimeError(
            "Failed to initialize Kokoro engine. Ensure `kokoro_onnx` assets are reachable "
            "or configure valid local model paths."
        ) from last_error

    def _init_pipeline_backend(self):
        from kokoro import KPipeline  # type: ignore

        self._engine = KPipeline(lang_code=self._lang_code)
        self.backend = "pipeline"

    def _init_onnx_backend(self):
        from kokoro_onnx import Kokoro  # type: ignore

        model_path, voices_path = ensure_onnx_assets()
        self._engine = Kokoro(model_path, voices_path)
        self.backend = "onnx"
        self._available_voices = set(self._engine.get_voices())

        if self.voice not in self._available_voices:
            if "af_sarah" in self._available_voices:
                print(f"Voice '{self.voice}' not found in ONNX voices; using 'af_sarah'.")
                self.voice = "af_sarah"
            elif self._available_voices:
                fallback = sorted(self._available_voices)[0]
                print(f"Voice '{self.voice}' not found in ONNX voices; using '{fallback}'.")
                self.voice = fallback

    def generate_audio(self, text: str):
        """Generate audio samples for text."""
        normalized_text = normalize_text(text or "")
        if not normalized_text.strip():
            return None

        if self.backend == "pipeline":
            try:
                generator = self._engine(normalized_text, voice=self.voice)
                for _, _, audio in generator:
                    return audio
            except Exception as e:
                # Pipeline backend can fail if model downloads are unavailable at runtime.
                print(f"Kokoro pipeline generation failed, switching to ONNX: {e}")
                self._init_onnx_backend()
                return self.generate_audio(text)
            return None

        if self.backend == "onnx":
            audio, sample_rate = self._engine.create(
                normalized_text,
                voice=self.voice,
                speed=1.0,
                lang=self._onnx_lang
            )
            self.sample_rate = sample_rate
            return audio

        return None
