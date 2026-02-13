# utils/tts_generator_pool.py
import soundfile as sf
import os
import torch
import threading
import numpy as np
import io
import requests
from queue import Queue
from utils.text_normalizer import normalize_text
from utils.kokoro_engine import KokoroEngine


class TTSInstance:
    """
    Single Kokoro KPipeline instance used by one thread at a time.
    This uses the SAME pipeline that works in your Jupyter test.
    """

    def __init__(self, instance_id, lang_code='a', voice='af_sarah', remote_server_url=''):
        self.instance_id = instance_id
        self.remote_server_url = (remote_server_url or '').strip().rstrip('/')
        self.request_timeout_sec = max(5.0, float(os.environ.get('WORKER_TTS_SERVER_TIMEOUT_SEC', '120')))
        self.engine = None
        if not self.remote_server_url:
            self.engine = KokoroEngine(lang_code=lang_code, voice=voice)
        self.voice = self.engine.voice if self.engine else voice
        self.sample_rate = 24000
        self.in_use = False

        # Lock to guarantee internal thread safety (KPipeline is not thread-safe)
        self.lock = threading.Lock()

    def generate_audio(self, text: str):
        """Generate audio using a KPipeline instance (thread-safe)."""
        try:
            with self.lock:
                if self.remote_server_url:
                    response = requests.post(
                        f"{self.remote_server_url}/generate",
                        json={'text': text},
                        timeout=self.request_timeout_sec
                    )
                    if response.status_code >= 400:
                        print(
                            f"[TTSInstance {self.instance_id}] Remote TTS error "
                            f"{response.status_code}: {response.text[:240]}"
                        )
                        return None
                    audio, sample_rate = sf.read(
                        io.BytesIO(response.content),
                        dtype='float32',
                        always_2d=False
                    )
                    if isinstance(audio, np.ndarray) and audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    self.sample_rate = int(sample_rate) if sample_rate else self.sample_rate
                    return audio

                audio = self.engine.generate_audio(text)
                self.sample_rate = self.engine.sample_rate
                return audio

        except Exception as e:
            print(f"[TTSInstance {self.instance_id}] Error: {e}")
            return None


class TTSPool:
    """
    Pool of KPipeline instances for parallel audio generation.
    Uses *multiple* pipelines, one per thread.
    """

    def __init__(self, pool_size=None, lang_code='a', voice='af_sarah'):
        # Auto-detect pool size (similar logic to your ONNX version)
        if pool_size is None:
            if torch.cuda.is_available():
                pool_size = min(8, torch.cuda.device_count() * 4)
            else:
                pool_size = min(4, (os.cpu_count() or 4))

        print(f"Initializing KPipeline TTS pool with {pool_size} instances...")

        self.pool_size = pool_size
        self.remote_tts_server_url = (os.environ.get('WORKER_TTS_SERVER_URL') or '').strip().rstrip('/')
        self.instances = []
        self.available_instances = Queue()
        self.long_text_chunking_enabled = self._env_bool('WORKER_LONG_TEXT_CHUNKING_ENABLED', True)
        self.long_text_chunk_max_chars = max(120, int(os.environ.get('WORKER_LONG_TEXT_CHUNK_MAX_CHARS', '240')))
        self.long_text_chunk_hard_max_chars = max(
            self.long_text_chunk_max_chars,
            int(os.environ.get('WORKER_LONG_TEXT_CHUNK_HARD_MAX_CHARS', '360'))
        )
        self.long_text_chunk_gap_ms = max(0, int(os.environ.get('WORKER_LONG_TEXT_CHUNK_GAP_MS', '60')))

        for i in range(pool_size):
            instance = TTSInstance(i, lang_code, voice, remote_server_url=self.remote_tts_server_url)
            self.instances.append(instance)
            self.available_instances.put(instance)

        if self.remote_tts_server_url:
            print(f"TTS pool configured to use shared remote model server: {self.remote_tts_server_url}")
        print("TTS pool ready.")

    # --------------------------------------------------------------------------
    # Instance handling
    # --------------------------------------------------------------------------

    def get_instance(self, timeout=30):
        """Fetch a free TTS instance."""
        try:
            instance = self.available_instances.get(timeout=timeout)
            instance.in_use = True
            return instance
        except Exception:
            print("Warning: No available TTS instance (timeout).")
            return None

    def return_instance(self, instance: TTSInstance):
        """Return instance to the pool."""
        if instance:
            instance.in_use = False
            self.available_instances.put(instance)

    # --------------------------------------------------------------------------
    # Main API
    # --------------------------------------------------------------------------

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in ('0', 'false', 'no', 'off')

    def _find_chunk_split_index(self, text_window: str) -> int:
        low = max(1, int(self.long_text_chunk_max_chars * 0.55))
        for token in ('. ', '! ', '? ', '; ', ': ', ', '):
            idx = text_window.rfind(token, low)
            if idx != -1:
                return idx + len(token)

        idx = text_window.rfind(' ', low)
        if idx != -1:
            return idx + 1

        return len(text_window)

    def _split_long_text(self, text: str):
        text = (text or '').strip()
        if not text:
            return []
        if not self.long_text_chunking_enabled or len(text) <= self.long_text_chunk_hard_max_chars:
            return [text]

        chunks = []
        remaining = text
        while len(remaining) > self.long_text_chunk_hard_max_chars:
            window = remaining[:self.long_text_chunk_hard_max_chars]
            split_idx = self._find_chunk_split_index(window)
            split_idx = max(1, min(split_idx, len(window)))
            chunk = window[:split_idx].strip()
            if not chunk:
                chunk = window.strip()
                split_idx = len(window)
            chunks.append(chunk)
            remaining = remaining[split_idx:].lstrip()

        if remaining:
            chunks.append(remaining)
        return chunks

    def _stitch_chunks(self, chunk_audio, sample_rate: int):
        if len(chunk_audio) == 1:
            return chunk_audio[0]

        gap_samples = int(sample_rate * (self.long_text_chunk_gap_ms / 1000.0))
        if gap_samples <= 0:
            return np.concatenate(chunk_audio)

        gap = np.zeros(gap_samples, dtype=np.float32)
        parts = []
        for idx, item in enumerate(chunk_audio):
            parts.append(item)
            if idx < len(chunk_audio) - 1:
                parts.append(gap)
        return np.concatenate(parts)

    def generate_single_sentence(self, sentence_data, output_dir):
        """Generate <sentence_id>.wav for a given sentence."""
        sent_id = sentence_data['id']
        text = sentence_data['text']

        if not text.strip():
            return False

        os.makedirs(output_dir, exist_ok=True)
        outfile = os.path.join(output_dir, f"{sent_id}.wav")

        # Skip existing files
        if os.path.exists(outfile):
            return True

        # Normalize text for better TTS pronunciation
        normalized_text = normalize_text(text)
        chunks = self._split_long_text(normalized_text)
        if not chunks:
            return False

        # Debug: Show what we're sending to TTS
        if "contrarian" in text.lower() or len(text) > 500:
            print(f"\n{'='*60}")
            print(f"TTS DEBUG: Generating audio for sentence")
            print(f"  ID: {sent_id}")
            print(f"  Original text length: {len(text)} chars")
            print(f"  Normalized text length: {len(normalized_text)} chars")
            print(f"  First 150 chars: {normalized_text[:150]}...")
            if len(normalized_text) > 300:
                print(f"  Last 100 chars: ...{normalized_text[-100:]}")
            print(f"  Output file: {outfile}")
            print(f"  Chunk count: {len(chunks)}")
            print(f"{'='*60}")
        elif len(chunks) > 1:
            print(f"TTS split: sentence {sent_id} into {len(chunks)} chunks ({len(normalized_text)} chars)")

        inst = self.get_instance()
        if not inst:
            return False

        try:
            chunk_audio = []
            for idx, chunk_text in enumerate(chunks):
                audio = inst.generate_audio(chunk_text)
                if audio is None:
                    print(f"WARNING: TTS returned None for sentence {sent_id} chunk {idx + 1}/{len(chunks)}")
                    return False
                arr = np.asarray(audio, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    print(f"WARNING: Empty audio for sentence {sent_id} chunk {idx + 1}/{len(chunks)}")
                    return False
                chunk_audio.append(arr)

            merged_audio = self._stitch_chunks(chunk_audio, inst.sample_rate)

            # Debug: Show audio generation result
            if "contrarian" in text.lower() or len(text) > 500:
                print(f"  → Audio generated: {len(merged_audio)} samples")
                print(f"  → Duration: ~{len(merged_audio)/inst.sample_rate:.2f} seconds")

            sf.write(outfile, merged_audio, inst.sample_rate)
            return True

        except Exception as e:
            print(f"Error generating audio for sentence {sent_id}: {e}")
            return False

        finally:
            self.return_instance(inst)

    # --------------------------------------------------------------------------

    def get_pool_status(self):
        """Return pool usage information."""
        available = self.available_instances.qsize()
        in_use = self.pool_size - available
        return {"total": self.pool_size, "available": available, "in_use": in_use}
