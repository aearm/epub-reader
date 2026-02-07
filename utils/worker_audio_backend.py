import os
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional

from utils.tts_generator_pool import TTSPool


@dataclass
class UploadAsset:
    upload_path: str
    audio_format: str
    content_type: str
    cleanup_paths: List[str]


class WorkerAudioBackend:
    """Encapsulates worker-side audio generation + upload-format conversion."""

    SUPPORTED_UPLOAD_FORMATS = ('wav', 'mp3', 'm4b')

    def __init__(self, default_upload_format='m4b', mp3_bitrate='64k', sample_rate='24000', tts_pool_size=None):
        self.default_upload_format = self.normalize_upload_format(default_upload_format, 'm4b')
        self.mp3_bitrate = mp3_bitrate
        self.sample_rate = str(int(sample_rate))
        self.tts_pool_size = int(tts_pool_size) if tts_pool_size else None
        self._tts_pool = None
        self._tts_pool_lock = threading.Lock()

    @classmethod
    def normalize_upload_format(cls, value: str, default='m4b') -> str:
        fmt = (value or '').strip().lower()
        if fmt in cls.SUPPORTED_UPLOAD_FORMATS:
            return fmt
        fallback = (default or '').strip().lower()
        if fallback in cls.SUPPORTED_UPLOAD_FORMATS:
            return fallback
        return 'wav'

    @classmethod
    def infer_format_from_url(cls, url: str) -> str:
        lower = (url or '').strip().lower()
        if lower.endswith('.mp3'):
            return 'mp3'
        if lower.endswith('.m4b') or lower.endswith('.m4a'):
            return 'm4b'
        return 'wav'

    @staticmethod
    def content_type_for_format(audio_format: str) -> str:
        if audio_format == 'mp3':
            return 'audio/mpeg'
        if audio_format == 'm4b':
            return 'audio/mp4'
        return 'audio/wav'

    @property
    def pool_size(self) -> int:
        with self._tts_pool_lock:
            if self._tts_pool:
                return int(getattr(self._tts_pool, 'pool_size', 0) or 0)
        return 0

    def preload(self):
        self._ensure_tts_pool()

    def reset_pool(self):
        with self._tts_pool_lock:
            self._tts_pool = None

    def generate_sentence_wav(self, sentence_id: str, text: str, output_dir: str, sentence_index=0) -> Optional[str]:
        if not (text or '').strip():
            return None

        os.makedirs(output_dir, exist_ok=True)
        wav_path = os.path.join(output_dir, f'{sentence_id}.wav')
        if os.path.exists(wav_path):
            return wav_path

        try:
            pool = self._ensure_tts_pool()
            sentence_data = {
                'id': sentence_id,
                'text': text,
                'sentence_index': sentence_index
            }
            success = pool.generate_single_sentence(sentence_data, output_dir)
            if not success:
                return None
            return wav_path if os.path.exists(wav_path) else None
        except Exception as e:
            print(f'Audio backend generation failed for {sentence_id}: {e}')
            self.reset_pool()
            return None

    def build_upload_asset(self, wav_path: str, requested_format: str) -> UploadAsset:
        upload_format = self.normalize_upload_format(requested_format, self.default_upload_format)
        upload_path = wav_path
        cleanup_paths = [wav_path]

        if upload_format in ('mp3', 'm4b'):
            converted_path = self._convert_from_wav(wav_path, upload_format)
            if converted_path:
                upload_path = converted_path
                cleanup_paths.append(converted_path)
            else:
                upload_format = 'wav'
                upload_path = wav_path

        unique_cleanup = []
        seen = set()
        for path in cleanup_paths:
            key = os.path.abspath(path)
            if key in seen:
                continue
            seen.add(key)
            unique_cleanup.append(path)

        return UploadAsset(
            upload_path=upload_path,
            audio_format=upload_format,
            content_type=self.content_type_for_format(upload_format),
            cleanup_paths=unique_cleanup
        )

    @staticmethod
    def cleanup_paths(paths):
        for path in paths or []:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f'Cleanup skipped for {path}: {e}')

    def _ensure_tts_pool(self):
        if self._tts_pool:
            return self._tts_pool
        with self._tts_pool_lock:
            if not self._tts_pool:
                self._tts_pool = TTSPool(pool_size=self.tts_pool_size)
            return self._tts_pool

    def _convert_from_wav(self, wav_path: str, target_format: str) -> Optional[str]:
        base, _ = os.path.splitext(wav_path)
        out_path = f'{base}.{target_format}'
        if os.path.exists(out_path):
            return out_path

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', wav_path,
            '-vn', '-ac', '1',
            '-ar', self.sample_rate
        ]
        if target_format == 'mp3':
            ffmpeg_cmd += ['-b:a', self.mp3_bitrate, out_path]
        elif target_format == 'm4b':
            ffmpeg_cmd += ['-c:a', 'aac', '-b:a', self.mp3_bitrate, '-movflags', '+faststart', out_path]
        else:
            return None

        try:
            subprocess.run(ffmpeg_cmd, check=True, timeout=60)
            return out_path if os.path.exists(out_path) else None
        except Exception as e:
            print(f'Audio conversion failed ({target_format}); falling back to wav: {e}')
            return None
