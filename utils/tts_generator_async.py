# utils/tts_generator_async.py
from kokoro import KPipeline
import soundfile as sf
import os
import threading


class AsyncTTSGenerator:
    """
    Thread-safe wrapper around a single Kokoro TTS pipeline.
    Useful if you want a simple async generator without a pool.
    """

    def __init__(self, lang_code='a', voice='af_heart'):
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.sample_rate = 24000
        self.lock = threading.Lock()

    def generate_single_sentence(self, sentence_data, output_dir: str) -> bool:
        """
        Generate audio for a single sentence.

        sentence_data: { "id": str, "text": str, ... }
        Output file: <output_dir>/<id>.wav
        """
        sent_id = sentence_data['id']
        text = sentence_data['text']

        if not text.strip():
            return False

        os.makedirs(output_dir, exist_ok=True)
        audio_file = os.path.join(output_dir, f"{sent_id}.wav")

        # Already generated
        if os.path.exists(audio_file):
            return True

        try:
            # Kokoro pipeline is not inherently thread-safe, so guard with lock
            with self.lock:
                audio = self._generate_audio(text)

            if audio is None:
                return False

            sf.write(audio_file, audio, self.sample_rate)
            return True

        except Exception as e:
            print(f"Error generating audio for sentence {sent_id}: {e}")
            return False

    def _generate_audio(self, text: str):
        """Internal Kokoro generator call."""
        try:
            generator = self.pipeline(text, voice=self.voice)
            for i, (gs, ps, audio) in enumerate(generator):
                return audio
        except Exception as e:
            print(f"Error in TTS generation: {e}")
            return None

    @staticmethod
    def check_audio_exists(sentence_id: str, output_dir: str) -> bool:
        """Utility to check if <sentence_id>.wav exists."""
        audio_file = os.path.join(output_dir, f"{sentence_id}.wav")
        return os.path.exists(audio_file)
