# utils/tts_generator.py
import soundfile as sf
import os
from utils.kokoro_engine import KokoroEngine


class TTSGenerator:
    """
    Simple, single-instance Kokoro TTS generator.

    - generate_book_audio: sequentially generate audio for a whole book.
    - generate_audio_stream: generator yielding chunks for streaming.
    """

    def __init__(self, lang_code='a', voice='af_sarah'):
        self.engine = KokoroEngine(lang_code=lang_code, voice=voice)
        self.voice = self.engine.voice
        self.sample_rate = self.engine.sample_rate

    # -------------------------------------------------------------------------
    # Batch generation
    # -------------------------------------------------------------------------
    def generate_book_audio(self, sentences, output_dir: str):
        """
        Generate audio files for every sentence in `sentences`.

        sentences: [{ "id": str, "text": str, "sentence_index": int }, ...]
        Output: dict mapping sentence_id -> { file, text, index }
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_mapping = {}
        total = len(sentences)

        for idx, sentence in enumerate(sentences):
            sent_id = sentence['id']
            text = sentence['text']

            if not text.strip():
                continue

            audio_file = os.path.join(output_dir, f"{sent_id}.wav")

            if not os.path.exists(audio_file):
                print(f"Generating audio {idx + 1}/{total}: {text[:50]}...")
                try:
                    audio = self._generate_single_audio(text)
                    if audio is not None:
                        sf.write(audio_file, audio, self.sample_rate)
                except Exception as e:
                    print(f"Error generating audio for sentence {sent_id}: {e}")

            if os.path.exists(audio_file):
                audio_mapping[sent_id] = {
                    'file': f"{sent_id}.wav",
                    'text': text,
                    'index': sentence.get('sentence_index', idx)
                }

        return audio_mapping

    def _generate_single_audio(self, text: str):
        """Generate audio for a single sentence."""
        try:
            audio = self.engine.generate_audio(text)
            self.sample_rate = self.engine.sample_rate
            return audio
        except Exception as e:
            print(f"Error in TTS generation: {e}")
            return None

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------
    def generate_audio_stream(self, text: str):
        """Yield audio chunks for streaming playback."""
        try:
            audio = self.engine.generate_audio(text)
            if audio is not None:
                self.sample_rate = self.engine.sample_rate
                yield audio
        except Exception as e:
            print(f"Error in TTS stream generation: {e}")
            yield None
