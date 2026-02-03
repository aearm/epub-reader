# utils/tts_generator_pool.py

from kokoro import KPipeline
import soundfile as sf
import os
import torch
import threading
from queue import Queue
from utils.text_normalizer import normalize_text


class TTSInstance:
    """
    Single Kokoro KPipeline instance used by one thread at a time.
    This uses the SAME pipeline that works in your Jupyter test.
    """

    def __init__(self, instance_id, lang_code='a', voice='af_heart'):
        self.instance_id = instance_id
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.sample_rate = 24000
        self.in_use = False

        # Lock to guarantee internal thread safety (KPipeline is not thread-safe)
        self.lock = threading.Lock()

    def generate_audio(self, text: str):
        """Generate audio using a KPipeline instance (thread-safe)."""
        try:
            with self.lock:
                generator = self.pipeline(text, voice=self.voice)
                for i, (gs, ps, audio) in enumerate(generator):
                    return audio

        except Exception as e:
            print(f"[TTSInstance {self.instance_id}] Error: {e}")
            return None


class TTSPool:
    """
    Pool of KPipeline instances for parallel audio generation.
    Uses *multiple* pipelines, one per thread.
    """

    def __init__(self, pool_size=None, lang_code='a', voice='af_heart'):
        # Auto-detect pool size (similar logic to your ONNX version)
        if pool_size is None:
            if torch.cuda.is_available():
                pool_size = min(8, torch.cuda.device_count() * 4)
            else:
                pool_size = min(4, (os.cpu_count() or 4))

        print(f"Initializing KPipeline TTS pool with {pool_size} instances...")

        self.pool_size = pool_size
        self.instances = []
        self.available_instances = Queue()

        for i in range(pool_size):
            instance = TTSInstance(i, lang_code, voice)
            self.instances.append(instance)
            self.available_instances.put(instance)

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
            print(f"{'='*60}")

        inst = self.get_instance()
        if not inst:
            return False

        try:
            audio = inst.generate_audio(normalized_text)
            if audio is None:
                print(f"WARNING: TTS returned None for sentence {sent_id}")
                return False

            # Debug: Show audio generation result
            if "contrarian" in text.lower() or len(text) > 500:
                print(f"  → Audio generated: {len(audio)} samples")
                print(f"  → Duration: ~{len(audio)/inst.sample_rate:.2f} seconds")

            sf.write(outfile, audio, inst.sample_rate)
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
