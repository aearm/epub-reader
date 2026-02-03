# utils/translation_service.py
"""
Translation service using Google Translate via deep-translator.
Provides word and phrase translation with caching.
"""

from deep_translator import GoogleTranslator
from functools import lru_cache
import threading

# Supported languages (common ones)
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh-CN': 'Chinese (Simplified)',
    'zh-TW': 'Chinese (Traditional)',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'pl': 'Polish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'el': 'Greek',
    'he': 'Hebrew',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'fa': 'Persian',
    'uk': 'Ukrainian',
    'cs': 'Czech',
    'ro': 'Romanian',
    'hu': 'Hungarian',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ur': 'Urdu',
}


class TranslationService:
    """Thread-safe translation service with caching."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}  # (text, source, target) -> translation
        self._max_cache_size = 10000

    def translate(self, text: str, target_lang: str, source_lang: str = 'auto') -> dict:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'es', 'fr')
            source_lang: Source language code or 'auto' for detection

        Returns:
            dict with 'translation', 'source_lang', 'target_lang'
        """
        if not text or not text.strip():
            return {'translation': '', 'source_lang': source_lang, 'target_lang': target_lang}

        text = text.strip()

        # Check cache
        cache_key = (text.lower(), source_lang, target_lang)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Create translator
            translator = GoogleTranslator(source=source_lang, target=target_lang)

            # Translate
            translation = translator.translate(text)

            result = {
                'translation': translation,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'original': text
            }

            # Cache result
            with self._lock:
                if len(self._cache) >= self._max_cache_size:
                    # Clear oldest entries (simple approach: clear half)
                    keys = list(self._cache.keys())[:self._max_cache_size // 2]
                    for k in keys:
                        del self._cache[k]
                self._cache[cache_key] = result

            return result

        except Exception as e:
            return {
                'translation': text,  # Return original on error
                'source_lang': source_lang,
                'target_lang': target_lang,
                'original': text,
                'error': str(e)
            }

    def get_supported_languages(self) -> dict:
        """Return dict of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()


# Global singleton
_service = None
_service_lock = threading.Lock()


def get_translation_service() -> TranslationService:
    """Get singleton translation service instance."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = TranslationService()
    return _service


def translate_text(text: str, target_lang: str, source_lang: str = 'auto') -> dict:
    """Convenience function for translation."""
    return get_translation_service().translate(text, target_lang, source_lang)


def get_languages() -> dict:
    """Get supported languages."""
    return get_translation_service().get_supported_languages()
