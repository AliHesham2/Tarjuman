"""DeepL translator — high quality, 500K chars/month free.

DeepL bills *characters*, not requests.  The engine reads the provider's
quota unit from QuotaAwareRouter and charges ``len(text)`` accordingly.
"""

from __future__ import annotations

import logging
import time
from xml.sax.saxutils import escape, unescape

from src.config.providers import TranslationProviderConfig
from src.translation.base import (
    TranslationProvider,
    TranslationRequest,
    TranslationResult,
    module_available,
)

logger = logging.getLogger(__name__)

# DeepL language codes: source and target differ for English (EN vs EN-US).
SOURCE_LANG_CODES = {"ar": "AR", "en": "EN"}
TARGET_LANG_CODES = {"ar": "AR", "en": "EN-US"}

# DeepL free-tier keys end with ":fx"
FREE_API_HOST = "https://api-free.deepl.com"
PRO_API_HOST = "https://api.deepl.com"


class DeepLProvider(TranslationProvider):
    supports_context = False  # plain MT API — no conversation context
    is_local = False

    def __init__(self, config: TranslationProviderConfig):
        self.name = "deepl"
        self.cfg = config
        self._translator = None

    def is_configured(self) -> bool:
        return bool(self.cfg.api_key)

    def is_available(self) -> bool:
        if not module_available("deepl"):
            logger.warning("DeepL skipped: the 'deepl' package is not installed")
            return False
        return True

    def _get_translator(self):
        if self._translator is None:
            import deepl
            host = FREE_API_HOST if self.cfg.api_key.endswith(":fx") else PRO_API_HOST
            self._translator = deepl.Translator(
                self.cfg.api_key, server_url=host
            )
        return self._translator

    @staticmethod
    def _wrap_terms(text: str, terms: dict) -> str:
        """Swap each "__T0__" placeholder for <keep>API</keep> (XML-escaped).

        DeepL translates or drops ordinary placeholder tokens, but leaves the
        content of an ignored XML tag exactly as it is.
        """
        text = escape(text)
        for placeholder, original in terms.items():
            text = text.replace(placeholder, f"<keep>{escape(original)}</keep>")
        return text

    @staticmethod
    def _unwrap_terms(text: str) -> str:
        return unescape(text.replace("<keep>", "").replace("</keep>", ""))

    def translate(self, request: TranslationRequest) -> TranslationResult:
        kwargs = {"target_lang": TARGET_LANG_CODES.get(request.target_lang, "AR")}
        source = SOURCE_LANG_CODES.get(request.source_lang)
        if source:
            kwargs["source_lang"] = source

        text = request.text
        keep_tags = bool(request.protected_terms)
        if keep_tags:
            text = self._wrap_terms(text, request.protected_terms)
            kwargs.update(tag_handling="xml", ignore_tags=["keep"])

        start = time.monotonic()
        result = self._get_translator().translate_text(text, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000.0
        translated = result.text or ""
        if keep_tags:
            translated = self._unwrap_terms(translated)
        return TranslationResult(
            translated=translated.strip(),
            provider=self.name,
            latency_ms=latency_ms,
        )
