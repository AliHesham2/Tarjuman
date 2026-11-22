"""Windows Credential Store-backed API key storage (keyring / DPAPI).

.env is for development only; the packaged app stores keys here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SERVICE_NAME = "liveTranslate"
_unavailable_warned = False


class CredentialStore:
    """Thin keyring wrapper — one credential per provider.

    Keyring is optional at runtime: a missing backend must never crash the
    settings dialog or provider startup. Callers can fall back to an
    in-process environment variable for the current session.
    """

    def __init__(self, service: str = SERVICE_NAME):
        self.service = service

    def set_key(self, provider: str, key: str) -> bool:
        try:
            import keyring
            keyring.set_password(self.service, provider, key)
        except Exception:
            logger.warning("Credential store unavailable for %s", provider, exc_info=True)
            return False
        logger.info("Credential stored for %s", provider)
        return True

    def get_key(self, provider: str) -> str | None:
        global _unavailable_warned
        try:
            import keyring
            return keyring.get_password(self.service, provider)
        except Exception:
            if not _unavailable_warned:
                _unavailable_warned = True
                logger.warning("Credential store unavailable — using .env keys only")
                logger.debug("Credential store error", exc_info=True)
            return None

    def delete_key(self, provider: str) -> bool:
        try:
            import keyring
            keyring.delete_password(self.service, provider)
            return True
        except Exception:
            logger.debug("Could not delete credential for %s", provider, exc_info=True)
            return False
