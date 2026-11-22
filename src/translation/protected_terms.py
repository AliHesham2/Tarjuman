"""Protected Terms — placeholder system for Arabic technical jargon.

"Deploy the API on Kubernetes" → translate → "نشر الـ API على Kubernetes"
Without protection, translators mangle "API" and "Kubernetes".
Pattern from references/Live-Translate, term list extended for Arabic
tech meetings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PROTECTED_TERMS = [
    # Cloud & DevOps
    "AWS", "GCP", "Azure", "Kubernetes", "Docker", "Terraform",
    "CI/CD", "DevOps", "MLOps", "GitOps", "Helm", "Ansible",
    "Lambda", "EC2", "S3", "EKS", "GKE", "AKS",

    # Software
    "API", "SDK", "CLI", "REST", "GraphQL", "gRPC", "WebSocket",
    "OAuth", "JWT", "CORS", "HTTP", "HTTPS", "DNS", "CDN",
    "PR", "MR", "Git", "GitHub", "GitLab", "Bitbucket",

    # AI/ML
    "LLM", "GPU", "CPU", "VRAM", "RAG", "RLHF", "LoRA",
    "PyTorch", "TensorFlow", "Whisper", "GPT", "Claude", "Gemini",
    "Transformer", "Embedding", "Tokenizer", "Inference",

    # Agile/PM
    "Sprint", "Backlog", "Scrum", "Kanban", "Standup", "Retro",
    "MVP", "KPI", "OKR", "SLA", "SLO", "P0", "P1",

    # Companies
    "Google", "Microsoft", "Amazon", "Apple", "Meta", "OpenAI",
    "Anthropic", "Stripe", "Twilio", "Vercel", "Cloudflare",

    # Databases & infra
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Kafka", "Elasticsearch",
    "Nginx", "Linux", "Windows", "Python", "JavaScript", "TypeScript",
]

# Words that are also ordinary English: protected only when written exactly
# like this ("Apple", never "apple"; "REST", never "rest"). Whisper writes
# technical terms in any case ("kubernetes", "api"), so everything else is
# matched case-insensitively and put back in its canonical spelling.
_CASE_SENSITIVE = {
    "Meta", "Apple", "Windows", "Python", "Google", "Amazon", "Microsoft",
    "Azure", "Sprint", "Retro", "Standup", "Backlog", "Lambda", "Transformer",
    "Inference", "Embedding", "Tokenizer", "Whisper", "Claude", "Gemini",
    "Helm", "Stripe", "REST", "RAG", "PR", "MR",
}
_CANONICAL = {t.lower(): t for t in PROTECTED_TERMS if t not in _CASE_SENSITIVE}

# Longest-first so "HTTPS" matches before "HTTP", "PostgreSQL" before "SQL"
_LOOSE = sorted(_CANONICAL.values(), key=len, reverse=True)
_STRICT = sorted(_CASE_SENSITIVE & set(PROTECTED_TERMS), key=len, reverse=True)

# Lookarounds keep neighbouring letters/digits from merging into a term.
_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:(?i:(" + "|".join(re.escape(t) for t in _LOOSE) + r"))|("
    + "|".join(re.escape(t) for t in _STRICT) + r"))(?![A-Za-z0-9_])"
)

# A placeholder is __T<number>__.  Some translators insert spaces or change
# case; the restore pass accepts "__ T 10 __" and "__t10__".
# The number itself may also be split ("__T1 0__") — normalise spaces after.
_PLACEHOLDER_RE = re.compile(r"__\s*T\s*(\d(?:\s*\d){0,5})\s*__", re.IGNORECASE)


@dataclass
class ProtectedText:
    """Text with terms swapped for placeholders, plus the restore map."""
    safe_text: str
    replacements: dict[str, str] = field(default_factory=dict)  # "__T0__" -> "API"


def _placeholder_index(match: re.Match) -> int:
    return int(re.sub(r"\s+", "", match.group(1)))


def protect(text: str) -> ProtectedText:
    """Replace every protected term with a __Tn__ placeholder.

    Placeholders already present in the input are never reused, so a
    literal "__T0__" cannot collide with a protected term.
    """
    replacements: dict[str, str] = {}
    used_indices = {_placeholder_index(m) for m in _PLACEHOLDER_RE.finditer(text)}
    next_index = 0

    def _sub(match: re.Match) -> str:
        nonlocal next_index
        while next_index in used_indices:
            next_index += 1
        index = next_index
        next_index += 1
        used_indices.add(index)
        placeholder = f"__T{index}__"
        loose, strict = match.group(1), match.group(2)
        replacements[placeholder] = (
            _CANONICAL[loose.lower()] if loose is not None else strict
        )
        return placeholder

    safe = _PATTERN.sub(_sub, text)
    return ProtectedText(safe_text=safe, replacements=replacements)


def restore(text: str, replacements: dict[str, str]) -> str:
    """Put the protected terms back after translation.

    A single regex consumes the *whole* placeholder, which fixes the old
    prefix bug where restoring "__T10__" with a map containing "__T1__"
    produced "TERM10__".  Spacing/case damage from translators is tolerated.
    """
    if not replacements:
        return text

    def _sub(match: re.Match) -> str:
        index = _placeholder_index(match)
        return replacements.get(f"__T{index}__", match.group(0))

    restored = _PLACEHOLDER_RE.sub(_sub, text)

    # A placeholder that was not in the map is either literal user text or a
    # term the translator moved.  Do not fail the translation; just make it
    # visible in debug logs.
    if _PLACEHOLDER_RE.search(restored):
        logger.debug(
            "Translation still contains placeholder-like text (%d chars)",
            len(restored),
        )

    return restored
