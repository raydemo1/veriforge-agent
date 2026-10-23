from __future__ import annotations

import re

_SECRETS = re.compile(
    r'''(?im)\b(?:api[_-]?key|(?:access|refresh)[_-]?token|password|secret)\b["']?\s*[:=]\s*'''
    r'''(?:"[^"\n]+"|'[^'\n]+'|[^\s,;]+)'''
    r"|\bsk-[A-Za-z0-9_-]{20,}\b"
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?(?:-----END [^-]+-----|\Z)"
)


def contains_secret(text: str) -> bool:
    return bool(_SECRETS.search(text))


def redact_secrets(text: str, *, preserve_length: bool = False) -> str:
    return _SECRETS.sub(lambda match: "*" * len(match.group()) if preserve_length else "[REDACTED]", text)
