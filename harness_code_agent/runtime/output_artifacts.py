"""Digest helpers for spilled tool output artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
