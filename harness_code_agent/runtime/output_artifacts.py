"""Digest helpers for spilled tool output artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()
