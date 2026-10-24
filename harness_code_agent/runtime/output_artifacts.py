"""Verified transfer of workspace-local tool output artifacts."""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def copy_artifact(source_root: Path, target_root: Path, reference: dict) -> None:
    relative = Path(reference["path"])
    if relative.is_absolute() or relative.parts[:2] != (".harness", "observations"):
        raise ValueError("Invalid tool output artifact reference")
    source = (source_root / relative).resolve()
    target = (target_root / relative).resolve()
    if not source.is_relative_to(source_root.resolve()) or not target.is_relative_to(target_root.resolve()):
        raise ValueError("Tool output artifact escapes its workspace")
    expected = reference.get("sha256")
    if not source.is_file() or (expected and file_digest(source) != expected):
        raise ValueError(f"Tool output artifact is missing or damaged: {relative}")
    if source == target:
        return
    if target.exists():
        if file_digest(source) == file_digest(target):
            return
        if not reference.get("append_only"):
            raise ValueError(f"Tool output artifact collision: {relative}")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            remaining = target.stat().st_size
            while remaining:
                chunk = stream.read(min(65536, remaining))
                if not chunk:
                    raise ValueError(f"Tool output log shrank: {relative}")
                digest.update(chunk)
                remaining -= len(chunk)
        if digest.hexdigest() != file_digest(target):
            raise ValueError(f"Tool output log was rewritten: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            while chunk := incoming.read(65536):
                outgoing.write(chunk)
                digest.update(chunk)
        if (expected and digest.hexdigest() != expected) or file_digest(temporary) != digest.hexdigest():
            raise ValueError(f"Tool output artifact changed during transfer: {relative}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
