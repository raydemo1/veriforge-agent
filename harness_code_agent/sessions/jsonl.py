"""JSON Lines I/O for append-only session records."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_records(path: Path, *, skip_invalid: bool = False) -> Iterator[Any]:
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n") and not skip_invalid:
                    raise


def append_record(path: Path, record: dict, *, durable: bool = False) -> None:
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                # An interrupted append can leave half a record. Only inspect
                # and repair the final line, leaving earlier records intact.
                stream.seek(0)
                while line := stream.readline():
                    if line.endswith(b"\n"):
                        continue
                    try:
                        json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        stream.truncate(stream.tell() - len(line))
                    else:
                        stream.write(b"\n")
                    break
        stream.write(payload)
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
