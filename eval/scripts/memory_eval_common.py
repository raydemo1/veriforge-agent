from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from harness_code_agent import config
from harness_code_agent.agent.providers import client_scope
from harness_code_agent.agent.utils import _usage_to_dict


class RequestRecorder:
    def __init__(self, path: Path, limit: int):
        self.path = path
        self.limit = limit
        self.records: list[dict] = []
        self.label = ""

    @contextmanager
    def scope(self):
        with client_scope() as client:
            original = client.chat.completions.create

            def create(**kwargs):
                if len(self.records) >= self.limit:
                    raise RuntimeError("Evaluation request cap reached")
                profile = config.resolve_model_profile("fast")
                payload = json.dumps(kwargs["messages"], ensure_ascii=False, sort_keys=True)
                record = {
                    "index": len(self.records) + 1, "label": self.label,
                    "provider": profile.provider, "model": kwargs["model"],
                    "input_chars": len(payload), "input_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                }
                self.records.append(record)
                started = time.monotonic()
                try:
                    result = original(**kwargs)
                    record["usage"] = _usage_to_dict(result.usage)
                    record["response"] = result.choices[0].message.content or ""
                    record["finish_reason"] = result.choices[0].finish_reason
                    return result
                except Exception as exc:
                    record["error_type"] = type(exc).__name__
                    raise
                finally:
                    record["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                    with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            with patch.object(client.chat.completions, "create", side_effect=create):
                yield client

