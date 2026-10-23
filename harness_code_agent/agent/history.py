"""Immutable messages and explicit request-context versions."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ContextEpoch:
    id: str
    reason: str
    contract: str
    base: tuple[str, ...]


class ContextHistory:
    def __init__(self):
        self.epoch: ContextEpoch | None = None
        self._messages: tuple[str, ...] = ()
        self._sequences: tuple[int, ...] = ()
        self._next_sequence = 0
        self._raw_messages: tuple[str, ...] = ()
        self._decoded: tuple[dict, ...] = ()

    def snapshot(self) -> list[dict]:
        return copy.deepcopy(list(self._decoded))

    def raw_snapshot(self) -> list[dict]:
        return [json.loads(message) for message in self._raw_messages]

    @property
    def sequences(self) -> tuple[int, ...]:
        return self._sequences

    def append(self, message: dict, *, journal=None, metadata=None) -> int:
        frozen = json.dumps(message, ensure_ascii=False)
        sequence = self._next_sequence + 1
        if journal is not None:
            sequence = journal.append_message(
                json.loads(frozen), metadata=metadata
            ).sequence
        self._next_sequence = sequence
        self._messages += (frozen,)
        self._decoded += (json.loads(frozen),)
        self._raw_messages += (frozen,)
        self._sequences += (sequence,)
        return sequence

    def start_epoch(
        self, reason: str, messages: list[dict], contract: dict, *, journal=None
    ):
        if not tool_pairs_closed(messages):
            raise ValueError("Cannot create an epoch with unmatched tool calls")
        frozen = tuple(json.dumps(message, ensure_ascii=False) for message in messages)
        epoch = ContextEpoch(uuid.uuid4().hex, reason, fingerprint(contract), frozen)
        # Reuse identities of retained messages, including duplicate messages in order.
        available: dict[str, list[int]] = {}
        sources = zip(self._messages, self._sequences)
        if journal is not None:
            sources = (
                (
                    json.dumps(entry.payload["message"], ensure_ascii=False),
                    entry.sequence,
                )
                for entry in journal.read()
                if entry.kind == "message"
            )
        for message, sequence in sources:
            available.setdefault(message, []).append(sequence)
        sequences = [
            available[item].pop() if available.get(item) else 0
            for item in reversed(frozen)
        ]
        sequences.reverse()
        if journal is not None:
            journal.append(
                "context_epoch",
                {
                    "epoch_id": epoch.id,
                    "reason": reason,
                    "contract_hash": epoch.contract,
                    "base_messages": copy.deepcopy(messages),
                    "source_sequences": sequences,
                },
            )
        self.epoch = epoch
        self._messages = frozen
        self._decoded = tuple(json.loads(message) for message in frozen)
        self._sequences = tuple(sequences)
        return epoch


def tool_pairs_closed(messages: list[dict]) -> bool:
    pending: set[str] = set()
    for message in messages:
        if pending and message.get("role") != "tool":
            return False
        for call in message.get("tool_calls") or []:
            pending.add(str(call.get("id")))
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id"))
            if call_id not in pending:
                return False
            pending.remove(call_id)
    return not pending
