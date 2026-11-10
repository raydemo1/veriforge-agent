"""Todo list model shared by the update_todo tool and agent runtime state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TODO_STATUSES = ("pending", "in_progress", "completed", "cancelled")
MAX_TODO_ITEMS = 20


@dataclass
class TodoItem:
    id: str
    text: str
    status: str = "pending"


@dataclass
class TodoList:
    """Agent-owned execution checklist.

    The list exists only once the agent creates it; a turn without a todo
    list simply has ``runtime_state.todo is None``.
    """

    items: list[TodoItem] = field(default_factory=list)
    revision: int = 0
    updated_at: str = ""
    next_seq: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "updated_at": self.updated_at,
            "next_seq": self.next_seq,
            "items": [
                {"id": item.id, "text": item.text, "status": item.status}
                for item in self.items
            ],
        }

    def render(self) -> str:
        markers = {
            "completed": "[done]",
            "in_progress": "[in progress]",
            "pending": "[pending]",
            "cancelled": "[cancelled]",
        }
        return "\n".join(
            f"{markers.get(item.status, '[pending]')} {item.text}"
            for item in self.items
        )


def normalize_todo_items(
    raw_items: Any,
    *,
    next_seq: int,
) -> tuple[list[TodoItem], int]:
    """Validate model-provided todo items and assign ids where missing.

    Returns the normalized items plus the advanced id sequence counter.
    Raises ValueError with an agent-readable message on invalid input.
    """
    if not isinstance(raw_items, list):
        raise TypeError("items must be a list")
    if not raw_items:
        raise ValueError("items must contain at least one entry")
    if len(raw_items) > MAX_TODO_ITEMS:
        raise ValueError(f"a todo list allows at most {MAX_TODO_ITEMS} items")

    items: list[TodoItem] = []
    seen_ids: set[str] = set()
    seq = int(next_seq)
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise TypeError("each todo item must be an object")
        text = str(raw.get("text") or "").strip()
        if not text:
            raise ValueError("each todo item requires non-empty text")
        if len(text) > 500:
            raise ValueError("each todo item text must be at most 500 characters")
        status = str(raw.get("status") or "pending").strip().lower()
        if status not in TODO_STATUSES:
            raise ValueError(
                "todo item status must be one of: " + ", ".join(TODO_STATUSES)
            )
        raw_id = str(raw.get("id") or "").strip()
        if raw_id:
            item_id = raw_id[:60]
        else:
            seq += 1
            item_id = f"todo_{seq}"
        if item_id in seen_ids:
            raise ValueError(f"duplicate todo item id: {item_id}")
        seen_ids.add(item_id)
        items.append(TodoItem(id=item_id, text=text, status=status))
    return items, seq
