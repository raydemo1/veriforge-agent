"""Incremental source change notifications for language servers."""

from __future__ import annotations

from pathlib import Path

from watchfiles._rust_notify import RustNotify

from .source_files import is_source_location, is_source_path, iter_source_files


class SourceWatch:
    def __init__(self, root: Path):
        self.root = root
        self._watcher = RustNotify([str(root)], False, False, 0, True, False)
        try:
            self._files = set(iter_source_files(root))
        except BaseException:
            self._watcher.close()
            raise

    def changes(self) -> list[dict]:
        events = self._watcher.watch(5, 1, 20, None)
        if events == "timeout":
            return []
        if isinstance(events, str):
            raise TypeError(f"Source watcher stopped: {events}")
        changed = set()
        for event_type, raw_path in events:
            path = Path(raw_path).absolute()
            if not is_source_location(self.root, path):
                continue
            if is_source_path(self.root, path):
                changed.add(path)
            elif path.is_dir():
                if event_type == 1:
                    changed.update(iter_source_files(self.root, path))
            else:
                changed.update(
                    file for file in self._files if file.is_relative_to(path)
                )
        results = []
        for path in sorted(changed):
            present = path.is_file() and path.resolve().is_relative_to(self.root)
            if not present and path not in self._files:
                continue
            results.append(
                {
                    "uri": path.as_uri(),
                    "type": 2
                    if path in self._files and present
                    else 1
                    if present
                    else 3,
                }
            )
            if present:
                self._files.add(path)
            else:
                self._files.discard(path)
        return results

    def close(self) -> None:
        self._watcher.close()
