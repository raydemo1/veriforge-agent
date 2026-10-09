from __future__ import annotations

import time

from harness_code_agent.runtime import source_watch
from harness_code_agent.runtime.source_watch import SourceWatch


def collect_changes(watch, expected):
    changes = {}
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        changes.update((item["uri"], item["type"]) for item in watch.changes())
        if all(changes.get(path.as_uri()) == kind for path, kind in expected.items()):
            return changes
    raise AssertionError(f"Missing source notifications: {changes}")


def test_native_watch_tracks_edits_atomic_restore_and_directory_moves(tmp_path):
    original = tmp_path / "main.py"
    original.write_text("before", encoding="utf-8")
    watch = SourceWatch(tmp_path)
    try:
        original.write_text("after", encoding="utf-8")
        collect_changes(watch, {original: 2})
        replacement = tmp_path / "restore.tmp"
        replacement.write_text("before", encoding="utf-8")
        replacement.replace(original)
        collect_changes(watch, {original: 2})
        folder = tmp_path / "package"
        folder.mkdir()
        child = folder / "library.py"
        child.write_text("value = 1", encoding="utf-8")
        collect_changes(watch, {child: 1})
        moved = tmp_path / "renamed"
        folder.rename(moved)
        moved_child = moved / child.name
        collect_changes(watch, {child: 3, moved_child: 1})
        original.unlink()
        collect_changes(watch, {original: 3})
    finally:
        watch.close()


def test_warm_watch_does_not_scan_sources_or_include_harbor(tmp_path, monkeypatch):
    source = tmp_path / "main.py"
    source.write_text("before", encoding="utf-8")
    watch = SourceWatch(tmp_path)
    try:

        def unexpected_scan(*_args):
            raise AssertionError("Warm queries must not rescan source files")

        monkeypatch.setattr(source_watch, "iter_source_files", unexpected_scan)
        assert watch.changes() == []
        artifact = tmp_path / ".harbor" / "project"
        artifact.mkdir(parents=True)
        (artifact / "main.py").write_text("artifact", encoding="utf-8")
        source.write_text("after", encoding="utf-8")
        changes = collect_changes(watch, {source: 2})
        assert changes == {source.as_uri(): 2}
        assert watch.changes() == []
    finally:
        watch.close()
