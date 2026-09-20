"""Headless in-container runner used by the Terminal-Bench Harbor adapter."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import Any


def export_session_artifacts(
    *,
    harness_root: str | Path,
    session_id: str,
    artifacts_root: str | Path = "/logs/artifacts",
    runner_error: str = "",
) -> Path:
    """Export the complete durable session and readable trajectory views."""
    harness_root = Path(harness_root).resolve()
    artifacts_root = Path(artifacts_root)
    session_root = harness_root / "sessions" / session_id
    if not session_root.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_root}")

    export_root = artifacts_root / "hca" / session_id
    export_root.mkdir(parents=True, exist_ok=True)
    for child_name in (
        "session",
        "observations",
        "traces",
        "trajectory.jsonl",
        "todo_history.jsonl",
        "manifest.json",
        "runner_error.txt",
    ):
        child = export_root / child_name
        if child.is_dir():
            shutil.rmtree(child)
        elif child.exists():
            child.unlink()

    shutil.copytree(session_root, export_root / "session")

    observations_root = harness_root / "observations" / session_id
    observations_exported = observations_root.is_dir()
    if observations_exported:
        shutil.copytree(observations_root, export_root / "observations")

    traces_root = harness_root / "traces"
    traces_exported = traces_root.is_dir()
    if traces_exported:
        shutil.copytree(traces_root, export_root / "traces")

    runner_error = str(runner_error or "").strip()
    runner_error_exported = bool(runner_error)
    if runner_error_exported:
        (export_root / "runner_error.txt").write_text(
            runner_error + "\n",
            encoding="utf-8",
        )

    events_path = session_root / "events.jsonl"
    events = _read_jsonl(events_path)
    _write_jsonl(export_root / "trajectory.jsonl", events)

    todo_events = [
        event
        for event in events
        if event.get("type") in {"tool_call", "tool_result"}
        and (event.get("payload") or {}).get("tool") == "update_todo"
    ]
    _write_jsonl(export_root / "todo_history.jsonl", todo_events)

    manifest = {
        "session_id": session_id,
        "event_count": len(events),
        "todo_event_count": len(todo_events),
        "observations_exported": observations_exported,
        "traces_exported": traces_exported,
        "runner_error_exported": runner_error_exported,
        "session_path": "session",
        "trajectory_path": "trajectory.jsonl",
        "todo_history_path": "todo_history.jsonl",
        "observations_path": "observations" if observations_exported else None,
        "traces_path": "traces" if traces_exported else None,
        "runner_error_path": "runner_error.txt" if runner_error_exported else None,
    }
    (export_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export_root


def write_session_manifest(
    *,
    session_id: str,
    workspace: str | Path,
    harness_root: str | Path,
    artifacts_root: str | Path = "/logs/artifacts",
    status: str = "started",
) -> Path:
    """Write a small artifact as soon as the session exists."""
    artifacts_root = Path(artifacts_root)
    manifest_root = artifacts_root / "hca" / session_id
    manifest_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "workspace": str(workspace),
        "harness_root": str(harness_root),
        "status": status,
    }
    (manifest_root / "early_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_root


def install_artifact_export_hooks(state: dict[str, Any]) -> None:
    """Register best-effort artifact export on normal exit and termination."""

    def export(reason: str) -> None:
        session_id = str(state.get("session_id") or "")
        session_store = state.get("session_store")
        if not session_id or session_store is None:
            return
        root = getattr(session_store, "root", None)
        if not root:
            return
        try:
            export_session_artifacts(
                harness_root=root,
                session_id=session_id,
                artifacts_root=os.environ.get("HCA_ARTIFACTS_ROOT", "/logs/artifacts"),
                runner_error=str(state.get("runner_error") or reason),
            )
        except BaseException:
            print("Failed best-effort VeriForge artifact export:", file=sys.stderr)
            traceback.print_exc()

    def on_exit() -> None:
        export("process exit before normal artifact export")

    atexit.register(on_exit)

    def handle_signal(signum, frame) -> None:
        state["runner_error"] = "\n".join(
            part
            for part in (
                str(state.get("runner_error") or "").strip(),
                f"received signal {signum}; attempting best-effort artifact export",
            )
            if part
        )
        export(f"received signal {signum}")
        raise SystemExit(128 + int(signum))

    for signame in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, signame, None)
        if sig is not None:
            try:
                signal.signal(sig, handle_signal)
            except (OSError, ValueError):
                pass


def start_periodic_artifact_export(
    state: dict[str, Any],
    *,
    interval_seconds: float | None = None,
) -> threading.Event:
    """Periodically refresh partial artifacts for timeout diagnostics."""
    try:
        interval = float(
            interval_seconds
            if interval_seconds is not None
            else os.environ.get("HCA_ARTIFACT_EXPORT_INTERVAL_SECONDS", "30")
        )
    except ValueError:
        interval = 30.0
    stop_event = threading.Event()
    if interval <= 0:
        return stop_event

    def loop() -> None:
        while not stop_event.wait(interval):
            session_id = str(state.get("session_id") or "")
            session_store = state.get("session_store")
            root = getattr(session_store, "root", None) if session_store is not None else None
            if not session_id or not root:
                continue
            try:
                export_session_artifacts(
                    harness_root=root,
                    session_id=session_id,
                    artifacts_root=os.environ.get("HCA_ARTIFACTS_ROOT", "/logs/artifacts"),
                    runner_error=str(state.get("runner_error") or "periodic partial artifact export"),
                )
            except BaseException:
                print("Failed periodic VeriForge artifact export:", file=sys.stderr)
                traceback.print_exc()

    thread = threading.Thread(target=loop, name="hca-artifact-export", daemon=True)
    thread.start()
    return stop_event


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).resolve()

    os.environ.setdefault("HARNESS_WORKSPACE", str(workspace))
    if args.task_name:
        os.environ["HARNESS_TERMINAL_TASK_NAME"] = args.task_name
    os.environ.setdefault("HARNESS_PERMISSION_MODE", "danger-full-access")
    os.environ.setdefault("HCA_TERMINAL_EVAL_MODE", "1")
    os.environ.setdefault("HARNESS_STREAM", "0")
    os.environ.setdefault("HARNESS_MEMORY_DISABLED", "1")

    session: Any | None = None
    session_store: Any | None = None
    session_id = ""
    result: Any | None = None
    session_started = False
    runner_error = ""
    hook_state: dict[str, Any] = {
        "session_id": "",
        "session_store": None,
        "runner_error": "",
    }
    install_artifact_export_hooks(hook_state)
    periodic_export_stop = start_periodic_artifact_export(hook_state)
    try:
        from eval.benchmarks.usage_metrics import (
            build_session_eval_metrics,
            print_eval_metrics,
        )
        from harness_code_agent.core.interactive import (
            InteractiveSession,
            print_turn_result,
        )

        session = InteractiveSession(
            cwd=workspace,
            profile_name="terminal",
            profile_explicit=True,
            stream_sink=None,
            allow_checkpoint_init_failure=True,
        )
        session_started = True
        session.checkpoint.auto = False
        session_id = session.session_id
        session_store = session.session_store
        hook_state["session_id"] = session_id
        hook_state["session_store"] = session_store
        if session.checkpoint_init_error:
            runner_error = "\n".join(
                part
                for part in (
                    runner_error.strip(),
                    f"checkpoint disabled: {session.checkpoint_init_error}",
                )
                if part
            )
            hook_state["runner_error"] = runner_error
        print(f"veriforge session: {session_id}", flush=True)
        print(f"workspace: {session.cwd}", flush=True)
        try:
            manifest_path = write_session_manifest(
                session_id=session_id,
                workspace=session.cwd,
                harness_root=session_store.root,
                artifacts_root=os.environ.get("HCA_ARTIFACTS_ROOT", "/logs/artifacts"),
            )
            print(f"veriforge early artifacts: {manifest_path}", flush=True)
        except Exception:
            print("Failed to write early VeriForge artifact manifest:", file=sys.stderr)
            traceback.print_exc()
        try:
            result = session.submit(args.prompt)
        except Exception:
            runner_error = traceback.format_exc()
            hook_state["runner_error"] = runner_error
            print(runner_error, file=sys.stderr, end="")
        finally:
            session_id = session_id or session.session_id
            session_store = session_store or session.session_store
            hook_state["session_id"] = session_id
            hook_state["session_store"] = session_store
            hook_state["runner_error"] = runner_error
            if session.session_id:
                print(f"veriforge session: {session.session_id}", flush=True)
            print(f"workspace: {session.cwd}", flush=True)
            if result is not None:
                print_turn_result(result)
            try:
                session.close()
            except Exception:
                close_error = traceback.format_exc()
                runner_error = "\n".join(
                    part for part in (runner_error.strip(), close_error.strip()) if part
                )
                hook_state["runner_error"] = runner_error
                print(close_error, file=sys.stderr, end="")
            if session_store is not None and session_id:
                try:
                    artifact_path = export_session_artifacts(
                        harness_root=session_store.root,
                        session_id=session_id,
                        artifacts_root=os.environ.get("HCA_ARTIFACTS_ROOT", "/logs/artifacts"),
                        runner_error=runner_error,
                    )
                    print(f"veriforge artifacts: {artifact_path}", flush=True)
                except Exception:
                    print("Failed to export VeriForge session artifacts:", file=sys.stderr)
                    traceback.print_exc()
            periodic_export_stop.set()
        if session_store is not None and session_id:
            metrics = build_session_eval_metrics(
                session_store,
                session_id,
                model=os.environ.get("HARNESS_MODEL", ""),
            )
            print_eval_metrics(metrics)
    except Exception:
        periodic_export_stop.set()
        traceback.print_exc()
        return 1
    finally:
        periodic_export_stop.set()
    if session_started:
        return 0
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VeriForge on a Terminal-Bench prompt.")
    parser.add_argument("prompt")
    parser.add_argument("--workspace", default="/app")
    parser.add_argument(
        "--task-name",
        default=os.environ.get("HARNESS_TERMINAL_TASK_NAME", ""),
        help="Terminal-Bench task name used for profile metadata such as per-task timeout.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
