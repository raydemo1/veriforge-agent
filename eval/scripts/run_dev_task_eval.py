"""P2.2: run deterministic development tasks with the real agent.

Each case copies a fixture into a fresh workspace, runs one headless turn in
``workspace-write`` mode, then applies an independent hidden test.  The
default model profile is ``.env`` (https://api.deepseek.com); ``--model-profile
ab`` selects the ``.env.ab`` overlay instead.

``--validate`` needs no model: it proves every hidden test fails on the broken
fixture and passes on the golden change.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.pop("HARNESS_EVAL_PROFILE", None)

from eval.scripts.eval_common import session_id_from_stdout
from eval.tasks.dev_tasks_v1 import TASKS as DEV_TASKS
from eval.tasks.windows_native_tasks_v1 import TASKS as WINDOWS_NATIVE_TASKS
from harness_code_agent import config
from harness_code_agent.sessions.observability import build_session_observability
from harness_code_agent.sessions.store import SessionStore
from harness_code_agent.workspace.state_paths import ensure_workspace_state

TASK_PACKS = {
    "dev": ("dev_tasks_v1", DEV_TASKS),
    "windows-native": ("windows_native_tasks_v1", WINDOWS_NATIVE_TASKS),
}

IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".harness"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _selected_tasks(args) -> list[dict]:
    tasks = list(TASK_PACKS[args.task_pack][1])
    if args.tasks:
        requested = {item.strip() for item in args.tasks.split(",") if item.strip()}
        tasks = [task for task in tasks if task["id"] in requested]
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def _materialize(task: dict, root: Path) -> None:
    for relative, content in task["files"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def _apply_golden(task: dict, root: Path) -> None:
    for relative, content in task["golden"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def _file_hashes(root: Path, paths: list[str]) -> dict[str, str]:
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in paths
    }


def _run_hidden(root: Path, task: dict, *, timeout: int = 180) -> tuple[bool, str]:
    path = root / "eval_hidden_test.py"
    path.write_text(task["hidden_test"], encoding="utf-8", newline="\n")
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", path.name],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"[hidden test timed out after {timeout}s]"
    finally:
        path.unlink(missing_ok=True)
    return completed.returncode == 0, (completed.stdout or "") + (completed.stderr or "")


def _snapshot(root: Path) -> dict[str, bytes]:
    snapshot = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix in IGNORED_SUFFIXES:
            continue
        snapshot[path.relative_to(root).as_posix()] = path.read_bytes()
    return snapshot


def _diff(before: dict[str, bytes], after: dict[str, bytes]) -> str:
    chunks: list[str] = []
    for path in sorted(set(before) | set(after)):
        old = before.get(path)
        new = after.get(path)
        if old == new:
            continue
        old_text = (old or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        new_text = (new or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        chunks.extend(difflib.unified_diff(old_text, new_text, fromfile=f"a/{path}", tofile=f"b/{path}"))
    return "".join(chunks)


def _state_root(workspace: Path, home: Path) -> Path:
    previous = os.environ.get("VERIFORGE_HOME")
    os.environ["VERIFORGE_HOME"] = str(home)
    try:
        return ensure_workspace_state(workspace)
    finally:
        if previous is None:
            os.environ.pop("VERIFORGE_HOME", None)
        else:
            os.environ["VERIFORGE_HOME"] = previous


def _session_metrics(
    workspace: Path, session_id: str, *, state_root: Path | None = None
) -> tuple[dict, list[dict]]:
    if not session_id:
        return {}, []
    try:
        store = SessionStore(state_root or ensure_workspace_state(workspace))
        metadata = store.read_metadata(session_id)
        events = store.read_events(session_id)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return {}, []
    return build_session_observability(metadata, events).to_dict(), events


def _event_counts(events: list[dict]) -> dict:
    tools: Counter[str] = Counter()
    shell_calls = 0
    background_jobs = 0
    failed_tools = 0
    agent_spawns = 0
    for event in events:
        event_type = str(event.get("type") or "")
        payload = event.get("payload") or {}
        if event_type == "agent_spawned":
            agent_spawns += 1
        if event_type != "tool_result":
            continue
        tool = str(payload.get("tool") or "")
        tools[tool] += 1
        if tool == "bash":
            shell_calls += 1
            metadata = payload.get("metadata") or {}
            output = str(payload.get("output") or "")
            if metadata.get("job_status") == "running" and output.startswith("Started background shell job"):
                background_jobs += 1
        if str(payload.get("status") or "") == "failed":
            failed_tools += 1
    return {
        "tool_calls": sum(tools.values()),
        "shell_calls": shell_calls,
        "background_jobs": background_jobs,
        "failed_tools": failed_tools,
        "agent_spawns": agent_spawns,
        "tool_breakdown": dict(tools),
    }


def _parse_cli_markers(stdout: str) -> tuple[dict[str, float], str]:
    timings: dict[str, float] = {}
    stop_reason = ""
    for line in stdout.splitlines():
        if line.startswith("veriforge timing: "):
            try:
                parsed = json.loads(line[len("veriforge timing: "):])
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    try:
                        timings[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
        elif line.startswith("veriforge stop: "):
            stop_reason = line[len("veriforge stop: "):].strip()
    return timings, stop_reason


def _copy_session_artifacts(
    workspace: Path, session_id: str, case_dir: Path, *, state_root: Path | None = None
) -> dict:
    """Keep the full journal, event stream and agent trace next to the case."""
    if not session_id:
        return {}
    state_root = state_root or ensure_workspace_state(workspace)
    session_dir = state_root / "sessions" / session_id
    artifacts: dict[str, str] = {}
    for name in ("journal.jsonl", "events.jsonl"):
        source = session_dir / name
        if source.is_file():
            target = case_dir / name
            shutil.copy2(source, target)
            artifacts[name] = str(target)
    trace_dir = state_root / "traces" / session_id
    if trace_dir.is_dir():
        target = case_dir / "trace"
        shutil.copytree(trace_dir, target, dirs_exist_ok=True)
        artifacts["trace_dir"] = str(target)
    return artifacts


def _tool_trace(events: list[dict]) -> list[dict]:
    """Pair tool_call and tool_result events, preferring the runtime call id."""
    pending_by_id: dict[str, dict] = {}
    pending_by_tool: dict[str, list[dict]] = {}
    rows: list[dict] = []

    def take(entry_id: str, tool: str) -> dict | None:
        if entry_id and entry_id in pending_by_id:
            entry = pending_by_id.pop(entry_id)
            queue = pending_by_tool.get(tool) or []
            for index, item in enumerate(queue):
                if item is entry:
                    queue.pop(index)
                    break
            return entry
        queue = pending_by_tool.get(tool) or []
        if not queue:
            return None
        entry = queue.pop(0)
        if entry.get("id"):
            pending_by_id.pop(entry["id"], None)
        return entry

    for event in events:
        payload = event.get("payload") or {}
        event_type = event.get("type")
        metadata = payload.get("metadata") or {}
        if event_type == "tool_call":
            entry = {
                "id": str(payload.get("tool_call_id") or ""),
                "args": payload.get("args"),
            }
            pending_by_tool.setdefault(str(payload.get("tool")), []).append(entry)
            if entry["id"]:
                pending_by_id[entry["id"]] = entry
        elif event_type == "tool_result":
            tool = str(payload.get("tool"))
            entry_id = str(
                payload.get("tool_call_id") or metadata.get("tool_call_id") or ""
            )
            entry = take(entry_id, tool)
            rows.append({
                "timestamp": event.get("timestamp"),
                "tool": tool,
                "tool_call_id": entry_id or (entry or {}).get("id"),
                "status": payload.get("status"),
                "error": payload.get("error"),
                "return_code": payload.get("return_code"),
                "output_tail": str(payload.get("output") or "")[-400:],
                "args": (entry or {}).get("args"),
                "file_changes": metadata.get("file_changes"),
                "file_change_summary": metadata.get("file_change_summary"),
                "stage_timings": metadata.get("stage_timings"),
            })
    return rows


def _has_file_changes(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    counts = payload.get("counts") or {}
    return any(int(counts.get(key) or 0) > 0 for key in ("created", "modified", "deleted"))


def _repeated_calls(rows: list[dict], limit: int = 10) -> list[dict]:
    counts: Counter[str] = Counter()
    examples: dict[str, dict] = {}
    for row in rows:
        if row.get("args") is None:
            continue
        key = f"{row.get('tool')}:{json.dumps(row['args'], sort_keys=True, ensure_ascii=False)}"
        counts[key] += 1
        examples[key] = {"tool": row.get("tool"), "args": row["args"]}
    return [
        {**examples[key], "count": count}
        for key, count in counts.most_common(limit)
        if count > 1
    ]


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * quantile + 0.999999) - 1)))
    return round(ordered[index], 2)


def validate(tasks: list[dict], output_root: Path, *, prefix: str = "dev-task-validation") -> int:
    stamp = datetime.now().astimezone()
    run_dir = output_root / f"{prefix}-{stamp:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix=f"vf-validate-{task['id']}-") as tmp:
            root = Path(tmp)
            _materialize(task, root)
            before_ok, before_log = _run_hidden(root, task)
            _apply_golden(task, root)
            after_ok, after_log = _run_hidden(root, task)
        ok = (not before_ok) and after_ok
        results.append({
            "id": task["id"],
            "category": task["category"],
            "broken_fails": not before_ok,
            "golden_passes": after_ok,
            "ok": ok,
            "before_log": before_log[-800:],
            "after_log": after_log[-800:],
        })
        print(f"{'PASS' if ok else 'FAIL'}  {task['id']}")
    (run_dir / "validation.json").write_text(
        json.dumps({"tasks": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {run_dir / 'validation.json'}")
    return 0 if all(item["ok"] for item in results) else 1


def _start_concurrent_edit(task: dict, workspace: Path, case_home: Path):
    """Apply a deterministic edit after the agent has read the target file.

    The controller watches the live session trace instead of racing a timer,
    so the agent's next write must go through WorkspaceService version
    protection and re-read before it can merge.
    """
    spec = task.get("concurrent_edit")
    if not spec:
        return None
    stop = threading.Event()
    state = {"applied": False, "error": ""}
    target_read = str(spec["after_read"]).replace("\\", "/").lower()
    target_path = workspace / str(spec["path"])

    def worker():
        seen: dict[str, int] = {}
        while not stop.wait(0.05):
            triggered = False
            for trace in case_home.rglob("main_agent_*.jsonl"):
                key = str(trace)
                try:
                    text = trace.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                lines = text.splitlines()
                if text and not text.endswith("\n"):
                    lines = lines[:-1]
                start = seen.get(key, 0)
                for line in lines[start:]:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("tool") != "read_file":
                        continue
                    args = row.get("args")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    read_path = str((args or {}).get("path") or "").replace("\\", "/").lower()
                    if target_read and target_read in read_path:
                        triggered = True
                        break
                seen[key] = len(lines)
                if triggered:
                    break
            if not triggered:
                continue
            try:
                content = target_path.read_text(encoding="utf-8")
                if spec["search"] not in content:
                    state["error"] = "search text not found"
                    return
                target_path.write_text(
                    content.replace(spec["search"], spec["replace"], 1),
                    encoding="utf-8",
                    newline="\n",
                )
                state["applied"] = True
            except OSError as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"
            return

    thread = threading.Thread(target=worker, name="concurrent-edit", daemon=True)
    thread.start()
    return {"stop": stop, "thread": thread, "state": state}


def _stop_concurrent_edit(handle) -> dict:
    if handle is None:
        return {"applied": False, "error": ""}
    handle["stop"].set()
    handle["thread"].join(timeout=5)
    return dict(handle["state"])


def _run_case(task: dict, rep: int, run_dir: Path, args) -> dict:
    workspace_root = Path(args.workspace_root)
    workspace = workspace_root / f"{task['id']}-r{rep}"
    case_home = workspace_root / "state" / f"{task['id']}-r{rep}"
    case_memory = workspace_root / "memory" / f"{task['id']}-r{rep}"
    case_home.mkdir(parents=True, exist_ok=True)
    case_memory.mkdir(parents=True, exist_ok=True)
    _materialize(task, workspace)
    protected = list(task.get("protected") or [])
    protected_before = _file_hashes(workspace, protected)
    snapshot = _snapshot(workspace)

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing else str(PROJECT_ROOT) + os.pathsep + existing
    env["HARNESS_PERMISSION_MODE"] = "workspace-write"
    env["HARNESS_MODEL_INTENSITY"] = "normal"
    env["VERIFORGE_HOME"] = str(case_home)
    env["VERIFORGE_MEMORY_ROOT"] = str(case_memory)
    # Evaluation must not depend on history from another task or from the host.
    env["HARNESS_MEMORY_DISABLED"] = "1"
    # No artificial budget: the old 100k cap counted cached input and stopped
    # runs mid-task.  The task timeout is the only bound.
    env["MAX_AGENT_TOTAL_TOKENS"] = "0"
    env["MAX_AGENT_TOOL_CALLS"] = "0"
    if getattr(args, "trace_file_changes", False):
        env["HARNESS_TRACE_FILE_CHANGES"] = "1"
    if getattr(args, "trace_stages", False):
        env["HARNESS_TRACE_STAGES"] = "1"
    if args.model_profile == "ab":
        env["HARNESS_EVAL_PROFILE"] = "ab"
    else:
        env.pop("HARNESS_EVAL_PROFILE", None)

    case_dir = run_dir / "cases" / f"{task['id']}-r{rep}"
    case_dir.mkdir(parents=True, exist_ok=True)
    timed_out = False
    controller = _start_concurrent_edit(task, workspace, case_home)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "harness_code_agent.cli", "-p", task["prompt"]],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.task_timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        stderr += f"\n[timeout] case exceeded {args.task_timeout}s"
        returncode = 124
    elapsed = time.perf_counter() - started
    concurrent_edit = _stop_concurrent_edit(controller)

    (case_dir / "stdout.txt").write_text(stdout, encoding="utf-8", newline="\n")
    (case_dir / "stderr.txt").write_text(stderr, encoding="utf-8", newline="\n")

    hidden_ok, hidden_log = _run_hidden(workspace, task)
    (case_dir / "hidden.txt").write_text(hidden_log, encoding="utf-8", newline="\n")
    protected_after = _file_hashes(workspace, protected)
    protected_ok = protected_after == protected_before
    changed = _snapshot(workspace)
    diff = _diff(snapshot, changed)
    (case_dir / "diff.patch").write_text(diff, encoding="utf-8", newline="\n")

    session_id = session_id_from_stdout(stdout)
    state_root = _state_root(workspace, case_home)
    metrics, events = _session_metrics(workspace, session_id, state_root=state_root)
    counts = _event_counts(events)
    tool_rows = _tool_trace(events)
    repeated = _repeated_calls(tool_rows)
    shell_rows = [
        row for row in tool_rows
        if row.get("tool") == "bash" and isinstance(row.get("file_change_summary"), dict)
    ]
    commands_with_changes = sum(
        1 for row in shell_rows if _has_file_changes(row.get("file_change_summary"))
    )
    commands_without_changes = sum(
        1 for row in shell_rows if not _has_file_changes(row.get("file_change_summary"))
    )
    (case_dir / "tool_trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tool_rows),
        encoding="utf-8",
        newline="\n",
    )
    artifacts = _copy_session_artifacts(
        workspace, session_id, case_dir, state_root=state_root
    )
    (case_dir / "observability.json").write_text(
        json.dumps({"metrics": metrics, "counts": counts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cli_timings, cli_stop = _parse_cli_markers(stdout)
    fallback_reasons = [
        str((event.get("payload") or {}).get("reason") or "")
        for event in events
        if event.get("type") == "agent_fallback"
    ]
    stop_reason = cli_stop or (fallback_reasons[-1] if fallback_reasons else "")
    tokens = metrics.get("tokens") or {}
    performance = metrics.get("performance") or {}
    failed_tool_details = [
        {"tool": row.get("tool"), "error": str(row.get("error") or "")}
        for row in tool_rows
        if row.get("status") == "failed"
    ][:5]
    shell_errors = [
        {
            "command": (row.get("args") or {}).get("command"),
            "return_code": row.get("return_code"),
            "output_tail": row.get("output_tail"),
        }
        for row in tool_rows
        if row.get("tool") == "bash" and row.get("return_code") not in (None, 0)
    ][:8]

    success = (
        returncode == 0
        and hidden_ok
        and protected_ok
        and not timed_out
        and not stop_reason
    )
    failure_kind = ""
    if timed_out:
        failure_kind = "timeout"
    elif stop_reason:
        failure_kind = f"stopped:{stop_reason}"
    elif not protected_ok:
        failure_kind = "protected_test_modified"
    elif not hidden_ok:
        failure_kind = "hidden_test_failed"
    elif returncode != 0:
        failure_kind = "runtime_exit_nonzero"

    return {
        "task_id": task["id"],
        "category": task["category"],
        "rep": rep,
        "status": "stopped" if stop_reason else ("failed" if returncode != 0 else "completed"),
        "success": success,
        "failure_kind": failure_kind,
        "stop_reason": stop_reason,
        "fallback_reasons": fallback_reasons,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 2),
        "timings": cli_timings,
        "session_id": session_id,
        "hidden_test": hidden_ok,
        "hidden_tail": hidden_log[-400:],
        "protected_files_intact": protected_ok,
        "concurrent_edit": concurrent_edit,
        "llm_calls": int(tokens.get("llm_calls") or 0),
        "prompt_tokens": int(tokens.get("prompt_tokens") or 0),
        "total_tokens": int(tokens.get("total_tokens") or 0),
        "tool_calls": counts["tool_calls"],
        "shell_calls": counts["shell_calls"],
        "background_jobs": counts["background_jobs"],
        "failed_tools": counts["failed_tools"],
        "failed_tool_details": failed_tool_details,
        "nonzero_shell": len([
            row for row in tool_rows
            if row.get("tool") == "bash" and row.get("return_code") not in (None, 0)
        ]),
        "shell_errors": shell_errors,
        "repeated_calls": repeated,
        "commands_with_changes": commands_with_changes,
        "commands_without_changes": commands_without_changes,
        "agent_spawns": counts["agent_spawns"],
        "tool_breakdown": counts["tool_breakdown"],
        "turn_duration_ms": performance.get("turn_duration_ms"),
        "llm_response_latency_ms": performance.get("llm_response_latency_ms"),
        "diff_lines": len(diff.splitlines()),
        "artifacts": artifacts,
        "case_dir": str(case_dir),
    }


def run(tasks: list[dict], args, output_root: Path) -> int:
    stamp = datetime.now().astimezone()
    prefix = "dev-task-eval" if args.task_pack == "dev" else f"{args.task_pack}-eval"
    run_dir = output_root / f"{prefix}-{stamp:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.workspace_root:
        args.workspace_root = tempfile.mkdtemp(prefix="vf-dev-eval-workspaces-")
    workspace_root = Path(args.workspace_root)
    results: list[dict] = []
    raw = run_dir / "results.jsonl"
    for task in tasks:
        for rep in range(1, args.repetitions + 1):
            result = _run_case(task, rep, run_dir, args)
            results.append(result)
            with raw.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"{'PASS' if result['success'] else 'FAIL'}  {task['id']} r{rep} "
                f"{result['elapsed_seconds']}s llm={result['llm_calls']} "
                f"tools={result['tool_calls']} shell={result['shell_calls']}"
            )
            for directory in (
                workspace_root / f"{task['id']}-r{rep}",
                workspace_root / "state" / f"{task['id']}-r{rep}",
                workspace_root / "memory" / f"{task['id']}-r{rep}",
            ):
                shutil.rmtree(directory, ignore_errors=True)

    per_task = []
    for task in tasks:
        runs = [item for item in results if item["task_id"] == task["id"]]
        per_task.append({
            "task_id": task["id"],
            "category": task["category"],
            "runs": len(runs),
            "passed": sum(1 for item in runs if item["success"]),
            "success_rate": round(sum(1 for item in runs if item["success"]) / len(runs), 3),
            "elapsed_p50": _percentile([item["elapsed_seconds"] for item in runs], 0.5),
            "llm_calls_p50": _percentile([item["llm_calls"] for item in runs], 0.5),
            "tool_calls_p50": _percentile([item["tool_calls"] for item in runs], 0.5),
            "total_tokens_p50": _percentile([item["total_tokens"] for item in runs], 0.5),
            "failure_kinds": dict(Counter(item["failure_kind"] for item in runs if item["failure_kind"])),
            "stopped": sum(1 for item in runs if item["status"] == "stopped"),
        })

    summary = {
        "suite": TASK_PACKS[args.task_pack][0],
        "generated_at": stamp.isoformat(timespec="seconds"),
        "model": {
            "profile": args.model_profile,
            "provider": config.MODEL_PROFILE.provider,
            "model": config.MODEL_PROFILE.model,
            "base_url": config.BASE_URL,
            "permission_mode": "workspace-write",
            "input_mode": config.MODEL_INPUT_MODE,
        },
        "repetitions": args.repetitions,
        "task_count": len(tasks),
        "overall": {
            "runs": len(results),
            "passed": sum(1 for item in results if item["success"]),
            "success_rate": round(
                sum(1 for item in results if item["success"]) / len(results), 3
            ) if results else 0.0,
            "elapsed_p50": _percentile([item["elapsed_seconds"] for item in results], 0.5),
            "elapsed_p95": _percentile([item["elapsed_seconds"] for item in results], 0.95),
            "llm_calls_total": sum(item["llm_calls"] for item in results),
            "tool_calls_total": sum(item["tool_calls"] for item in results),
            "shell_calls_total": sum(item["shell_calls"] for item in results),
            "total_tokens": sum(item["total_tokens"] for item in results),
            "unexpected_agent_spawns": sum(item["agent_spawns"] for item in results),
            "protected_test_modifications": sum(
                1 for item in results if not item["protected_files_intact"]
            ),
            "stopped_runs": sum(1 for item in results if item["status"] == "stopped"),
            "init_seconds_p50": _percentile(
                [item["timings"]["init_seconds"] for item in results
                 if item["timings"].get("init_seconds") is not None], 0.5
            ),
            "turn_seconds_p50": _percentile(
                [item["timings"]["turn_seconds"] for item in results
                 if item["timings"].get("turn_seconds") is not None], 0.5
            ),
            "close_seconds_p50": _percentile(
                [item["timings"]["close_seconds"] for item in results
                 if item["timings"].get("close_seconds") is not None], 0.5
            ),
            "tool_failures_total": sum(item["failed_tools"] for item in results),
            "nonzero_shell_total": sum(item["nonzero_shell"] for item in results),
            "repeated_calls_total": sum(len(item["repeated_calls"]) for item in results),
            "shell_commands_with_changes": sum(item["commands_with_changes"] for item in results),
            "shell_commands_without_changes": sum(
                item["commands_without_changes"] for item in results
            ),
        },
        "tasks": per_task,
        "shell_error_counts": dict(Counter(
            str(error.get("command") or "")
            for item in results
            for error in item["shell_errors"]
        )),
        "failure_kinds": dict(Counter(item["failure_kind"] for item in results if item["failure_kind"])),
        "failures": [
            {
                "task_id": item["task_id"],
                "rep": item["rep"],
                "status": item["status"],
                "failure_kind": item["failure_kind"],
                "stop_reason": item["stop_reason"],
                "failed_tools": item["failed_tool_details"],
                "hidden_tail": item["hidden_tail"],
                "case_dir": item["case_dir"],
            }
            for item in results
            if not item["success"]
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "REPORT.md").write_text(
        _report(summary, per_task, summary["failures"]), encoding="utf-8", newline="\n"
    )
    print(f"\nWrote {run_dir / 'summary.json'}")
    return 0 if summary["overall"]["success_rate"] == 1.0 else 1


def _report(summary: dict, per_task: list[dict], failures: list[dict]) -> str:
    lines = [
        "# P2.2 真实开发任务评测",
        "",
        f"- 生成时间：{summary['generated_at']}",
        (
            f"- 模型：`{summary['model']['model']}`（{summary['model']['base_url']}，"
            f"profile `{summary['model']['profile']}`）"
        ),
        f"- 权限模式：`{summary['model']['permission_mode']}`；输入模式：`{summary['model']['input_mode']}`",
        f"- 重复次数：{summary['repetitions']}；任务数：{summary['task_count']}",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 运行数 | {summary['overall']['runs']} |",
        f"| 成功率 | {summary['overall']['success_rate']:.1%} |",
        (
            f"| 端到端 P50 / P95 | {summary['overall']['elapsed_p50']} s / "
            f"{summary['overall']['elapsed_p95']} s |"
        ),
        f"| LLM 调用 | {summary['overall']['llm_calls_total']} |",
        (
            f"| 工具调用（Shell） | {summary['overall']['tool_calls_total']} "
            f"（{summary['overall']['shell_calls_total']}） |"
        ),
        f"| Token 总量 | {summary['overall']['total_tokens']} |",
        (
            f"| 初始化 / Turn / 关闭 P50 | {summary['overall']['init_seconds_p50']} s / "
            f"{summary['overall']['turn_seconds_p50']} s / "
            f"{summary['overall']['close_seconds_p50']} s |"
        ),
        f"| 被运行时停止 | {summary['overall']['stopped_runs']} |",
        (
            f"| 工具失败 / 非零退出 / 重复调用组 | {summary['overall']['tool_failures_total']} / "
            f"{summary['overall']['nonzero_shell_total']} / "
            f"{summary['overall']['repeated_calls_total']} |"
        ),
        f"| 未要求的子 Agent | {summary['overall']['unexpected_agent_spawns']} |",
        f"| 篡改受保护测试 | {summary['overall']['protected_test_modifications']} |",
        "",
        "## 逐任务",
        "",
        "| 任务 | 类别 | 通过 | P50 秒 | LLM | 工具 | Token | 失败类型 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in per_task:
        lines.append(
            f"| {item['task_id']} | {item['category']} | {item['passed']}/{item['runs']} | "
            f"{item['elapsed_p50']} | {item['llm_calls_p50']} | {item['tool_calls_p50']} | "
            f"{item['total_tokens_p50']} | {item['failure_kinds'] or '-'} |"
        )
    lines += ["", "## 停止与失败", ""]
    if failures:
        lines += [
            "| 任务/次 | 状态 | 类型 | 停止原因 | 首个工具失败 | 隐藏测试摘要 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in failures:
            first_error = ""
            if item["failed_tools"]:
                first = item["failed_tools"][0]
                first_error = f"{first.get('tool')}: {str(first.get('error') or '')[:120]}"
            hidden = " ".join(str(item["hidden_tail"] or "").split())[:160].replace("|", "\\|")
            lines.append(
                f"| {item['task_id']} r{item['rep']} | {item['status']} | {item['failure_kind']} | "
                f"{item['stop_reason'] or '-'} | {first_error or '-'} | {hidden or '-'} |"
            )
    else:
        lines.append("无失败。")
    shell_errors = summary.get("shell_error_counts") or {}
    if shell_errors:
        lines += ["", "## 非零退出 Shell（含已恢复）", "", "| 命令 | 次数 |", "| --- | --- |"]
        for command, count in sorted(shell_errors.items(), key=lambda item: -item[1])[:8]:
            lines.append(f"| `{command[:120]}` | {count} |")
    lines += [
        "",
        "## 轨迹文件",
        "",
        (
            "每个 `cases/<任务>-r<N>/` 目录保存：`journal.jsonl`、`events.jsonl`、`trace/`、"
            "`tool_trace.jsonl`（工具调用/结果/参数/错误）、`hidden.txt`、`diff.patch`、"
            "`stdout.txt`、`stderr.txt`、`observability.json`。"
        ),
        "",
        "## 判读",
        "",
        "- 成功率优先于速度；失败先按 `failure_kinds` 排查轨迹。",
        "- `stopped:*` 表示运行时主动停止，不是正常完成；预算项已从评测中移除。",
        "- 若 LLM 时间占主导，下一轮优化上下文与模型往返，而不是 Shell。",
        "- 若 Shell 调用次数高且耗时占比明显，优先减少重复搜索、重复读取和无效测试。",
        "- 未要求的子 Agent 应保持为 0；非零时检查系统提示词与委派触发。",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validate", action="store_true", help="No-model fixture and hidden-test check.")
    group.add_argument("--run", action="store_true", help="Run the real-agent evaluation.")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--task-timeout", type=int, default=900)
    parser.add_argument("--tasks", default="", help="Comma-separated task ids.")
    parser.add_argument(
        "--task-pack",
        choices=sorted(TASK_PACKS),
        default="dev",
        help="Task pack: the original development tasks or the Windows-native regression set.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-profile", choices=["default", "ab"], default="default")
    parser.add_argument(
        "--workspace-root",
        default="",
        help="Directory for task workspaces outside the repository (created under temp by default).",
    )
    parser.add_argument(
        "--trace-file-changes",
        action="store_true",
        help="Snapshot the workspace before/after each Shell command (diagnostic).",
    )
    parser.add_argument(
        "--trace-stages",
        action="store_true",
        help="Record preparation, lease, launch, spawn, wait and result timings.",
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "eval" / "results"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = _selected_tasks(args)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 2
    output_root = Path(args.output_root)
    if args.validate:
        prefix = (
            "dev-task-validation"
            if args.task_pack == "dev"
            else f"{args.task_pack}-validation"
        )
        return validate(tasks, output_root, prefix=prefix)
    if not args.run:
        print(json.dumps({"tasks": [task["id"] for task in tasks],
                          "repetitions": args.repetitions,
                          "model": config.MODEL_PROFILE.model,
                          "base_url": config.BASE_URL}, ensure_ascii=False, indent=2))
        return 0
    return run(tasks, args, output_root)


if __name__ == "__main__":
    raise SystemExit(main())
