"""Formatting and diagnostic helpers for interactive CLI/TUI commands."""
from __future__ import annotations

import os
from pathlib import Path

from .. import config
from ..profiles import list_profiles
from ..runtime.mcp import McpClientManager, McpConfigError, load_mcp_config
from ..sessions.journal import SessionJournal
from ..sessions.store import SessionStore
from ..sessions.summary import load_session_summary
from ..workspace.service import WorkspaceService
from ..workspace.shell_session import (
    docker_cli_path,
    docker_info_check,
    docker_shell_hint,
    sandbox_mode,
    windows_shell_hint,
    windows_shell_path,
)


def print_help() -> None:
    from ..tui.commands import default_command_registry

    print(default_command_registry().format_help())


def format_sessions(store: SessionStore) -> str:
    sessions = store.list_sessions()
    if not sessions:
        return "No sessions found."
    lines = [f"{'ID':28s} {'PROFILE':15s} {'MODE':18s} CREATED"]
    for item in sessions:
        lines.append(
            f"{item.get('id', ''):28s} "
            f"{item.get('profile', ''):15s} "
            f"{item.get('permission_mode', ''):18s} "
            f"{item.get('created_at', '')}"
        )
    return "\n".join(lines)


def print_sessions(store: SessionStore) -> None:
    print(format_sessions(store))


def print_session(store: SessionStore, session_id: str) -> None:
    print(load_session_summary(store, session_id))


def format_fork(store: SessionStore, session_id: str) -> str:
    session = store.fork(session_id)
    metadata = store.read_metadata(session.id)
    return "\n".join([
        f"forked_session: {session.id}",
        f"forked_from: {metadata.get('forked_from', session_id)}",
        f"profile: {metadata.get('profile', '')}",
        f"cwd: {metadata.get('cwd', '')}",
    ])


def print_fork(store: SessionStore, session_id: str) -> None:
    print(format_fork(store, session_id))


def format_rollback_session_file(store: SessionStore, session_id: str, path: str) -> str:
    metadata = store.read_metadata(session_id)
    workspace = WorkspaceService(
        root=metadata["cwd"],
        snapshots_dir=store.sessions_dir / session_id / "snapshots",
    )
    result = workspace.rollback_latest_snapshot(path)
    lines = [
        f"rolled_back: {path}",
        f"workspace: {workspace.root}",
    ]
    if result.snapshot_path:
        lines.append(f"pre_rollback_snapshot: {result.snapshot_path}")
    return "\n".join(lines)


def rollback_session_file(store: SessionStore, session_id: str, path: str) -> None:
    print(format_rollback_session_file(store, session_id, path))


def format_profiles() -> str:
    lines = ["Available profiles:", ""]
    for profile in list_profiles():
        lines.append(f"  {profile['name']:15s} {profile['description']}")
    return "\n".join(lines)


def print_profiles() -> None:
    print(format_profiles())


def format_config_show(workspace: Path) -> str:
    lines = [
        "Harness config",
        f"api_key: {_redact_secret(config.API_KEY)}",
        f"base_url: {config.BASE_URL}",
        f"model: {config.MODEL}",
        f"model_intensity: {config.MODEL_INTENSITY}",
        *[
            f"model_profile_{intensity}: {profile.model}"
            f" thinking={profile.thinking}"
            f" reasoning_effort={profile.reasoning_effort or 'none'}"
            for intensity, profile in config.MODEL_PROFILES.items()
        ],
        f"workspace: {workspace}",
        f"permission_mode: {os.environ.get('HARNESS_PERMISSION_MODE', 'workspace-write')}",
        f"sandbox_mode: {config.SANDBOX_MODE}",
        f"docker_image: {config.DOCKER_IMAGE}",
        f"docker_network: {config.DOCKER_NETWORK}",
        f"docker_user: {config.DOCKER_USER or 'auto'}",
        f"provider: {config.PROVIDER}",
        f"stream: {config.STREAM}",
    ]
    if os.name == "nt":
        lines.append(f"windows_shell: {config.WINDOWS_SHELL} ({windows_shell_hint()})")
    lines.extend([
        "checkpoint_auto: off by default",
        f"compress_threshold: {config.COMPRESS_THRESHOLD}",
        "auto_reset: disabled",
        f"max_agent_iterations: {config.MAX_AGENT_ITERATIONS}",
        f"max_agent_total_tokens: {config.MAX_AGENT_TOTAL_TOKENS}",
        f"max_agent_tool_calls: {config.MAX_AGENT_TOOL_CALLS}",
        f"agent_budget_warn_fraction: {config.AGENT_BUDGET_WARN_FRACTION}",
    ])
    return "\n".join(lines)


def print_config_show(workspace: Path) -> None:
    print(format_config_show(workspace))


def format_doctor(workspace: Path, *, mcp_manager: McpClientManager | None = None) -> tuple[str, int]:
    rows = []
    rows.append(("API key", bool(config.API_KEY), "configured" if config.API_KEY else "missing OPENAI_API_KEY"))
    rows.append(("API base URL", bool(config.BASE_URL), config.BASE_URL or "missing OPENAI_BASE_URL"))
    rows.append(("Workspace", workspace.exists() and workspace.is_dir(), str(workspace)))
    rows.append(("Git", shutil_which("git") is not None, shutil_which("git") or "not installed"))
    if sandbox_mode() == "docker":
        docker = docker_cli_path()
        rows.append(("Docker CLI", docker is not None, docker_shell_hint()))
        rows.append(("Docker daemon", *_docker_daemon_doctor_status(docker)))
    else:
        shell = shell_path()
        rows.append(("Shell", shell is not None, shell or "no shell found"))
    rows.append(("MCP", *_mcp_doctor_status(workspace, mcp_manager=mcp_manager)))
    failures = sum(0 if ok else 1 for _, ok, _ in rows)
    lines = ["Harness doctor"]
    lines.extend(_format_doctor_line(label, ok, detail) for label, ok, detail in rows)
    return "\n".join(lines), failures


def run_doctor(workspace: Path) -> int:
    text, failures = format_doctor(workspace)
    print(text)
    return 0 if failures == 0 else 1


def _format_doctor_line(label: str, ok: bool, detail: str) -> str:
    return f"{'OK' if ok else 'FAIL':4s} {label:18s} {detail}"


def _mcp_doctor_status(workspace: Path, *, mcp_manager: McpClientManager | None = None) -> tuple[bool, str]:
    if mcp_manager is not None:
        return mcp_manager.doctor_status()
    try:
        cfg = load_mcp_config(workspace)
    except McpConfigError as exc:
        return False, str(exc)
    if not cfg.path.exists():
        return True, "not configured"
    return True, f"{len(cfg.servers)} configured server(s)"


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def _docker_daemon_doctor_status(docker_cli: str | None) -> tuple[bool, str]:
    if not docker_cli:
        return False, "Docker CLI not available"
    ok, detail = docker_info_check()
    return ok, detail


def shell_path() -> str | None:
    if sandbox_mode() == "docker":
        return docker_cli_path()
    if os.name == "nt":
        return windows_shell_path()
    return shutil_which("pwsh") or shutil_which("powershell") or os.environ.get("ComSpec")


def _redact_secret(value: str) -> str:
    if not value:
        return "unset"
    if len(value) <= 8:
        return "set"
    return f"{value[:4]}...{value[-4:]}"


def _build_resume_context(
    store: SessionStore,
    session_id: str,
    *,
    max_recent_events: int = 8,
) -> str:
    lineage = store.read_lineage(session_id)
    current = lineage[-1]
    journal_path = store.root / "sessions" / session_id / "journal.jsonl"
    if journal_path.exists() and journal_path.stat().st_size:
        recovered = SessionJournal(journal_path).recovery_messages("")
        lines = [
            f"Resuming session: {current.get('id', session_id)}",
            "The following state was rebuilt from the append-only session journal.",
            "Re-check current files before relying on old tool output.",
            "",
        ]
        for message in recovered[1:]:
            role = message.get("role", "unknown")
            content = str(message.get("content") or "")
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)
    lines = [
        f"Resuming session: {current.get('id', session_id)}",
        "Lineage: " + " -> ".join(item.get("id", "") for item in lineage),
        f"Workspace: {current.get('cwd', '')}",
        f"Profile: {current.get('profile', '')}",
        f"Permission mode: {current.get('permission_mode', '')}",
    ]
    if current.get("forked_from"):
        lines.append(f"Forked from: {current.get('forked_from')}")
    lines.append("")
    lines.append("Recent session events:")
    for metadata in lineage:
        events = store.read_events(metadata["id"])
        if not events:
            lines.append(f"- {metadata['id']}: no events")
            continue
        lines.append(f"- {metadata['id']}:")
        for event in events[-max_recent_events:]:
            lines.append(f"  - {_event_summary(event)}")
    return "\n".join(lines)


def _event_summary(event: dict) -> str:
    payload = event.get("payload") or {}
    payload_bits = []
    for key in sorted(payload)[:4]:
        value = payload[key]
        text = str(value).replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        payload_bits.append(f"{key}={text}")
    suffix = f" ({', '.join(payload_bits)})" if payload_bits else ""
    return f"#{event.get('sequence')} {event.get('type')} agent={event.get('agent')}{suffix}"
