"""VeriForge command-line interface."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .sessions.store import SessionStore

def TuiApp(**kwargs):
    """Lazy OpenTUI constructor so batch commands stay lightweight."""
    from .opentui_launcher import OpenTuiApp as App

    return App(**kwargs)


def _configure_stdio_encoding() -> None:
    """Reconfigure stdout/stderr to use UTF-8 on Windows.

    Without this, print() of CJK characters fails with UnicodeEncodeError
    on GBK terminals, or produces garbled output.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):  # non-TTY or already configured — best-effort
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio_encoding()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        _print_error("Error: 'run' is no longer supported. Use 'veriforge \"<task>\"' or start 'veriforge' interactively.")
        return 1
    if argv == ["session", "show", "latest"]:
        return show_latest_session(Path.cwd())
    if len(argv) >= 2 and argv[:2] == ["session", "observe"]:
        return observe_session(Path.cwd(), argv[2:])
    if argv and argv[0] == "mcp":
        return mcp_command(Path.cwd(), argv[1:])

    parser = argparse.ArgumentParser(prog="veriforge", description="VeriForge interactive local coding agent")
    parser.add_argument("task", nargs="*", help="Optional first task to submit after startup")
    parser.add_argument("-p", "--print", dest="print_mode", action="store_true",
        help="Execute a single task and print results (no REPL)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-alt-screen", action="store_true",
        help="Keep the OpenTUI frontend in the main terminal screen")
    parser.add_argument("--theme", choices=("auto", "dark", "light"),
        default=os.environ.get("VERIFORGE_THEME", "auto"), help="OpenTUI color theme")
    parser.add_argument("--icons", choices=("auto", "nerd", "unicode"),
        default=os.environ.get("VERIFORGE_ICONS", "auto"), help="OpenTUI icon set")
    args = parser.parse_args(argv)
    is_tty = _is_interactive_tty()

    from .core.logging_config import setup_logging
    setup_logging(
        verbose=args.verbose,
        console=args.print_mode or not is_tty,
    )

    first_task = " ".join(args.task).strip()

    if not config.API_KEY:
        _print_error("Error: Set OPENAI_API_KEY in .env or environment.")
        return 1

    cwd = Path.cwd()

    if args.print_mode or not is_tty:
        if not first_task and not args.print_mode:
            # Only read stdin when auto-degrading (no TTY). -p requires argv.
            first_task = sys.stdin.read().strip()
        if not first_task:
            _print_error("Error: no task provided (pass a task argument or pipe input)")
            return 2
        try:
            stream_sink = _build_stream_callback()
        except ValueError as e:
            _print_error(f"Error: {e}")
            return 2
        return run_batch(cwd=cwd, first_task=first_task, stream_sink=stream_sink)

    try:
        app = TuiApp(
            cwd=cwd,
            first_task=first_task,
            no_alt_screen=args.no_alt_screen,
            theme=args.theme,
            icons=args.icons,
        )
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1
    return app.run()


def run_batch(
    *,
    cwd: Path,
    first_task: str,
    stream_sink=None,
) -> int:
    from .headless import print_run_result, run_task

    result = run_task(cwd=cwd, task=first_task, stream_sink=stream_sink)
    print_run_result(result)
    if result.error:
        _print_error(result.error.rstrip())
    return result.exit_code


def _is_interactive_tty() -> bool:
    return bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )


def _build_stream_callback():
    value = (config.STREAM or "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        enabled = True
    elif value in {"0", "false", "no", "off"}:
        enabled = False
    elif value == "auto":
        enabled = bool(getattr(sys.stdout, "isatty", lambda: False)())
    else:
        raise ValueError("HARNESS_STREAM must be auto, 1, or 0")
    if not enabled:
        return None

    def stream_callback(delta: str) -> None:
        sys.stdout.write(delta)
        sys.stdout.flush()

    return stream_callback


def show_latest_session(cwd: Path) -> int:
    from .core.formatters import print_session
    from .sessions.store import SessionStore

    store = SessionStore(cwd / ".harness")
    try:
        print_session(store, _latest_session_id(store))
    except (FileNotFoundError, ValueError, KeyError) as e:
        _print_error(f"Error: {e}")
        return 1
    return 0


def observe_session(cwd: Path, args: list[str]) -> int:
    from .sessions.observability import (
        export_observability_report,
        format_export_result,
        format_project_observability,
        format_session_observability,
    )
    from .sessions.store import SessionStore

    export = False
    target_args = list(args)
    if "--export" in target_args:
        export = True
        target_args.remove("--export")
    if len(target_args) != 1:
        _print_error("Error: Usage: veriforge session observe latest|<session-id>|project [--export]")
        return 2

    target = target_args[0]
    store = SessionStore(cwd / ".harness")
    try:
        if target == "project":
            if export:
                print(format_export_result(export_observability_report(store, mode="project")))
            else:
                print(format_project_observability(store))
            return 0

        session_id = _latest_session_id(store) if target == "latest" else target
        if export:
            print(format_export_result(export_observability_report(store, mode="current", session_id=session_id)))
        else:
            print(format_session_observability(store, session_id))
    except (FileNotFoundError, ValueError, KeyError) as e:
        _print_error(f"Error: {e}")
        return 1
    return 0


def mcp_command(cwd: Path, args: list[str]) -> int:
    from .runtime.mcp import McpClientManager, McpConfigError, build_server_config

    parser = argparse.ArgumentParser(prog="veriforge mcp", description="Manage registered MCP servers")
    subparsers = parser.add_subparsers(dest="command")
    add = subparsers.add_parser("add", help="Register an MCP server")
    add.add_argument("name")
    add.add_argument("--url", help="Streamable HTTP endpoint")
    add.add_argument("--command", dest="stdio_command", help="stdio server command")
    add.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="stdio environment variable")
    add.add_argument("--header", action="append", default=[], metavar="KEY=VALUE", help="HTTP header")
    add.add_argument("--cwd", help="stdio working directory")
    add.add_argument("--permission", help="Server permission (read, network_read, dangerous)")
    add.add_argument("--title", default="")
    add.add_argument("--description", default="")
    add.add_argument("args", nargs="*", help="stdio command arguments")
    subparsers.add_parser("list", help="List registered MCP servers")
    remove = subparsers.add_parser("remove", help="Remove a registered MCP server")
    remove.add_argument("name")
    describe = subparsers.add_parser("describe", help="Generate one server description")
    describe.add_argument("name")
    describe.add_argument("--refresh", action="store_true", help="Regenerate an existing description")

    parsed, extras = parser.parse_known_args(args)
    if parsed.command is None:
        parser.print_help()
        return 2
    if extras:
        command_args = list(extras)
        if command_args[0] == "--":
            command_args = command_args[1:]
        if parsed.command != "add" or parsed.stdio_command is None:
            parser.error(f"unrecognized arguments: {' '.join(extras)}")
        parsed.args = list(parsed.args) + command_args

    manager = McpClientManager.from_workspace(cwd)
    try:
        if parsed.command == "list":
            print(manager.registered_report())
            return 0
        if parsed.command == "remove":
            try:
                print(manager.remove_server(parsed.name))
            except McpConfigError as exc:
                _print_error(f"Error: {exc}")
                return 1
            return 0
        if parsed.command == "describe":
            if manager.config_error:
                _print_error(f"Error: {manager.config_error}")
                return 1
            if error := manager.config.server_errors.get(parsed.name):
                _print_error(f"Error: {error}")
                return 1
            if parsed.name not in manager.config.servers:
                _print_error(f"Error: MCP server is not registered or is disabled: {parsed.name}")
                return 1
            manager.discover(parsed.name)
            try:
                print(manager.generate_server_description(parsed.name, refresh=parsed.refresh))
            except McpConfigError as exc:
                _print_error(f"Error: {exc}")
                return 1
            return 0

        try:
            server = build_server_config(
                name=parsed.name,
                url=parsed.url,
                command=parsed.stdio_command,
                args=parsed.args,
                env=_parse_pairs(parsed.env),
                cwd=parsed.cwd,
                headers=_parse_pairs(parsed.header),
                permission=parsed.permission,
                title=parsed.title,
                description=parsed.description,
            )
            registration = manager.register_server(server)
        except ValueError as exc:
            _print_error(f"Error: {exc}")
            return 1

        state = "connected" if registration.state == "connected" else f"connection failed: {registration.error}"
        action = "updated" if registration.replaced else "registered"
        print(f"MCP server {action}: {server.name} ({state}, {registration.tool_count} tool(s))")
        if registration.state == "connected" and not registration.server.description:
            try:
                print(manager.generate_server_description(server.name))
            except McpConfigError as exc:
                print(f"Description not generated: {exc}", file=sys.stderr)
        return 0 if registration.state == "connected" else 1
    finally:
        # One-shot CLI processes own the stdio/HTTP resources they started.
        manager.close()


def _parse_pairs(values: list[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        pairs[key.strip()] = value
    return pairs


def _latest_session_id(store: SessionStore) -> str:
    latest = store.latest_session()
    if latest is None:
        raise FileNotFoundError("No sessions found.")
    return latest["id"]


def _print_error(message: str) -> None:
    print(message, file=sys.stderr)



if __name__ == "__main__":
    raise SystemExit(main())
