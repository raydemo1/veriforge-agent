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
    from .core.interactive import InteractiveSession
    from .sessions.store import SessionStore

PRODUCT_DEFAULT_PROFILE = "general"


def InteractiveSession(**kwargs):
    """Lazy session constructor kept patchable for CLI tests and embedders."""
    from .core.interactive import InteractiveSession as Session

    return Session(**kwargs)


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

    parser = argparse.ArgumentParser(prog="veriforge", description="VeriForge interactive local coding agent")
    parser.add_argument("task", nargs="*", help="Optional first task to submit after startup")
    parser.add_argument("--profile", default=PRODUCT_DEFAULT_PROFILE, help="Profile name to use before the session starts")
    parser.add_argument("-p", "--print", dest="print_mode", action="store_true",
        help="Execute a single task and print results (no REPL)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--list-profiles", action="store_true", help="List profiles and exit")
    parser.add_argument("--no-alt-screen", action="store_true",
        help="Keep the OpenTUI frontend in the main terminal screen")
    parser.add_argument("--theme", choices=("auto", "dark", "light"),
        default=os.environ.get("VERIFORGE_THEME", "auto"), help="OpenTUI color theme")
    parser.add_argument("--icons", choices=("auto", "nerd", "unicode"),
        default=os.environ.get("VERIFORGE_ICONS", "auto"), help="OpenTUI icon set")
    args = parser.parse_args(argv)
    profile_explicit = any(arg == "--profile" or arg.startswith("--profile=") for arg in argv)
    is_tty = _is_interactive_tty()

    from .core.logging_config import setup_logging
    setup_logging(
        verbose=args.verbose,
        console=args.list_profiles or args.print_mode or not is_tty,
    )

    if args.list_profiles:
        from .core.formatters import print_profiles
        print_profiles()
        return 0

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
        return run_batch(
            cwd=cwd,
            profile_name=args.profile,
            profile_explicit=profile_explicit,
            first_task=first_task,
            stream_sink=stream_sink,
        )

    try:
        app = TuiApp(
            cwd=cwd,
            profile_name=args.profile,
            profile_explicit=profile_explicit,
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
    profile_name: str,
    first_task: str,
    stream_sink=None,
    profile_explicit: bool = False,
) -> int:
    from .attachments import AttachmentError
    from .core.interactive import print_turn_result
    from .core.mentions import MentionResolutionError

    try:
        session = InteractiveSession(
            cwd=cwd,
            profile_name=profile_name,
            profile_explicit=profile_explicit,
            stream_sink=stream_sink,
        )
    except Exception as e:
        _print_error(f"Error: {e}")
        return 1

    try:
        if first_task.startswith("/"):
            from .tui.commands import default_command_registry

            registry = default_command_registry(skill_registry=session.skill_registry)
            if not registry.is_agent_command(first_task):
                if session.session_id:
                    print(f"veriforge session: {session.session_id}")
                print(f"workspace: {session.cwd}")
                session.handle_slash_command(first_task)
                return 0

        result = session.submit(first_task)
        if session.session_id:
            print(f"veriforge session: {session.session_id}")
        print(f"workspace: {session.cwd}")
        print_turn_result(result)
    except (AttachmentError, MentionResolutionError) as e:
        _print_error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        _print_error("\nInterrupted.")
        return 130
    finally:
        session.close()
    return 0


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


def _latest_session_id(store: SessionStore) -> str:
    latest = store.latest_session()
    if latest is None:
        raise FileNotFoundError("No sessions found.")
    return latest["id"]


def _print_error(message: str) -> None:
    print(message, file=sys.stderr)



if __name__ == "__main__":
    raise SystemExit(main())
