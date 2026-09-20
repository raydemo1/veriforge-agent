"""Headless in-container runner used by the Claw-SWE-Bench adapter."""
from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    workspace = Path(args.workspace).resolve()

    os.environ.setdefault("HARNESS_WORKSPACE", str(workspace))
    os.environ.setdefault("HARNESS_PERMISSION_MODE", "danger-full-access")
    os.environ.setdefault("HARNESS_STREAM", "0")
    os.environ.setdefault("HARNESS_MEMORY_DISABLED", "1")
    os.environ.setdefault("HARNESS_MENTION_MODE", "off")

    try:
        from harness_code_agent.core.interactive import (
            InteractiveSession,
            print_turn_result,
        )

        session = InteractiveSession(
            cwd=workspace,
            profile_name="coding-agent",
            profile_explicit=True,
            stream_sink=None,
        )
        session.checkpoint.auto = False
        try:
            result = session.submit(prompt)
            if session.session_id:
                print(f"veriforge session: {session.session_id}")
            print(f"workspace: {session.cwd}")
            print_turn_result(result)
        finally:
            session.close()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VeriForge on a Claw-SWE-Bench prompt.")
    parser.add_argument("prompt_file")
    parser.add_argument("--workspace", default="/testbed")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
