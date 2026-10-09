"""Headless in-container runner used by the Claw-SWE-Bench adapter."""
from __future__ import annotations

import argparse
import os
import sys
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
        from harness_code_agent.headless import print_run_result, run_task

        result = run_task(
            cwd=workspace,
            task=prompt,
            profile="coding-agent",
            profile_explicit=True,
        )
        print_run_result(result)
        if result.error:
            print(result.error.rstrip(), file=sys.stderr)
        return result.exit_code
    except Exception:
        traceback.print_exc()
        return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VeriForge on a Claw-SWE-Bench prompt.")
    parser.add_argument("prompt_file")
    parser.add_argument("--workspace", default="/testbed")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
