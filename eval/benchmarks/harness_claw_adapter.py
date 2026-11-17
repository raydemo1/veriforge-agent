"""Claw-SWE-Bench adapter for VeriForge.

This module is imported by eval/benchmarks/run_claw_swe_bench.py after the
upstream claw-swe-bench repository has been added to PYTHONPATH.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.benchmarks.container_python import (
    CONTAINER_PYTHON,
    PYTHON_STANDALONE_TARBALL,
    bootstrap_python,
)

try:
    from claw_swebench.claws.base import BaseClawAdapter
    from claw_swebench.types import AgentResult
except ImportError:  # Allows local unit tests to import helper code without upstream installed.
    class BaseClawAdapter:  # type: ignore[no-redef]
        def __init__(self, model: str, timeout: int, max_turns: int | None = None):
            self.model = model
            self.timeout = timeout
            self.max_turns = max_turns

    @dataclass
    class AgentResult:  # type: ignore[no-redef]
        success: bool
        timeout: bool
        exit_code: int
        finish_reason: str
        stdout_path: Path | None = None
        stderr_path: Path | None = None
        session_id: str | None = None
        duration_seconds: float = 0.0
        usage: dict[str, Any] = field(default_factory=dict)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_REPO = "/opt/harness-code-agent"
CONTAINER_PROMPT = "/tmp/hca_claw_prompt.txt"
RUNNER = f"{CONTAINER_REPO}/eval/benchmarks/hca_claw_runner.py"

FORWARDED_ENV_PREFIXES = (
    "HARNESS_",
    "OPENAI_",
    "DEEPSEEK_",
    "MAX_AGENT_",
    "COMPRESS_THRESHOLD",
)


class HarnessCodeAgentAdapter(BaseClawAdapter):
    """Runs VeriForge inside each SWE-bench container."""

    name = "veriforge"

    def __init__(
        self,
        model: str,
        timeout: int,
        max_turns: int | None = None,
        *,
        repo_root: str | Path | None = None,
        install_deps: bool = True,
    ):
        super().__init__(model=model, timeout=timeout, max_turns=max_turns)
        self.repo_root = Path(repo_root or os.environ.get("HARNESS_CLAW_AGENT_REPO") or PROJECT_ROOT).resolve()
        self.install_deps = install_deps
        self.python = CONTAINER_PYTHON if install_deps else "python3.14"

    def container_run_args(self, instance_id: str) -> list[str]:
        args = ["-v", f"{_docker_path(self.repo_root)}:{CONTAINER_REPO}:ro"]
        for key, value in _forwarded_env().items():
            args.extend(["-e", f"{key}={value}"])
        args.extend([
            "-e",
            "HARNESS_WORKSPACE=/testbed",
            "-e",
            "HARNESS_PERMISSION_MODE=danger-full-access",
            "-e",
            "HARNESS_STREAM=0",
            "-e",
            "HARNESS_MEMORY_DISABLED=1",
            "-e",
            "HARNESS_MENTION_MODE=off",
            "-e",
            "MAX_AGENT_TOTAL_TOKENS=900000",
            "-e",
            f"HARNESS_MODEL={self.model}",
            "-e",
            f"HARNESS_MODEL_FAST={self.model}",
            "-e",
            f"HARNESS_MODEL_NORMAL={self.model}",
            "-e",
            f"HARNESS_MODEL_HARD={self.model}",
            "-e",
            f"HARNESS_MODEL_MAX={self.model}",
            "-e",
            f"HARNESS_MODEL_INTENSITY={os.environ.get('HARNESS_MODEL_INTENSITY', 'normal')}",
        ])
        # max_turns stays accepted for upstream CLI compatibility but is
        # intentionally not applied: runs are bounded by the task time budget
        # (HARNESS_TASK_BUDGET) plus the hard process timeout, not by a step
        # guess.
        return args

    def post_container_start(self, workspace) -> None:
        if not self.install_deps:
            return
        result = workspace.run_in_container(_bootstrap_command(), timeout=900)
        if result.exit_code != 0:
            raise RuntimeError(
                "Failed to bootstrap VeriForge runtime in container: "
                f"{result.stderr or result.stdout}"
            )

    def send_task(
        self,
        prompt: str,
        agent_id: str,
        container_name: str,
        artifact_dir: Path | None = None,
        instance_id: str | None = None,
    ) -> AgentResult:
        artifact_dir = artifact_dir or Path(tempfile.mkdtemp(prefix="hca-claw-"))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = artifact_dir / "hca_prompt.txt"
        stdout_path = artifact_dir / "agent_stdout.log"
        stderr_path = artifact_dir / "agent_stderr.log"
        prompt_path.write_text(prompt, encoding="utf-8")

        copy = subprocess.run(
            ["docker", "cp", str(prompt_path), f"{container_name}:{CONTAINER_PROMPT}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if copy.returncode != 0:
            stderr_path.write_text(copy.stderr, encoding="utf-8")
            return AgentResult(
                success=False,
                timeout=False,
                exit_code=copy.returncode,
                finish_reason="copy_prompt_failed",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )

        command = _agent_command(timeout=self.timeout, python=self.python)
        started = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(
                ["docker", "exec", container_name, "bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=self.timeout + 120,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            stdout = _decode(exc.stdout)
            stderr = _decode(exc.stderr) + f"\nTIMEOUT after {self.timeout + 120}s"

        duration = time.perf_counter() - started
        stdout_path.write_text(stdout, encoding="utf-8", newline="\n")
        stderr_path.write_text(stderr, encoding="utf-8", newline="\n")
        session_id = _session_id(stdout)
        return AgentResult(
            success=exit_code == 0 and not timed_out,
            timeout=timed_out,
            exit_code=exit_code,
            finish_reason="timeout" if timed_out else ("stop" if exit_code == 0 else "error"),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            session_id=session_id,
            duration_seconds=duration,
        )

    def collect_usage(self, workspace, artifact_dir: Path) -> dict:
        usage_result = workspace.run_in_container(_usage_command(python=self.python), timeout=120)
        usage: dict[str, Any] = {}
        if usage_result.exit_code == 0 and usage_result.stdout.strip():
            try:
                usage = json.loads(usage_result.stdout)
            except json.JSONDecodeError:
                usage = {"raw_usage_error": usage_result.stdout[:2000]}
        elif usage_result.stderr:
            usage = {"usage_error": usage_result.stderr[:2000]}

        state_dest = artifact_dir / "harness_state"
        if state_dest.exists():
            shutil.rmtree(state_dest, ignore_errors=True)
        subprocess.run(
            ["docker", "cp", f"{workspace.container_name}:/root/.veriforge/workspaces", str(state_dest)],
            capture_output=True,
            check=False,
        )
        if usage:
            (artifact_dir / "harness_usage.json").write_text(
                json.dumps(usage, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return usage


def _agent_command(*, timeout: int, python: str = CONTAINER_PYTHON) -> str:
    env = {
        "PYTHONPATH": CONTAINER_REPO,
        "HARNESS_WORKSPACE": "/testbed",
        "HARNESS_PERMISSION_MODE": "danger-full-access",
        "HARNESS_STREAM": "0",
        "HARNESS_MEMORY_DISABLED": "1",
        "HARNESS_MENTION_MODE": "off",
        "MAX_AGENT_TOTAL_TOKENS": "900000",
        "HARNESS_TASK_BUDGET": str(timeout),
    }
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return f"cd /testbed && {prefix} {shlex.quote(python)} {shlex.quote(RUNNER)} {shlex.quote(CONTAINER_PROMPT)}"


def _bootstrap_command() -> str:
    provision = bootstrap_python(
        f"{CONTAINER_REPO}/vendor_wheels/{PYTHON_STANDALONE_TARBALL}",
        download_url=os.environ.get("HARNESS_CLAW_PYTHON_URL"),
    )
    return f"""
set -e
command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq curl)
{provision}
{CONTAINER_PYTHON} -m ensurepip --upgrade
# Build from a writable snapshot of the package files because the repo mount is read-only.
source_dir=$(mktemp -d)
trap 'rm -rf "$source_dir"' EXIT
cp -a {CONTAINER_REPO}/pyproject.toml {CONTAINER_REPO}/README.md {CONTAINER_REPO}/harness_code_agent "$source_dir/"
{CONTAINER_PYTHON} -m pip install -q --find-links={CONTAINER_REPO}/vendor_wheels "$source_dir"
""".strip()


def _usage_command(*, python: str = CONTAINER_PYTHON) -> str:
    return f"""
cd /testbed
PYTHONPATH={shlex.quote(CONTAINER_REPO)} {shlex.quote(python)} - <<'PY'
import json
from pathlib import Path
from harness_code_agent.sessions.observability import build_session_observability
from harness_code_agent.sessions.store import SessionStore
from harness_code_agent.workspace.state_paths import ensure_workspace_state

store = SessionStore(ensure_workspace_state('/testbed'))
sessions = store.list_sessions()
if not sessions:
    print(json.dumps({{}}))
    raise SystemExit(0)
latest = sessions[0]['id']
metadata = store.read_metadata(latest)
events = store.read_events(latest)
snapshot = build_session_observability(metadata, events).to_dict()
snapshot['session_id'] = latest
print(json.dumps(snapshot, ensure_ascii=False))
PY
""".strip()


def _forwarded_env() -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in os.environ.items():
        if value and any(key == prefix.rstrip("_") or key.startswith(prefix) for prefix in FORWARDED_ENV_PREFIXES):
            result[key] = value
    return result


def _docker_path(path: Path) -> str:
    return path.resolve().as_posix()


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _session_id(stdout: str) -> str | None:
    match = re.search(r"^veriforge session:\s*(\S+)", stdout, flags=re.MULTILINE)
    return match.group(1) if match else None
