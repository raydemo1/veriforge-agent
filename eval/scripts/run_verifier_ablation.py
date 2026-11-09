"""Paired static-verifier and coding-agent prompt pilots. Use --run for model requests."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "eval" / "tasks" / "verifier_ablation.json"
IMAGE = "veriforge-verifier-eval:local"
PILOT_RUNTIME_REVISION = "b2391f303663549903b8d578f71da2016e0826d6"
EXPERIMENT_GROUPS = {"verifier": ("A", "B", "C")}
TEST_COMMAND = re.compile(
    r"\b(?:pytest|unittest|ctest)\b|\b(?:python\S*\s+\S*test\S*\.py|(?:npm|cargo|go)\s+test)\b"
    r"|\bpython\S*\s+.*(?:\n|;|\s)\s*assert\s",
    re.DOTALL,
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def worker(group, prompt):
    import shutil
    from unittest.mock import patch

    from harness_code_agent.core import interactive
    from harness_code_agent.runtime.approvals import NoApprovalProvider
    from harness_code_agent.runtime.middleware.verification import (
        StaticVerifierMiddleware,
    )
    from harness_code_agent.runtime.questions import NoQuestionProvider

    interventions = []
    capture_seconds = 0.0
    build_stack = interactive.build_main_agent_middlewares
    verifier_group = group

    def stack(**kwargs):
        if verifier_group == "A":
            # Build the production stack without paying for the verifier's initial scan.
            with patch.object(StaticVerifierMiddleware, "__init__", return_value=None):
                middlewares = build_stack(**kwargs)
        else:
            middlewares = build_stack(**kwargs)
        verifier = next(
            m for m in middlewares if isinstance(m, StaticVerifierMiddleware)
        )
        if verifier_group == "A":
            middlewares.remove(verifier)
            return middlewares

        pre_exit = verifier.pre_exit

        def observe(messages, runtime_state=None, agent_name=None):
            nonlocal capture_seconds
            injection = pre_exit(messages, runtime_state, agent_name)
            if verifier_group == "B" or not injection:
                return None
            capture_started = time.monotonic()
            if not interventions:
                shutil.copytree(
                    Path.cwd(),
                    "/trial/first-intervention",
                    symlinks=True,
                    ignore=shutil.ignore_patterns(".git", ".harness", "__pycache__"),
                )
            interventions.append({"timestamp": time.time(), "message": injection})
            write_json(Path("/trial/interventions.json"), interventions)
            capture_seconds += time.monotonic() - capture_started
            return injection

        verifier.pre_exit = observe
        return middlewares

    # Docker Desktop's Windows bind mounts can make every fixture executable.
    shutil.copytree(
        "/trial/seed", Path.cwd(), copy_function=shutil.copyfile, dirs_exist_ok=True
    )
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Eval",
            "-c",
            "user.email=eval@localhost",
            "commit",
            "-qm",
            "Initial fixture",
        ],
        check=True,
    )
    with patch.object(interactive, "build_main_agent_middlewares", stack):
        session = interactive.InteractiveSession(
            cwd=Path.cwd(),
            enable_turn_summary=False,
            approval_provider=NoApprovalProvider(),
            question_provider=NoQuestionProvider(),
        )
        started = time.monotonic()
        try:
            interactive.print_turn_result(session.submit(prompt))
        finally:
            elapsed = time.monotonic() - started - capture_seconds
            session.close()
            write_json(
                Path("/trial/session.json"),
                {"id": session.session_id, "elapsed": elapsed},
            )


def plan_trials(tasks, repeats, seed, groups):
    rng = random.Random(seed)
    offsets = {task["name"]: index + seed for index, task in enumerate(tasks)}
    plan = []
    for repeat in range(1, repeats + 1):
        order = list(tasks)
        rng.shuffle(order)
        for task in order:
            offset = (repeat - 1 + offsets[task["name"]]) % len(groups)
            plan.extend(
                (task["name"], repeat, group)
                for group in groups[offset:] + groups[:offset]
            )
    return plan


def run_container(
    arguments, *, log, timeout, env=None, input_text=None, export_workspace=None
):
    name = "vf-abl-" + os.urandom(8).hex()
    command = ["docker", "run", "--name", name, "-i"]
    if export_workspace is None:
        command.append("--rm")
    with log.open("w", encoding="utf-8") as stream:
        try:
            status = subprocess.run(
                [*command, *arguments],
                env=env,
                input=input_text or "",
                encoding="utf-8",
                errors="replace",
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            ).returncode
        except (KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
            subprocess.run(
                ["docker", "kill", name],
                capture_output=True,
                check=False,
                timeout=30,
            )
            if isinstance(exc, KeyboardInterrupt):
                raise
            status = 124
        finally:
            if export_workspace is not None:
                export_workspace.mkdir()
                try:
                    copied = subprocess.run(
                        ["docker", "cp", f"{name}:/workspace/.", str(export_workspace)],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=30,
                    )
                finally:
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        capture_output=True,
                        check=False,
                        timeout=30,
                    )
    return 125 if export_workspace is not None and copied.returncode else status


def mount(source, target, *, readonly=True):
    return [
        "--mount",
        f"type=bind,source={source},target={target}"
        + (",readonly" if readonly else ""),
    ]


def judge(workspace, task, log):
    marker = "hidden-test-completed-" + os.urandom(16).hex()
    code = (
        "import sys\nsys.path.insert(0, '/candidate')\n"
        + task["hidden_test"]
        + f"\nprint({marker!r})\n"
    )
    status = run_container(
        [
            "--network",
            "none",
            *mount(workspace, "/candidate"),
            "-w",
            "/candidate",
            IMAGE,
            "python",
            "-IB",
            "-",
        ],
        log=log,
        timeout=30,
        input_text=code,
    )
    if status in {124, 125, 126, 127}:
        return None
    return (
        status == 0
        and marker in log.read_text(encoding="utf-8", errors="replace").splitlines()
    )


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def trial_metrics(trial, args):
    workspace = trial / "workspace"
    traces = [
        row
        for path in (workspace / ".harness" / "traces").glob("*/*.jsonl")
        for row in read_lines(path)
    ]
    requests = [row for row in traces if row["event"] == "request_diagnostics"]
    first_request = min(requests, key=lambda row: row["timestamp"], default={})
    started = sum(
        row["event"] == "llm_request" or row.get("purpose") == "summary"
        for row in traces
    )
    interventions_path = trial / "interventions.json"
    interventions = (
        json.loads(interventions_path.read_text(encoding="utf-8"))
        if interventions_path.exists()
        else []
    )
    first = interventions[0]["timestamp"] if interventions else math.inf
    session_path = trial / "session.json"
    session = (
        json.loads(session_path.read_text(encoding="utf-8"))
        if session_path.exists()
        else {}
    )
    events = [
        row
        for path in (workspace / ".harness" / "sessions").glob("*/events.jsonl")
        for row in read_lines(path)
    ]
    checks = [
        row["payload"]["checks"] for row in events if row["type"] == "checks_recorded"
    ]
    tools = [row["payload"] for row in events if row["type"] == "tool_call"]
    fallback_reasons = [
        row["payload"].get("reason")
        for row in events
        if row["type"] == "agent_fallback"
    ]
    test_commands = [
        command
        for call in tools
        if call["tool"] == "run_bash"
        and isinstance(command := call["args"].get("command"), str)
        and TEST_COMMAND.search(command)
    ]
    usage = [
        row
        for row in requests
        if row.get("prompt_tokens") is not None
        and row.get("completion_tokens") is not None
    ]
    tokens = Counter()
    for row in usage:
        hits = row.get("cache_hit_tokens") or 0
        tokens.update(
            {
                "cache_hit": hits,
                "cache_miss": row.get("cache_miss_tokens")
                or row["prompt_tokens"] - hits,
                "output": row["completion_tokens"],
            }
        )
    complete = len(usage) == len(requests) == started and started > 0
    model_cost = (
        sum(tokens[key] * rate for key, rate in args.rates.items()) / 1_000_000
        if complete
        else None
    )
    tool_cost = len(tools) * args.tool_cost
    return {
        "agent_seconds": session.get("elapsed"),
        "fallback_reasons": fallback_reasons,
        "llm_calls": started,
        "llm_calls_after_first_intervention": sum(
            datetime.fromisoformat(row["timestamp"]).timestamp() >= first
            for row in requests
        ),
        "usage_complete": complete,
        "first_request_prompt_tokens": first_request.get("prompt_tokens"),
        "first_request_tools_hash": first_request.get("tools_hash"),
        "model_errors": [
            {"type": row["error_type"], "status": row.get("error_status")}
            for row in requests
            if row.get("error_type")
        ],
        "tokens": dict(tokens),
        "model_cost_usd": model_cost,
        "tool_calls": len(tools),
        "tool_cost_usd": tool_cost,
        "total_cost_usd": None if model_cost is None else model_cost + tool_cost,
        "test_commands": test_commands,
        "repeated_test_commands": len(test_commands) - len(set(test_commands)),
        "static_check_batches": checks,
        "interventions": interventions,
    }


def summarize(rows, groups):
    summaries = {}
    for group in groups:
        trials = [row for row in rows if row["group"] == group]
        passed = sum(row["passed"] is True for row in trials)
        judged = sum(row["passed"] is not None for row in trials)
        latencies = sorted(
            row["agent_seconds"] for row in trials if row["agent_seconds"] is not None
        )
        costs = [row["total_cost_usd"] for row in trials]
        total = (
            sum(costs) if costs and all(cost is not None for cost in costs) else None
        )
        summaries[group] = {
            "trials": len(trials),
            "judged": judged,
            "timeouts": sum(row["agent_exit"] == 124 for row in trials),
            "fallback_trials": sum(bool(row["fallback_reasons"]) for row in trials),
            "passed": passed,
            "success_rate": passed / judged if judged else None,
            "p50_agent_seconds": latencies[math.ceil(len(latencies) * 0.50) - 1]
            if latencies
            else None,
            "p95_agent_seconds": latencies[math.ceil(len(latencies) * 0.95) - 1]
            if latencies
            else None,
            "total_cost_usd": total,
            "cost_per_success_usd": total / passed
            if total is not None and passed and judged == len(trials)
            else None,
            "llm_calls": sum(row["llm_calls"] for row in trials),
            "tool_calls": sum(row["tool_calls"] for row in trials),
            "llm_calls_after_first_intervention": sum(
                row["llm_calls_after_first_intervention"] for row in trials
            ),
            "repeated_test_commands": sum(
                row["repeated_test_commands"] for row in trials
            ),
            "static_check_runs": sum(
                len(row["static_check_batches"]) for row in trials
            ),
            "interventions": sum(len(row["interventions"]) for row in trials),
            "first_intervention_already_passing": sum(
                row["before_intervention_passed"] is True for row in trials
            ),
            "failed_before_intervention_passed_after": sum(
                row["before_intervention_passed"] is False and row["passed"] is True
                for row in trials
            ),
            "passed_before_intervention_failed_after": sum(
                row["before_intervention_passed"] is True and row["passed"] is False
                for row in trials
            ),
        }
    return summaries


def make_workspace(path, task):
    for name, content in task["files"].items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def build_image():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = shlex.join([*project["project"]["dependencies"], "ruff"])
    dockerfile = (
        f"FROM python:3.12-bookworm\nRUN pip install --no-cache-dir {packages}\n"
    )
    subprocess.run(
        ["docker", "build", "-t", IMAGE, "-"], input=dockerfile, text=True, check=True
    )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.stdin.read())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENT_GROUPS, default="verifier")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--tool-budget", type=int, default=24)
    parser.add_argument("--tool-cost", type=float, default=0, help="USD per tool call")
    parser.add_argument(
        "--rates",
        type=float,
        nargs=3,
        metavar=("HIT", "MISS", "OUTPUT"),
        help="Common USD rates per million tokens, applied to all groups",
    )
    args = parser.parse_args()
    if min(args.repeats, args.timeout, args.tool_budget) < 1:
        parser.error("repeats, timeout, and tool budget must be positive")
    if args.build_image:
        build_image()
    tasks = {
        task["name"]: task for task in json.loads(TASKS.read_text(encoding="utf-8"))
    }
    groups = EXPERIMENT_GROUPS[args.experiment]
    plan = plan_trials(list(tasks.values()), args.repeats, args.seed, groups)
    if not args.run:
        print(
            json.dumps(
                {"plan": plan, "settings": vars(args)},
                indent=2,
            )
        )
        return
    if args.rates is None or any(
        not math.isfinite(rate) or rate < 0 for rate in [*args.rates, args.tool_cost]
    ):
        parser.error("--run requires nonnegative finite --rates and --tool-cost")
    args.rates = dict(zip(("cache_hit", "cache_miss", "output"), args.rates))
    sys.path.insert(0, str(ROOT))
    from eval.benchmarks.harbor_env import runner_env_vars
    from harness_code_agent import config

    settings = runner_env_vars()
    if not settings.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is missing")
    settings.update(
        {
            "PYTHONPATH": "/runtime",
            "PYTHONUNBUFFERED": "1",
            "HARNESS_PERMISSION_MODE": "danger-full-access",
            "HARNESS_MEMORY_DISABLED": "1",
            "HARNESS_MEMORY_GENERATION_DISABLED": "1",
            "HARNESS_STREAM": "0",
            "HARNESS_MODEL_INPUT_MODE": config.MODEL_INPUT_MODE,
            "HARNESS_LLM_MAX_RETRIES": "0",
            "HARNESS_LLM_REQUEST_TIMEOUT_SECONDS": str(min(args.timeout, 90)),
            "MAX_AGENT_TOTAL_TOKENS": "0",
            "MAX_AGENT_TOOL_CALLS": str(args.tool_budget),
        }
    )
    run_root = (
        ROOT
        / "eval"
        / "results"
        / (
            args.experiment
            + datetime.now(timezone.utc).strftime("-ablation-%Y%m%d-%H%M%S-%f")
        )
    )
    run_root.mkdir(parents=True)
    runtime = run_root / "runtime"
    # Keep the original gates and prompts available for reproduction without
    # adding legacy behavior to the production runtime.
    snapshot = subprocess.check_output(
        [
            "git",
            "archive",
            "--format=zip",
            PILOT_RUNTIME_REVISION,
            "harness_code_agent",
        ],
        cwd=ROOT,
    )
    with ZipFile(BytesIO(snapshot)) as archive:
        archive.extractall(runtime)
    metadata = {
        "settings": vars(args),
        "runtime_revision": PILOT_RUNTIME_REVISION,
        "permission_mode": settings["HARNESS_PERMISSION_MODE"],
        "cumulative_token_budget": None,
        "model": vars(config.MODEL_PROFILE),
        "plan": plan,
        "image": subprocess.check_output(
            ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True
        ).strip(),
    }
    write_json(run_root / "protocol.json", metadata)
    for name, task in tasks.items():
        baseline = run_root / "baseline" / name
        make_workspace(baseline, task)
        if judge(baseline, task, baseline.parent / f"{name}.log") is not False:
            raise SystemExit(
                f"Invalid baseline or unavailable judge for {name}. See {baseline.parent}."
            )

    rows = []
    for name, repeat, group in plan:
        print(f"{len(rows) + 1}/{len(plan)} {name} r{repeat} {group}", flush=True)
        task = tasks[name]
        trial = run_root / f"r{repeat}-{name}-{group}"
        workspace = trial / "workspace"
        make_workspace(trial / "seed", task)
        code = run_container(
            [
                *mount(trial, "/trial", readonly=False),
                *mount(runtime / "harness_code_agent", "/runtime/harness_code_agent"),
                *mount(
                    Path(__file__).resolve(),
                    "/runtime/eval/scripts/run_verifier_ablation.py",
                ),
                "-w",
                "/workspace",
                *[flag for key in settings for flag in ("--env", key)],
                IMAGE,
                "python",
                "-B",
                "/runtime/eval/scripts/run_verifier_ablation.py",
                "--worker",
                group,
            ],
            env={**os.environ, **settings},
            input_text=task["prompt"],
            log=trial / "agent.log",
            timeout=args.timeout,
            export_workspace=workspace,
        )
        first = trial / "first-intervention"
        row = {
            "task": name,
            "category": task["category"],
            "repeat": repeat,
            "group": group,
            "agent_exit": code,
            "passed": None
            if code in {125, 126, 127}
            else judge(workspace, task, trial / "judge.log"),
            "before_intervention_passed": judge(first, task, trial / "first-judge.log")
            if first.exists()
            else None,
            **trial_metrics(trial, args),
        }
        rows.append(row)
        write_json(
            run_root / "ablation.json",
            {**metadata, "groups": summarize(rows, groups), "trials": rows},
        )
        if code in {125, 126, 127} or (code and row["llm_calls"] == 0):
            raise SystemExit(f"Agent setup failed. See {trial / 'agent.log'}.")
        if any(error["status"] in {401, 402, 403} for error in row["model_errors"]):
            raise SystemExit(
                f"Model API rejected access or billing. See {trial / 'agent.log'}."
            )
    print(run_root)


if __name__ == "__main__":
    main()
