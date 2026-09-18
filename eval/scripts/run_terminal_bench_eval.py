"""Run the Terminal-Bench eval subset as its own heavy benchmark entrypoint."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from eval.benchmarks.usage_metrics import aggregate_usage
from eval.scripts.eval_common import (
    PROJECT_ROOT,
    TBENCH_TASK_FILES,
    add_common_args,
    base_env,
    category_results,
    load_tbench_metadata,
    load_tbench_task_config,
    make_run_dir,
    number,
    safe_name,
    write_eval_reports,
    write_suite_summary,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args), ensure_ascii=False, indent=2))
        return 0

    run_tbench_suite(args)
    write_eval_reports(args.output_root)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Terminal-Bench eval tasks.")
    add_common_args(parser)
    parser.add_argument(
        "--task-wall-timeout",
        dest="task_wall_timeout",
        type=int,
        default=7200,
        help=(
            "Wall-clock timeout, in seconds, for each Terminal-Bench launcher "
            "subprocess. This is an outer watchdog, separate from a task's "
            "agent_timeout_sec budget."
        ),
    )
    parser.add_argument(
        "--tbench-task-set",
        choices=sorted(TBENCH_TASK_FILES),
        default="8task",
        help="Terminal-Bench task set size: 8task by default, or 24task for the larger subset.",
    )
    parser.add_argument(
        "--runner-env",
        default="",
        help="Optional Harbor environment backend, for example daytona.",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Ask Harbor to build task environments locally instead of pulling prebuilt images.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Optional task override. Repeat to run only selected tasks from the configured task set.",
    )
    parser.add_argument(
        "--tbench-parallelism",
        type=int,
        default=1,
        help="Number of Terminal-Bench tasks to run concurrently. Each task gets an isolated Harbor jobs dir.",
    )
    return parser.parse_args(argv)


def run_tbench_suite(args: argparse.Namespace) -> Path:
    payload = load_tbench_task_config(args.tbench_task_set)
    metadata = load_tbench_metadata()
    configured_tasks = list(payload["tasks"])
    tasks = list(args.task or configured_tasks)
    unknown_tasks = [task for task in tasks if task not in configured_tasks]
    if unknown_tasks:
        raise SystemExit(
            f"Unknown task(s) for {args.tbench_task_set}: {', '.join(unknown_tasks)}"
        )
    run_dir = make_run_dir(args, "tbench")
    started = time.perf_counter()
    task_results: list[dict[str, Any]] = []
    outputs_dir = run_dir / "task_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    parallelism = max(1, int(args.tbench_parallelism or 1))
    if parallelism == 1 or len(tasks) <= 1:
        task_results = [
            _run_tbench_task(args, run_dir=run_dir, outputs_dir=outputs_dir, metadata=metadata, task=task)
            for task in tasks
        ]
    else:
        results_by_task: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = {
                executor.submit(
                    _run_tbench_task,
                    args,
                    run_dir=run_dir,
                    outputs_dir=outputs_dir,
                    metadata=metadata,
                    task=task,
                ): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                results_by_task[task] = future.result()
        task_results = [results_by_task[task] for task in tasks]
    elapsed = time.perf_counter() - started
    passed = sum(1 for item in task_results if item["status"] == "passed")
    summary = {
        "suite": "tbench",
        "benchmark_name": str(payload.get("benchmark_name") or "Terminal-Bench 2.1 subset"),
        "task_set": str(payload.get("task_set") or args.tbench_task_set),
        "status": "passed" if passed == len(tasks) else "failed",
        "task_count": len(tasks),
        "passed": passed,
        "pass_rate": passed / len(tasks) if tasks else 0.0,
        "elapsed_seconds": elapsed,
        "tasks": tasks,
        "task_results": task_results,
        "category_results": category_results(task_results),
        **aggregate_usage(task_results),
    }
    write_suite_summary(run_dir, summary, [])
    return run_dir


def _run_tbench_task(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    outputs_dir: Path,
    metadata: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    task_jobs_dir = run_dir / "harbor_jobs" / safe_name(task)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "eval" / "benchmarks" / "run_terminal_bench.py"),
        "--jobs-dir",
        str(task_jobs_dir),
        "--task",
        task,
    ]
    if args.runner_env:
        command.extend(["--env", args.runner_env])
    if args.force_build:
        command.append("--force-build")
    task_started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=base_env(),
            capture_output=True,
            text=True,
            timeout=args.task_wall_timeout,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = _output_text(exc.stdout or exc.output)
        stderr = _output_text(exc.stderr)
        timeout_message = f"Terminal-Bench launcher timed out after {args.task_wall_timeout} seconds."
        stderr = f"{stderr.rstrip()}\n{timeout_message}\n" if stderr else f"{timeout_message}\n"
    task_elapsed = time.perf_counter() - task_started
    stdout_path = outputs_dir / f"{safe_name(task)}.stdout.txt"
    stderr_path = outputs_dir / f"{safe_name(task)}.stderr.txt"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(stderr, encoding="utf-8", newline="\n")
    task_meta = metadata.get(task) or {}
    launcher_result = _launcher_task_result(stdout, task)
    launcher_passed = launcher_result.get("passed") if launcher_result else None
    status = "failed" if timed_out else _task_status(returncode, launcher_passed)
    diagnostics = _task_diagnostics(
        task=task,
        status=status,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        launcher_result=launcher_result,
    )
    return {
        "task": task,
        "category": str(task_meta.get("category") or "unknown"),
        "difficulty": str(task_meta.get("difficulty") or "unknown"),
        "agent_timeout_sec": number(task_meta.get("agent_timeout_sec")),
        "returncode": returncode,
        "status": status,
        "timed_out": timed_out,
        "elapsed_seconds": task_elapsed,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": command,
        "reward": launcher_result.get("reward") if launcher_result else None,
        "trial_name": launcher_result.get("trial_name") if launcher_result else "",
        "session_id": launcher_result.get("session_id") if launcher_result else "",
        "metrics": (launcher_result.get("metrics") or {}) if launcher_result else {},
        **diagnostics,
    }


def _dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_tbench_task_config(args.tbench_task_set)
    return {
        "runner": "terminal_bench",
        "output_root": str(Path(args.output_root)),
        "terminal_bench_task_set": args.tbench_task_set,
        "terminal_bench_benchmark_name": payload.get("benchmark_name"),
        "terminal_bench_tasks": list(args.task or payload["tasks"]),
        "runner_env": args.runner_env,
        "force_build": args.force_build,
        "tbench_parallelism": max(1, int(args.tbench_parallelism or 1)),
    }


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _launcher_task_result(stdout: str, task: str) -> dict[str, Any]:
    prefix = "HCA_TERMINAL_BENCH_RESULT:"
    for line in reversed((stdout or "").splitlines()):
        if prefix not in line:
            continue
        try:
            payload = json.loads(line.split(prefix, 1)[1].strip())
        except json.JSONDecodeError:
            continue
        for item in payload.get("task_results") or []:
            if item.get("task") == task:
                return item if isinstance(item, dict) else {}
    return {}


def _task_status(returncode: int, launcher_passed: Any) -> str:
    if launcher_passed is True:
        return "passed"
    if launcher_passed is False:
        return "failed"
    return "passed" if returncode == 0 else "failed"


def _task_diagnostics(
    *,
    task: str,
    status: str,
    returncode: int,
    stdout: str,
    stderr: str,
    launcher_result: dict[str, Any],
) -> dict[str, Any]:
    metrics = launcher_result.get("metrics") if isinstance(launcher_result, dict) else {}
    trial_name = str(launcher_result.get("trial_name") or "") if launcher_result else ""
    session_id = str(launcher_result.get("session_id") or "") if launcher_result else ""
    job_dir = str((launcher_result or {}).get("job_dir") or _extract_job_dir(stdout) or "")
    artifact_info = _resolve_hca_artifacts(
        trial_name=trial_name,
        session_id=session_id,
        job_dir=job_dir,
    )
    if artifact_info:
        job_dir = str(artifact_info.get("job_dir") or job_dir)
        session_id = str(artifact_info.get("session_id") or session_id)
    verifier_headline = _extract_verifier_headline(stdout)
    final_report_summary = _extract_final_report_summary(
        stdout=stdout,
        job_dir=job_dir,
        session_id=session_id,
    )
    failure_kind = "passed"
    if status != "passed":
        exception_type = str((launcher_result or {}).get("exception_type") or "")
        exception_message = str((launcher_result or {}).get("exception_message") or "")
        combined_error_text = "\n".join([exception_type, exception_message, stdout or "", stderr or ""])
        if "AgentTimeoutError" in combined_error_text:
            failure_kind = "agent_timeout"
        elif returncode == 124 or "Terminal-Bench launcher timed out" in combined_error_text:
            failure_kind = "launcher_timeout"
        elif (
            "AgentSetupTimeoutError" in combined_error_text
            or "NonZeroAgentExitCodeError" in combined_error_text
            or "FATAL: failed to install harness dependencies" in combined_error_text
            or "Docker daemon is not running" in combined_error_text
            or (returncode != 0 and not launcher_result)
        ):
            failure_kind = "infra_or_setup_failure"
        else:
            failure_kind = "failed_verifier"
    missing_metrics = status != "passed" and not bool(metrics)
    return {
        "failure_kind": failure_kind,
        "missing_metrics": missing_metrics,
        "has_hca_artifacts": bool(artifact_info),
        "resolved_session_id": session_id,
        "resolved_job_dir": job_dir,
        "verifier_failure_headline": verifier_headline,
        "final_report_summary": final_report_summary,
        "diagnostic": {
            "task": task,
            "trial_name": trial_name,
            "session_id": session_id,
            "job_dir": job_dir,
            "failure_kind": failure_kind,
            "missing_metrics": missing_metrics,
            "verifier_failure_headline": verifier_headline,
            "final_report_summary": final_report_summary,
        },
    }


def _extract_job_dir(stdout: str) -> str:
    match = re.search(r"Results written to ([^\r\n]+)", stdout or "")
    if not match:
        return ""
    path = match.group(1).strip()
    if path.endswith("result.json"):
        return str(Path(path).parent)
    return path


def _resolve_hca_artifacts(
    *,
    trial_name: str,
    session_id: str,
    job_dir: str,
) -> dict[str, str]:
    for trial_dir in _candidate_trial_dirs(trial_name=trial_name, job_dir=job_dir):
        hca_root = trial_dir / "artifacts" / "hca"
        if not hca_root.is_dir():
            continue
        session_dirs = []
        if session_id:
            session_dirs.append(hca_root / session_id)
        session_dirs.extend(
            sorted(
                (path for path in hca_root.iterdir() if path.is_dir() and path.name != session_id),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
        for path in session_dirs:
            if (path / "manifest.json").exists() or (path / "early_manifest.json").exists():
                return {
                    "job_dir": str(trial_dir.parent),
                    "trial_dir": str(trial_dir),
                    "session_id": path.name,
                    "artifact_dir": str(path),
                }
    return {}


def _candidate_trial_dirs(*, trial_name: str, job_dir: str) -> list[Path]:
    candidates: list[Path] = []
    if job_dir:
        base = PROJECT_ROOT / job_dir if not Path(job_dir).is_absolute() else Path(job_dir)
        if trial_name and (base / trial_name).is_dir():
            candidates.append(base / trial_name)
        candidates.extend(path for path in base.iterdir() if path.is_dir()) if base.is_dir() else None
    if trial_name:
        jobs_root = PROJECT_ROOT / "jobs"
        if jobs_root.is_dir():
            candidates.extend(path for path in jobs_root.glob(f"*/{trial_name}") if path.is_dir())
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _extract_final_report_summary(*, stdout: str, job_dir: str, session_id: str) -> str:
    for event in _read_hca_trajectory_events(job_dir=job_dir, session_id=session_id):
        if event.get("type") == "final_report":
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            summary = str(payload.get("summary") or "").strip()
            if summary:
                return summary[:500]
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("final report:"):
            return stripped.split(":", 1)[1].strip()[:500]
    return ""


def _read_hca_trajectory_events(*, job_dir: str, session_id: str) -> list[dict[str, Any]]:
    if not job_dir or not session_id:
        return []
    job_path = PROJECT_ROOT / job_dir if not Path(job_dir).is_absolute() else Path(job_dir)
    events: list[dict[str, Any]] = []
    for path in job_path.glob(f"*/artifacts/hca/{session_id}/trajectory.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _extract_verifier_headline(stdout: str) -> str:
    lines = (stdout or "").splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("FAILED "):
            return stripped[:500]
    for line in lines:
        stripped = line.strip()
        if "AssertionError:" in stripped or "ValueError:" in stripped or "FileNotFoundError:" in stripped:
            return stripped[:500]
    for line in lines:
        stripped = line.strip()
        if "AgentTimeoutError" in stripped:
            return stripped[:500]
    return ""


def run_self_test() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        assert len(load_tbench_task_config("8task")["tasks"]) == 8
        assert len(load_tbench_task_config("24task")["tasks"]) == 24
        root = Path(temp_dir) / "results"
        run_dir = root / "tbench"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps({"suite": "tbench", "task_count": 1, "passed": 1, "pass_rate": 1.0}),
            encoding="utf-8",
        )
        write_eval_reports(root, report_output_dir=Path(temp_dir) / "out")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("self-test passed")


if __name__ == "__main__":
    raise SystemExit(main())
