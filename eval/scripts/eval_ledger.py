"""Rebuild a task-level Terminal-Bench ledger from raw eval artifacts.

This module is intentionally independent from the older resume summarizer.  It
normalizes every discovered run attempt first, then derives task-level final
results and retention decisions from that ledger.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

from eval.benchmarks.usage_metrics import parse_eval_metrics_from_text

RUN_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{6})")
HARBOR_JOB_RE = re.compile(r"^\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}$")
FAILED_KEEP_COUNT = 3
EXCLUDED_TASKS = {
    "chess-best-move": "vision_required",
    "code-from-image": "vision_required",
    "extract-moves-from-video": "vision_required",
    "gcode-to-text": "vision_required",
    "path-tracing": "vision_required",
    "sam-cell-seg": "vision_required",
    "video-processing": "vision_required",
}


@dataclass
class Attempt:
    attempt_id: str
    benchmark_version: str
    benchmark_name: str
    task_set: str
    task: str
    run_name: str
    run_timestamp: str
    status: str
    reward: float | None = None
    failure_kind: str = ""
    source: str = ""
    summary_path: str = ""
    result_path: str = ""
    trial_dir: str = ""
    artifact_dir: str = ""
    trajectory_path: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    tool_calls: int | None = None
    token_totals: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskFinal:
    task: str
    attempt_id: str
    source_version: str
    selection_reason: str
    status: str
    reward: float | None
    failure_kind: str
    cost_usd: float | None
    tool_calls: int | None
    token_totals: dict[str, int]
    run_name: str
    run_timestamp: str
    source: str
    trajectory_path: str = ""
    stdout_path: str = ""
    stderr_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rebuild_eval_ledger(
    *,
    results_root: str | Path,
    jobs_root: str | Path | None = None,
    include_jobs: bool = True,
) -> dict[str, Any]:
    results_root = Path(results_root)
    jobs_root_path = Path(jobs_root) if jobs_root else None
    warnings: list[str] = []
    attempts = scan_attempts(
        results_root=results_root,
        jobs_root=jobs_root_path,
        include_jobs=include_jobs,
        warnings=warnings,
    )
    attempts = dedupe_attempts(attempts)
    all_finals = select_final_results(attempts)
    finals = [item for item in all_finals if item.task not in EXCLUDED_TASKS]
    excluded_finals = [item for item in all_finals if item.task in EXCLUDED_TASKS]
    retention = build_retention_plan(
        attempts=attempts,
        finals=all_finals,
        allowed_roots=[path for path in (results_root, jobs_root_path) if path is not None],
    )
    summary = build_summary(
        attempts=attempts,
        finals=finals,
        excluded_finals=excluded_finals,
        warnings=warnings,
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "jobs_root": str(jobs_root_path) if jobs_root_path else "",
        "summary": summary,
        "final_results": [item.to_dict() for item in finals],
        "excluded_results": [
            {**item.to_dict(), "exclusion_reason": EXCLUDED_TASKS[item.task]}
            for item in excluded_finals
        ],
        "attempts": [item.to_dict() for item in sorted_attempts(attempts)],
        "retention_plan": retention,
        "warnings": warnings,
    }


def scan_attempts(
    *,
    results_root: Path,
    jobs_root: Path | None,
    include_jobs: bool,
    warnings: list[str],
) -> list[Attempt]:
    attempts: list[Attempt] = []
    attempts.extend(_scan_summary_attempts(results_root, warnings))
    attempts.extend(_scan_eval_harbor_attempts(results_root, warnings))
    if include_jobs and jobs_root is not None:
        attempts.extend(_scan_top_level_jobs(jobs_root, warnings))
    return attempts


def _scan_summary_attempts(results_root: Path, warnings: list[str]) -> list[Attempt]:
    attempts: list[Attempt] = []
    for summary_path in sorted(results_root.glob("*/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"skip unreadable summary: {summary_path} ({exc})")
            continue
        if _suite_name(payload, summary_path.parent.name) != "tbench":
            continue
        benchmark_name = str(payload.get("benchmark_name") or "")
        benchmark_version = _benchmark_version(benchmark_name, summary_path.parent.name, payload)
        task_set = str(payload.get("task_set") or "")
        run_name = summary_path.parent.name
        run_timestamp = _run_timestamp(run_name, summary_path)
        for item in payload.get("task_results") or []:
            if not isinstance(item, dict):
                continue
            task = _task_name(item)
            if not task:
                continue
            status = _status_from_summary_item(item)
            reward = _optional_float(item.get("reward"))
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            stdout_path = str(item.get("stdout_path") or "")
            stderr_path = str(item.get("stderr_path") or "")
            attempt_warnings: list[str] = []
            if not metrics:
                attempt_warnings.append("missing metrics")
            attempt = Attempt(
                attempt_id="",
                benchmark_version=benchmark_version,
                benchmark_name=benchmark_name,
                task_set=task_set,
                task=task,
                run_name=run_name,
                run_timestamp=run_timestamp,
                status=status,
                reward=reward,
                failure_kind=_failure_kind(status, item, metrics),
                source="summary.json",
                summary_path=str(summary_path),
                result_path="",
                trial_dir=str(item.get("trial_dir") or ""),
                artifact_dir=str(item.get("artifact_dir") or ""),
                trajectory_path=_first_existing_path([
                    _trajectory_path_from_item(item),
                ]),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                metrics=metrics,
                cost_usd=_cost_from_item(item, metrics),
                tool_calls=_tool_calls(metrics),
                token_totals=_token_totals(metrics, item),
                warnings=attempt_warnings,
            )
            attempt.attempt_id = _attempt_id(attempt)
            attempts.append(attempt)
    return attempts


def _scan_eval_harbor_attempts(results_root: Path, warnings: list[str]) -> list[Attempt]:
    attempts: list[Attempt] = []
    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        harbor_root = run_dir / "harbor_jobs"
        if not harbor_root.is_dir():
            continue
        summary_payload = _read_json(run_dir / "summary.json")
        benchmark_name = str((summary_payload or {}).get("benchmark_name") or "")
        task_set = str((summary_payload or {}).get("task_set") or "")
        version_hint = _benchmark_version(benchmark_name, run_dir.name, summary_payload or {})
        for task_dir in _safe_iterdirs(harbor_root, warnings):
            for timestamp_dir in _safe_iterdirs(task_dir, warnings):
                if not HARBOR_JOB_RE.match(timestamp_dir.name):
                    continue
                trial_results = [
                    trial_dir / "result.json"
                    for trial_dir in _safe_iterdirs(timestamp_dir, warnings)
                    if (trial_dir / "result.json").is_file()
                ]
                if trial_results:
                    for result_path in trial_results:
                        attempt = _attempt_from_harbor_result(
                            result_path,
                            run_name=run_dir.name,
                            run_timestamp=_timestamp_from_harbor_dir(timestamp_dir.name),
                            benchmark_name=benchmark_name,
                            benchmark_version=version_hint,
                            task_set=task_set,
                            source="harbor_result.json",
                            warnings=warnings,
                        )
                        if attempt:
                            attempts.append(attempt)
                elif (timestamp_dir / "result.json").is_file():
                    attempt = _attempt_from_harbor_job_summary(
                        timestamp_dir / "result.json",
                        config_path=timestamp_dir / "config.json",
                        run_name=run_dir.name,
                        run_timestamp=_timestamp_from_harbor_dir(timestamp_dir.name),
                        benchmark_name=benchmark_name,
                        benchmark_version=version_hint,
                        task_set=task_set,
                        source="harbor_job_result.json",
                        warnings=warnings,
                    )
                    if attempt:
                        attempts.append(attempt)
    return attempts


def _scan_top_level_jobs(jobs_root: Path, warnings: list[str]) -> list[Attempt]:
    attempts: list[Attempt] = []
    if not jobs_root.exists():
        return attempts
    for job_dir in _safe_iterdirs(jobs_root, warnings):
        if not HARBOR_JOB_RE.match(job_dir.name):
            continue
        result_path = job_dir / "result.json"
        if not result_path.is_file():
            continue
        attempt = _attempt_from_harbor_job_summary(
            result_path,
            config_path=job_dir / "config.json",
            run_name=job_dir.name,
            run_timestamp=_timestamp_from_harbor_dir(job_dir.name),
            benchmark_name="",
            benchmark_version="",
            task_set="",
            source="jobs_result.json",
            warnings=warnings,
        )
        if attempt:
            attempts.append(attempt)
    return attempts


def _attempt_from_harbor_result(
    result_path: Path,
    *,
    run_name: str,
    run_timestamp: str,
    benchmark_name: str,
    benchmark_version: str,
    task_set: str,
    source: str,
    warnings: list[str],
) -> Attempt | None:
    payload = _read_json(result_path)
    if not isinstance(payload, dict):
        warnings.append(f"skip unreadable harbor result: {result_path}")
        return None
    result_text = _read_text(result_path)
    task = _task_from_harbor_payload(payload, result_path)
    if not task:
        warnings.append(f"skip harbor result without task: {result_path}")
        return None
    config_payload = _read_json(_nearest_job_config(result_path))
    benchmark_version = benchmark_version or _benchmark_version(benchmark_name, run_name, config_payload or payload)
    task_set = task_set or _task_set_from_config(config_payload)
    reward = _reward_from_harbor_payload(payload)
    status = _status_from_reward_and_exception(reward, payload.get("exception_info"))
    metrics = _metrics_from_harbor(result_path.parent, result_text, payload)
    artifact_dir = _artifact_dir(result_path.parent)
    attempt_warnings: list[str] = []
    if not metrics:
        attempt_warnings.append("missing metrics")
    attempt = Attempt(
        attempt_id="",
        benchmark_version=benchmark_version or "unknown",
        benchmark_name=benchmark_name or _benchmark_name_from_version(benchmark_version),
        task_set=task_set,
        task=task,
        run_name=run_name,
        run_timestamp=run_timestamp,
        status=status,
        reward=reward,
        failure_kind=_failure_kind(status, payload, metrics),
        source=source,
        result_path=str(result_path),
        trial_dir=str(result_path.parent),
        artifact_dir=str(artifact_dir),
        trajectory_path=str(artifact_dir / "trajectory.jsonl") if artifact_dir else "",
        metrics=metrics,
        cost_usd=_cost_from_harbor(payload, metrics),
        tool_calls=_tool_calls(metrics),
        token_totals=_token_totals(metrics, payload),
        warnings=attempt_warnings,
    )
    attempt.attempt_id = _attempt_id(attempt)
    return attempt


def _attempt_from_harbor_job_summary(
    result_path: Path,
    *,
    config_path: Path,
    run_name: str,
    run_timestamp: str,
    benchmark_name: str,
    benchmark_version: str,
    task_set: str,
    source: str,
    warnings: list[str],
) -> Attempt | None:
    payload = _read_json(result_path)
    if not isinstance(payload, dict):
        warnings.append(f"skip unreadable job result: {result_path}")
        return None
    config_payload = _read_json(config_path)
    task = _task_from_config(config_payload) or _task_from_job_stats(payload)
    if not task:
        warnings.append(f"skip job result without task: {result_path}")
        return None
    benchmark_version = benchmark_version or _benchmark_version(benchmark_name, run_name, config_payload or payload)
    task_set = task_set or _task_set_from_config(config_payload)
    reward = _reward_from_job_result(payload)
    status = _status_from_reward_and_exception(reward, None)
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    metrics = _metrics_from_job_stats(stats)
    attempt_warnings: list[str] = []
    if not metrics:
        attempt_warnings.append("missing metrics")
    attempt = Attempt(
        attempt_id="",
        benchmark_version=benchmark_version or "unknown",
        benchmark_name=benchmark_name or _benchmark_name_from_version(benchmark_version),
        task_set=task_set,
        task=task,
        run_name=run_name,
        run_timestamp=run_timestamp,
        status=status,
        reward=reward,
        failure_kind=_failure_kind(status, payload, metrics),
        source=source,
        result_path=str(result_path),
        trial_dir=str(result_path.parent),
        metrics=metrics,
        cost_usd=_optional_float(stats.get("cost_usd")),
        tool_calls=_tool_calls(metrics),
        token_totals=_token_totals(metrics, payload),
        warnings=attempt_warnings,
    )
    attempt.attempt_id = _attempt_id(attempt)
    return attempt


def dedupe_attempts(attempts: list[Attempt]) -> list[Attempt]:
    by_key: dict[tuple[str, str, str, str, str], Attempt] = {}
    source_rank = {
        "harbor_result.json": 4,
        "summary.json": 3,
        "jobs_result.json": 2,
        "harbor_job_result.json": 1,
    }
    for attempt in attempts:
        key = (
            attempt.benchmark_version,
            attempt.task,
            attempt.run_timestamp,
            str(attempt.reward),
            attempt.status,
        )
        existing = by_key.get(key)
        if existing is None or source_rank.get(attempt.source, 0) > source_rank.get(existing.source, 0):
            by_key[key] = attempt
    return list(by_key.values())


def select_final_results(attempts: list[Attempt]) -> list[TaskFinal]:
    finals: list[TaskFinal] = []
    tasks = sorted({attempt.task for attempt in attempts if _is_terminal_bench_version(attempt.benchmark_version)})
    for task in tasks:
        task_attempts = [attempt for attempt in attempts if attempt.task == task]
        v21 = [attempt for attempt in task_attempts if attempt.benchmark_version == "2.1"]
        if not v21:
            continue
        selected = _select_best_attempt(v21)
        reason = (
            "latest_success_2.1"
            if selected.status == "passed"
            else "latest_failure_2.1"
        )
        finals.append(
            TaskFinal(
                task=task,
                attempt_id=selected.attempt_id,
                source_version=selected.benchmark_version,
                selection_reason=reason,
                status=selected.status,
                reward=selected.reward,
                failure_kind=selected.failure_kind,
                cost_usd=selected.cost_usd,
                tool_calls=selected.tool_calls,
                token_totals=selected.token_totals,
                run_name=selected.run_name,
                run_timestamp=selected.run_timestamp,
                source=selected.source,
                trajectory_path=selected.trajectory_path,
                stdout_path=selected.stdout_path,
                stderr_path=selected.stderr_path,
            )
        )
    return finals


def _select_best_attempt(attempts: list[Attempt]) -> Attempt:
    passed = [attempt for attempt in attempts if attempt.status == "passed"]
    pool = passed or [attempt for attempt in attempts if attempt.status in {"failed", "incomplete", "unknown"}]
    return max(pool, key=lambda item: (item.run_timestamp, item.run_name, item.attempt_id))


def build_summary(
    *,
    attempts: list[Attempt],
    finals: list[TaskFinal],
    excluded_finals: list[TaskFinal] | None = None,
    warnings: list[str],
) -> dict[str, Any]:
    excluded_finals = excluded_finals or []
    included_attempts = [attempt for attempt in attempts if attempt.task not in EXCLUDED_TASKS]
    excluded_attempts = [attempt for attempt in attempts if attempt.task in EXCLUDED_TASKS]
    excluded_final_tasks = {item.task for item in excluded_finals}
    excluded_passed_tasks = {item.task for item in excluded_finals if item.status == "passed"}
    configured_excluded_tasks = set(EXCLUDED_TASKS)
    benchmark_tasks = _terminal_bench_task_inventory()
    observed_final_tasks = {item.task for item in finals} | excluded_final_tasks
    if benchmark_tasks and observed_final_tasks and observed_final_tasks.issubset(benchmark_tasks):
        official_total_tasks = len(benchmark_tasks)
    else:
        official_total_tasks = len(finals) + len(excluded_finals)
    main_passed = sum(1 for item in finals if item.status == "passed")
    official_passed = main_passed + len(excluded_passed_tasks)
    official_failed = max(official_total_tasks - official_passed, 0)
    final_totals = _totals_from_finals(finals)
    attempt_totals = _totals_from_attempts(included_attempts)
    failed = [item for item in finals if item.status != "passed"]
    source_versions: dict[str, int] = {}
    for item in finals:
        source_versions[item.source_version] = source_versions.get(item.source_version, 0) + 1
    failure_kinds: dict[str, int] = {}
    for item in failed:
        key = item.failure_kind or "unknown"
        failure_kinds[key] = failure_kinds.get(key, 0) + 1
    excluded_failure_count = len(configured_excluded_tasks - excluded_passed_tasks)
    if excluded_failure_count:
        failure_kinds["strong_vision_or_not_attempted"] = excluded_failure_count
    v21_tasks = {
        attempt.task
        for attempt in included_attempts
        if attempt.benchmark_version == "2.1"
    }
    return {
        "benchmark_task_count": len(benchmark_tasks) if benchmark_tasks else official_total_tasks,
        "total_tasks": official_total_tasks,
        "passed": official_passed,
        "failed": official_failed,
        "pass_rate": _safe_rate(official_passed, official_total_tasks),
        "evaluated_main_tasks": len(finals),
        "main_passed": main_passed,
        "main_failed": len(failed),
        "source_versions": dict(sorted(source_versions.items())),
        "excluded_task_count": len(excluded_finals),
        "configured_excluded_task_count": len(configured_excluded_tasks),
        "excluded_with_results_count": len(excluded_finals),
        "excluded_tasks": [
            {"task": item.task, "reason": EXCLUDED_TASKS.get(item.task, "excluded")}
            for item in sorted(excluded_finals, key=lambda item: item.task)
        ],
        "configured_excluded_tasks": [
            {"task": task, "reason": EXCLUDED_TASKS[task]}
            for task in sorted(configured_excluded_tasks)
        ],
        "excluded_without_results": [
            {"task": task, "reason": EXCLUDED_TASKS[task]}
            for task in sorted(configured_excluded_tasks - excluded_final_tasks)
        ],
        "terminal_bench_2_1_attempted_tasks": len(v21_tasks),
        "terminal_bench_2_1_attempted_tasks_including_excluded": len({
            attempt.task
            for attempt in attempts
            if attempt.benchmark_version == "2.1"
        }),
        "failure_kinds": dict(sorted(failure_kinds.items())),
        "final_selected_totals": final_totals,
        "all_attempt_totals": attempt_totals,
        "attempt_count": len(included_attempts),
        "excluded_attempt_count": len(excluded_attempts),
        "warning_count": len(warnings) + sum(len(item.warnings) for item in attempts),
    }


def _terminal_bench_task_inventory() -> set[str]:
    path = Path(__file__).resolve().parents[2] / "eval" / "benchmarks" / "data" / "tb2_tasks.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(data, dict):
        return {str(key) for key in data}
    if isinstance(data, list):
        return {str(item) for item in data}
    return set()


def build_retention_plan(
    *,
    attempts: list[Attempt],
    finals: list[TaskFinal],
    allowed_roots: list[Path],
) -> dict[str, Any]:
    final_ids = {item.attempt_id for item in finals}
    keep_attempt_ids: set[str] = set()
    task_attempts: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        if _is_terminal_bench_version(attempt.benchmark_version):
            task_attempts.setdefault(attempt.task, []).append(attempt)
    for items in task_attempts.values():
        selected = [item for item in items if item.attempt_id in final_ids]
        if selected and selected[0].status == "passed":
            keep_attempt_ids.add(selected[0].attempt_id)
            continue
        failed = [item for item in items if item.status != "passed"]
        failed_sorted = sorted(failed, key=lambda item: (item.run_timestamp, item.run_name), reverse=True)
        keep_attempt_ids.update(item.attempt_id for item in failed_sorted[:FAILED_KEEP_COUNT])
    deletions: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for attempt in sorted_attempts(attempts):
        paths = _retention_paths(attempt)
        if attempt.attempt_id in keep_attempt_ids:
            kept.append({"attempt_id": attempt.attempt_id, "task": attempt.task, "paths": paths})
            continue
        safe_paths = [
            path for path in paths
            if Path(path).exists() and _is_safe_path(path, allowed_roots)
        ]
        if safe_paths:
            deletions.append({"attempt_id": attempt.attempt_id, "task": attempt.task, "paths": safe_paths})
    return {
        "mode": "dry_run",
        "failed_keep_count": FAILED_KEEP_COUNT,
        "kept_attempts": kept,
        "delete_attempts": deletions,
        "delete_path_count": sum(len(item["paths"]) for item in deletions),
    }


def apply_retention_plan(plan: dict[str, Any]) -> list[str]:
    deleted: list[str] = []
    for item in plan.get("delete_attempts") or []:
        for raw_path in item.get("paths") or []:
            path = Path(raw_path)
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    deleted.append(str(path))
                elif path.exists():
                    path.unlink()
                    deleted.append(str(path))
            except OSError:
                continue
    plan["mode"] = "applied"
    plan["deleted_paths"] = deleted
    return deleted


def write_outputs(ledger: dict[str, Any], *, output_root: str | Path) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "results.json").write_text(
        json.dumps(_results_json(ledger), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "retention_plan.json").write_text(
        json.dumps(ledger["retention_plan"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "SUMMARY.md").write_text(
        render_summary_markdown(ledger),
        encoding="utf-8",
        newline="\n",
    )


def render_summary_markdown(ledger: dict[str, Any]) -> str:
    summary = ledger.get("summary") or {}
    finals = ledger.get("final_results") or []
    attempts = ledger.get("attempts") or []
    excluded = ledger.get("excluded_results") or []
    retention = ledger.get("retention_plan") or {}
    lines = [
        "# Terminal-Bench Task Ledger",
        "",
        f"Generated at: {ledger.get('generated_at')}",
        "",
        "## Combined Task-Level Result",
        "",
        f"- Total tasks: **{summary.get('total_tasks', 0)}**",
        f"- Passed: **{summary.get('passed', 0)}**",
        f"- Failed: **{summary.get('failed', 0)}**",
        f"- Pass rate: **{_percent(summary.get('pass_rate'))}**",
        f"- Main task results recorded: **{summary.get('evaluated_main_tasks', 0)}**",
        f"- Source versions: {_source_version_line(summary.get('source_versions') or {})}",
        f"- Terminal-Bench 2.1 attempted main tasks: **{summary.get('terminal_bench_2_1_attempted_tasks', 0)}**",
        f"- Terminal-Bench 2.1 attempted tasks including excluded: **{summary.get('terminal_bench_2_1_attempted_tasks_including_excluded', 0)}**",
        f"- Strong-vision tasks counted as failures: **{summary.get('configured_excluded_task_count', summary.get('excluded_task_count', 0))}**",
        f"- Strong-vision tasks with results: **{summary.get('excluded_with_results_count', summary.get('excluded_task_count', 0))}**",
        "",
        "## Excluded Tasks",
        "",
        "| Task | Reason | Latest Status | Source Version |",
        "| --- | --- | --- | --- |",
    ]
    if excluded:
        for item in sorted(excluded, key=lambda row: str(row.get("task") or "")):
            lines.append(
                f"| {_cell(item.get('task'))} | {_cell(item.get('exclusion_reason'))} | "
                f"{_cell(item.get('status'))} | {_cell(item.get('source_version'))} |"
            )
    else:
        lines.append("| none |  |  |  |")
    excluded_without_results = summary.get("excluded_without_results") or []
    if excluded_without_results:
        lines.extend([
            "",
            "## Excluded Tasks Without Results",
            "",
            "| Task | Reason |",
            "| --- | --- |",
        ])
        for item in excluded_without_results:
            lines.append(f"| {_cell(item.get('task'))} | {_cell(item.get('reason'))} |")
    lines.extend([
        "",
        "## Aggregate Stats",
        "",
        "| Scope | Cost | Tokens | Tool calls | Coverage |",
        "| --- | ---: | ---: | ---: | --- |",
        _totals_row("Final selected", summary.get("final_selected_totals") or {}),
        _totals_row("All attempts", summary.get("all_attempt_totals") or {}),
        "",
        "## Failure Kinds",
        "",
    ])
    failure_kinds = summary.get("failure_kinds") or {}
    if failure_kinds:
        lines.extend(f"- {kind}: {count}" for kind, count in sorted(failure_kinds.items()))
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Final Task Table",
        "",
        "| Task | Status | Source Version | Selection | Cost | Tokens | Tools | Failure Kind | Source |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ])
    for item in finals:
        lines.append(_final_row(item))
    lines.extend([
        "",
        "## Terminal-Bench 2.1 Coverage",
        "",
        "| Task | Latest Status | Attempts | Latest Run |",
        "| --- | --- | ---: | --- |",
    ])
    lines.extend(_coverage_rows(attempts, version="2.1"))
    lines.extend([
        "",
        "## Failed Task Trajectories",
        "",
        "| Task | Failure Kind | Kept Paths |",
        "| --- | --- | --- |",
    ])
    {item.get("attempt_id") for item in finals if item.get("status") != "passed"}
    kept_by_id = {
        item.get("attempt_id"): item.get("paths") or []
        for item in retention.get("kept_attempts") or []
    }
    failed_finals = [item for item in finals if item.get("status") != "passed"]
    if failed_finals:
        for item in failed_finals:
            paths = kept_by_id.get(item.get("attempt_id")) or _final_paths(item)
            lines.append(
                f"| {_cell(item.get('task'))} | {_cell(item.get('failure_kind'))} | "
                f"{_cell('; '.join(paths) if paths else 'not captured')} |"
            )
    else:
        lines.append("| none |  |  |")
    lines.extend([
        "",
        "## Retention Plan",
        "",
        f"- Mode: {retention.get('mode', 'dry_run')}",
        f"- Delete attempts: {len(retention.get('delete_attempts') or [])}",
        f"- Delete paths: {retention.get('delete_path_count', 0)}",
        f"- Warnings: {summary.get('warning_count', 0)}",
        "",
    ])
    return "\n".join(lines)


def sorted_attempts(attempts: list[Attempt]) -> list[Attempt]:
    return sorted(attempts, key=lambda item: (item.task, item.run_timestamp, item.source, item.attempt_id))


def _suite_name(payload: dict[str, Any], dirname: str) -> str:
    explicit = str(payload.get("suite") or "").strip()
    if explicit == "tbench":
        return "tbench"
    lowered = (explicit + " " + dirname).lower()
    if "tbench" in lowered or "terminal_bench" in lowered:
        return "tbench"
    return explicit or "unknown"


def _benchmark_version(name: str, dirname: str, payload: dict[str, Any] | None) -> str:
    text = f"{name} {dirname} {json.dumps(payload or {}, ensure_ascii=False)[:2000]}".lower()
    if "2.1" in text or "2-1" in text or "tbench21" in text or "terminal-bench-2-1" in text:
        return "2.1"
    if "tbench" in text or "terminal-bench" in text or "terminal_bench" in text:
        return "2.1"
    return "unknown"


def _benchmark_name_from_version(version: str) -> str:
    if version == "2.1":
        return "Terminal-Bench 2.1"
    return "Terminal-Bench"


def _is_terminal_bench_version(version: str) -> bool:
    return version == "2.1"


def _run_timestamp(run_name: str, summary_path: Path) -> str:
    match = RUN_TIMESTAMP_RE.match(run_name)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    try:
        return datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    except OSError:
        return "0000-00-00_000000"


def _timestamp_from_harbor_dir(name: str) -> str:
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})__(\d{2})-(\d{2})-(\d{2})$", name)
    if match:
        year, month, day, hour, minute, second = match.groups()
        return f"{year}-{month}-{day}_{hour}{minute}{second}"
    return name.replace("__", "_")


def _task_name(item: dict[str, Any]) -> str:
    return str(item.get("task") or item.get("task_name") or "").rsplit("/", 1)[-1].strip()


def _task_from_harbor_payload(payload: dict[str, Any], result_path: Path) -> str:
    task = _task_name(payload)
    if task:
        return task
    trial_name = str(payload.get("trial_name") or result_path.parent.name)
    return trial_name.split("__", 1)[0] if "__" in trial_name else trial_name


def _task_from_config(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    for dataset in payload.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        names = dataset.get("task_names") or []
        if names:
            return str(names[0]).rsplit("/", 1)[-1]
    return ""


def _task_set_from_config(payload: dict[str, Any] | None) -> str:
    task = _task_from_config(payload)
    return "manual" if task else ""


def _task_from_job_stats(payload: dict[str, Any]) -> str:
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    evals = stats.get("evals") if isinstance(stats.get("evals"), dict) else {}
    for eval_payload in evals.values():
        if not isinstance(eval_payload, dict):
            continue
        reward_stats = eval_payload.get("reward_stats") if isinstance(eval_payload.get("reward_stats"), dict) else {}
        reward_bucket = reward_stats.get("reward") if isinstance(reward_stats.get("reward"), dict) else {}
        for names in reward_bucket.values():
            if isinstance(names, list) and names:
                return str(names[0]).split("__", 1)[0]
    return ""


def _status_from_summary_item(item: dict[str, Any]) -> str:
    status = str(item.get("status") or "").lower().strip()
    if status in {"passed", "failed", "incomplete"}:
        return status
    reward = _optional_float(item.get("reward"))
    return _status_from_reward_and_exception(reward, item.get("exception_info"))


def _status_from_reward_and_exception(reward: float | None, exception_info: Any) -> str:
    if reward is not None:
        return "passed" if reward >= 1.0 else "failed"
    if exception_info:
        return "failed"
    return "unknown"


def _reward_from_harbor_payload(payload: dict[str, Any]) -> float | None:
    verifier = payload.get("verifier_result") if isinstance(payload.get("verifier_result"), dict) else {}
    rewards = verifier.get("rewards") if isinstance(verifier.get("rewards"), dict) else {}
    return _optional_float(rewards.get("reward"))


def _reward_from_job_result(payload: dict[str, Any]) -> float | None:
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    evals = stats.get("evals") if isinstance(stats.get("evals"), dict) else {}
    for eval_payload in evals.values():
        if not isinstance(eval_payload, dict):
            continue
        metrics = eval_payload.get("metrics") or []
        if metrics and isinstance(metrics[0], dict):
            return _optional_float(metrics[0].get("mean"))
    return None


def _failure_kind(status: str, payload: dict[str, Any], metrics: dict[str, Any]) -> str:
    if status == "passed":
        return "passed"
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if "AgentTimeoutError" in text:
        return "agent_timeout"
    if "Terminal-Bench launcher timed out" in text:
        return "launcher_timeout"
    if (
        "AgentSetupTimeoutError" in text
        or "NonZeroAgentExitCodeError" in text
        or "Docker daemon is not running" in text
        or "failed to install harness dependencies" in text
    ):
        return "infra_or_setup_failure"
    if status == "unknown":
        return "incomplete_or_unknown"
    return "failed_verifier" if metrics else "failed_without_metrics"


def _metrics_from_harbor(trial_dir: Path, result_text: str, payload: dict[str, Any]) -> dict[str, Any]:
    agent_result = payload.get("agent_result") if isinstance(payload.get("agent_result"), dict) else {}
    metadata = agent_result.get("metadata") if isinstance(agent_result.get("metadata"), dict) else {}
    metrics = metadata.get("hca_eval_metrics") if isinstance(metadata.get("hca_eval_metrics"), dict) else None
    if metrics:
        return metrics
    for text in _payload_text_fields(payload) + [result_text]:
        metrics = parse_eval_metrics_from_text(text)
        if metrics:
            return metrics
    for name in ("trial.log", "exception.txt", "runner_error.txt"):
        metrics = parse_eval_metrics_from_text(_read_text(trial_dir / name))
        if metrics:
            return metrics
    return {}


def _metrics_from_job_stats(stats: dict[str, Any]) -> dict[str, Any]:
    prompt = _optional_int(stats.get("n_input_tokens"))
    cached = _optional_int(stats.get("n_cache_tokens"))
    completion = _optional_int(stats.get("n_output_tokens"))
    if prompt is None and cached is None and completion is None:
        return {}
    prompt = prompt or 0
    cached = cached or 0
    completion = completion or 0
    return {
        "tokens": {
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "cache_miss_tokens": max(0, prompt - cached),
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
        "tools": {},
        "usage_cost": {"estimated_cost_usd": _optional_float(stats.get("cost_usd"))},
    }


def _payload_text_fields(payload: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for section_name in ("exception_info", "agent_result"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        for value in section.values():
            if isinstance(value, str):
                fields.append(value)
    return fields


def _cost_from_item(item: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    usage = metrics.get("usage_cost") if isinstance(metrics.get("usage_cost"), dict) else {}
    return _optional_float(usage.get("estimated_cost_usd")) or _optional_float(item.get("cost_usd"))


def _cost_from_harbor(payload: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    usage = metrics.get("usage_cost") if isinstance(metrics.get("usage_cost"), dict) else {}
    cost = _optional_float(usage.get("estimated_cost_usd"))
    if cost is not None:
        return cost
    agent_result = payload.get("agent_result") if isinstance(payload.get("agent_result"), dict) else {}
    return _optional_float(agent_result.get("cost_usd"))


def _tool_calls(metrics: dict[str, Any]) -> int | None:
    tools = metrics.get("tools") if isinstance(metrics.get("tools"), dict) else {}
    value = _optional_int(tools.get("tool_calls"))
    return value


def _token_totals(metrics: dict[str, Any], payload: dict[str, Any]) -> dict[str, int]:
    tokens = metrics.get("tokens") if isinstance(metrics.get("tokens"), dict) else {}
    result = {
        key: _optional_int(tokens.get(key)) or 0
        for key in (
            "prompt_tokens",
            "cached_tokens",
            "cache_miss_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }
    if any(result.values()):
        return result
    agent_result = payload.get("agent_result") if isinstance(payload.get("agent_result"), dict) else {}
    prompt = _optional_int(agent_result.get("n_input_tokens")) or 0
    cached = _optional_int(agent_result.get("n_cache_tokens")) or 0
    completion = _optional_int(agent_result.get("n_output_tokens")) or 0
    if prompt or cached or completion:
        return {
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "cache_miss_tokens": max(0, prompt - cached),
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    return {}


def _totals_from_finals(finals: list[TaskFinal]) -> dict[str, Any]:
    return _totals([item.to_dict() for item in finals])


def _totals_from_attempts(attempts: list[Attempt]) -> dict[str, Any]:
    return _totals([item.to_dict() for item in attempts])


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cost = 0.0
    cost_count = 0
    tools = 0
    tools_count = 0
    tokens = 0
    tokens_count = 0
    for item in rows:
        if item.get("cost_usd") is not None:
            cost += float(item["cost_usd"])
            cost_count += 1
        if item.get("tool_calls") is not None:
            tools += int(item["tool_calls"])
            tools_count += 1
        token_totals = item.get("token_totals") if isinstance(item.get("token_totals"), dict) else {}
        total = _optional_int(token_totals.get("total_tokens"))
        if total is not None and total > 0:
            tokens += total
            tokens_count += 1
    return {
        "estimated_cost_usd": cost if cost_count else None,
        "cost_coverage": f"{cost_count}/{len(rows)}",
        "total_tokens": tokens,
        "token_coverage": f"{tokens_count}/{len(rows)}",
        "tool_calls": tools,
        "tool_coverage": f"{tools_count}/{len(rows)}",
    }


def _results_json(ledger: dict[str, Any]) -> dict[str, Any]:
    summary = ledger.get("summary") or {}
    return {
        "schema_version": ledger.get("schema_version"),
        "generated_at": ledger.get("generated_at"),
        "benchmark_task_count": summary.get("benchmark_task_count", 0),
        "total_tasks": summary.get("total_tasks", 0),
        "passed": summary.get("passed", 0),
        "failed": summary.get("failed", 0),
        "pass_rate": summary.get("pass_rate", 0.0),
        "evaluated_main_tasks": summary.get("evaluated_main_tasks", 0),
        "main_passed": summary.get("main_passed", 0),
        "main_failed": summary.get("main_failed", 0),
        "source_versions": summary.get("source_versions") or {},
        "excluded_task_count": summary.get("excluded_task_count", 0),
        "configured_excluded_task_count": summary.get("configured_excluded_task_count", 0),
        "excluded_with_results_count": summary.get("excluded_with_results_count", 0),
        "excluded_tasks": summary.get("excluded_tasks") or [],
        "configured_excluded_tasks": summary.get("configured_excluded_tasks") or [],
        "excluded_without_results": summary.get("excluded_without_results") or [],
        "terminal_bench_2_1_attempted_tasks": summary.get("terminal_bench_2_1_attempted_tasks", 0),
        "terminal_bench_2_1_attempted_tasks_including_excluded": summary.get(
            "terminal_bench_2_1_attempted_tasks_including_excluded",
            0,
        ),
        "final_selected_totals": summary.get("final_selected_totals") or {},
        "all_attempt_totals": summary.get("all_attempt_totals") or {},
        "tasks": ledger.get("final_results") or [],
        "excluded_results": ledger.get("excluded_results") or [],
        "warnings": ledger.get("warnings") or [],
    }


def _retention_paths(attempt: Attempt) -> list[str]:
    paths = [
        attempt.trial_dir,
        attempt.artifact_dir,
        attempt.trajectory_path,
        attempt.stdout_path,
        attempt.stderr_path,
        attempt.result_path,
    ]
    return sorted({path for path in paths if path})


def _is_safe_path(raw_path: str, allowed_roots: list[Path]) -> bool:
    try:
        resolved = Path(raw_path).resolve()
    except OSError:
        return False
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def _first_existing_path(paths: list[str]) -> str:
    for raw_path in paths:
        if raw_path and Path(raw_path).exists():
            return raw_path
    return paths[0] if paths and paths[0] else ""


def _trajectory_path_from_item(item: dict[str, Any]) -> str:
    for key in ("trajectory_path", "artifact_dir"):
        raw = str(item.get(key) or "")
        if raw and key == "trajectory_path":
            return raw
        if raw and key == "artifact_dir":
            return str(Path(raw) / "trajectory.jsonl")
    job_dir = str(item.get("resolved_job_dir") or "")
    session_id = str(item.get("resolved_session_id") or item.get("session_id") or "")
    if job_dir and session_id:
        matches = list(Path(job_dir).glob(f"*/artifacts/hca/{session_id}/trajectory.jsonl"))
        if matches:
            return str(matches[0])
    return ""


def _artifact_dir(trial_dir: Path) -> Path | None:
    hca_root = trial_dir / "artifacts" / "hca"
    if not hca_root.is_dir():
        return None
    sessions = sorted((path for path in _safe_iterdirs(hca_root, []) if path.is_dir()), key=lambda p: p.name)
    for session_dir in sessions:
        if (session_dir / "manifest.json").exists() or (session_dir / "early_manifest.json").exists():
            return session_dir
    return sessions[0] if sessions else None


def _nearest_job_config(result_path: Path) -> Path:
    for parent in result_path.parents:
        candidate = parent / "config.json"
        if candidate.exists():
            return candidate
    return result_path.parent / "config.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _safe_iterdirs(path: Path, warnings: list[str]) -> list[Path]:
    try:
        return [entry for entry in path.iterdir() if entry.is_dir()]
    except OSError as exc:
        warnings.append(f"skip unreadable directory: {path} ({exc})")
        return []


def _attempt_id(attempt: Attempt) -> str:
    raw = "|".join(
        [
            attempt.benchmark_version,
            attempt.task,
            attempt.run_name,
            attempt.run_timestamp,
            attempt.summary_path,
            attempt.result_path,
            attempt.trial_dir,
            str(attempt.reward),
            attempt.status,
        ]
    )
    return sha1(raw.encode("utf-8")).hexdigest()[:16]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percent(value: Any) -> str:
    try:
        return f"{float(value or 0.0) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _source_version_line(source_versions: dict[str, int]) -> str:
    if not source_versions:
        return "none"
    return ", ".join(f"{version}: {count}" for version, count in sorted(source_versions.items()))


def _totals_row(label: str, totals: dict[str, Any]) -> str:
    cost = totals.get("estimated_cost_usd")
    cost_cell = f"${float(cost):.4f}" if cost is not None else "not captured"
    tokens = int(totals.get("total_tokens") or 0)
    tools = int(totals.get("tool_calls") or 0)
    coverage = (
        f"cost {totals.get('cost_coverage', '0/0')}, "
        f"tokens {totals.get('token_coverage', '0/0')}, "
        f"tools {totals.get('tool_coverage', '0/0')}"
    )
    return f"| {label} | {cost_cell} | {tokens:,} | {tools:,} | {coverage} |"


def _final_row(item: dict[str, Any]) -> str:
    tokens = item.get("token_totals") if isinstance(item.get("token_totals"), dict) else {}
    cost = item.get("cost_usd")
    return (
        f"| {_cell(item.get('task'))} | {_cell(item.get('status'))} | "
        f"{_cell(item.get('source_version'))} | {_cell(item.get('selection_reason'))} | "
        f"{_money(cost)} | {_int_cell(tokens.get('total_tokens'))} | "
        f"{_int_cell(item.get('tool_calls'))} | {_cell(item.get('failure_kind'))} | "
        f"{_cell(item.get('source'))} |"
    )


def _coverage_rows(attempts: list[dict[str, Any]], *, version: str) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        if attempt.get("benchmark_version") == version:
            if attempt.get("task") in EXCLUDED_TASKS:
                continue
            grouped.setdefault(str(attempt.get("task") or ""), []).append(attempt)
    if not grouped:
        return ["| none |  | 0 |  |"]
    rows: list[str] = []
    for task, items in sorted(grouped.items()):
        latest = max(items, key=lambda item: (item.get("run_timestamp") or "", item.get("run_name") or ""))
        rows.append(
            f"| {_cell(task)} | {_cell(latest.get('status'))} | {len(items)} | {_cell(latest.get('run_name'))} |"
        )
    return rows


def _final_paths(item: dict[str, Any]) -> list[str]:
    return [
        str(path)
        for path in (item.get("trajectory_path"), item.get("stdout_path"), item.get("stderr_path"))
        if path
    ]


def _cell(value: Any) -> str:
    text = str(value or "").replace("|", "\\|").strip()
    return text or "not captured"


def _money(value: Any) -> str:
    parsed = _optional_float(value)
    return f"${parsed:.4f}" if parsed is not None else "not captured"


def _int_cell(value: Any) -> str:
    parsed = _optional_int(value)
    return f"{parsed:,}" if parsed is not None and parsed > 0 else "not captured"
