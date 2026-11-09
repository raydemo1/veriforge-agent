"""Shared helpers for eval entrypoints."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = PROJECT_ROOT / "eval" / "results"
TASKS_ROOT = PROJECT_ROOT / "eval" / "tasks"
TBENCH_METADATA_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "data" / "tb2_tasks.json"
TBENCH_TASK_FILES = {
    "8task": "terminal_bench_8task.json",
    "24task": "terminal_bench_24task.json",
    "full": "terminal_bench_full.json",
}
CLAW_TASK_FILES = {
    "lite80": "claw_swe_bench_lite80.json",
}


@dataclass
class CaseResult:
    suite: str
    case_id: str
    variant: str
    returncode: int
    elapsed_seconds: float
    success: bool
    session_id: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = dict(self.metrics or {})
        return payload


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", default=str(RESULTS_ROOT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")


def write_eval_reports(output_root: str | Path, *, report_output_dir: str | Path | None = None) -> None:
    from eval.scripts.eval_ledger import rebuild_eval_ledger, write_outputs

    results_root = Path(output_root)
    ledger = rebuild_eval_ledger(results_root=results_root, jobs_root=PROJECT_ROOT / "jobs")
    output_dir = Path(report_output_dir) if report_output_dir is not None else results_root
    write_outputs(ledger, output_root=output_dir)
    print(f"Wrote eval ledger: {output_dir / 'SUMMARY.md'}")


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing else str(PROJECT_ROOT) + os.pathsep + existing
    return env


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_task_list(filename: str) -> dict[str, Any]:
    path = TASKS_ROOT / filename
    return json.loads(path.read_text(encoding="utf-8"))


def load_tbench_task_config(task_set: str) -> dict[str, Any]:
    return load_task_list(TBENCH_TASK_FILES[task_set])


def load_claw_task_config(task_set: str) -> dict[str, Any]:
    return load_task_list(CLAW_TASK_FILES[task_set])


def load_tbench_metadata() -> dict[str, Any]:
    return json.loads(TBENCH_METADATA_PATH.read_text(encoding="utf-8"))


def make_run_dir(args: argparse.Namespace, suite: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{safe_name(args.run_name)}" if args.run_name else ""
    run_dir = Path(args.output_root) / f"{timestamp}_{suite}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_suffix(args: argparse.Namespace, suite: str) -> str:
    return f"{suite}_{safe_name(args.run_name)}" if args.run_name else suite


def suite_names(value: str, default: list[str]) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    return names or default


def write_suite_summary(run_dir: Path, summary: dict[str, Any], results: list[CaseResult]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if results:
        (run_dir / "cases.json").write_text(
            json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def session_id_from_stdout(stdout: str) -> str:
    match = re.search(r"^veriforge session:\s*(\S+)", stdout, flags=re.MULTILINE)
    return match.group(1) if match else ""


def nested_int(payload: dict[str, Any] | None, *keys: str) -> int:
    value: Any = payload or {}
    for key in keys:
        if not isinstance(value, dict):
            return 0
        value = value.get(key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def success_rate(results: list[CaseResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for item in results if item.success) / len(results)


def reduction(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before


def category_results(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = {}
    for item in task_results:
        category = str(item.get("category") or "unknown")
        bucket = grouped.setdefault(category, {"task_count": 0, "passed": 0})
        bucket["task_count"] += 1
        if item.get("status") == "passed":
            bucket["passed"] += 1
    return {
        category: {
            "task_count": values["task_count"],
            "passed": values["passed"],
            "pass_rate": values["passed"] / values["task_count"] if values["task_count"] else 0.0,
        }
        for category, values in sorted(grouped.items())
    }


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    idx = max(0, min(len(values) - 1, int((len(values) * quantile + 0.999999) - 1)))
    return sorted(values)[idx]


def safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return text.strip("_") or "run"
