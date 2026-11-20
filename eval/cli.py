"""One entry point for every evaluation suite: ``python -m eval``.

    python -m eval list
    python -m eval validate veriloop
    python -m eval run veriloop --split held_out --repetitions 3 --intensity fast
    python -m eval run veriloop --dry-run
    python -m eval run tbench --task-set tb41-20task
    python -m eval report
    python -m eval compare eval/results/<run-a> eval/results/<run-b>

The suites keep their own runners under ``eval/scripts``; this module gives
them one vocabulary. ``run`` always performs the paid run and ``--dry-run``
always only prints the plan, whatever the underlying script's own default is.
Shared options (``--repetitions``, ``--split``, ``--intensity``, ``--tasks``,
``--task-set``, ``--output-root``, ``--run-name``) are translated per suite;
anything after ``--`` goes to the suite's runner unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = PROJECT_ROOT / "eval" / "results"
SCRIPTS = PROJECT_ROOT / "eval" / "scripts"
INTENSITIES = ("fast", "normal", "hard", "max")
SPLITS = ("all", "evolve", "held_out")


@dataclass(frozen=True)
class Suite:
    name: str
    script: str
    summary: str
    cost: str
    fixed_args: tuple[str, ...] = ()
    # How the runner distinguishes a plan from a paid run: "run-flag" means the
    # script only spends money with --run; "dry-run-flag" means it runs by
    # default and plans with --dry-run; "none" means it has no plan mode, so
    # --dry-run prints the command instead of running it.
    mode: str = "run-flag"
    validate_args: tuple[str, ...] | None = None
    validate_scripts: tuple[str, ...] = ()
    # Shared option -> runner flag. Options missing here are rejected.
    options: dict[str, str] = field(default_factory=dict)


SUITES: dict[str, Suite] = {
    suite.name: suite
    for suite in (
        Suite(
            name="veriloop",
            script="run_dev_task_eval.py",
            summary="VeriLoop-Bench: 24 multi-trap tasks inside three base projects (evolve / held_out), no Docker",
            cost="about ¥0.1-0.5 per task run (estimate)",
            fixed_args=("--task-pack", "veriloop"),
            validate_args=("--validate",),
            options={
                "repetitions": "--repetitions", "split": "--split", "intensity": "--intensity",
                "tasks": "--tasks", "output_root": "--output-root",
            },
        ),
        Suite(
            name="dev",
            script="run_dev_task_eval.py",
            summary="Original 8 local development tasks with hidden tests",
            cost="about ¥0.05-0.25 per task run",
            fixed_args=("--task-pack", "dev"),
            validate_args=("--validate",),
            options={
                "repetitions": "--repetitions", "intensity": "--intensity",
                "tasks": "--tasks", "output_root": "--output-root",
            },
        ),
        Suite(
            name="windows-native",
            script="run_dev_task_eval.py",
            summary="Windows-native regression tasks (PowerShell, paths, encodings)",
            cost="about ¥0.05-0.25 per task run",
            fixed_args=("--task-pack", "windows-native"),
            validate_args=("--validate",),
            options={
                "repetitions": "--repetitions", "intensity": "--intensity",
                "tasks": "--tasks", "output_root": "--output-root",
            },
        ),
        Suite(
            name="tbench",
            script="run_terminal_bench_eval.py",
            summary="Terminal-Bench through Harbor: tb2.1 sets (8task/24task/full) or tb41-20task",
            cost="Docker per task; TB4.1 roughly ¥2 per task",
            mode="dry-run-flag",
            validate_args=("--self-test",),
            options={
                "task_set": "--tbench-task-set", "split": "--split", "tasks": "--task",
                "output_root": "--output-root", "run_name": "--run-name",
                "intensity": "env:HARNESS_MODEL_INTENSITY",
            },
        ),
        Suite(
            name="metrics",
            script="run_basic_metrics_eval.py",
            summary="Local basic metrics: memory and latency (add -- --suites cache for cache)",
            cost="small",
            mode="dry-run-flag",
            validate_args=("--self-test",),
            options={
                "output_root": "--output-root", "run_name": "--run-name",
                "intensity": "env:HARNESS_MODEL_INTENSITY",
            },
        ),
        Suite(
            name="context",
            script="deepseek_context_eval.py",
            summary="DeepSeek prompt-cache behaviour across the context lifecycle",
            cost="small",
            mode="none",
            validate_args=("--self-test",),
            options={"output_root": "--output-root", "run_name": "--run-name"},
        ),
        Suite(
            name="ablation",
            script="run_verifier_ablation.py",
            summary="Paired verifier / prompt ablation in Docker with hidden tests",
            cost="small per trial; needs the ablation image",
            options={"repetitions": "--repeats"},
        ),
        Suite(
            name="memory",
            script="run_memory_consolidation_eval.py",
            summary="Memory extraction, consolidation and retrieval with real models",
            cost="bounded request count",
            validate_scripts=("check_memory_consolidation.py", "check_memory_runtime.py"),
        ),
        Suite(
            name="memory-runtime",
            script="run_memory_runtime_eval.py",
            summary="Two real sessions exercising memory at runtime",
            cost="bounded: six model and five embedding requests",
            validate_scripts=("check_memory_runtime.py",),
        ),
        Suite(
            name="memory-candidate",
            script="run_memory_candidate_eval.py",
            summary="Two-request check of candidate extraction",
            cost="two model requests",
        ),
    )
}


class UsageError(Exception):
    pass


def build_run_command(suite: Suite, args: argparse.Namespace) -> tuple[list[str], dict[str, str], bool]:
    """Return (command, extra env, execute) for ``run``.

    ``execute`` is False only when --dry-run is asked of a runner without a
    plan mode; the caller then prints the command instead of running it.
    """
    command = [sys.executable, str(SCRIPTS / suite.script), *suite.fixed_args]
    env: dict[str, str] = {}
    execute = True
    if suite.mode == "run-flag":
        if not args.dry_run:
            command.append("--run")
    elif suite.mode == "dry-run-flag":
        if args.dry_run:
            command.append("--dry-run")
    elif args.dry_run:
        execute = False

    for option in ("repetitions", "split", "intensity", "tasks", "task_set", "output_root", "run_name"):
        value = getattr(args, option, None)
        if value in (None, "", "all") or (option == "split" and value == "all"):
            continue
        flag = suite.options.get(option)
        if flag is None:
            raise UsageError(f"suite {suite.name!r} does not support --{option.replace('_', '-')}")
        if flag.startswith("env:"):
            env[flag[4:]] = str(value)
        elif option == "tasks" and flag == "--task":
            for task in _split_csv(value):
                command.extend([flag, task])
        else:
            command.extend([flag, str(value)])
    command.extend(args.passthrough)
    return command, env, execute


def build_validate_commands(suite: Suite, args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []
    if suite.validate_args is not None:
        command = [sys.executable, str(SCRIPTS / suite.script), *suite.fixed_args, *suite.validate_args]
        if getattr(args, "output_root", None) and "output_root" in suite.options:
            command.extend([suite.options["output_root"], args.output_root])
        commands.append(command)
    commands.extend([sys.executable, str(SCRIPTS / script)] for script in suite.validate_scripts)
    if not commands:
        raise UsageError(f"suite {suite.name!r} has no offline validation")
    return commands


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _child_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing else str(PROJECT_ROOT) + os.pathsep + existing
    env.update(extra)
    return env


def _execute(command: list[str], extra_env: dict[str, str] | None = None) -> int:
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=_child_env(extra_env or {}), check=False).returncode


# ----------------------------------------------------------------------
# Result index and comparison
# ----------------------------------------------------------------------


def normalize_summary(run_dir: Path) -> dict[str, Any] | None:
    """Reduce any suite's summary.json to one row of the results index."""
    path = run_dir / "summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    runs = overall.get("runs", payload.get("task_count", payload.get("case_count")))
    passed = overall.get("passed", payload.get("passed"))
    rate = overall.get("success_rate", payload.get("pass_rate"))
    if rate is None and isinstance(runs, (int, float)) and runs and isinstance(passed, (int, float)):
        rate = passed / runs
    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "suite": str(payload.get("suite") or "unknown"),
        "benchmark": str(payload.get("benchmark") or payload.get("benchmark_name") or ""),
        "split": str(payload.get("split") or "all"),
        "model": str(model.get("model") or ""),
        "intensity": str(model.get("intensity") or ""),
        "repetitions": payload.get("repetitions"),
        "runs": runs,
        "passed": passed,
        "success_rate": rate,
        "trap_failures": overall.get("trap_failures"),
        "generated_at": str(payload.get("generated_at") or ""),
    }


def collect_index(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    if results_root.is_dir():
        for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
            row = normalize_summary(run_dir)
            if row is not None:
                rows.append(row)
    return rows


def render_index(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Eval results index",
        "",
        "Generated by `python -m eval report`. One row per run directory with a summary.json.",
        "",
        "| Run | Suite | Split | Model / intensity | Passed | Success | Low-level failures |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        rate = row["success_rate"]
        model = " / ".join(item for item in (row["model"], row["intensity"]) if item) or "-"
        lines.append(
            f"| {row['run']} | {row['suite']} | {row['split']} | {model} | "
            f"{_fmt(row['passed'])}/{_fmt(row['runs'])} | "
            f"{f'{rate:.1%}' if isinstance(rate, (int, float)) else '-'} | {_fmt(row['trap_failures'])} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def task_outcomes(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-task pass counts for a run, from the dev-task or Terminal-Bench format."""
    outcomes: dict[str, dict[str, Any]] = {}
    raw = run_dir / "results.jsonl"
    if raw.is_file():
        for line in raw.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            entry = outcomes.setdefault(
                item["task_id"],
                {"runs": 0, "passed": 0, "trap_failures": 0, "split": item.get("split", "")},
            )
            entry["runs"] += 1
            entry["passed"] += 1 if item.get("success") else 0
            entry["trap_failures"] += 1 if item.get("trap_failure") else 0
        return outcomes
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for item in summary.get("task_results") or []:
        entry = outcomes.setdefault(
            item["task"], {"runs": 0, "passed": 0, "trap_failures": 0, "split": item.get("split", "")}
        )
        entry["runs"] += 1
        entry["passed"] += 1 if item.get("status") == "passed" else 0
    return outcomes


def compare_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired comparison over tasks present in both runs.

    The unit is the task: its success rate in each run. The confidence
    interval is a percentile bootstrap over tasks of the mean per-task
    difference, so repeated runs of one task are not treated as independent.
    """
    base = task_outcomes(baseline_dir)
    cand = task_outcomes(candidate_dir)
    shared = sorted(set(base) & set(cand))
    if not shared:
        raise UsageError("the two runs share no tasks")
    diffs = {
        task: cand[task]["passed"] / cand[task]["runs"] - base[task]["passed"] / base[task]["runs"]
        for task in shared
    }
    result = {
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
        "overall": _paired_stats(shared, base, cand, diffs, resamples, seed),
        "by_split": {},
        "tasks": [
            {
                "task": task,
                "split": cand[task]["split"] or base[task]["split"],
                "baseline": f"{base[task]['passed']}/{base[task]['runs']}",
                "candidate": f"{cand[task]['passed']}/{cand[task]['runs']}",
                "delta": round(diffs[task], 3),
            }
            for task in shared
        ],
        "only_in_baseline": sorted(set(base) - set(cand)),
        "only_in_candidate": sorted(set(cand) - set(base)),
    }
    splits = sorted({cand[t]["split"] or base[t]["split"] for t in shared} - {""})
    for split in splits:
        members = [t for t in shared if (cand[t]["split"] or base[t]["split"]) == split]
        result["by_split"][split] = _paired_stats(members, base, cand, diffs, resamples, seed)
    return result


def _paired_stats(
    tasks: list[str],
    base: dict[str, dict[str, Any]],
    cand: dict[str, dict[str, Any]],
    diffs: dict[str, float],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = [diffs[task] for task in tasks]
    mean = sum(values) / len(values)
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(max(1, resamples))
    )
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]

    def rate(source: dict[str, dict[str, Any]]) -> float:
        runs = sum(source[task]["runs"] for task in tasks)
        return sum(source[task]["passed"] for task in tasks) / runs if runs else 0.0

    return {
        "tasks": len(tasks),
        "baseline_success": round(rate(base), 3),
        "candidate_success": round(rate(cand), 3),
        "mean_task_delta": round(mean, 3),
        "ci95": [round(low, 3), round(high, 3)],
        "improved": sum(1 for value in values if value > 0),
        "regressed": sum(1 for value in values if value < 0),
        "baseline_trap_failures": sum(base[task]["trap_failures"] for task in tasks),
        "candidate_trap_failures": sum(cand[task]["trap_failures"] for task in tasks),
    }


def render_comparison(result: dict[str, Any]) -> str:
    def row(name: str, stats: dict[str, Any]) -> str:
        return (
            f"| {name} | {stats['tasks']} | {stats['baseline_success']:.1%} → {stats['candidate_success']:.1%} | "
            f"{stats['mean_task_delta']:+.3f} [{stats['ci95'][0]:+.3f}, {stats['ci95'][1]:+.3f}] | "
            f"{stats['improved']} / {stats['regressed']} | "
            f"{stats['baseline_trap_failures']} → {stats['candidate_trap_failures']} |"
        )

    lines = [
        f"Baseline:  {result['baseline']}",
        f"Candidate: {result['candidate']}",
        "",
        "| Scope | Tasks | Success | Mean task delta [95% CI] | Improved / regressed | Low-level failures |",
        "| --- | --- | --- | --- | --- | --- |",
        row("all", result["overall"]),
    ]
    lines += [row(split, stats) for split, stats in result["by_split"].items()]
    changed = [item for item in result["tasks"] if item["delta"]]
    if changed:
        lines += ["", "| Task | Split | Baseline | Candidate | Delta |", "| --- | --- | --- | --- | --- |"]
        lines += [
            f"| {item['task']} | {item['split'] or '-'} | {item['baseline']} | {item['candidate']} | {item['delta']:+.3f} |"
            for item in sorted(changed, key=lambda item: item["delta"])
        ]
    for key, label in (("only_in_baseline", "Only in baseline"), ("only_in_candidate", "Only in candidate")):
        if result[key]:
            lines += ["", f"{label}: {', '.join(result[key])}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        index = argv.index("--")
        argv, passthrough = argv[:index], argv[index + 1:]

    parser = argparse.ArgumentParser(prog="python -m eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List the suites, what they measure and what they cost.")

    validate = sub.add_parser("validate", help="Offline checks that need no model.")
    validate.add_argument("suite", choices=sorted(SUITES))
    validate.add_argument("--output-root", default="")

    run = sub.add_parser("run", help="Run a suite (paid); add --dry-run to only print the plan.")
    run.add_argument("suite", choices=sorted(SUITES))
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--repetitions", type=int)
    run.add_argument("--split", choices=SPLITS, default="all")
    run.add_argument("--intensity", choices=INTENSITIES)
    run.add_argument("--tasks", default="", help="Comma-separated task ids.")
    run.add_argument("--task-set", default="", help="Task set for suites that have several (tbench).")
    run.add_argument("--output-root", default="")
    run.add_argument("--run-name", default="")

    report = sub.add_parser("report", help="Index every run under the results root (writes INDEX.md).")
    report.add_argument("--results-root", default=str(RESULTS_ROOT))
    report.add_argument("--json", action="store_true")

    compare = sub.add_parser("compare", help="Paired per-task comparison of two runs, with a bootstrap CI.")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--resamples", type=int, default=10000)
    compare.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    args.passthrough = passthrough
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "list":
            for suite in SUITES.values():
                checks = "validate" if suite.validate_args is not None or suite.validate_scripts else "-"
                print(f"{suite.name:16} {suite.summary}\n{'':16} cost: {suite.cost}; offline: {checks}")
            return 0
        if args.command == "validate":
            status = 0
            for command in build_validate_commands(SUITES[args.suite], args):
                status = status or _execute(command)
            return status
        if args.command == "run":
            command, env, execute = build_run_command(SUITES[args.suite], args)
            if not execute:
                print("Plan (this runner has no dry-run mode; nothing was started):")
                print("$ " + " ".join(command))
                return 0
            return _execute(command, env)
        if args.command == "report":
            root = Path(args.results_root)
            rows = collect_index(root)
            if args.json:
                print(json.dumps(rows, ensure_ascii=False, indent=2))
                return 0
            text = render_index(rows)
            root.mkdir(parents=True, exist_ok=True)
            (root / "INDEX.md").write_text(text, encoding="utf-8", newline="\n")
            print(text, end="")
            print(f"\nWrote {root / 'INDEX.md'}")
            return 0
        if args.command == "compare":
            result = compare_runs(Path(args.baseline), Path(args.candidate), resamples=args.resamples)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else render_comparison(result), end="")
            return 0
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
