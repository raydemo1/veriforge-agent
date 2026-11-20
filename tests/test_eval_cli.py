import json
from collections import Counter
from pathlib import Path

import pytest

from eval import cli


def _run_args(*argv):
    return cli.parse_args(["run", *argv])


def test_run_flag_suites_only_spend_with_run_and_translate_shared_options():
    suite = cli.SUITES["veriloop"]
    command, env, execute = cli.build_run_command(
        suite, _run_args("veriloop", "--split", "held_out", "--intensity", "fast", "--repetitions", "2")
    )
    assert execute and env == {}
    assert command[1].endswith("run_dev_task_eval.py")
    assert command[2:5] == ["--task-pack", "veriloop", "--run"]
    assert command[command.index("--split") + 1] == "held_out"
    assert command[command.index("--intensity") + 1] == "fast"
    assert command[command.index("--repetitions") + 1] == "2"

    dry, _, _ = cli.build_run_command(suite, _run_args("veriloop", "--dry-run"))
    assert "--run" not in dry


def test_dry_run_flag_suites_and_env_options():
    suite = cli.SUITES["tbench"]
    command, env, execute = cli.build_run_command(
        suite,
        _run_args("tbench", "--task-set", "tb41-20task", "--tasks", "music-harmony,cad-model", "--intensity", "fast"),
    )
    assert execute and "--dry-run" not in command
    assert env == {"HARNESS_MODEL_INTENSITY": "fast"}
    assert command[command.index("--tbench-task-set") + 1] == "tb41-20task"
    assert [command[i + 1] for i, item in enumerate(command) if item == "--task"] == ["music-harmony", "cad-model"]

    planned, _, _ = cli.build_run_command(suite, _run_args("tbench", "--dry-run"))
    assert "--dry-run" in planned


def test_runner_without_plan_mode_is_not_started_on_dry_run():
    _, _, execute = cli.build_run_command(cli.SUITES["context"], _run_args("context", "--dry-run"))
    assert execute is False


def test_unsupported_options_are_rejected_and_passthrough_is_kept():
    with pytest.raises(cli.UsageError):
        cli.build_run_command(cli.SUITES["ablation"], _run_args("ablation", "--split", "evolve"))
    command, _, _ = cli.build_run_command(
        cli.SUITES["metrics"], cli.parse_args(["run", "metrics", "--", "--suites", "cache"])
    )
    assert command[-2:] == ["--suites", "cache"]


def test_validate_commands():
    veriloop = cli.build_validate_commands(cli.SUITES["veriloop"], cli.parse_args(["validate", "veriloop"]))
    assert veriloop == [[*veriloop[0][:2], "--task-pack", "veriloop", "--validate"]]
    memory = cli.build_validate_commands(cli.SUITES["memory"], cli.parse_args(["validate", "memory"]))
    assert [Path(command[1]).name for command in memory] == ["check_memory_consolidation.py", "check_memory_runtime.py"]
    with pytest.raises(cli.UsageError):
        cli.build_validate_commands(cli.SUITES["memory-candidate"], cli.parse_args(["validate", "memory-candidate"]))


def _write_dev_run(root: Path, name: str, outcomes: dict[str, list[tuple[bool, bool]]], split_of: dict[str, str]) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    lines = []
    for task, reps in outcomes.items():
        for success, trap in reps:
            lines.append(json.dumps({
                "task_id": task, "split": split_of[task], "success": success, "trap_failure": trap,
            }))
    (run_dir / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    runs = sum(len(reps) for reps in outcomes.values())
    passed = sum(success for reps in outcomes.values() for success, _ in reps)
    (run_dir / "summary.json").write_text(json.dumps({
        "suite": "veriloop_bench_v1",
        "split": "all",
        "repetitions": 2,
        "model": {"model": "deepseek-flash", "intensity": "fast"},
        "overall": {"runs": runs, "passed": passed, "success_rate": passed / runs,
                    "trap_failures": sum(trap for reps in outcomes.values() for _, trap in reps)},
    }), encoding="utf-8")
    return run_dir


def test_report_index_normalizes_dev_and_tbench_summaries(tmp_path):
    _write_dev_run(tmp_path, "a_dev", {"t1": [(True, False), (False, True)]}, {"t1": "evolve"})
    tb = tmp_path / "b_tbench"
    tb.mkdir()
    (tb / "summary.json").write_text(json.dumps({
        "suite": "tbench", "benchmark": "tb4.1", "split": "held_out", "task_count": 4, "passed": 1, "pass_rate": 0.25,
    }), encoding="utf-8")
    (tmp_path / "not_a_run").mkdir()

    rows = cli.collect_index(tmp_path)
    assert [row["run"] for row in rows] == ["a_dev", "b_tbench"]
    assert rows[0]["success_rate"] == 0.5 and rows[0]["trap_failures"] == 1 and rows[0]["intensity"] == "fast"
    assert rows[1]["runs"] == 4 and rows[1]["success_rate"] == 0.25 and rows[1]["split"] == "held_out"
    assert "| b_tbench | tbench | held_out |" in cli.render_index(rows)


def test_compare_is_paired_by_task_with_split_breakdown(tmp_path):
    split_of = {"a": "evolve", "b": "evolve", "c": "held_out", "d": "held_out"}
    base = _write_dev_run(tmp_path, "base", {
        "a": [(False, True), (False, True)], "b": [(True, False), (True, False)],
        "c": [(False, True), (True, False)], "d": [(False, False), (False, False)],
    }, split_of)
    cand = _write_dev_run(tmp_path, "cand", {
        "a": [(True, False), (True, False)], "b": [(True, False), (True, False)],
        "c": [(True, False), (True, False)], "d": [(False, False), (False, False)],
    }, split_of)

    result = cli.compare_runs(base, cand, resamples=2000)
    overall = result["overall"]
    assert overall["tasks"] == 4
    assert overall["baseline_success"] == 0.375 and overall["candidate_success"] == 0.75
    assert overall["mean_task_delta"] == 0.375
    assert overall["ci95"][0] <= 0.375 <= overall["ci95"][1]
    assert (overall["improved"], overall["regressed"]) == (2, 0)
    assert (overall["baseline_trap_failures"], overall["candidate_trap_failures"]) == (3, 0)
    assert result["by_split"]["evolve"]["mean_task_delta"] == 0.5
    assert result["by_split"]["held_out"]["mean_task_delta"] == 0.25
    assert cli.compare_runs(base, cand, resamples=2000) == result
    assert "| all | 4 | 37.5% → 75.0% |" in cli.render_comparison(result)


def test_trap_failure_needs_only_trap_checks_failing():
    from eval.scripts.run_dev_task_eval import _failed_tests, _is_trap_failure

    task = {"traps": ["deliverable"]}
    trap_only = "FAILED eval_hidden_x_test.py::test_trap_report_file - FileNotFoundError: x\n1 failed, 1 passed"
    mixed = trap_only + "\nFAILED eval_hidden_x_test.py::test_summary_values - assert 1 == 2"
    collection = "ERROR eval_hidden_x_test.py - ModuleNotFoundError: No module named 'stats'"

    assert _failed_tests(mixed) == (["test_summary_values", "test_trap_report_file"], False)
    assert _failed_tests(collection) == ([], True)
    assert _is_trap_failure(task, False, trap_only) is True
    assert _is_trap_failure(task, False, mixed) is False
    assert _is_trap_failure(task, False, collection) is False
    assert _is_trap_failure(task, True, trap_only) is False
    assert _is_trap_failure({}, False, trap_only) is False


def test_veriloop_tasks_share_projects_and_check_several_traps():
    from eval.scripts.run_dev_task_eval import TRAP_TEST_PREFIX, _failed_trap_categories, _hidden_test_names
    from eval.tasks import veriloop_bench_v1 as bench

    assert len(bench.TASKS) == 24
    assert (len(bench.SPLITS["evolve"]), len(bench.SPLITS["held_out"])) == (12, 12)
    for project in bench.PROJECTS:
        splits = Counter(task["split"] for task in bench.TASKS if task["project"] == project)
        assert splits == {"evolve": 4, "held_out": 4}, project
    for task in bench.TASKS:
        names = _hidden_test_names(task)
        traps = [name for name in names if name.startswith(TRAP_TEST_PREFIX)]
        assert len(traps) >= 2, task["id"]
        assert all(bench.trap_category(name) for name in traps), task["id"]
        assert task["near_miss_fails_all_traps"] and "@golden" not in task["near_miss"].values()
        assert "CONTRIBUTING.md" in task["files"] and "CONTRIBUTING" not in task["prompt"], task["id"]

    assert bench.trap_category("test_trap_verify_claim_x") == "verify_claim"
    assert bench.trap_category("test_trap_edge_case_y") == "edge_case"
    assert bench.trap_category("test_core") == ""
    task = {"traps": ["cross_file", "deliverable"]}
    assert _failed_trap_categories(task, ["test_trap_deliverable_changelog", "test_core"]) == ["deliverable"]
    assert _failed_trap_categories({"traps": ["interface"]}, ["test_trap_unnamed"]) == []
