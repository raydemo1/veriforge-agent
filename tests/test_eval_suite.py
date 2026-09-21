import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class EvalSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_terminal_bench_8task_subset_is_fixed_and_lightweight(self):
        payload = json.loads(Path("eval/tasks/terminal_bench_8task.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["suite"], "terminal_bench_8task")
        self.assertEqual(payload["task_set"], "8task")
        self.assertEqual(payload["benchmark_name"], "Terminal-Bench 2.1 8-task subset")
        self.assertEqual(
            payload["tasks"],
            [
                "fix-git",
                "overfull-hbox",
                "build-cython-ext",
                "custom-memory-heap-crash",
                "git-leak-recovery",
                "log-summary-date-ranges",
                "large-scale-text-editing",
                "query-optimize",
            ],
        )

    def test_terminal_bench_24task_subset_is_fixed_and_bounded(self):
        payload = json.loads(Path("eval/tasks/terminal_bench_24task.json").read_text(encoding="utf-8"))
        metadata = json.loads(Path("harness_code_agent/profiles/tb2_tasks.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["suite"], "terminal_bench_24task")
        self.assertEqual(payload["task_set"], "24task")
        self.assertEqual(payload["benchmark_name"], "Terminal-Bench 2.1 24-task subset")
        self.assertEqual(
            payload["tasks"],
            [
                "fix-git",
                "overfull-hbox",
                "build-cython-ext",
                "custom-memory-heap-crash",
                "git-leak-recovery",
                "log-summary-date-ranges",
                "large-scale-text-editing",
                "query-optimize",
                "cancel-async-tasks",
                "kv-store-grpc",
                "polyglot-c-py",
                "sqlite-db-truncate",
                "nginx-request-logging",
                "git-multibranch",
                "configure-git-webserver",
                "fix-code-vulnerability",
                "sanitize-git-repo",
                "openssl-selfsigned-cert",
                "multi-source-data-merger",
                "sparql-university",
                "db-wal-recovery",
                "extract-elf",
                "adaptive-rejection-sampler",
                "model-extraction-relu-logits",
            ],
        )
        self.assertEqual(len(payload["tasks"]), 24)
        self.assertEqual(len(set(payload["tasks"])), 24)
        for task in payload["tasks"]:
            self.assertIn(task, metadata)
            self.assertLessEqual(float(metadata[task]["agent_timeout_sec"]), 1800.0)

    def test_terminal_bench_eval_dry_run_defaults_to_8task_and_can_select_24task(self):
        from eval.scripts.run_terminal_bench_eval import _dry_run_plan, parse_args

        default_args = parse_args(["--dry-run"])
        default_plan = _dry_run_plan(default_args)
        self.assertEqual(default_plan["terminal_bench_task_set"], "8task")
        self.assertEqual(len(default_plan["terminal_bench_tasks"]), 8)

        larger_args = parse_args(["--dry-run", "--tbench-task-set", "24task"])
        larger_plan = _dry_run_plan(larger_args)
        self.assertEqual(larger_plan["terminal_bench_task_set"], "24task")
        self.assertEqual(len(larger_plan["terminal_bench_tasks"]), 24)

    def test_claw_swe_bench_lite80_config_is_fixed(self):
        payload = json.loads(Path("eval/tasks/claw_swe_bench_lite80.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["suite"], "claw_swe_bench_lite80")
        self.assertEqual(payload["task_set"], "lite80")
        self.assertEqual(payload["benchmark_name"], "Claw-SWE-Bench Lite80")
        self.assertEqual(payload["dataset_name"], "TokenRhythm/Claw-SWE-Bench")
        self.assertEqual(payload["dataset_config"], "lite")
        self.assertEqual(payload["split"], "test")

    def test_memory_ab_seed_records_cover_success_markers(self):
        payload = json.loads(Path("eval/tasks/memory_ab.json").read_text(encoding="utf-8"))

        for task in payload["tasks"]:
            seed_text = " ".join(
                [
                    task.get("prompt", ""),
                    *[
                        " ".join(
                            [
                                record.get("title", ""),
                                record.get("summary", ""),
                                " ".join(record.get("tags", [])),
                                " ".join(record.get("source_paths", [])),
                            ]
                        )
                        for record in task.get("memory_records", [])
                    ],
                ]
            ).lower()
            for marker in task.get("success_markers", []):
                self.assertIn(str(marker).lower(), seed_text, task["id"])

    def test_deepseek_v4_flash_cost_uses_official_cache_split(self):
        from eval.benchmarks.usage_metrics import estimate_usage_cost

        cost = estimate_usage_cost(
            {
                "prompt_tokens": 1000,
                "cached_tokens": 800,
                "cache_miss_tokens": 200,
                "completion_tokens": 500,
            },
            model="deepseek-v4-flash",
        )

        self.assertEqual(cost["pricing_source"], "https://api-docs.deepseek.com/quick_start/pricing")
        self.assertEqual(cost["pricing_per_1m_tokens"]["input_cache_hit"], 0.0028)
        self.assertEqual(cost["pricing_per_1m_tokens"]["input_cache_miss"], 0.14)
        self.assertEqual(cost["pricing_per_1m_tokens"]["output"], 0.28)
        self.assertAlmostEqual(cost["estimated_cost_usd"], (800 * 0.0028 + 200 * 0.14 + 500 * 0.28) / 1_000_000)

    def test_claw_swe_bench_eval_dry_run_includes_lite80(self):
        from eval.scripts.run_claw_swe_bench_eval import _dry_run_plan, parse_args

        args = parse_args(["--dry-run", "--claw-limit", "3"])
        plan = _dry_run_plan(args)

        self.assertEqual(plan["claw_swe_bench_task_set"], "lite80")
        self.assertEqual(plan["claw_swe_bench_benchmark_name"], "Claw-SWE-Bench Lite80")
        self.assertEqual(plan["claw_swe_bench_dataset"]["dataset_name"], "TokenRhythm/Claw-SWE-Bench")
        self.assertEqual(plan["claw_swe_bench_dataset"]["limit"], 3)

    def test_run_tbench_suite_records_per_task_category_results(self):
        from eval.scripts.run_terminal_bench_eval import parse_args, run_tbench_suite

        args = parse_args([
            "--tbench-task-set",
            "8task",
            "--output-root",
            str(self.root / "results"),
            "--run-name",
            "unit",
        ])

        call_count = 0

        def fake_run(command, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                for path in (self.root / "results").glob("*/task_outputs"):
                    shutil.rmtree(path, ignore_errors=True)
            task = command[-1]
            returncode = 1 if task == "overfull-hbox" else 0
            stdout = f"stdout {task}"
            if task == "fix-git":
                stdout += "\nHCA_TERMINAL_BENCH_RESULT:" + json.dumps({
                    "task_results": [{
                        "task": "fix-git",
                        "passed": True,
                        "reward": 1.0,
                        "metrics": {
                            "turns": {"started": 1, "finished": 1},
                            "tokens": {
                                "llm_calls": 2,
                                "prompt_tokens": 100,
                                "cached_tokens": 40,
                                "cache_miss_tokens": 60,
                                "completion_tokens": 30,
                                "total_tokens": 130,
                            },
                            "tools": {"tool_calls": 3},
                            "usage_cost": {"estimated_cost_usd": 0.00002},
                        },
                    }]
                })
            if task == "build-cython-ext":
                stdout += "\nHCA_TERMINAL_BENCH_RESULT:" + json.dumps({
                    "task_results": [{
                        "task": "build-cython-ext",
                        "passed": False,
                        "reward": 0.0,
                        "metrics": {},
                    }]
                })
            return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=f"stderr {task}")

        with patch("eval.scripts.run_terminal_bench_eval.subprocess.run", side_effect=fake_run) as run_mock:
            run_dir = run_tbench_suite(args)

        self.assertEqual(run_mock.call_count, 8)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["task_set"], "8task")
        self.assertEqual(summary["task_count"], 8)
        self.assertEqual(summary["passed"], 6)
        self.assertAlmostEqual(summary["pass_rate"], 6 / 8)
        self.assertEqual(summary["category_results"]["debugging"]["task_count"], 3)
        self.assertEqual(summary["category_results"]["debugging"]["passed"], 1)
        self.assertEqual(summary["turn_totals"]["finished"], 1)
        self.assertEqual(summary["token_totals"]["total_tokens"], 130)
        self.assertEqual(summary["tool_calls"], 3)
        self.assertAlmostEqual(summary["estimated_cost_usd"], 0.00002)
        cython = next(item for item in summary["task_results"] if item["task"] == "build-cython-ext")
        self.assertEqual(cython["status"], "failed")
        self.assertEqual(cython["reward"], 0.0)
        self.assertIn("--jobs-dir", cython["command"])
        self.assertTrue(Path(cython["stdout_path"]).exists())
        self.assertTrue(Path(cython["stderr_path"]).exists())

    def test_run_tbench_suite_records_timeout_and_continues(self):
        from eval.scripts.run_terminal_bench_eval import parse_args, run_tbench_suite

        args = parse_args([
            "--tbench-task-set",
            "8task",
            "--task",
            "fix-git",
            "--task",
            "overfull-hbox",
            "--output-root",
            str(self.root / "results"),
            "--run-name",
            "unit-timeout",
            "--task-wall-timeout",
            "1",
        ])

        def fake_run(command, **kwargs):
            task = command[-1]
            if task == "fix-git":
                raise subprocess.TimeoutExpired(
                    command,
                    timeout=kwargs["timeout"],
                    output=b"partial stdout",
                    stderr=b"partial stderr",
                )
            stdout = "\nHCA_TERMINAL_BENCH_RESULT:" + json.dumps({
                "task_results": [{
                    "task": task,
                    "passed": True,
                    "reward": 1.0,
                    "metrics": {},
                }]
            })
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        with patch("eval.scripts.run_terminal_bench_eval.subprocess.run", side_effect=fake_run) as run_mock:
            run_dir = run_tbench_suite(args)

        self.assertEqual(run_mock.call_count, 2)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["passed"], 1)

        timed_out = next(item for item in summary["task_results"] if item["task"] == "fix-git")
        self.assertEqual(timed_out["status"], "failed")
        self.assertTrue(timed_out["timed_out"])
        self.assertEqual(timed_out["returncode"], 124)
        self.assertEqual(timed_out["failure_kind"], "launcher_timeout")
        self.assertIn("partial stdout", Path(timed_out["stdout_path"]).read_text(encoding="utf-8"))
        timeout_stderr = Path(timed_out["stderr_path"]).read_text(encoding="utf-8")
        self.assertIn("partial stderr", timeout_stderr)
        self.assertIn("Terminal-Bench launcher timed out after 1 seconds.", timeout_stderr)

        continued = next(item for item in summary["task_results"] if item["task"] == "overfull-hbox")
        self.assertEqual(continued["status"], "passed")

    def test_tbench_dry_run_includes_parallelism(self):
        from eval.scripts.run_terminal_bench_eval import _dry_run_plan, parse_args

        args = parse_args([
            "--tbench-task-set",
            "24task",
            "--task",
            "fix-git",
            "--tbench-parallelism",
            "3",
            "--dry-run",
        ])

        plan = _dry_run_plan(args)

        self.assertEqual(plan["terminal_bench_task_set"], "24task")
        self.assertEqual(plan["terminal_bench_tasks"], ["fix-git"])
        self.assertEqual(plan["tbench_parallelism"], 3)

    def test_harbor_usage_collects_metrics_from_result_text_fallback(self):
        from eval.benchmarks.usage_metrics import collect_harbor_job_usage

        trial_dir = self.root / "jobs" / "job-1" / "task-a__abc"
        trial_dir.mkdir(parents=True)
        metrics = {
            "session_id": "session-a",
            "turns": {"started": 1, "finished": 1},
            "tokens": {"total_tokens": 42},
            "tools": {"tool_calls": 2},
            "usage_cost": {"estimated_cost_usd": 0.0001},
        }
        payload = {
            "task_name": "terminal-bench/task-a",
            "trial_name": "task-a__abc",
            "agent_result": {"metadata": None},
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": {
                "exception_message": "stdout: HCA_EVAL_METRICS:" + json.dumps(metrics)
            },
        }
        (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

        usage = collect_harbor_job_usage(self.root / "jobs" / "job-1")

        self.assertEqual(usage["task_results"][0]["session_id"], "session-a")
        self.assertTrue(usage["task_results"][0]["passed"])
        self.assertEqual(usage["turn_totals"]["finished"], 1)
        self.assertEqual(usage["tool_calls"], 2)

    def test_harbor_usage_falls_back_to_hca_artifact_session_id(self):
        from eval.benchmarks.usage_metrics import collect_harbor_job_usage

        trial_dir = self.root / "jobs" / "job-1" / "task-a__abc"
        artifact_dir = trial_dir / "artifacts" / "hca" / "session-artifact"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "early_manifest.json").write_text("{}", encoding="utf-8")
        payload = {
            "task_name": "terminal-bench/task-a",
            "trial_name": "task-a__abc",
            "agent_result": {"metadata": None},
            "verifier_result": {"rewards": {"reward": 0.0}},
        }
        (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

        usage = collect_harbor_job_usage(self.root / "jobs" / "job-1")

        item = usage["task_results"][0]
        self.assertEqual(item["session_id"], "session-artifact")
        self.assertTrue(item["has_hca_artifacts"])
        self.assertEqual(item["job_dir"], str(self.root / "jobs" / "job-1"))
        self.assertEqual(item["trial_dir"], str(trial_dir))

    def test_basic_metrics_memory_cases_run_with_full_access_permissions(self):
        from eval.scripts.run_basic_metrics_eval import parse_args, run_memory_suite

        args = parse_args([
            "--suites",
            "memory",
            "--memory-limit",
            "1",
            "--output-root",
            str(self.root / "results"),
            "--run-name",
            "unit",
        ])

        real_run = subprocess.run

        def fake_run(command, **kwargs):
            if command[:1] == ["git"]:
                # Memory seeding resolves the repo key through git; let it run.
                return real_run(command, **kwargs)
            self.assertEqual(kwargs["env"]["HARNESS_PERMISSION_MODE"], "danger-full-access")
            return subprocess.CompletedProcess(command, 0, stdout="veriforge session: missing\nmarker", stderr="")

        with (
            patch("eval.scripts.run_basic_metrics_eval.subprocess.run", side_effect=fake_run),
            patch("eval.scripts.run_basic_metrics_eval._session_metrics", return_value={}),
        ):
            run_dir = run_memory_suite(args)

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["task_count"], 1)

    def test_run_claw_suite_invokes_launcher(self):
        from eval.scripts.run_claw_swe_bench_eval import parse_args, run_claw_suite

        args = parse_args([
            "--claw-limit",
            "2",
            "--output-root",
            str(self.root / "results"),
            "--run-name",
            "unit",
            "--claw-no-install-deps",
        ])

        with patch("eval.scripts.run_claw_swe_bench_eval.subprocess.run") as run_mock:
            run_claw_suite(args)

        command = run_mock.call_args.args[0]
        self.assertIn("run_claw_swe_bench.py", command[1])
        self.assertIn("--limit", command)
        self.assertIn("2", command)
        self.assertIn("--no-install-deps", command)

    def test_harness_claw_adapter_builds_container_args_and_command(self):
        from eval.benchmarks.harness_claw_adapter import (
            HarnessCodeAgentAdapter,
            _agent_command,
        )

        adapter = HarnessCodeAgentAdapter(
            model="deepseek-v4-flash",
            timeout=120,
            max_turns=12,
            repo_root=Path.cwd(),
            install_deps=False,
        )
        args = adapter.container_run_args("example__repo-1")
        joined = " ".join(args)
        self.assertIn("/opt/harness-code-agent:ro", joined)
        self.assertIn("HARNESS_WORKSPACE=/testbed", joined)
        self.assertIn("HARNESS_MODEL=deepseek-v4-flash", joined)
        self.assertIn("HARNESS_MODEL_HARD=deepseek-v4-flash", joined)
        self.assertIn("HARNESS_MODEL_INTENSITY=normal", joined)
        self.assertNotIn("MAX_AGENT_ITERATIONS", joined)

        command = _agent_command(timeout=120)
        self.assertIn("PROFILE_CODING_AGENT_TASK_BUDGET=120", command)
        self.assertIn("hca_claw_runner.py", command)

    def test_eval_scripts_self_test(self):
        for script in (
            "eval/scripts/run_basic_metrics_eval.py",
            "eval/scripts/run_terminal_bench_eval.py",
            "eval/scripts/run_claw_swe_bench_eval.py",
            "eval/scripts/rebuild_eval_results.py",
        ):
            completed = subprocess.run(
                [sys.executable, script, "--self-test"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)

    def test_terminal_bench_task_diagnostics_classifies_timeout(self):
        from eval.scripts.run_terminal_bench_eval import _task_diagnostics

        diagnostic = _task_diagnostics(
            task="example",
            status="failed",
            returncode=0,
            stdout="harbor.trial.errors.AgentTimeoutError: Agent execution timed out after 900.0 seconds",
            stderr="",
            launcher_result={"trial_name": "example__abc", "metrics": {}},
        )

        self.assertEqual(diagnostic["failure_kind"], "agent_timeout")
        self.assertTrue(diagnostic["missing_metrics"])

    def test_terminal_bench_task_diagnostics_classifies_agent_setup_failure(self):
        from eval.scripts.run_terminal_bench_eval import _task_diagnostics

        diagnostic = _task_diagnostics(
            task="example",
            status="failed",
            returncode=1,
            stdout="",
            stderr="",
            launcher_result={
                "trial_name": "example__abc",
                "exception_type": "NonZeroAgentExitCodeError",
                "exception_message": "FATAL: failed to install harness dependencies",
                "metrics": {},
            },
        )

        self.assertEqual(diagnostic["failure_kind"], "infra_or_setup_failure")
        self.assertTrue(diagnostic["missing_metrics"])

    def test_terminal_bench_task_diagnostics_classifies_verifier_failure(self):
        from eval.scripts.run_terminal_bench_eval import _task_diagnostics

        diagnostic = _task_diagnostics(
            task="example",
            status="failed",
            returncode=0,
            stdout="FAILED ../tests/test_outputs.py::test_x - AssertionError: bad output",
            stderr="",
            launcher_result={"trial_name": "example__abc", "metrics": {"tokens": {}}},
        )

        self.assertEqual(diagnostic["failure_kind"], "failed_verifier")
        self.assertFalse(diagnostic["missing_metrics"])
        self.assertIn("FAILED", diagnostic["verifier_failure_headline"])

    def test_terminal_bench_task_diagnostics_extracts_final_report_summary(self):
        from eval.scripts.run_terminal_bench_eval import _task_diagnostics

        job_dir = self.root / "jobs" / "job-final"
        artifact_dir = job_dir / "trial-a" / "artifacts" / "hca" / "session-final"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (artifact_dir / "trajectory.jsonl").write_text(
            json.dumps({
                "type": "final_report",
                "payload": {"summary": "Verifier failed after final validation."},
            }) + "\n",
            encoding="utf-8",
        )

        diagnostic = _task_diagnostics(
            task="example",
            status="failed",
            returncode=0,
            stdout="FAILED ../tests/test_outputs.py::test_x - AssertionError: bad output",
            stderr="",
            launcher_result={
                "trial_name": "example__abc",
                "session_id": "session-final",
                "job_dir": str(job_dir),
                "metrics": {"tokens": {}},
            },
        )

        self.assertTrue(diagnostic["has_hca_artifacts"])
        self.assertEqual(diagnostic["final_report_summary"], "Verifier failed after final validation.")
        self.assertEqual(
            diagnostic["diagnostic"]["final_report_summary"],
            "Verifier failed after final validation.",
        )

    def test_terminal_bench_task_diagnostics_finds_artifacts_across_job_timestamps(self):
        from eval.scripts.run_terminal_bench_eval import _task_diagnostics

        trial_dir = self.root / "jobs" / "2026-06-27__12-35-11" / "example__abc"
        artifact_dir = trial_dir / "artifacts" / "hca" / "session-cross-job"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "early_manifest.json").write_text("{}", encoding="utf-8")

        with patch("eval.scripts.run_terminal_bench_eval.PROJECT_ROOT", self.root):
            diagnostic = _task_diagnostics(
                task="example",
                status="failed",
                returncode=0,
                stdout="FAILED ../tests/test_outputs.py::test_x - AssertionError: bad output",
                stderr="",
                launcher_result={
                    "trial_name": "example__abc",
                    "metrics": {},
                },
            )

        self.assertTrue(diagnostic["has_hca_artifacts"])
        self.assertEqual(diagnostic["resolved_session_id"], "session-cross-job")
        self.assertEqual(diagnostic["diagnostic"]["job_dir"], str(trial_dir.parent))


if __name__ == "__main__":
    unittest.main()
