import os
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from eval.benchmarks.harbor_env import runner_env_vars
from eval.benchmarks.run_terminal_bench import (
    DEFAULT_VERIFIER_NO_PROXY_HOSTS,
    build_harbor_run_command,
    build_launch_environment,
    default_local_dataset_path,
    docker_daemon_running,
    ensure_local_dataset,
    is_valid_harbor_dataset,
    main,
    patch_verifier_proxy_env,
    pre_pull_task_images,
    repair_task_images,
    resolve_harbor_dataset_path,
    resolve_harbor_executable,
)


class TerminalBenchLauncherTests(unittest.TestCase):
    def _workspace_path(self, name: str) -> Path:
        path = Path(os.getcwd()) / "workspace" / f"{name}-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_build_command_targets_registry_or_local_dataset(self):
        agent = ["--agent-import-path", "eval.benchmarks.harbor_agent:HarnessAgent"]
        local = Path("E:/tmp/terminal-bench-2-1")

        def build(tasks, **kwargs):
            options = {"runner_env": None, "dataset_path": None, "force_build": False, **kwargs}
            return build_harbor_run_command(harbor_executable="harbor", tasks=tasks, **options)

        self.assertEqual(
            build(["fix-git", "headless-terminal"]),
            ["harbor", "run", "-d", "terminal-bench@2.1", *agent,
             "--include-task-name", "fix-git", "--include-task-name", "headless-terminal"],
        )
        self.assertEqual(
            build(["fix-git"], runner_env="daytona", dataset_path=local),
            ["harbor", "run", "--path", str(local), *agent, "--env", "daytona", "--include-task-name", "fix-git"],
        )
        self.assertEqual(
            build(["fix-git"], dataset_path=local, force_build=True),
            ["harbor", "run", "--path", str(local), *agent, "--force-build", "--include-task-name", "fix-git"],
        )

    def test_build_environment_uses_repo_local_temp_and_loads_missing_dotenv_values(self):
        repo_root = self._workspace_path("test-terminal-bench-launcher-env")
        try:
            dotenv_path = repo_root / ".env"
            dotenv_path.write_text(
                "OPENAI_API_KEY=test-key\nHARNESS_MODEL=dotenv-model\n",
                encoding="utf-8",
            )
            base_env = {
                "PATH": "base-path",
                "HARNESS_MODEL": "existing-model",
                "PYTHONPATH": "existing-pythonpath",
            }

            env = build_launch_environment(repo_root, base_env=base_env, dotenv_path=dotenv_path)

            expected_temp = str((repo_root / ".harbor" / "tmp").resolve())
            self.assertEqual(env["TEMP"], expected_temp)
            self.assertEqual(env["TMP"], expected_temp)
            self.assertEqual(env["OPENAI_API_KEY"], "test-key")
            self.assertEqual(env["HARNESS_MODEL"], "existing-model")
            self.assertEqual(
                env["PYTHONPATH"],
                os.pathsep.join([str(repo_root.resolve()), "existing-pythonpath"]),
            )
            self.assertNotIn("MAX_AGENT_ITERATIONS", env)
            self.assertEqual(env["MAX_AGENT_TOOL_CALLS"], "400")
            self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
            self.assertEqual(env["PYTHONUTF8"], "1")
            self.assertEqual(env["NO_COLOR"], "1")
            self.assertEqual(env["TERM"], "dumb")
            self.assertEqual(env["RICH_FORCE_TERMINAL"], "0")
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_build_environment_preserves_explicit_agent_budget(self):
        repo_root = self._workspace_path("test-terminal-bench-launcher-budget")
        try:
            env = build_launch_environment(
                repo_root,
                base_env={
                    "PATH": "base-path",
                    "MAX_AGENT_TOOL_CALLS": "500",
                },
            )

            self.assertEqual(env["MAX_AGENT_TOOL_CALLS"], "500")
            self.assertEqual(env["PYTHONPATH"], str(repo_root.resolve()))
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_harbor_agent_forwards_agent_budget_environment(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "secret",
                "OPENAI_BASE_URL": "https://api.deepseek.com",
                "HARNESS_MODEL": "deepseek-v4-flash",
                "MAX_AGENT_TOTAL_TOKENS": "900000",
                "MAX_AGENT_TOOL_CALLS": "400",
                "AGENT_BUDGET_WARN_FRACTION": "0.9",
            },
            clear=True,
        ):
            env = runner_env_vars()

        self.assertNotIn("MAX_AGENT_ITERATIONS", env)
        self.assertEqual(env["MAX_AGENT_TOTAL_TOKENS"], "900000")
        self.assertEqual(env["MAX_AGENT_TOOL_CALLS"], "400")
        self.assertEqual(env["AGENT_BUDGET_WARN_FRACTION"], "0.9")
        self.assertEqual(env["HARNESS_MODEL"], "deepseek-v4-flash")
        self.assertEqual(env["HCA_MODEL_API_NO_PROXY_HOSTS"], "api.deepseek.com")

    def test_harbor_agent_snapshot_excludes_runtime_artifacts_and_large_python_tarball(self):
        harbor_agent = self._import_harbor_agent_with_fakes()
        repo_root = self._workspace_path("test-harbor-agent-snapshot-source")
        dest_parent = self._workspace_path("test-harbor-agent-snapshot-dest")
        snapshot = dest_parent / "snapshot"
        try:
            (repo_root / "harness_code_agent").mkdir()
            (repo_root / "harness_code_agent" / "__init__.py").write_text("", encoding="utf-8")
            (repo_root / "vendor_wheels").mkdir()
            (repo_root / "vendor_wheels" / "openai-1.0.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
            python_archive = Path("vendor_wheels") / harbor_agent.PYTHON_STANDALONE_TARBALL
            (repo_root / python_archive).write_text(
                "large runtime archive",
                encoding="utf-8",
            )
            (repo_root / ".veriforge" / "traces").mkdir(parents=True)
            (repo_root / ".veriforge" / "traces" / "trace.jsonl").write_text("trace", encoding="utf-8")
            (repo_root / ".harbor" / "tmp").mkdir(parents=True)
            (repo_root / ".harbor" / "tmp" / "leftover.txt").write_text("tmp", encoding="utf-8")
            (repo_root / "eval" / "results" / "old-run").mkdir(parents=True)
            (repo_root / "eval" / "results" / "old-run" / "summary.json").write_text("{}", encoding="utf-8")
            (repo_root / ".env").write_text("SECRET=1\n", encoding="utf-8")
            (repo_root / ".venv").mkdir()
            (repo_root / "frontend" / "node_modules").mkdir(parents=True)

            harbor_agent._copy_repo_snapshot(repo_root, snapshot)

            self.assertTrue((snapshot / "harness_code_agent" / "__init__.py").exists())
            self.assertTrue((snapshot / "vendor_wheels" / "openai-1.0.0-py3-none-any.whl").exists())
            self.assertFalse((snapshot / python_archive).exists())
            self.assertFalse((snapshot / ".veriforge").exists())
            self.assertFalse((snapshot / ".harbor").exists())
            self.assertFalse((snapshot / "eval" / "results").exists())
            self.assertFalse((snapshot / ".env").exists())
            self.assertFalse((snapshot / ".venv").exists())
            self.assertFalse((snapshot / "frontend" / "node_modules").exists())
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)
            shutil.rmtree(dest_parent, ignore_errors=True)

    def test_harbor_agent_apt_mirror_snippet_rewrites_debian_sources(self):
        harbor_agent = self._import_harbor_agent_with_fakes()

        command = harbor_agent._configure_debian_apt_mirror_command()

        self.assertIn("HCA_APT_MIRROR", command)
        self.assertIn("mirrors.tuna.tsinghua.edu.cn", command)
        self.assertIn("/etc/apt/sources.list.d/debian.sources", command)
        self.assertIn("/etc/apt/sources.list.d/ubuntu.sources", command)
        self.assertIn("/etc/apt/sources.list", command)
        self.assertIn("deb.debian.org/debian-security", command)
        self.assertIn("${APT_MIRROR}/debian-security", command)
        self.assertIn("archive.ubuntu.com/ubuntu", command)
        self.assertIn("${UBUNTU_APT_MIRROR}", command)

    def test_harbor_agent_apt_install_uses_fallback_mirrors_and_no_proxy(self):
        harbor_agent = self._import_harbor_agent_with_fakes()

        command = harbor_agent._apt_get_install_command("curl git")

        self.assertIn("HCA_NO_PROXY_HOSTS", command)
        self.assertIn("astral.sh", command)
        self.assertIn("download.pytorch.org", command)
        self.assertIn("download-r2.pytorch.org", command)
        self.assertIn("huggingface.co", command)
        self.assertIn("us.aws.cdn.hf.co", command)
        self.assertIn("HCA_APT_MIRROR_PAIRS", command)
        self.assertIn("mirrors.tuna.tsinghua.edu.cn", command)
        self.assertIn("mirrors.aliyun.com", command)
        self.assertIn("mirrors.ustc.edu.cn", command)
        self.assertIn("deb.debian.org", command)
        self.assertIn("archive.ubuntu.com", command)
        self.assertIn("for APT_PAIR in $APT_MIRROR_PAIRS", command)
        self.assertIn("Dpkg::Lock::Timeout=300", command)
        self.assertIn("curl git", command)

    def _import_harbor_agent_with_fakes(self):
        fake_base = types.ModuleType("harbor.agents.installed.base")
        fake_base.BaseInstalledAgent = object
        fake_base.with_prompt_template = lambda fn: fn

        fake_environment_base = types.ModuleType("harbor.environments.base")
        fake_environment_base.BaseEnvironment = object

        fake_context = types.ModuleType("harbor.models.agent.context")
        fake_context.AgentContext = object

        fake_modules = {
            "harbor": types.ModuleType("harbor"),
            "harbor.agents": types.ModuleType("harbor.agents"),
            "harbor.agents.installed": types.ModuleType("harbor.agents.installed"),
            "harbor.agents.installed.base": fake_base,
            "harbor.environments": types.ModuleType("harbor.environments"),
            "harbor.environments.base": fake_environment_base,
            "harbor.models": types.ModuleType("harbor.models"),
            "harbor.models.agent": types.ModuleType("harbor.models.agent"),
            "harbor.models.agent.context": fake_context,
        }

        previous = sys.modules.pop("eval.benchmarks.harbor_agent", None)
        with patch.dict(sys.modules, fake_modules):
            from eval.benchmarks import harbor_agent

        sys.modules.pop("eval.benchmarks.harbor_agent", None)
        if previous is not None:
            sys.modules["eval.benchmarks.harbor_agent"] = previous
        return harbor_agent

    def test_docker_daemon_running_returns_false_on_cli_failure(self):
        with patch("eval.benchmarks.run_terminal_bench.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 1

            self.assertFalse(docker_daemon_running())

    def test_main_fails_fast_when_docker_daemon_is_unavailable(self):
        repo_root = self._workspace_path("test-terminal-bench-docker-preflight")
        dataset_path = repo_root / "dataset"
        task_dir = dataset_path / "tasks" / "overfull-hbox"
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text("name='overfull-hbox'\n", encoding="utf-8")
        try:
            with (
                patch("eval.benchmarks.run_terminal_bench.resolve_harbor_executable", return_value="harbor"),
                patch("eval.benchmarks.run_terminal_bench.repair_task_images", return_value=0),
                patch("eval.benchmarks.run_terminal_bench.docker_daemon_running", return_value=False),
                patch("eval.benchmarks.run_terminal_bench.subprocess.run") as run_mock,
                patch("sys.argv", ["run_terminal_bench.py", "--task", "overfull-hbox", "--dataset-path", str(dataset_path)]),
            ):
                result = main()

            self.assertEqual(result, 125)
            run_mock.assert_not_called()
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_resolve_harbor_executable_falls_back_to_user_scripts(self):
        repo_root = self._workspace_path("test-terminal-bench-launcher-bin")
        try:
            appdata = repo_root / "AppData" / "Roaming"
            scripts_dir = appdata / "Python" / "Python312" / "Scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            harbor_path = scripts_dir / "harbor.exe"
            harbor_path.write_text("", encoding="utf-8")

            with patch("eval.benchmarks.run_terminal_bench.shutil.which", return_value=None):
                resolved = resolve_harbor_executable(
                    {
                        "APPDATA": str(appdata),
                        "USERPROFILE": str(repo_root / "UserProfile"),
                    }
                )

            self.assertEqual(resolved, str(harbor_path))
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_default_local_dataset_path_is_repo_scoped(self):
        repo_root = Path("E:/repo-root")

        resolved = default_local_dataset_path(repo_root)

        self.assertEqual(
            resolved,
            (repo_root / ".harbor" / "datasets" / "terminal-bench-2-1").resolve(),
        )

    def test_is_valid_harbor_dataset_requires_2_1_tasks_layout(self):
        repo_root = self._workspace_path("test-terminal-bench-validity")
        try:
            invalid_dataset = repo_root / "terminal-bench-2-1"
            invalid_dataset.mkdir(parents=True, exist_ok=True)
            (invalid_dataset / ".git").mkdir(exist_ok=True)

            self.assertFalse(is_valid_harbor_dataset(invalid_dataset))

            valid_task_dir = invalid_dataset / "fix-git"
            valid_task_dir.mkdir(parents=True, exist_ok=True)
            (valid_task_dir / "task.toml").write_text("name='fix-git'\n", encoding="utf-8")

            self.assertFalse(is_valid_harbor_dataset(invalid_dataset))

            task_dir = invalid_dataset / "tasks" / "fix-git"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.toml").write_text("name='fix-git'\n", encoding="utf-8")
            self.assertTrue(is_valid_harbor_dataset(invalid_dataset))
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_resolve_harbor_dataset_path_returns_2_1_tasks_directory(self):
        repo_root = self._workspace_path("test-terminal-bench-harbor-path")
        try:
            dataset_path = repo_root / "terminal-bench-2-1"
            task_dir = dataset_path / "tasks" / "fix-git"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.toml").write_text("name='fix-git'\n", encoding="utf-8")

            self.assertEqual(resolve_harbor_dataset_path(dataset_path), dataset_path / "tasks")
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_resolve_harbor_dataset_path_rejects_legacy_root_task_layout(self):
        repo_root = self._workspace_path("test-terminal-bench-harbor-path-legacy")
        try:
            dataset_path = repo_root / "terminal-bench-2-1"
            task_dir = dataset_path / "fix-git"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.toml").write_text("name='fix-git'\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "tasks/\\*/task.toml"):
                resolve_harbor_dataset_path(dataset_path)
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_ensure_local_dataset_reclones_invalid_partial_checkout(self):
        repo_root = self._workspace_path("test-terminal-bench-reclone")
        dataset_path = repo_root / ".harbor" / "datasets" / "terminal-bench-2-1"
        dataset_path.mkdir(parents=True, exist_ok=True)
        downloaded_paths = []

        def fake_download(path):
            downloaded_paths.append(path)
            task_dir = dataset_path / "tasks" / "fix-git"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.toml").write_text("name='fix-git'\n", encoding="utf-8")

        try:
            with patch("eval.benchmarks.run_terminal_bench.shutil.rmtree") as rmtree_mock, patch(
                "eval.benchmarks.run_terminal_bench._download_and_extract_dataset_archive",
                side_effect=fake_download,
            ) as download_mock:
                resolved = ensure_local_dataset(dataset_path)

            self.assertEqual(resolved, dataset_path.resolve())
            self.assertTrue(is_valid_harbor_dataset(dataset_path))
            self.assertEqual(download_mock.call_count, 1)
            self.assertEqual(rmtree_mock.call_count, 1)
            self.assertEqual(downloaded_paths, [dataset_path.resolve()])
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_repair_task_images_rewrites_only_unavailable_selected_images(self):
        repo_root = self._workspace_path("test-terminal-bench-image-repair")
        dataset_path = repo_root / ".harbor" / "datasets" / "terminal-bench-2-1"
        try:
            for task in ("overfull-hbox", "fix-git", "custom-memory-heap-crash"):
                task_dir = dataset_path / "tasks" / task
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "task.toml").write_text(
                    f'[environment]\ndocker_image = "ghcr.io/laude-institute/terminal-bench/{task}:2.0"', encoding="utf-8")
            seen: list[str] = []

            def fake_exists(image: str) -> bool:
                seen.append(image)
                return image in {"alexgshaw/overfull-hbox:20251031", "ghcr.io/laude-institute/terminal-bench/fix-git:2.0"}

            with patch("eval.benchmarks.run_terminal_bench._docker_image_exists", side_effect=fake_exists):
                repaired = repair_task_images(dataset_path, ["overfull-hbox", "fix-git"])

            def image(task):
                return (dataset_path / "tasks" / task / "task.toml").read_text(encoding="utf-8")

            self.assertEqual(repaired, 1)
            self.assertIn('docker_image = "alexgshaw/overfull-hbox:20251031"', image("overfull-hbox"))
            self.assertIn("terminal-bench/fix-git:2.0", image("fix-git"))
            self.assertNotIn("custom-memory-heap-crash", "\n".join(seen))
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_patch_verifier_proxy_env_fills_merges_and_appends_sections(self):
        repo_root = self._workspace_path("test-terminal-bench-verifier-proxy")
        dataset_path = repo_root / ".harbor" / "datasets" / "terminal-bench-2-1"
        image = '[environment]\ndocker_image = "alexgshaw/circuit-fibsqrt:20251031"\n'
        tasks = {
            "empty": "[verifier]\ntimeout_sec = 3600.0\n\n[verifier.env]\n\n" + image,
            "existing": '[verifier.env]\nNO_PROXY = "localhost,astral.sh"\n\n' + image,
            "missing": "[environment]\n",
            "defaults": '[verifier.env]\nNO_PROXY = "localhost,astral.sh"\nno_proxy = "localhost,astral.sh"\n',
        }
        try:
            for name, content in tasks.items():
                (dataset_path / "tasks" / name).mkdir(parents=True, exist_ok=True)
                (dataset_path / "tasks" / name / "task.toml").write_text(content, encoding="utf-8")

            def read(name):
                return (dataset_path / "tasks" / name / "task.toml").read_text(encoding="utf-8")

            hosts = ("astral.sh", "github.com")
            selected = ["empty", "existing", "missing"]
            self.assertEqual(patch_verifier_proxy_env(dataset_path, selected, no_proxy_hosts=hosts), 3)
            self.assertEqual(patch_verifier_proxy_env(dataset_path, selected, no_proxy_hosts=hosts), 0)

            empty = read("empty")
            self.assertIn('[verifier.env]\nNO_PROXY = "astral.sh,github.com"\nno_proxy = "astral.sh,github.com"', empty)
            self.assertLess(empty.index("NO_PROXY"), empty.index("[environment]"))
            self.assertIn('NO_PROXY = "localhost,astral.sh,github.com"', read("existing"))
            self.assertIn('no_proxy = "astral.sh,github.com"', read("existing"))
            self.assertTrue(read("missing").endswith(
                '[verifier.env]\nNO_PROXY = "astral.sh,github.com"\nno_proxy = "astral.sh,github.com"\n'))

            # The default host list covers the model and dataset download hosts.
            self.assertEqual(patch_verifier_proxy_env(dataset_path, ["defaults"]), 1)
            defaults = read("defaults")
            for host in ("download.pytorch.org", "download-r2.pytorch.org", "huggingface.co",
                         "us.aws.cdn.hf.co", "cas-bridge.xethub.hf.co"):
                self.assertIn(host, defaults)
            self.assertIn('NO_PROXY = "localhost,astral.sh,127.0.0.1', defaults)
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)

    def test_verifier_proxy_defaults_include_huggingface_download_hosts(self):
        self.assertIn("download.pytorch.org", DEFAULT_VERIFIER_NO_PROXY_HOSTS)
        self.assertIn("download-r2.pytorch.org", DEFAULT_VERIFIER_NO_PROXY_HOSTS)
        self.assertIn("huggingface.co", DEFAULT_VERIFIER_NO_PROXY_HOSTS)
        self.assertIn("us.aws.cdn.hf.co", DEFAULT_VERIFIER_NO_PROXY_HOSTS)
        self.assertIn("cas-bridge.xethub.hf.co", DEFAULT_VERIFIER_NO_PROXY_HOSTS)

    def test_pre_pull_task_images_pulls_only_missing_selected_images(self):
        repo_root = self._workspace_path("test-terminal-bench-image-prepull")
        dataset_path = repo_root / ".harbor" / "datasets" / "terminal-bench-2-1"
        try:
            for task, image in {
                "missing-task": "alexgshaw/missing-task:20251031",
                "present-task": "alexgshaw/present-task:20251031",
                "unselected-task": "alexgshaw/unselected-task:20251031",
            }.items():
                task_dir = dataset_path / "tasks" / task
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "task.toml").write_text(
                    "\n".join(["[environment]", f'docker_image = "{image}"']),
                    encoding="utf-8",
                )

            def fake_present(image: str) -> bool:
                return image == "alexgshaw/present-task:20251031"

            with (
                patch("eval.benchmarks.run_terminal_bench._docker_image_present", side_effect=fake_present),
                patch("eval.benchmarks.run_terminal_bench.subprocess.run") as run_mock,
            ):
                run_mock.return_value.returncode = 0
                pulled = pre_pull_task_images(
                    dataset_path,
                    ["missing-task", "present-task"],
                    timeout_sec=123,
                )

            self.assertEqual(pulled, ["alexgshaw/missing-task:20251031"])
            run_mock.assert_called_once()
            self.assertEqual(
                run_mock.call_args.args[0],
                ["docker", "pull", "alexgshaw/missing-task:20251031"],
            )
            self.assertEqual(run_mock.call_args.kwargs["timeout"], 123)
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
