import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness_code_agent import config
from harness_code_agent.workspace.shell_session import (
    PersistentShellSession,
    _docker_resource_args,
    _docker_user_arg,
    _DockerShellBackend,
    _posix_capture_script,
    docker_shell_hint,
    sandbox_mode,
    validate_shell_configuration,
    windows_shell_kind,
)


def _commands_for_platform(temp_dir: str) -> dict[str, str]:
    if os.name == "nt":
        if windows_shell_kind() == "pwsh":
            return {
                "make_and_cd": "Set-Location ..",
                "pwd": "(Get-Location).Path",
                "pwd_expected": str(Path(temp_dir).resolve().parent),
                "set_env": "$env:FOO='bar'",
                "get_env": "$env:FOO",
                "sleep": "Start-Sleep -Seconds 5",
                "alive": "'alive'",
            }
        return {
            "make_and_cd": "mkdir -p subdir && cd subdir",
            "pwd": "wslpath -w \"$(pwd)\"",
            "pwd_expected": str(Path(temp_dir).resolve() / "subdir"),
            "set_env": "export FOO=bar",
            "get_env": "printf '%s' \"$FOO\"",
            "sleep": "sleep 5",
            "alive": "echo alive",
        }
    return {
        "make_and_cd": "mkdir -p subdir && cd subdir",
        "pwd": "pwd",
        "pwd_expected": os.path.abspath(os.path.join(temp_dir, "subdir")),
        "set_env": "export FOO=bar",
        "get_env": "printf '%s' \"$FOO\"",
        "sleep": "sleep 5",
        "alive": "echo alive",
    }


class PersistentShellSessionTests(unittest.TestCase):
    def _make_temp_dir(self) -> str:
        return tempfile.mkdtemp(dir=os.getcwd())

    def test_commands_for_platform_returns_posix_commands(self):
        temp_dir = self._make_temp_dir()
        try:
            with patch("os.name", "posix"):
                commands = _commands_for_platform(temp_dir)

            self.assertEqual(commands["make_and_cd"], "mkdir -p subdir && cd subdir")
            self.assertEqual(commands["pwd"], "pwd")
            self.assertEqual(commands["set_env"], "export FOO=bar")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_shell_preserves_working_directory(self):
        temp_dir = self._make_temp_dir()
        commands = _commands_for_platform(temp_dir)
        shell = PersistentShellSession(cwd=temp_dir)
        try:
            shell.run(commands["make_and_cd"])
            result = shell.run(commands["pwd"])

            if os.name != "nt":
                self.assertEqual(result.exit_code, 0)
            self.assertEqual(Path(result.stdout.strip()).resolve(), Path(commands["pwd_expected"]).resolve())
        finally:
            shell.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_shell_preserves_environment_variables(self):
        temp_dir = self._make_temp_dir()
        commands = _commands_for_platform(temp_dir)
        shell = PersistentShellSession(cwd=temp_dir)
        try:
            shell.run(commands["set_env"])
            result = shell.run(commands["get_env"])

            if os.name != "nt":
                self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.stdout.strip(), "bar")
        finally:
            shell.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_shell_timeout_interrupts_command_but_keeps_session_alive(self):
        temp_dir = self._make_temp_dir()
        commands = _commands_for_platform(temp_dir)
        shell = PersistentShellSession(cwd=temp_dir)
        try:
            timed_out = shell.run(commands["sleep"], timeout=1)
            follow_up = shell.run(commands["alive"])

            self.assertTrue(timed_out.timed_out)
            self.assertIn("alive", follow_up.stdout)
            if os.name != "nt":
                self.assertEqual(follow_up.exit_code, 0)
        finally:
            shell.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_windows_shell_rejects_auto_instead_of_falling_back(self):
        if os.name != "nt":
            self.skipTest("Windows-only shell selection")
        from harness_code_agent.workspace import shell_session

        with (
            patch.object(config, "WINDOWS_SHELL", "auto"),
            self.assertRaisesRegex(ValueError, "must be pwsh or wsl"),
        ):
            shell_session.windows_shell_path()

    def test_windows_shell_wsl_uses_dedicated_backend(self):
        if os.name != "nt":
            self.skipTest("Windows-only shell selection")

        temp_dir = self._make_temp_dir()
        try:
            with (
                patch.object(config, "SANDBOX_MODE", "host"),
                patch.object(config, "WINDOWS_SHELL", "wsl"),
                patch("harness_code_agent.workspace.shell_session.shutil.which", return_value="C:/Windows/System32/wsl.exe"),
                patch("harness_code_agent.workspace.shell_session._WslShellBackend") as wsl_backend,
            ):
                shell = PersistentShellSession(cwd=temp_dir)

            wsl_backend.assert_called_once_with(
                str(Path(temp_dir).resolve()),
                executable="C:/Windows/System32/wsl.exe",
            )
            self.assertIs(shell._backend, wsl_backend.return_value)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_selected_windows_shell_must_exist(self):
        if os.name != "nt":
            self.skipTest("Windows-only shell selection")

        with (
            patch.object(config, "SANDBOX_MODE", "host"),
            patch.object(config, "WINDOWS_SHELL", "pwsh"),
            patch("harness_code_agent.workspace.shell_session.shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "pwsh selected.*not found"),
        ):
            validate_shell_configuration()

    def test_windows_powershell_backend_preserves_utf8_output(self):
        if os.name != "nt" or windows_shell_kind() != "pwsh":
            self.skipTest("PowerShell backend not active")
        temp_dir = self._make_temp_dir()
        shell = PersistentShellSession(cwd=temp_dir)
        try:
            result = shell.run("'中文 output'")

            self.assertEqual(result.exit_code, 0)
            self.assertIn("中文 output", result.stdout)
        finally:
            shell.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_windows_powershell_backend_uses_final_statement_status(self):
        if os.name != "nt" or windows_shell_kind() != "pwsh":
            self.skipTest("PowerShell backend not active")
        temp_dir = self._make_temp_dir()
        shell = PersistentShellSession(cwd=temp_dir)
        try:
            recovered = shell.run(
                'python -c "import sys; sys.exit(2)"; Write-Output recovered'
            )
            failed = shell.run('python -c "import sys; sys.exit(2)"')

            self.assertEqual(recovered.exit_code, 0)
            self.assertIn("recovered", recovered.stdout)
            self.assertEqual(failed.exit_code, 2)
        finally:
            shell.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_docker_sandbox_mode_uses_docker_backend(self):

        temp_dir = self._make_temp_dir()
        try:
            with (
                patch.object(config, "SANDBOX_MODE", "docker"),
                patch("harness_code_agent.workspace.shell_session._DockerShellBackend") as docker_backend,
            ):
                shell = PersistentShellSession(cwd=temp_dir)

            docker_backend.assert_called_once_with(str(Path(temp_dir).resolve()))
            self.assertIs(shell._backend, docker_backend.return_value)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_docker_run_args_mount_workspace_read_write_with_offline_network(self):
        from harness_code_agent.workspace import shell_session

        temp_dir = self._make_temp_dir()
        try:
            with (
                patch.object(config, "DOCKER_IMAGE", "python:3.12"),
                patch.object(config, "DOCKER_NETWORK", "none"),
                patch.object(config, "DOCKER_USER", ""),
                patch("harness_code_agent.workspace.shell_session._docker_user_arg", return_value=[]),
            ):
                args = shell_session._docker_run_args("hca-test", str(Path(temp_dir).resolve()))

            self.assertEqual(args[:4], ["docker", "run", "-d", "--name"])
            self.assertIn("hca-test", args)
            self.assertIn("--network", args)
            self.assertIn("none", args)
            self.assertIn("--security-opt=no-new-privileges", args)
            self.assertIn("-v", args)
            self.assertIn(f"{Path(temp_dir).resolve()}:/workspace", args)
            self.assertIn("-w", args)
            self.assertIn("/workspace", args)
            self.assertEqual(args[-3:], ["python:3.12", "sleep", "infinity"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_docker_run_args_explicit_user(self):
        from harness_code_agent.workspace import shell_session

        temp_dir = self._make_temp_dir()
        try:
            with (
                patch.object(config, "DOCKER_IMAGE", "python:3.12"),
                patch.object(config, "DOCKER_NETWORK", "none"),
                patch.object(config, "DOCKER_USER", "1000:1000"),
            ):
                args = shell_session._docker_run_args("hca-test", str(Path(temp_dir).resolve()))

            self.assertIn("--user", args)
            user_idx = args.index("--user")
            self.assertEqual(args[user_idx + 1], "1000:1000")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_docker_user_arg_auto_posix_non_root(self):
        with (
            patch.object(config, "DOCKER_USER", ""),
            patch("os.name", "posix"),
            patch("os.getuid", return_value=1000, create=True),
            patch("os.getgid", return_value=1000, create=True),
        ):
            result = _docker_user_arg()
            self.assertEqual(result, ["--user", "1000:1000"])

    def test_docker_user_arg_no_mapping_for_root(self):
        with (
            patch.object(config, "DOCKER_USER", ""),
            patch("os.name", "posix"),
            patch("os.getuid", return_value=0, create=True),
            patch("os.getgid", return_value=0, create=True),
        ):
            result = _docker_user_arg()
            self.assertEqual(result, [])

    def test_docker_user_arg_maps_non_root_uid_even_with_root_gid(self):
        with (
            patch.object(config, "DOCKER_USER", ""),
            patch("os.name", "posix"),
            patch("os.getuid", return_value=1000, create=True),
            patch("os.getgid", return_value=0, create=True),
        ):
            result = _docker_user_arg()
            self.assertEqual(result, ["--user", "1000:0"])

    def test_docker_user_arg_no_mapping_on_windows(self):
        with (
            patch.object(config, "DOCKER_USER", ""),
            patch("os.name", "nt"),
        ):
            result = _docker_user_arg()
            self.assertEqual(result, [])

    def test_docker_user_arg_no_uid_available(self):
        with (
            patch.object(config, "DOCKER_USER", ""),
            patch("os.name", "posix"),
            patch("os.getuid", side_effect=AttributeError, create=True),
        ):
            result = _docker_user_arg()
            self.assertEqual(result, [])

    def test_sandbox_mode_invalid_value_raises(self):
        with (
            patch.object(config, "SANDBOX_MODE", "lxc"),
            self.assertRaises(ValueError),
        ):
            sandbox_mode()

    def test_docker_shell_hint_host_mode(self):
        with patch.object(config, "SANDBOX_MODE", "host"):
            self.assertEqual(docker_shell_hint(), "host shell")

    def test_docker_shell_hint_cli_missing(self):
        with (
            patch.object(config, "SANDBOX_MODE", "docker"),
            patch("harness_code_agent.workspace.shell_session.docker_cli_path", return_value=None),
        ):
            self.assertEqual(docker_shell_hint(), "Docker CLI not found")

    def test_docker_shell_hint_docker_available(self):
        with (
            patch.object(config, "SANDBOX_MODE", "docker"),
            patch.object(config, "DOCKER_IMAGE", "python:3.12"),
            patch.object(config, "DOCKER_NETWORK", "none"),
            patch("harness_code_agent.workspace.shell_session.docker_cli_path", return_value="/usr/bin/docker"),
        ):
            hint = docker_shell_hint()
            self.assertIn("Docker sandbox", hint)
            self.assertIn("python:3.12", hint)

    def test_docker_shell_hint_includes_user_when_set(self):
        with (
            patch.object(config, "SANDBOX_MODE", "docker"),
            patch.object(config, "DOCKER_IMAGE", "python:3.12"),
            patch.object(config, "DOCKER_NETWORK", "none"),
            patch.object(config, "DOCKER_USER", "1000:1000"),
            patch("harness_code_agent.workspace.shell_session.docker_cli_path", return_value="/usr/bin/docker"),
        ):
            hint = docker_shell_hint()
            self.assertIn("user=1000:1000", hint)

    def test_docker_cleanup_swallows_docker_rm_errors(self):

        backend = _DockerShellBackend.__new__(_DockerShellBackend)
        backend.container_name = "hca-test-xyz"
        backend._container_started = True

        def fake_run(*args, **kwargs):
            raise RuntimeError("docker rm failed")

        with patch("harness_code_agent.workspace.shell_session.subprocess.run", side_effect=fake_run):
            # Should not raise
            backend._cleanup_impl()
            self.assertFalse(backend._container_started)

    def test_stop_exec_process_waits_for_reader_thread(self):
        import threading

        backend = _DockerShellBackend.__new__(_DockerShellBackend)
        # Simulate a stopped process
        proc = unittest.mock.MagicMock()
        proc.poll.return_value = 0
        proc.stdin = unittest.mock.MagicMock()
        proc.stdout = unittest.mock.MagicMock()
        backend._process = proc

        reader_started = threading.Event()
        reader_exited = threading.Event()

        def reader_run():
            reader_started.set()
            reader_exited.wait()  # block until told to exit

        reader = threading.Thread(target=reader_run)
        reader.start()
        reader_started.wait()
        backend._reader = reader

        backend._stop_exec_process(timeout=2)

        reader_exited.set()
        reader.join(timeout=2)
        self.assertFalse(reader.is_alive())

    def test_docker_backend_interrupt_does_not_set_closed(self):
        backend = _DockerShellBackend.__new__(_DockerShellBackend)
        backend._closed = False
        backend.container_name = "hca-test-interrupt"
        backend._container_started = False
        backend._reader = unittest.mock.MagicMock()
        backend._reader.is_alive.return_value = False
        backend._queue = unittest.mock.MagicMock()
        backend._process = unittest.mock.MagicMock()
        backend._process.poll.return_value = 0
        backend._process.stdin = unittest.mock.MagicMock()
        backend._process.stdout = unittest.mock.MagicMock()

        with (
            patch.object(backend, "_interrupt_impl") as mock_interrupt,
            patch.object(backend, "_sync") as mock_sync,
        ):
            backend.interrupt()
            mock_interrupt.assert_called_once()
            mock_sync.assert_called_once()

        # interrupt() should NOT set _closed
        self.assertFalse(backend._closed)

    def test_docker_backend_close_sets_closed(self):
        import queue

        backend = _DockerShellBackend.__new__(_DockerShellBackend)
        backend._closed = False
        backend.container_name = "hca-test-close"
        backend._container_started = False
        backend._reader = unittest.mock.MagicMock()
        backend._queue = unittest.mock.MagicMock()
        backend._queue.get_nowait.side_effect = queue.Empty

        with patch.object(backend, "_cleanup_impl"):
            backend.close()
            self.assertTrue(backend._closed)

        # Second close is a no-op
        with patch.object(backend, "_cleanup_impl") as mock_cleanup:
            backend.close()
            mock_cleanup.assert_not_called()


class CaptureScriptAndResourceTests(unittest.TestCase):
    def test_posix_script_caps_output_and_preserves_braces_without_ulimits(self):
        with patch.object(config, "SHELL_MAX_OUTPUT_BYTES", 1234), patch.object(
            config, "SHELL_CPU_SECONDS", 0
        ), patch.object(config, "SHELL_FSIZE_BLOCKS", 0), patch.object(
            config, "SHELL_MEMORY_KB", 0
        ):
            script = _posix_capture_script("echo hi", "OUT", "ERR", "EXIT", prefix="__hca")
        self.assertIn("__hca_max=1234", script)
        self.assertIn('head -c "$__hca_max" "$1"', script)
        self.assertIn("{\n", script)
        self.assertNotIn("ulimit", script)

    def test_posix_script_runs_command_in_subshell_when_ulimits_enabled(self):
        with patch.object(config, "SHELL_MAX_OUTPUT_BYTES", 0), patch.object(
            config, "SHELL_CPU_SECONDS", 10
        ), patch.object(config, "SHELL_FSIZE_BLOCKS", 0), patch.object(
            config, "SHELL_MEMORY_KB", 0
        ):
            script = _posix_capture_script("echo hi", "OUT", "ERR", "EXIT", prefix="__hca")
        self.assertIn("ulimit -t 10", script)
        self.assertIn("(\n", script)
        self.assertIn("__hca_max=0", script)

    def test_docker_resource_args_include_default_caps(self):
        with patch.object(config, "DOCKER_MEMORY_MB", 2048), patch.object(
            config, "DOCKER_CPUS", 2.0
        ):
            self.assertEqual(
                _docker_resource_args(), ["--memory", "2048m", "--cpus", "2"]
            )

    def test_docker_resource_args_empty_when_caps_disabled(self):
        with patch.object(config, "DOCKER_MEMORY_MB", 0), patch.object(
            config, "DOCKER_CPUS", 0.0
        ):
            self.assertEqual(_docker_resource_args(), [])


class ShellOutputCapIntegrationTests(unittest.TestCase):
    @staticmethod
    def _big_output_command() -> str:
        if os.name == "nt" and windows_shell_kind() == "pwsh":
            return "Write-Output ('a' * 50000)"
        return "head -c 50000 /dev/zero | tr '\\0' 'a'"

    def test_run_interrupts_output_over_cap_and_session_recovers(self):
        temp_dir = tempfile.mkdtemp(dir=os.getcwd())
        shell = PersistentShellSession(cwd=temp_dir)
        try:
            with patch.object(config, "SHELL_MAX_OUTPUT_BYTES", 20000):
                result = shell.run(self._big_output_command(), timeout=30)
            self.assertTrue(result.output_truncated)
            self.assertGreaterEqual(result.output_bytes, 20000)
            self.assertIn("a", result.stdout)

            recovered = shell.run("echo recovered", timeout=30)
            self.assertEqual(recovered.exit_code, 0)
            self.assertIn("recovered", recovered.stdout)
        finally:
            shell.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
