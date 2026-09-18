from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness_code_agent import config
from harness_code_agent.agent.runtime_state import AgentRuntimeState
from harness_code_agent.runtime.builtins.filesystem import (
    list_files,
    repo_search,
)
from harness_code_agent.runtime.middleware.terminal_shell_edit import (
    TerminalShellEditPolicyMiddleware,
)
from harness_code_agent.runtime.middleware.tool_guard import ToolGuardMiddleware
from harness_code_agent.runtime.tool_result import ToolResult


def _result(text: str, *, status: str | None = None) -> ToolResult:
    """Build a ToolResult from the legacy text conventions used in these tests."""
    if status is None:
        status = "failed" if text.startswith(("[error]", "[blocked]")) else "success"
    metadata = {"status_source": "permission"} if text.startswith("[blocked]") else {}
    error = text.removeprefix("[error] ").removeprefix("[blocked] ") if status == "failed" else None
    return ToolResult(tool="run_bash", status=status, output=text, error=error, metadata=metadata)


class RepositoryToolPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_workspace = config.WORKSPACE
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        config.WORKSPACE = str(self.workspace)
        (self.workspace / "pkg").mkdir()
        (self.workspace / "pkg" / "target.py").write_text("VALUE = 'needle'\n", encoding="utf-8")
        (self.workspace / "pkg" / "__pycache__").mkdir()
        (self.workspace / "pkg" / "__pycache__" / "ignored.pyc").write_text("needle", encoding="utf-8")

    def tearDown(self) -> None:
        config.WORKSPACE = self._old_workspace
        self._tmp.cleanup()

    def test_repo_search_uses_bounded_explicit_path_and_excludes_generated_dirs(self) -> None:
        result = repo_search("needle", path=".", max_results=5)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.metadata["explicit_path"], ".")
        self.assertIn("pkg/target.py", result.output.replace("\\", "/"))
        self.assertNotIn("__pycache__", result.output)

    def test_list_files_defaults_to_depth_two_and_hides_internal_dirs(self) -> None:
        result = list_files(".")

        self.assertEqual(result.status, "success")
        self.assertIn("pkg/", result.output.replace("\\", "/"))
        self.assertIn("pkg/target.py", result.output.replace("\\", "/"))
        self.assertNotIn(".harness", result.output)


class ShellPolicyTests(unittest.TestCase):
    def test_bare_rg_without_path_is_blocked_without_mutating_args(self) -> None:
        middleware = ToolGuardMiddleware()
        args = {"command": 'rg -n "needle" --type py', "timeout": 300}
        original = dict(args)

        blocked = middleware.before_tool("run_bash", args, [], runtime_state=AgentRuntimeState())

        self.assertIsNotNone(blocked)
        self.assertIn("[blocked]", blocked.output)
        # The guard is a pure decision point: the model's request is unchanged.
        self.assertEqual(args, original)

    def test_rg_with_explicit_path_is_allowed(self) -> None:
        middleware = ToolGuardMiddleware()
        args = {"command": 'rg -n "needle" --type py src/pkg', "timeout": 300}

        blocked = middleware.before_tool("run_bash", args, [], runtime_state=AgentRuntimeState())

        self.assertIsNone(blocked)

    def test_recursive_shell_browse_is_blocked_without_stopping(self) -> None:
        # ToolPolicy only intercepts; repeated-block stops are owned by
        # ToolFailurePolicyMiddleware via the canonical failure model.
        middleware = ToolGuardMiddleware()
        state = AgentRuntimeState()
        args = {"command": "Get-ChildItem -Recurse"}

        first = middleware.before_tool("run_bash", args, [], runtime_state=state)
        second = middleware.before_tool("run_bash", args, [], runtime_state=state)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIn("[blocked]", first.output)
        self.assertIn("[blocked]", second.output)
        self.assertFalse(state.fallback.stop_requested)

    def test_recursive_grep_over_explicit_file_globs_is_allowed(self) -> None:
        middleware = ToolGuardMiddleware()
        state = AgentRuntimeState()
        args = {
            "command": (
                "cd /app/project && "
                "grep -rn 'n\\.\\(int\\|bool\\|float\\)' "
                "pkg/*.pyx pkg/submodule/*.pyx 2>/dev/null"
            )
        }

        blocked = middleware.before_tool("run_bash", args, [], runtime_state=state)

        self.assertIsNone(blocked)
        self.assertFalse(state.fallback.stop_requested)

    def test_bounded_recursive_grep_on_explicit_absolute_path_is_allowed(self) -> None:
        middleware = ToolGuardMiddleware()
        state = AgentRuntimeState()
        args = {
            "command": (
                "grep -rn '_Facet_Register_impl' "
                "/build/gcc/libstdc++-v3/ 2>/dev/null | head -20"
            )
        }

        blocked = middleware.before_tool("run_bash", args, [], runtime_state=state)

        self.assertIsNone(blocked)
        self.assertFalse(state.fallback.stop_requested)

    def test_unbounded_recursive_grep_on_absolute_directory_remains_blocked(self) -> None:
        middleware = ToolGuardMiddleware()
        state = AgentRuntimeState()
        args = {"command": "grep -rn needle /build/gcc/libstdc++-v3/"}

        blocked = middleware.before_tool("run_bash", args, [], runtime_state=state)

        self.assertIsNotNone(blocked)
        self.assertIn("[blocked]", blocked.output)

    def test_repeated_blocks_never_stop_inside_guard_itself(self) -> None:
        # Batch de-duplication and stop decisions moved to FailureTracker /
        # ToolFailurePolicyMiddleware; the guard stays side-effect free here.
        middleware = ToolGuardMiddleware()
        state = AgentRuntimeState()
        args = {"command": "grep -rn needle ."}
        first_batch = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-a", "function": {"name": "run_bash"}},
                    {"id": "call-b", "function": {"name": "run_bash"}},
                ],
            }
        ]

        first = middleware.before_tool("run_bash", args, first_batch, runtime_state=state)
        second = middleware.before_tool("run_bash", args, first_batch, runtime_state=state)
        third = middleware.before_tool("run_bash", args, [{"role": "assistant", "tool_calls": []}],
                                       runtime_state=state)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNotNone(third)
        self.assertFalse(state.fallback.stop_requested)


class TerminalShellEditPolicyTests(unittest.TestCase):
    def test_allows_explicit_shell_file_edit_inside_workspace(self):
        middleware = TerminalShellEditPolicyMiddleware()

        blocked = middleware.before_tool(
            "run_bash",
            {"command": "Set-Content -Path app.py -Value 'changed'"},
            messages=[],
            runtime_state=AgentRuntimeState(),
            agent_name="main_agent",
        )

        self.assertIsNone(blocked)

    def test_allows_build_command_that_generates_artifacts(self):
        middleware = TerminalShellEditPolicyMiddleware()

        blocked = middleware.before_tool(
            "run_bash",
            {"command": "python -m pip install -e . && python -m pytest"},
            messages=[],
            runtime_state=AgentRuntimeState(),
            agent_name="main_agent",
        )

        self.assertIsNone(blocked)

    def test_allows_stderr_to_stdout_redirection(self):
        middleware = TerminalShellEditPolicyMiddleware()

        blocked = middleware.before_tool(
            "run_bash",
            {"command": "python -m pytest 2>&1"},
            messages=[],
            runtime_state=AgentRuntimeState(),
            agent_name="main_agent",
        )

        self.assertIsNone(blocked)

    def test_allows_output_redirection_to_dev_null(self):
        middleware = TerminalShellEditPolicyMiddleware()

        commands = [
            "ls /app/polyglot 2>/dev/null || echo missing",
            "find /app -name '*.js' 2>/dev/null",
            "command -v rg >/dev/null && rg foo",
            "curl -sf http://localhost:8080/index.html &>/dev/null",
        ]

        for command in commands:
            with self.subTest(command=command):
                blocked = middleware.before_tool(
                    "run_bash",
                    {"command": command},
                    messages=[],
                    runtime_state=AgentRuntimeState(),
                    agent_name="main_agent",
                )
                self.assertIsNone(blocked)

    def test_allows_shell_file_writes_inside_workspace(self):
        middleware = TerminalShellEditPolicyMiddleware()

        commands = [
            "echo x > file.txt",
            "python test.py 2> error.log",
            "cat > file.txt <<'EOF'\nhello\nEOF",
            "printf x &> combined.log",
            "rg foo . | tee out.txt",
            "python -c \"open('out.txt','w').write('x')\"",
        ]

        for command in commands:
            with self.subTest(command=command):
                blocked = middleware.before_tool(
                    "run_bash",
                    {"command": command},
                    messages=[],
                    runtime_state=AgentRuntimeState(),
                    agent_name="main_agent",
                )
                self.assertIsNone(blocked)

    def test_allows_formatter_and_patch_style_workspace_edits(self):
        middleware = TerminalShellEditPolicyMiddleware()

        commands = [
            "sed -i 's/foo/bar/g' app.py",
            "python -m ruff check --fix .",
            "gofmt -w main.go",
            "git apply fix.patch",
        ]

        for command in commands:
            with self.subTest(command=command):
                blocked = middleware.before_tool(
                    "run_bash",
                    {"command": command},
                    messages=[],
                    runtime_state=AgentRuntimeState(),
                    agent_name="main_agent",
                )
                self.assertIsNone(blocked)

    def test_blocks_destructive_or_outside_workspace_shell_commands(self):
        middleware = TerminalShellEditPolicyMiddleware()

        commands = [
            "rm -rf /",
            "rm -r -f /",
            "rm -rf -- /",
            "rm --recursive --force /",
            "rm -rf ~",
            'rm -rf "$HOME"',
            "rm -rf C:\\",
            "Remove-Item -Force -Recurse C:\\",
            "Remove-Item -LiteralPath C:\\ -Recurse -Force",
            "git reset --hard HEAD",
            "git clean -fd",
            "git clean -xdf",
            "git restore --source HEAD -- .",
            "echo x > /etc/profile",
            "python -c \"open('/etc/passwd','w').write('x')\"",
        ]

        for command in commands:
            with self.subTest(command=command):
                blocked = middleware.before_tool(
                    "run_bash",
                    {"command": command},
                    messages=[],
                    runtime_state=AgentRuntimeState(),
                    agent_name="main_agent",
                )
                self.assertIsNotNone(blocked)

    def test_blocks_system_config_writes_without_system_admin_context(self):
        middleware = TerminalShellEditPolicyMiddleware()

        blocked = middleware.before_tool(
            "run_bash",
            {"command": "cat > /etc/nginx/conf.d/git-site.conf <<'EOF'\nserver {}\nEOF"},
            messages=[],
            runtime_state=AgentRuntimeState(),
            agent_name="main_agent",
        )

        self.assertIsNotNone(blocked)

    def test_allows_container_absolute_path_mutations_in_eval_danger_full_access(self):
        middleware = TerminalShellEditPolicyMiddleware()
        state = AgentRuntimeState(permission_mode="danger-full-access")

        commands = [
            "cat > /etc/nginx/conf.d/git-site.conf <<'EOF'\nserver {}\nEOF",
            "python -c \"open('/usr/local/bin/x','w').write('x')\"",
            "rm -rf /var/www/app",
        ]

        with patch.dict("os.environ", {"HCA_TERMINAL_EVAL_MODE": "1"}):
            for command in commands:
                with self.subTest(command=command):
                    blocked = middleware.before_tool(
                        "run_bash",
                        {"command": command},
                        messages=[],
                        runtime_state=state,
                        agent_name="main_agent",
                    )
                    self.assertIsNone(blocked)

    def test_danger_full_access_without_eval_marker_still_blocks_system_path_writes(self):
        middleware = TerminalShellEditPolicyMiddleware()
        state = AgentRuntimeState(permission_mode="danger-full-access")

        blocked = middleware.before_tool(
            "run_bash",
            {"command": "cat > /etc/nginx/conf.d/git-site.conf <<'EOF'\nserver {}\nEOF"},
            messages=[],
            runtime_state=state,
            agent_name="main_agent",
        )

        self.assertIsNotNone(blocked)

    def test_danger_full_access_still_blocks_destructive_system_commands(self):
        middleware = TerminalShellEditPolicyMiddleware()
        state = AgentRuntimeState(permission_mode="danger-full-access")

        commands = [
            "rm -rf /etc",
            "git reset --hard HEAD",
        ]

        with patch.dict("os.environ", {"HCA_TERMINAL_EVAL_MODE": "1"}):
            for command in commands:
                with self.subTest(command=command):
                    blocked = middleware.before_tool(
                        "run_bash",
                        {"command": command},
                        messages=[],
                        runtime_state=state,
                        agent_name="main_agent",
                    )
                    self.assertIsNotNone(blocked)

    def test_container_system_config_write_requires_danger_full_access(self):
        middleware = TerminalShellEditPolicyMiddleware()
        state = AgentRuntimeState(permission_mode="workspace-write")

        blocked = middleware.before_tool(
            "run_bash",
            {"command": "cat > /etc/nginx/conf.d/git-site.conf <<'EOF'\nserver {}\nEOF"},
            messages=[],
            runtime_state=state,
            agent_name="main_agent",
        )

        self.assertIsNotNone(blocked)


if __name__ == "__main__":
    unittest.main()
