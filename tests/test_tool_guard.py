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
from harness_code_agent.runtime.middleware.tool_guard import ToolGuardMiddleware
from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.shell_classification import (
    ShellEffect,
    ShellTrait,
    TargetScope,
    analyze_shell_command,
)
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


class ShellPermissionPolicyTests(unittest.TestCase):
    """Facts come from the analyzer; allow/ask/deny comes only from the policy."""

    def setUp(self) -> None:
        self.workspace = PermissionPolicy("workspace-write", sandbox_mode="host")
        self.eval = PermissionPolicy("terminal-eval", sandbox_mode="docker")
        self.full = PermissionPolicy("danger-full-access", sandbox_mode="host")

    def _decide(self, policy, command):
        return policy.decide_tool_call("run_bash", {"command": command})

    # --- reads: allowed in every mode -------------------------------------

    def test_read_only_commands_are_allowed_everywhere(self):
        commands = [
            "python -m pytest 2>&1",
            "git status --short",
            "ls /app/polyglot 2>/dev/null || echo missing",
            "find /app -name '*.js' 2>/dev/null",
            "command -v rg >/dev/null && rg foo",
            "curl -sf http://localhost:8080/index.html &>/dev/null",
        ]
        for policy in (self.workspace, self.eval, self.full):
            for command in commands:
                with self.subTest(mode=policy.mode, command=command):
                    self.assertTrue(self._decide(policy, command).allowed)

    # --- workspace writes: ask in workspace-write, allow in eval ----------

    def test_workspace_writes_ask_in_workspace_mode_and_run_in_eval(self):
        commands = [
            "Set-Content -Path app.py -Value 'changed'",
            "echo x > file.txt",
            "python test.py 2> error.log",
            "cat > file.txt <<'EOF'\nhello\nEOF",
            "printf x &> combined.log",
            "rg foo . | tee out.txt",
            "sed -i 's/foo/bar/g' app.py",
            "python -m ruff check --fix .",
            "gofmt -w main.go",
            "git apply fix.patch",
            "git commit --allow-empty -m test",
            "rm -rf build",
        ]
        for command in commands:
            with self.subTest(command=command):
                decision = self._decide(self.workspace, command)
                self.assertTrue(decision.requires_approval, command)
                self.assertTrue(self._decide(self.eval, command).allowed, command)

    def test_container_absolute_paths_are_workspace_targets_in_docker(self):
        analysis = analyze_shell_command("rm -rf /tests/build", sandbox_mode="docker")
        target = analysis.targets[0]
        self.assertIs(target.scope, TargetScope.WORKSPACE)
        self.assertIn(ShellEffect.DELETE, analysis.effects)
        self.assertTrue(
            self._decide(self.eval, "rm -rf /tests/build").allowed
        )
        # The same absolute path on a host is outside the workspace.
        host_analysis = analyze_shell_command("rm -rf /tests/build", sandbox_mode="host")
        self.assertIs(host_analysis.targets[0].scope, TargetScope.EXTERNAL)
        self.assertFalse(
            self._decide(self.workspace, "rm -rf /tests/build").allowed
        )

    # --- unknown execution: ask / allow, never proven safe ---------------

    def test_interpreter_payloads_are_unknown_execution(self):
        analysis = analyze_shell_command("python script.py")
        self.assertIn(ShellEffect.EXECUTE, analysis.effects)
        self.assertIn(ShellTrait.UNKNOWN_EFFECT, analysis.traits)
        self.assertTrue(
            self._decide(self.workspace, "python script.py").requires_approval
        )
        self.assertTrue(self._decide(self.eval, "python script.py").allowed)
        # We deliberately do not parse Python to find embedded writes.
        decision = self._decide(
            self.workspace, "python -c \"open('/etc/passwd','w').write('x')\""
        )
        self.assertTrue(decision.requires_approval)

    def test_sudo_wrapper_is_normalized_before_dispatch(self):
        analysis = analyze_shell_command("sudo rm -rf /etc")
        self.assertIn(ShellTrait.PRIVILEGED, analysis.traits)
        self.assertIn(ShellEffect.DELETE, analysis.effects)
        self.assertIs(analysis.targets[0].scope, TargetScope.SYSTEM)

    # --- catastrophic guardrail: denied in EVERY mode ---------------------

    def test_catastrophic_commands_are_denied_in_every_mode(self):
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
            "sudo rm -rf /etc",
            "git reset --hard HEAD",
            "git clean -fd",
            "git clean -xdf",
            "git restore --source HEAD -- .",
            "git push --force origin main",
            "mkfs.ext4 /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
        ]
        for policy in (self.workspace, self.eval, self.full):
            for command in commands:
                with self.subTest(mode=policy.mode, command=command):
                    decision = self._decide(policy, command)
                    self.assertFalse(decision.allowed)
                    self.assertFalse(decision.requires_approval)
                    self.assertEqual(decision.risk, "shell_blocked")

    # --- system, non-destructive writes -----------------------------------

    def test_system_config_writes_follow_mode_presets(self):
        command = "cat > /etc/nginx/conf.d/git-site.conf <<'EOF'\nserver {}\nEOF"
        self.assertFalse(self._decide(self.workspace, command).allowed)
        self.assertFalse(self._decide(self.eval, command).allowed)
        # Full access permits ordinary system writes but not destructive ones.
        self.assertTrue(self._decide(self.full, command).allowed)
        self.assertFalse(self._decide(self.full, "rm -rf /etc").allowed)

    def test_legacy_eval_env_marker_selects_eval_preset(self):
        with patch.dict("os.environ", {"HCA_TERMINAL_EVAL_MODE": "1"}):
            policy = PermissionPolicy("danger-full-access")
            self.assertEqual(policy.sandbox_mode, "docker")
            self.assertTrue(
                self._decide(policy, "rm -rf /tests/build").allowed
            )
            self.assertFalse(
                self._decide(policy, "cat > /etc/profile").allowed
            )
            self.assertFalse(
                self._decide(policy, "git reset --hard").allowed
            )


if __name__ == "__main__":
    unittest.main()
