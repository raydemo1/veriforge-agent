"""Shell permission decisions per mode.

Facts come from the analyzer; allow/ask/deny comes only from the policy.
"""
from __future__ import annotations

import unittest

from harness_code_agent.runtime.permissions import PermissionPolicy
from harness_code_agent.runtime.shell_classification import (
    ShellEffect,
    ShellTrait,
    TargetScope,
    analyze_shell_command,
)


class ShellPermissionPolicyTests(unittest.TestCase):
    """Facts come from the analyzer; allow/ask/deny comes only from the policy."""

    def setUp(self):
        self.workspace = PermissionPolicy("workspace-write")
        self.eval = PermissionPolicy("terminal-eval")
        self.full = PermissionPolicy("danger-full-access")

    def _decide(self, policy, command):
        return policy.decide_tool_call("bash", {"command": command})

    # --- reads: allowed in every mode -------------------------------------

    def test_read_only_commands_are_allowed_everywhere(self):
        commands = [
            "python -m pytest 2>&1",
            "git status --short",
            "ls /app/polyglot 2>/dev/null || echo missing",
            "find /app -name '*.js' 2>/dev/null",
            "command -v rg >/dev/null && rg foo",
        ]
        for policy in (self.workspace, self.eval, self.full):
            for command in commands:
                with self.subTest(mode=policy.mode, command=command):
                    self.assertTrue(self._decide(policy, command).allowed)

    def test_network_access_is_a_separate_grant(self):
        command = "curl -sf http://localhost:8080/index.html"
        self.assertTrue(self._decide(self.workspace, command).requires_approval)
        self.assertTrue(self._decide(self.eval, command).allowed)
        self.assertTrue(self._decide(self.full, command).allowed)

    # --- workspace writes: proven workspace-scoped writes run without a
    # prompt in workspace-write mode; only unrecognized executors still ask ---

    def test_workspace_writes_run_in_workspace_mode_and_eval(self):
        commands = [
            "Set-Content -Path app.py -Value 'changed'",
            "echo x > file.txt",
            "rg foo . | tee out.txt",
            "sed -i 's/foo/bar/g' app.py",
            "gofmt -w main.go",
            "git apply fix.patch",
            "git commit --allow-empty -m test",
            "rm -rf build",
        ]
        for policy in (self.workspace, self.eval):
            for command in commands:
                with self.subTest(mode=policy.mode, command=command):
                    self.assertTrue(self._decide(policy, command).allowed, command)

    def test_private_temp_writes_run_without_approval(self):
        commands = [
            'Set-Content -Path "$env:TEMP\\check.py" -Value "x"',
            "Set-Content -Path $env:TMP\\check.py -Value 'x'",
            'Remove-Item "$env:TEMP\\old.txt"',
        ]
        for command in commands:
            with self.subTest(command=command):
                analysis = analyze_shell_command(command)
                self.assertIn(TargetScope.TEMP, analysis.scopes)
                for policy in (self.workspace, self.eval):
                    decision = self._decide(policy, command)
                    self.assertTrue(decision.allowed, decision.reason)
        read_only = PermissionPolicy("read-only")
        self.assertFalse(self._decide(read_only, commands[0]).allowed)

    def test_temp_variable_reassignment_is_not_proof_of_private_temp(self):
        commands = [
            '$env:TEMP = "C:\\Windows"; Set-Content -Path "$env:TEMP\\x.txt" -Value x',
            'Set-Item Env:TMP C:\\Windows; Set-Content -Path "$env:TMP\\x.txt" -Value x',
        ]
        for command in commands:
            with self.subTest(command=command):
                analysis = analyze_shell_command(command)
                self.assertNotIn(TargetScope.TEMP, analysis.scopes)
                self.assertFalse(self._decide(self.workspace, command).allowed)
                self.assertFalse(self._decide(self.eval, command).allowed)

    def test_temp_traversal_is_not_proof_of_private_temp(self):
        command = 'Set-Content -Path "$env:TEMP\\..\\..\\Windows\\x.txt" -Value x'
        analysis = analyze_shell_command(command)
        self.assertNotIn(TargetScope.TEMP, analysis.scopes)
        self.assertFalse(self._decide(self.workspace, command).allowed)

    def test_plain_temp_like_variables_are_not_trusted(self):
        commands = [
            "$tmpdir = 'C:\\somewhere-outside'; Set-Content \"$tmpdir\\escape.txt\" 'data'",
            "$temp = 'C:\\somewhere-outside'; Set-Content \"$temp\\escape.txt\" 'data'",
        ]
        for command in commands:
            with self.subTest(command=command):
                analysis = analyze_shell_command(command)
                self.assertNotIn(TargetScope.TEMP, analysis.scopes)
                self.assertFalse(self._decide(self.workspace, command).allowed)

    def test_dynamic_temp_suffix_is_not_proof_of_private_temp(self):
        commands = [
            '$suffix = "escape.txt"; Set-Content "$env:TEMP\\$suffix" -Value x',
            'Set-Content "%TEMP%\\%suffix%" -Value x',
        ]
        for command in commands:
            with self.subTest(command=command):
                analysis = analyze_shell_command(command)
                self.assertNotIn(TargetScope.TEMP, analysis.scopes)
                self.assertFalse(self._decide(self.workspace, command).allowed)

    def test_temp_derived_variables_are_proven_private(self):
        commands = [
            '$tmp = Join-Path $env:TEMP "retry_check.py"; Set-Content -Path $tmp -Value x',
            '$tmp = "$env:TEMP\\retry_check.py"; Set-Content -Path $tmp -Value x',
            (
                '$tmp = Join-Path $env:TEMP "a.py"; '
                'Remove-Item $tmp, (Join-Path $env:TEMP "b.py") -ErrorAction SilentlyContinue'
            ),
        ]
        for command in commands:
            with self.subTest(command=command):
                analysis = analyze_shell_command(command)
                self.assertIn(TargetScope.TEMP, analysis.scopes)
                self.assertTrue(self._decide(self.workspace, command).allowed)

    def test_temp_variable_reassignment_or_dynamic_suffix_is_not_proven(self):
        commands = [
            '$tmp = Join-Path $env:TEMP $name; Set-Content -Path $tmp -Value x',
            (
                '$tmp = Join-Path $env:TEMP "sub\\..\\..\\x.txt"; '
                'Set-Content -Path $tmp -Value x'
            ),
            (
                '$tmp = Join-Path $env:TEMP "ok.txt"; $tmp = "C:\\Windows\\x.txt"; '
                'Set-Content -Path $tmp -Value x'
            ),
            (
                '$tmp = Join-Path $env:TEMP "a.py"; '
                'Remove-Item $tmp, "C:\\Windows\\b.py"'
            ),
        ]
        for command in commands:
            with self.subTest(command=command):
                analysis = analyze_shell_command(command)
                self.assertNotIn(TargetScope.TEMP, analysis.scopes)
                self.assertFalse(self._decide(self.workspace, command).allowed)

    def test_unknown_program_is_bounded_by_the_sandbox(self):
        # Semantic uncertainty is not dangerousness: an unknown program runs in
        # the default sandbox, and the network stays off unless granted.
        for command in (
            "python test.py 2> error.log",
            "node build.js > out.log",
        ):
            with self.subTest(command=command):
                decision = self._decide(self.workspace, command)
                self.assertTrue(decision.allowed, command)
                self.assertFalse(decision.network_access, command)

    def test_unrecognized_executor_runs_inside_the_workspace_sandbox(self):
        decision = self._decide(self.workspace, "python -m ruff check --fix .")
        self.assertTrue(decision.allowed)
        self.assertTrue(self._decide(self.eval, "python -m ruff check --fix .").allowed)


    # --- unknown execution: ask / allow, never proven safe ---------------

    def test_interpreter_payloads_are_unknown_execution(self):
        analysis = analyze_shell_command("python script.py")
        self.assertIn(ShellEffect.EXECUTE, analysis.effects)
        self.assertIn(ShellTrait.UNKNOWN_EFFECT, analysis.traits)
        # Unknown code runs, but only inside the workspace sandbox.
        self.assertTrue(self._decide(self.workspace, "python script.py").allowed)
        self.assertTrue(self._decide(self.eval, "python script.py").allowed)
        # We deliberately do not parse Python to find embedded writes; the
        # filesystem sandbox is the boundary for those.
        decision = self._decide(
            self.workspace, "python -c \"open('/etc/passwd','w').write('x')\""
        )
        self.assertTrue(decision.allowed)

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

    def test_destructive_git_is_an_approval_question(self):
        commands = [
            "git reset --hard HEAD",
            "git clean -fd",
            "git clean -xdf",
            "git restore --source HEAD -- .",
            "git push --force origin main",
        ]
        for command in commands:
            with self.subTest(command=command, mode="workspace-write"):
                self.assertTrue(self._decide(self.workspace, command).requires_approval)
            with self.subTest(command=command, mode="terminal-eval"):
                eval_decision = self._decide(self.eval, command)
                if command.startswith("git push"):
                    # Remote mutation keeps the network-upload grant in eval.
                    self.assertTrue(eval_decision.requires_approval)
                else:
                    self.assertTrue(eval_decision.allowed)
            with self.subTest(command=command, mode="danger-full-access"):
                self.assertTrue(self._decide(self.full, command).allowed)

    # --- system, non-destructive writes -----------------------------------

    def test_system_config_writes_follow_mode_presets(self):
        command = "cat > /etc/nginx/conf.d/git-site.conf <<'EOF'\nserver {}\nEOF"
        self.assertFalse(self._decide(self.workspace, command).allowed)
        self.assertFalse(self._decide(self.eval, command).allowed)
        # Full access permits ordinary system writes but not destructive ones.
        self.assertTrue(self._decide(self.full, command).allowed)
        self.assertFalse(self._decide(self.full, "rm -rf /etc").allowed)


if __name__ == "__main__":
    unittest.main()
