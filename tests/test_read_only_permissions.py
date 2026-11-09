"""Read-only shell boundary for permission presets.

The golden table was captured empirically from the composition of a session
``PermissionPolicy`` with a read-only preset, and is now a plain permission
regression: the preset may only tighten the session mode, never loosen it.

Structured tools (write_file / apply_patch / update_todo / ask_user) are not
part of this boundary. A read-only workflow removes them from the tool surface
instead, so the executor rejects them before any permission decision.
"""
from __future__ import annotations

from harness_code_agent.runtime.permissions import (
    TOOL_PERMISSION_SHELL,
    PermissionPolicy,
    PermissionPreset,
)

MODES = ("read-only", "workspace-write", "danger-full-access")

# (case id, command) -> effective action per session mode
# (read-only, workspace-write, danger-full-access)
SHELL_GOLDEN_TABLE: dict[str, tuple[str, str, str, str]] = {
    # Plain reads: session allows in every mode; middleware never blocks.
    "read_ls": ("ls", "allow", "allow", "allow"),
    "read_cat": ("cat README.md", "allow", "allow", "allow"),
    "read_grep_recursive": ("grep -r foo .", "allow", "allow", "allow"),
    "read_network": ("curl http://example.com", "allow", "allow", "allow"),
    # Workspace writes: middleware denied regardless of session mode.
    "ws_redirect": ("echo hi > out.txt", "deny", "deny", "deny"),
    "ws_rm": ("rm out.txt", "deny", "deny", "deny"),
    "ws_mkdir": ("mkdir newdir", "deny", "deny", "deny"),
    "ws_sed_inplace": ("sed -i s/a/b/ f.txt", "deny", "deny", "deny"),
    # Local git mutation counts as a workspace write for the review boundary.
    "git_commit": ("git commit -m x", "deny", "deny", "deny"),
    # Remote git mutation is NOT a workspace write; the session decides.
    "git_push": ("git push", "deny", "deny", "allow"),
    # External writes are outside the review predicate; the session decides.
    "external_tmp_write": (
        "echo hi > /tmp/hca_characterization_out.txt",
        "deny",
        "deny",
        "allow",
    ),
    # Unknown effects: middleware never blocked these; the session decides.
    "unknown_program": ("./mystery.sh", "deny", "ask", "allow"),
    # Pre-existing quirk locked deliberately: the preset predicate runs
    # without a workspace root, so an absolute path resolving inside the
    # workspace is classified external and the preset does not deny it; the
    # session policy (which does have the root) decides instead. Read-only
    # denies the write, workspace-write trusts proven workspace-scoped writes,
    # and full access allows.
    "absolute_inside_workspace": None,  # filled in at runtime (needs tmp root)
}

EXPECTED_ABSOLUTE_INSIDE = ("deny", "allow", "allow")


def _iter_shell_cases(workspace_root: str):
    for case_id, entry in SHELL_GOLDEN_TABLE.items():
        if entry is None:
            continue
        command, *expected = entry
        yield case_id, command, tuple(expected)
    abs_inside = f"{workspace_root.replace(chr(92), '/')}/nested/f.txt"
    yield (
        "absolute_inside_workspace",
        f"echo hi > {abs_inside}",
        EXPECTED_ABSOLUTE_INSIDE,
    )


def _new_effective_action(mode: str, command: str, workspace_root: str) -> str:
    """The migrated composition: session policy restricted by the preset."""
    policy = PermissionPolicy(mode).restricted_by(
        PermissionPreset.NO_WORKSPACE_WRITES
    )
    return policy.decide_tool_call(
        "run_bash",
        {"command": command},
        TOOL_PERMISSION_SHELL,
        workspace_root,
    ).action


def test_new_preset_policy_matches_golden_table(tmp_path):
    for case_id, command, expected in _iter_shell_cases(str(tmp_path)):
        actual = tuple(
            _new_effective_action(mode, command, str(tmp_path)) for mode in MODES
        )
        assert actual == expected, f"{case_id}: {command} -> {actual}"


def test_restricted_by_none_returns_same_policy():
    policy = PermissionPolicy("workspace-write")
    assert policy.restricted_by(None) is policy


def test_preset_only_tightens_never_loosens(tmp_path):
    """For every corpus case the preset result is >= session strictness."""
    rank = {"allow": 0, "ask": 1, "deny": 2}
    for _, command, _ in _iter_shell_cases(str(tmp_path)):
        for mode in MODES:
            session = PermissionPolicy(mode)
            session_action = session.decide_tool_call(
                "run_bash", {"command": command}, TOOL_PERMISSION_SHELL, str(tmp_path)
            ).action
            preset_action = _new_effective_action(mode, command, str(tmp_path))
            assert rank[preset_action] >= rank[session_action]


def test_preset_mode_stays_session_mode():
    policy = PermissionPolicy("danger-full-access").restricted_by(
        PermissionPreset.NO_WORKSPACE_WRITES
    )
    assert policy.mode == "danger-full-access"


