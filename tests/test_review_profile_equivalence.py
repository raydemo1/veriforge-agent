"""Characterization for the review profile read-only boundary.

The golden table below was captured empirically from the old composition
(per-mode ``PermissionPolicy`` + the deleted ``ReviewOnlyMiddleware``) during
the migration that moved that boundary into
``PermissionPreset.NO_WORKSPACE_WRITES``.  The preset policy wrapper must keep
producing exactly these effective actions.

The four structured tools (write_file / apply_patch / update_todo / ask_user)
are not part of the policy boundary: the profile hides them via
``blocked_tool_names``, so the executor capability gate rejects them before
any permission decision.  The preset intentionally does not duplicate that
list.
"""
from __future__ import annotations

from harness_code_agent.profiles import get_profile
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
    # Pre-existing quirk locked deliberately: the old predicate ran without a
    # workspace root, so an absolute path resolving inside the workspace was
    # classified external and NOT blocked under full access.  The migration
    # preserves this exactly rather than quietly tightening it.
    "absolute_inside_workspace": None,  # filled in at runtime (needs tmp root)
}

EXPECTED_ABSOLUTE_INSIDE = ("deny", "ask", "allow")


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
    policy = PermissionPolicy(mode, sandbox_mode="host").restricted_by(
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
            session = PermissionPolicy(mode, sandbox_mode="host")
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


def test_review_profile_owns_structured_tools_via_blocked_names():
    config = get_profile("review").main_agent()
    assert config.permission_preset is PermissionPreset.NO_WORKSPACE_WRITES
    assert config.blocked_tool_names == {
        "write_file",
        "apply_patch",
        "update_todo",
        "ask_user",
    }


def _session_with_profile(profile_name: str, mode: str):
    # InteractiveSession pulls in skills/approval/session-store machinery in
    # __init__; the preset composition only needs profile + mode, so bypass it.
    from harness_code_agent.core.interactive import InteractiveSession

    session = object.__new__(InteractiveSession)
    session.profile = get_profile(profile_name)
    session.permission_mode = mode
    return session


def test_effective_policy_review_applies_preset_even_under_full_access(tmp_path):
    session = _session_with_profile("review", "danger-full-access")
    policy = session._effective_permission_policy()
    decision = policy.decide_tool_call(
        "run_bash",
        {"command": "echo hi > out.txt"},
        TOOL_PERMISSION_SHELL,
        str(tmp_path),
    )
    assert decision.action == "deny"
    assert policy.mode == "danger-full-access"


def test_effective_policy_leaving_review_restores_session_authority(tmp_path):
    session = _session_with_profile("general", "danger-full-access")
    policy = session._effective_permission_policy()
    decision = policy.decide_tool_call(
        "run_bash",
        {"command": "echo hi > out.txt"},
        TOOL_PERMISSION_SHELL,
        str(tmp_path),
    )
    assert decision.action == "allow"


def test_effective_policy_tracks_permission_mode_changes(tmp_path):
    session = _session_with_profile("review", "danger-full-access")
    assert (
        session._effective_permission_policy()
        .decide_tool_call(
            "run_bash", {"command": "ls"}, TOOL_PERMISSION_SHELL, str(tmp_path)
        )
        .action
        == "allow"
    )
    session.permission_mode = "read-only"
    policy = session._effective_permission_policy()
    assert policy.mode == "read-only"
    # External write stays governed by the read-only session mode (deny)...
    assert (
        policy.decide_tool_call(
            "run_bash",
            {"command": "echo hi > /tmp/hca_mode_switch.txt"},
            TOOL_PERMISSION_SHELL,
            str(tmp_path),
        )
        .action
        == "deny"
    )
