"""Shell decisions per permission mode.

The golden table pins how each mode classifies a shell command: reads pass in
every mode, workspace writes follow the mode, external writes and executables
of unknown effect only pass under full access. Plan mode runs on the
read-only column, so a regression here would silently widen planning.
"""
from __future__ import annotations

from harness_code_agent.runtime.permissions import (
    TOOL_PERMISSION_CONTROL,
    TOOL_PERMISSION_DANGEROUS,
    TOOL_PERMISSION_EDIT,
    TOOL_PERMISSION_READ,
    TOOL_PERMISSION_SHELL,
    PermissionPolicy,
)

MODES = ("read-only", "workspace-write", "danger-full-access")

# (case id, command) -> effective action per mode in MODES order
SHELL_GOLDEN_TABLE: dict[str, tuple[str, str, str, str]] = {
    # Plain reads and network reads.
    "read_ls": ("ls", "allow", "allow", "allow"),
    "read_cat": ("cat README.md", "allow", "allow", "allow"),
    "read_grep_recursive": ("grep -r foo .", "allow", "allow", "allow"),
    "read_network": ("curl http://example.com", "allow", "allow", "allow"),
    # Workspace writes: read-only denies, the write modes trust them.
    "ws_redirect": ("echo hi > out.txt", "deny", "allow", "allow"),
    "ws_rm": ("rm out.txt", "deny", "allow", "allow"),
    "ws_mkdir": ("mkdir newdir", "deny", "allow", "allow"),
    "ws_sed_inplace": ("sed -i s/a/b/ f.txt", "deny", "allow", "allow"),
    # Local git mutation writes the workspace; remote mutation is external.
    "git_commit": ("git commit -m x", "deny", "allow", "allow"),
    "git_push": ("git push", "deny", "deny", "allow"),
    # Writes outside the workspace are never trusted before full access.
    "external_tmp_write": (
        "echo hi > /tmp/hca_characterization_out.txt",
        "deny",
        "deny",
        "allow",
    ),
    # Unknown effects ask in workspace-write and pass only under full access.
    "unknown_program": ("./mystery.sh", "deny", "ask", "allow"),
    # An absolute path that resolves inside the workspace is a workspace
    # write once the policy knows the root.
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


def _decide(mode: str, command: str, workspace_root: str) -> str:
    return PermissionPolicy(mode).decide_tool_call(
        "run_bash",
        {"command": command},
        TOOL_PERMISSION_SHELL,
        workspace_root,
    ).action


def test_shell_decisions_match_golden_table(tmp_path):
    for case_id, command, expected in _iter_shell_cases(str(tmp_path)):
        actual = tuple(_decide(mode, command, str(tmp_path)) for mode in MODES)
        assert actual == expected, f"{case_id}: {command} -> {actual}"


MUTATING_CASES = frozenset({
    "ws_redirect",
    "ws_rm",
    "ws_mkdir",
    "ws_sed_inplace",
    "git_commit",
    "git_push",
    "external_tmp_write",
    "unknown_program",
    "absolute_inside_workspace",
})


def test_read_only_denies_every_mutating_shell_command(tmp_path):
    for case_id, command, _ in _iter_shell_cases(str(tmp_path)):
        if case_id in MUTATING_CASES:
            assert _decide("read-only", command, str(tmp_path)) == "deny", case_id


def test_read_only_denies_structured_writes_and_dangerous_tools(tmp_path):
    policy = PermissionPolicy("read-only")
    root = str(tmp_path)

    assert policy.decide_tool_call(
        "read_file", {"path": "a.py"}, TOOL_PERMISSION_READ, root
    ).action == "allow"
    assert policy.decide_tool_call(
        "write_file", {"path": "a.py", "content": "x"}, TOOL_PERMISSION_EDIT, root
    ).action == "deny"
    assert policy.decide_tool_call(
        "apply_patch", {"path": "a.py"}, TOOL_PERMISSION_EDIT, root
    ).action == "deny"
    assert policy.decide_tool_call(
        "mcp__docs__delete", {}, TOOL_PERMISSION_DANGEROUS, root
    ).action == "deny"
    assert policy.decide_tool_call(
        "update_todo", {}, TOOL_PERMISSION_CONTROL, root
    ).action == "allow"
