import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness_code_agent.runtime.middleware import (
    AgentMiddleware,
    StaticVerifierMiddleware,
    ToolFailurePolicyMiddleware,
)
from harness_code_agent.runtime.middleware.stack import (
    build_main_agent_middlewares,
    build_subagent_middlewares,
)
from harness_code_agent.runtime.permission_middleware import PermissionMiddleware
from harness_code_agent.runtime.verification import VerificationEngine
from harness_code_agent.workspace.service import WorkspaceService


class _MarkerMiddleware(AgentMiddleware):
    """A user-provided middleware that runs before the structural guards."""


class MiddlewareStackFactoryTests(unittest.TestCase):
    def _context(self, root: Path) -> tuple[SimpleNamespace, object, WorkspaceService]:
        workspace = WorkspaceService(root=root, snapshots_dir=root / ".harness" / "snapshots")
        registry = object()
        context = SimpleNamespace(workspace=workspace)
        return context, registry, workspace

    def test_main_agent_stack_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context, registry, workspace = self._context(root)

            stack = build_main_agent_middlewares(
                user_middlewares=[],
                tool_context=context,
                tool_registry=registry,
                workspace=root,
            )

            self.assertEqual(
                [type(mw) for mw in stack],
                [
                    ToolFailurePolicyMiddleware,
                    PermissionMiddleware,
                    StaticVerifierMiddleware,
                ],
            )
            self.assertIs(stack[0].tool_registry, registry)
            self.assertIs(stack[1]._ctx, context)
            self.assertIs(stack[1]._registry, registry)
            self.assertEqual(stack[2]._root, workspace.root)
            self.assertIsInstance(stack[2].engine, VerificationEngine)

    def test_user_middlewares_run_before_guards(self):
        # User middlewares from ~/.veriforge/middlewares.json run before the
        # structural guards. Memory lives outside the middleware stack (its
        # index is injected into the system prompt).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context, registry, _ = self._context(root)
            user_middleware = _MarkerMiddleware()

            stack = build_main_agent_middlewares(
                user_middlewares=[user_middleware],
                tool_context=context,
                tool_registry=registry,
                workspace=root,
            )

            self.assertEqual(
                [type(mw) for mw in stack],
                [
                    _MarkerMiddleware,
                    ToolFailurePolicyMiddleware,
                    PermissionMiddleware,
                    StaticVerifierMiddleware,
                ],
            )
            # The same instance, loaded once per session.
            self.assertIs(stack[0], user_middleware)

    def test_subagent_stack_is_permission_only(self):
        context = SimpleNamespace(workspace=object())
        registry = object()

        stack = build_subagent_middlewares(tool_context=context, tool_registry=registry)

        self.assertEqual([type(mw) for mw in stack], [PermissionMiddleware])
        self.assertIs(stack[0]._ctx, context)
        self.assertIs(stack[0]._registry, registry)


def test_shell_risky_denial_includes_the_decision_reason():
    middleware = PermissionMiddleware(
        tool_context=SimpleNamespace(
            permission_policy=SimpleNamespace(mode="workspace-write")
        ),
        tool_registry=None,
    )
    decision = SimpleNamespace(
        risk="shell_risky",
        reason="shell command mutates paths it cannot prove are inside the workspace",
        analysis=SimpleNamespace(effects=frozenset(), targets=()),
    )

    message = middleware._deny_message("bash", decision)

    assert "当前权限模式不允许此命令" in message
    assert "原因：shell command mutates paths it cannot prove" in message


if __name__ == "__main__":
    unittest.main()
