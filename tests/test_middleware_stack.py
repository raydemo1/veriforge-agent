import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness_code_agent.runtime.middleware import (
    AgentMiddleware,
    StaticVerifierMiddleware,
    ToolFailurePolicyMiddleware,
    ToolGuardMiddleware,
)
from harness_code_agent.runtime.middleware.stack import (
    build_main_agent_middlewares,
    build_subagent_middlewares,
)
from harness_code_agent.runtime.permission_middleware import PermissionMiddleware
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
                    ToolGuardMiddleware,
                    ToolFailurePolicyMiddleware,
                    PermissionMiddleware,
                    StaticVerifierMiddleware,
                ],
            )
            self.assertIs(stack[1].tool_registry, registry)
            self.assertIs(stack[2]._ctx, context)
            self.assertIs(stack[2]._registry, registry)
            self.assertEqual(stack[3]._workspace_root, str(root))
            self.assertIs(stack[3]._workspace, workspace)

    def test_user_middlewares_run_before_guards(self):
        # User middlewares from ~/.harness/middlewares.json run before the
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
                    ToolGuardMiddleware,
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


if __name__ == "__main__":
    unittest.main()
