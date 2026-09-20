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
    """A profile-provided middleware that must stay first in the stack."""


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
            marker = _MarkerMiddleware()
            agent_config = SimpleNamespace(middlewares=[marker])

            stack = build_main_agent_middlewares(
                agent_config=agent_config,
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
            # Profile-provided middleware is the same instance, not copied.
            self.assertIs(stack[0], marker)
            self.assertIs(stack[2].tool_registry, registry)
            self.assertIs(stack[3]._ctx, context)
            self.assertIs(stack[3]._registry, registry)
            self.assertEqual(stack[4]._workspace_root, str(root))
            self.assertIs(stack[4]._workspace, workspace)

    def test_main_agent_stack_has_no_memory_middleware(self):
        # Memory lives outside the middleware stack (index injected into
        # the system prompt), so memory_enabled must not alter the stack.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context, registry, _ = self._context(root)

            expected = [
                ToolGuardMiddleware,
                ToolFailurePolicyMiddleware,
                PermissionMiddleware,
                StaticVerifierMiddleware,
            ]
            for config in (
                SimpleNamespace(middlewares=[], memory_enabled=True),
                SimpleNamespace(middlewares=[], memory_enabled=False),
                SimpleNamespace(middlewares=[]),
            ):
                with self.subTest(config=config):
                    stack = build_main_agent_middlewares(
                        agent_config=config,
                        tool_context=context,
                        tool_registry=registry,
                        workspace=root,
                    )
                    self.assertEqual([type(mw) for mw in stack], expected)

    def test_subagent_stack_is_permission_only(self):
        context = SimpleNamespace(workspace=object())
        registry = object()

        stack = build_subagent_middlewares(tool_context=context, tool_registry=registry)

        self.assertEqual([type(mw) for mw in stack], [PermissionMiddleware])
        self.assertIs(stack[0]._ctx, context)
        self.assertIs(stack[0]._registry, registry)


if __name__ == "__main__":
    unittest.main()
