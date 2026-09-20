import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness_code_agent.runtime.middleware import (
    AgentMiddleware,
    MemoryMiddleware,
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

    def test_main_agent_stack_order_with_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context, registry, workspace = self._context(root)
            marker = _MarkerMiddleware()
            agent_config = SimpleNamespace(middlewares=[marker], memory_enabled=True)

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
                    MemoryMiddleware,
                    PermissionMiddleware,
                    StaticVerifierMiddleware,
                ],
            )
            # Profile-provided middleware is the same instance, not copied.
            self.assertIs(stack[0], marker)
            self.assertIs(stack[2].tool_registry, registry)
            self.assertEqual(stack[3].workspace, root.resolve())
            self.assertIs(stack[4]._ctx, context)
            self.assertIs(stack[4]._registry, registry)
            self.assertEqual(stack[5]._workspace_root, str(root))
            self.assertIs(stack[5]._workspace, workspace)

    def test_main_agent_stack_omits_memory_when_disabled_and_defaults_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context, registry, _ = self._context(root)

            disabled = build_main_agent_middlewares(
                agent_config=SimpleNamespace(middlewares=[], memory_enabled=False),
                tool_context=context,
                tool_registry=registry,
                workspace=root,
            )
            self.assertEqual(
                [type(mw) for mw in disabled],
                [
                    ToolGuardMiddleware,
                    ToolFailurePolicyMiddleware,
                    PermissionMiddleware,
                    StaticVerifierMiddleware,
                ],
            )

            # AgentConfig without an explicit memory_enabled keeps memory on.
            default = build_main_agent_middlewares(
                agent_config=SimpleNamespace(middlewares=[]),
                tool_context=context,
                tool_registry=registry,
                workspace=root,
            )
            self.assertIn(MemoryMiddleware, [type(mw) for mw in default])

    def test_subagent_stack_is_permission_only(self):
        context = SimpleNamespace(workspace=object())
        registry = object()

        stack = build_subagent_middlewares(tool_context=context, tool_registry=registry)

        self.assertEqual([type(mw) for mw in stack], [PermissionMiddleware])
        self.assertIs(stack[0]._ctx, context)
        self.assertIs(stack[0]._registry, registry)


if __name__ == "__main__":
    unittest.main()
