"""Tests for the main-agent proposal integration exit gate."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness_code_agent.agent.change_proposal import ChangeProposalStore
from harness_code_agent.runtime.middleware import ProposalIntegrationMiddleware
from harness_code_agent.workspace.service import WorkspaceService


class ProposalIntegrationGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="hca-gate-test-"))
        self.root = self.temp / "workspace"
        self.root.mkdir()
        self.workspace = WorkspaceService(root=self.root)
        self.store = ChangeProposalStore()
        self.tool_context = SimpleNamespace(
            agent_coordinator=SimpleNamespace(changes=self.store)
        )
        self.middleware = ProposalIntegrationMiddleware(self.tool_context)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp, ignore_errors=True)

    def _finalize(self, base="one\n", worker="ONE\n", path="app.py"):
        (self.root / path).write_text(base, encoding="utf-8")
        sandbox = self.store.create_sandbox("agent_worker", self.root)
        (sandbox.workspace / path).write_text(worker, encoding="utf-8")
        return self.store.finalize("agent_worker", [path])

    def test_no_coordinator_allows_exit(self):
        middleware = ProposalIntegrationMiddleware(SimpleNamespace(agent_coordinator=None))
        self.assertIsNone(middleware.pre_exit([]))

    def test_no_pending_proposals_allows_exit(self):
        self.assertIsNone(self.middleware.pre_exit([]))

    def test_ready_proposal_blocks_exit_with_integration_directive(self):
        proposal = self._finalize()
        injection = self.middleware.pre_exit([])
        self.assertIsNotNone(injection)
        self.assertIn("UNINTEGRATED WORKER CHANGES", injection)
        self.assertIn(proposal.id, injection)
        self.assertIn("agent_worker", injection)
        self.assertIn("app.py", injection)
        self.assertIn("read_agent_changes", injection)
        self.assertIn("apply_agent_changes", injection)

    def test_open_conflict_blocks_exit_with_conflict_directive(self):
        proposal = self._finalize(base="value = 1\n", worker="value = 2\n")
        # Diverge the main workspace to force a true three-way conflict.
        (self.root / "app.py").write_text("value = 3\n", encoding="utf-8")
        result = self.store.apply(proposal.id, self.workspace)
        self.assertEqual(result["status"], "conflict")

        injection = self.middleware.pre_exit([])
        self.assertIsNotNone(injection)
        self.assertIn(result["conflict_id"], injection)
        self.assertIn("read_agent_conflicts", injection)
        self.assertIn("resolve_agent_conflicts", injection)
        self.assertIn("question interaction", injection)

    def test_applied_proposal_allows_exit(self):
        proposal = self._finalize()
        self.assertEqual(self.store.apply(proposal.id, self.workspace)["status"], "applied")
        self.assertIsNone(self.middleware.pre_exit([]))


if __name__ == "__main__":
    unittest.main()
