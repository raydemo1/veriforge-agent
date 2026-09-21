from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from harness_code_agent.attachments import (
    ExternalPathConfirmationRequired,
    PreparedTurn,
)
from harness_code_agent.tui_bridge import BridgeServer


class AttachmentBridgeTests(unittest.TestCase):
    def _server(self, session):
        server = object.__new__(BridgeServer)
        server._session = session
        server._tasks = queue.Queue()
        server._stopping = threading.Event()
        server._active_lock = threading.Lock()
        server._active_token = None
        server._send_event = lambda event: None
        server.responses = []
        server._response = lambda request_id, result=None, error=None: server.responses.append(
            {"id": request_id, "result": result, "error": error}
        )
        return server

    def test_submit_queues_the_complete_prepared_turn(self):
        prepared = PreparedTurn("compare", ())
        received = []
        session = SimpleNamespace(
            prepare_submission=lambda submission: received.append(submission) or prepared
        )
        server = self._server(session)

        BridgeServer._handle_request(server, {
            "id": "submit-1",
            "method": "submit",
            "params": {"text": "compare", "attachmentIds": ["a1"], "authorizedPaths": [r"C:\outside.pdf"]},
        })

        self.assertIs(server._tasks.get_nowait(), prepared)
        self.assertEqual(received[0].attachment_ids, ("a1",))
        self.assertEqual(received[0].authorized_paths, (r"C:\outside.pdf",))
        self.assertTrue(server.responses[0]["result"]["accepted"])

    def test_external_path_confirmation_does_not_queue_or_fail_the_draft(self):
        def prepare(_submission):
            raise ExternalPathConfirmationRequired([r"C:\outside.pdf"])

        server = self._server(SimpleNamespace(prepare_submission=prepare))
        BridgeServer._handle_request(server, {
            "id": "submit-2",
            "method": "submit",
            "params": {"text": r"read C:\outside.pdf"},
        })

        self.assertTrue(server._tasks.empty())
        self.assertIsNone(server.responses[0]["error"])
        self.assertEqual(
            server.responses[0]["result"]["confirmation"],
            {"kind": "external_paths", "paths": [r"C:\outside.pdf"]},
        )

    def test_attachment_only_submission_is_allowed(self):
        prepared = PreparedTurn("", ())
        server = self._server(SimpleNamespace(prepare_submission=lambda _submission: prepared))
        BridgeServer._handle_request(server, {
            "id": "submit-3",
            "method": "submit",
            "params": {"text": "", "attachmentIds": ["a1"]},
        })
        self.assertTrue(server.responses[0]["result"]["accepted"])

    def test_remove_attachment_action_targets_the_exact_id(self):
        removed = []
        server = self._server(SimpleNamespace(
            remove_attachment=lambda attachment_id: removed.append(attachment_id) or True
        ))
        result = BridgeServer._action(server, "remove_attachment", {"attachmentId": "a1"})
        self.assertEqual(removed, ["a1"])
        self.assertEqual(result, {"ok": True})

    def test_text_model_mention_keeps_pdf_and_docx_but_filters_images(self):
        session = SimpleNamespace(session_store=object())
        server = self._server(session)
        server.cwd = r"C:\workspace"
        server._mention_index = None
        candidates = [
            SimpleNamespace(
                insert_text="file:manual.pdf",
                display="manual.pdf",
                description="file",
                kind="file",
            ),
            SimpleNamespace(
                insert_text="file:brief.docx",
                display="brief.docx",
                description="file",
                kind="file",
            ),
            SimpleNamespace(
                insert_text="file:screen.png",
                display="screen.png",
                description="file",
                kind="file",
            ),
        ]

        class FakeMentionIndex:
            def __init__(self, root, store) -> None:
                pass

            def candidates(self, prefix, *, limit):
                return candidates

        with (
            patch("harness_code_agent.tui_bridge.model_input_mode", return_value="text"),
            patch("harness_code_agent.tui_bridge.MentionIndex", FakeMentionIndex),
        ):
            result = BridgeServer._action(server, "complete_mention", {"prefix": ""})

        self.assertEqual(
            [item["insertText"] for item in result["candidates"]],
            ["file:manual.pdf", "file:brief.docx"],
        )


if __name__ == "__main__":
    unittest.main()
