from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from harness_code_agent.attachments import (
    AttachmentError,
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
        server._rewind_pending = False
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

    def test_external_path_confirmation_queues_interaction_and_keeps_draft(self):
        def prepare(_submission):
            raise ExternalPathConfirmationRequired([r"C:\outside.pdf"])

        server = self._server(SimpleNamespace(prepare_submission=prepare))
        BridgeServer._handle_request(server, {
            "id": "submit-2",
            "method": "submit",
            "params": {"text": r"read C:\outside.pdf"},
        })

        # The approval runs as a worker task (the worker blocks on the
        # interaction while the stdin loop stays free). The client keeps the
        # draft and gets no bespoke confirmation payload.
        task = server._tasks.get_nowait()
        self.assertTrue(callable(task))
        self.assertIsNone(server.responses[0]["error"])
        result = server.responses[0]["result"]
        self.assertFalse(result["accepted"])
        self.assertNotIn("confirmation", result)

    def test_external_task_denial_notices_and_does_not_submit(self):
        events = []
        notices = []

        def prepare_with_approval(_submission):
            raise AttachmentError("已拒绝读取工作区外文件，本次任务未提交")

        server = self._server(SimpleNamespace(
            prepare_with_external_approval=prepare_with_approval
        ))
        server._send_event = events.append
        server._notice = lambda level, text: notices.append((level, text))
        server._enqueue_external_submission(SimpleNamespace(text="x", attachment_ids=(), authorized_paths=()))
        task = server._tasks.get_nowait()
        task()

        self.assertEqual(notices, [("info", "已拒绝读取工作区外文件，本次任务未提交")])
        self.assertEqual(events, [])
        self.assertTrue(server._tasks.empty())

    def test_external_task_approval_emits_user_row_and_queues_prepared(self):
        attachment = SimpleNamespace(public_dict=lambda: {
            "kind": "text", "name": "outside.txt", "size": 2048,
        })
        prepared = PreparedTurn("compare", (attachment,))
        events = []
        server = self._server(SimpleNamespace(
            prepare_with_external_approval=lambda _submission: prepared
        ))
        server._send_event = events.append
        server._notice = lambda level, text: None
        server._enqueue_external_submission(SimpleNamespace(text="compare", attachment_ids=(), authorized_paths=()))
        task = server._tasks.get_nowait()
        task()

        self.assertEqual(events[0]["type"], "transcript")
        self.assertEqual(events[0]["item"]["kind"], "user")
        self.assertIn("compare", events[0]["item"]["body"])
        self.assertIn("[text] outside.txt (2.0 KB)", events[0]["item"]["body"])
        # The prepared turn re-enters the normal task path.
        self.assertIs(server._tasks.get_nowait(), prepared)

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
